"""ChromaDB storage backend for ReasoningBank."""

from __future__ import annotations

import asyncio
import logging
import os

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "reasoning_bank"


class ChromaStorage(StorageBackend):
    """Stores memories in a ChromaDB collection.

    Connects to a standalone ChromaDB instance via CHROMA_HOST / CHROMA_PORT env vars
    using the native async HTTP client. Falls back to a local persistent client
    (sync, wrapped with ``asyncio.to_thread`` for non-blocking operation).

    Use ``ChromaStorage.create()`` to obtain an instance.
    """

    def __init__(self, collection, *, is_async: bool) -> None:
        """Private constructor. Use ``ChromaStorage.create()`` instead."""
        self._collection = collection
        self._is_async = is_async

    @classmethod
    async def create(cls, storage_path: str = "./memories") -> ChromaStorage:
        """Create a ChromaStorage instance.

        For remote ChromaDB (CHROMA_HOST + CHROMA_PORT), uses the synchronous
        ``HttpClient`` wrapped with ``asyncio.to_thread`` to avoid a
        ``StopIteration`` bug in chromadb's ``AsyncHttpClient`` on Python 3.12+.
        For local embedded mode, uses ``PersistentClient`` similarly wrapped.
        """
        import chromadb  # noqa: PLC0415

        host = os.environ.get("CHROMA_HOST")
        port = os.environ.get("CHROMA_PORT")

        if host and port:
            logger.info("Connecting to ChromaDB at %s:%s (sync+to_thread)", host, port)
            client = chromadb.HttpClient(host=host, port=int(port))
            try:
                collection = await asyncio.to_thread(
                    client.get_or_create_collection,
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                msg = (
                    f"Failed to connect to ChromaDB at {host}:{port}. "
                    f"Ensure the ChromaDB server is running and reachable."
                )
                raise RuntimeError(msg) from exc
            return cls(collection=collection, is_async=False)

        logger.info("Using local ChromaDB at %s (sync+to_thread)", storage_path)
        client = chromadb.PersistentClient(path=storage_path)
        collection = await asyncio.to_thread(
            client.get_or_create_collection,
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return cls(collection=collection, is_async=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call(self, fn, *args, **kwargs):
        """Dispatch to await (async) or asyncio.to_thread (sync)."""
        if self._is_async:
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    async def add(self, item: MemoryItem, embedding: list[float] | None = None) -> None:
        doc = item.to_prompt_text()
        kwargs: dict = {
            "ids": [item.id],
            "documents": [doc],
            "metadatas": [
                {
                    "query": item.query,
                    "status": item.status,
                    "domain": item.domain,
                    "created_at": item.created_at.isoformat(),
                    "memory_items_json": "\n\n".join(item.memory_items),
                }
            ],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        await self._call(self._collection.upsert, **kwargs)

    async def add_batch(self, items: list[MemoryItem], embeddings: list[list[float]] | None = None) -> None:
        if not items:
            return
        ids, docs, metas = [], [], []
        for item in items:
            doc = item.to_prompt_text()
            ids.append(item.id)
            docs.append(doc)
            metas.append(
                {
                    "query": item.query,
                    "status": item.status,
                    "domain": item.domain,
                    "created_at": item.created_at.isoformat(),
                    "memory_items_json": "\n\n".join(item.memory_items),
                }
            )
        kwargs: dict = {"ids": ids, "documents": docs, "metadatas": metas}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        await self._call(self._collection.upsert, **kwargs)

    async def retrieve(self, query_embedding: list[float], top_k: int) -> list[MemoryItem]:
        cnt = await self._call(self._collection.count)
        if cnt == 0:
            return []
        results = await self._call(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=min(top_k, cnt),
            include=["metadatas", "distances"],
        )
        items: list[MemoryItem] = []
        if not results or not results.get("metadatas"):
            return items
        ids = results.get("ids", [[]])[0]
        for i, meta in enumerate(results["metadatas"][0]):
            items.append(self._meta_to_item(meta, ids[i] if i < len(ids) else ""))
        return items

    async def delete(self, item_id: str) -> None:
        """Delete a single memory item by its ID."""
        await self._call(self._collection.delete, ids=[item_id])

    async def list_all(self) -> list[MemoryItem]:
        results = await self._call(self._collection.get, include=["metadatas"])
        if not results or not results.get("metadatas"):
            return []
        ids = results.get("ids", [])
        return [self._meta_to_item(m, ids[i] if i < len(ids) else "") for i, m in enumerate(results["metadatas"])]

    async def count(self) -> int:
        return await self._call(self._collection.count)

    @staticmethod
    def _meta_to_item(meta: dict, item_id: str = "") -> MemoryItem:
        from datetime import datetime  # noqa: PLC0415

        memory_text = meta.get("memory_items_json", "")
        memory_items = memory_text.split("\n\n") if memory_text else []
        created_at = meta.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return MemoryItem(
            id=item_id,
            query=meta.get("query", ""),
            status=meta.get("status", ""),
            domain=meta.get("domain", "web"),
            memory_items=memory_items,
            created_at=created_at,
        )
