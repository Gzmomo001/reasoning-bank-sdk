"""ChromaDB storage backend for ReasoningBank."""

from __future__ import annotations

import logging
import os
from typing import Sequence

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "reasoning_bank"


class ChromaStorage(StorageBackend):
    """Stores memories in a ChromaDB collection.

    Connects to a standalone ChromaDB instance via CHROMA_HOST / CHROMA_PORT env vars,
    or falls back to a local persistent client at ``storage_path``.
    """

    def __init__(self, storage_path: str = "./memories") -> None:
        self._storage_path = storage_path
        self._client = self._init_client()
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_client(self):
        host = os.environ.get("CHROMA_HOST")
        port = os.environ.get("CHROMA_PORT")
        if host and port:
            import chromadb

            logger.info("Connecting to ChromaDB at %s:%s", host, port)
            return chromadb.HttpClient(host=host, port=int(port))
        else:
            import chromadb

            logger.info("Using local ChromaDB at %s", self._storage_path)
            return chromadb.PersistentClient(path=self._storage_path)

    def add(self, item: MemoryItem, embedding: list[float] | None = None) -> None:
        doc = item.to_prompt_text()
        kwargs: dict = {
            "ids": [item.id],
            "documents": [doc],
            "metadatas": [{
                "task_id": item.task_id,
                "query": item.query,
                "status": item.status,
                "domain": item.domain,
                "created_at": item.created_at.isoformat(),
                "memory_items_json": "\n\n".join(item.memory_items),
            }],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        self._collection.upsert(**kwargs)

    def add_batch(self, items: list[MemoryItem], embeddings: list[list[float]] | None = None) -> None:
        if not items:
            return
        ids, docs, metas = [], [], []
        for item in items:
            doc = item.to_prompt_text()
            ids.append(item.id)
            docs.append(doc)
            metas.append({
                "task_id": item.task_id,
                "query": item.query,
                "status": item.status,
                "domain": item.domain,
                "created_at": item.created_at.isoformat(),
                "memory_items_json": "\n\n".join(item.memory_items),
            })
        kwargs: dict = {"ids": ids, "documents": docs, "metadatas": metas}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._collection.upsert(**kwargs)

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[MemoryItem]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances"],
        )
        items: list[MemoryItem] = []
        if not results or not results.get("metadatas"):
            return items
        for meta in results["metadatas"][0]:
            items.append(self._meta_to_item(meta))
        return items

    def delete(self, task_id: str) -> None:
        # Find all items with the given task_id
        results = self._collection.get(
            where={"task_id": task_id},
            include=["metadatas"],
        )
        if results and results.get("ids"):
            self._collection.delete(ids=results["ids"])

    def list_all(self) -> list[MemoryItem]:
        results = self._collection.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return []
        return [self._meta_to_item(m) for m in results["metadatas"]]

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _meta_to_item(meta: dict) -> MemoryItem:
        from datetime import datetime

        memory_text = meta.pop("memory_items_json", "")
        memory_items = memory_text.split("\n\n") if memory_text else []
        created_at = meta.pop("created_at", None)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return MemoryItem(
            id="",  # We don't store id in metadata; task_id serves as key
            task_id=meta.get("task_id", ""),
            query=meta.get("query", ""),
            status=meta.get("status", ""),
            domain=meta.get("domain", "web"),
            memory_items=memory_items,
            created_at=created_at,
        )
