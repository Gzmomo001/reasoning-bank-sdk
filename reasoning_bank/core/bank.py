"""MemoryBank — main entry point for the ReasoningBank SDK."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from reasoning_bank.core.embedding import (
    EmbeddingProvider,
    GeminiEmbedding,
    OpenAIEmbedding,
)
from reasoning_bank.core.induction import induce
from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.core.rate_limiter import RateLimiter, get_embedding_rpm, get_llm_rpm
from reasoning_bank.core.scaling import induce_scaling
from reasoning_bank.storage.chroma import ChromaStorage
from reasoning_bank.storage.jsonl import JsonlStorage

if TYPE_CHECKING:
    from reasoning_bank.llm.base import LLMClient
    from reasoning_bank.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_EMBEDDING_MAP: dict[str, type[EmbeddingProvider]] = {
    "gemini": GeminiEmbedding,
    "openai": OpenAIEmbedding,
}


class MemoryBank:
    """Main interface for ReasoningBank persistent agent memory.

    Usage::

        bank = await MemoryBank.create(
            storage="chroma",
            storage_path="./memories",
            embedding_provider="gemini",
            embedding_model="gemini-embedding-001",
            llm_client=OpenAIClient(api_key="..."),
        )

        # Retrieve relevant memories
        memories = await bank.retrieve(query="fix login bug", top_k=3)

        # Induce memories from a trajectory
        items = await bank.induce(
            query="Navigate to shopping cart",
            trajectory="...",
            status="success",
            domain="web",
        )
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedding: EmbeddingProvider,
        llm: LLMClient | None,
        llm_limiter: RateLimiter,
        embedding_limiter: RateLimiter,
    ) -> None:
        """Private constructor. Use ``MemoryBank.create()`` instead."""
        self._storage = storage
        self._embedding = embedding
        self._llm = llm
        self._llm_limiter = llm_limiter
        self._embedding_limiter = embedding_limiter

    @classmethod
    async def create(
        cls,
        storage: str = "chroma",
        storage_path: str = "./memories",
        embedding_provider: str | EmbeddingProvider = "gemini",
        embedding_model: str | None = None,
        llm_client: LLMClient | None = None,
    ) -> MemoryBank:
        """Create and initialize a MemoryBank instance."""
        storage_backend = await cls._init_storage(storage, storage_path)
        embedding = cls._init_embedding(embedding_provider, embedding_model)
        llm_limiter = RateLimiter(get_llm_rpm())
        embedding_limiter = RateLimiter(get_embedding_rpm())
        return cls(
            storage=storage_backend,
            embedding=embedding,
            llm=llm_client,
            llm_limiter=llm_limiter,
            embedding_limiter=embedding_limiter,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        """Retrieve the top-k most relevant memories for a query."""
        await self._embedding_limiter.wait()
        embeddings = await self._embedding.embed([query])
        return await self._storage.retrieve(embeddings[0], top_k)

    async def induce(
        self,
        query: str,
        trajectory: str,
        status: str,
        domain: str = "web",
    ) -> list[MemoryItem]:
        """Extract memory items from a single trajectory and store them."""
        self._require_llm()
        await self._llm_limiter.wait()
        items = await induce(
            llm=self._llm,
            query=query,
            trajectory=trajectory,
            status=status,
            domain=domain,
        )
        await self._store_with_embeddings(items)
        return items

    async def induce_scaling(
        self,
        query: str,
        trajectories: list[dict],
        domain: str = "web",
    ) -> list[MemoryItem]:
        """Extract memory items by comparing multiple trajectories and store them."""
        self._require_llm()
        await self._llm_limiter.wait()
        items = await induce_scaling(
            llm=self._llm,
            query=query,
            trajectories=trajectories,
            domain=domain,
        )
        await self._store_with_embeddings(items)
        return items

    async def add(
        self,
        query: str,
        memory_items: list[str],
        status: str = "success",
        domain: str = "general",
    ) -> MemoryItem:
        """Directly add a memory item (no LLM induction)."""
        item = MemoryItem(
            query=query,
            status=status,
            domain=domain,
            memory_items=memory_items,
        )
        await self._store_with_embeddings([item])
        return item

    async def delete(self, item_id: str) -> None:
        """Delete a memory item by its ID."""
        await self._storage.delete(item_id)

    async def list(self) -> list[MemoryItem]:
        """List all stored memory items."""
        return await self._storage.list_all()

    async def count(self) -> int:
        """Return the total number of stored memory items."""
        return await self._storage.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _store_with_embeddings(self, items: list[MemoryItem]) -> None:
        """Compute embeddings and persist items."""
        if not items:
            return
        texts = [item.to_prompt_text() for item in items]
        await self._embedding_limiter.wait()
        embeddings = await self._embedding.embed(texts)

        if isinstance(self._storage, JsonlStorage):
            await self._storage.add_batch(items)
            for item, emb in zip(items, embeddings, strict=False):
                await self._storage.store_embedding(item.id, emb)
        else:
            await self._storage.add_batch(items, embeddings=embeddings)

    def _require_llm(self) -> None:
        if self._llm is None:
            msg = "An LLM client is required for induction. Pass llm_client= to MemoryBank constructor."
            raise ValueError(msg)

    @staticmethod
    async def _init_storage(backend: str, path: str) -> StorageBackend:
        if backend == "chroma":
            return await ChromaStorage.create(storage_path=path)
        if backend == "jsonl":
            return JsonlStorage(storage_path=path)
        msg = f"Unknown storage backend: {backend!r}. Use 'chroma' or 'jsonl'."
        raise ValueError(msg)

    @staticmethod
    def _init_embedding(
        provider: str | EmbeddingProvider,
        model: str | None,
    ) -> EmbeddingProvider:
        if isinstance(provider, EmbeddingProvider):
            return provider

        cls = _EMBEDDING_MAP.get(provider)
        if cls is None:
            msg = (
                f"Unknown embedding provider: {provider!r}. "
                f"Use one of: {', '.join(_EMBEDDING_MAP)} or pass an EmbeddingProvider instance."
            )
            raise ValueError(msg)
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        elif provider == "gemini":
            kwargs["model"] = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
        elif provider == "openai":
            kwargs["model"] = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        return cls(**kwargs)
