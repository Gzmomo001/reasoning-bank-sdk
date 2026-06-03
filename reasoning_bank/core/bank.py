"""MemoryBank — main entry point for the ReasoningBank SDK."""

from __future__ import annotations

import logging
import os
from typing import Union

from reasoning_bank.core.embedding import (
    CustomEmbedding,
    EmbeddingProvider,
    GeminiEmbedding,
    OpenAIEmbedding,
)
from reasoning_bank.core.induction import induce
from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.core.rate_limiter import RateLimiter, get_embedding_rpm, get_llm_rpm
from reasoning_bank.core.scaling import induce_scaling
from reasoning_bank.llm.anthropic_client import AnthropicClient
from reasoning_bank.llm.base import LLMClient
from reasoning_bank.llm.gemini_client import GeminiClient
from reasoning_bank.llm.openai_client import OpenAIClient
from reasoning_bank.storage.base import StorageBackend
from reasoning_bank.storage.chroma import ChromaStorage
from reasoning_bank.storage.jsonl import JsonlStorage

logger = logging.getLogger(__name__)

_EMBEDDING_MAP: dict[str, type[EmbeddingProvider]] = {
    "gemini": GeminiEmbedding,
    "openai": OpenAIEmbedding,
}


class MemoryBank:
    """Main interface for ReasoningBank persistent agent memory.

    Usage::

        bank = MemoryBank(
            storage="chroma",
            storage_path="./memories",
            embedding_provider="gemini",
            embedding_model="gemini-embedding-001",
            llm_client=OpenAIClient(api_key="..."),
        )

        # Retrieve relevant memories
        memories = bank.retrieve(query="fix login bug", top_k=3)

        # Induce memories from a trajectory
        items = bank.induce(
            query="Navigate to shopping cart",
            trajectory="...",
            status="success",
            domain="web",
        )
    """

    def __init__(
        self,
        storage: str = "chroma",
        storage_path: str = "./memories",
        embedding_provider: Union[str, EmbeddingProvider] = "gemini",
        embedding_model: str | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._storage = self._init_storage(storage, storage_path)
        self._embedding = self._init_embedding(embedding_provider, embedding_model)
        self._llm = llm_client
        self._llm_limiter = RateLimiter(get_llm_rpm())
        self._embedding_limiter = RateLimiter(get_embedding_rpm())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        """Retrieve the top-k most relevant memories for a query."""
        self._embedding_limiter.wait()
        embeddings = self._embedding.embed([query])
        return self._storage.retrieve(embeddings[0], top_k)

    def induce(
        self,
        query: str,
        trajectory: str,
        status: str,
        domain: str = "web",
    ) -> list[MemoryItem]:
        """Extract memory items from a single trajectory and store them."""
        self._require_llm()
        self._llm_limiter.wait()
        items = induce(
            llm=self._llm,
            query=query,
            trajectory=trajectory,
            status=status,
            domain=domain,
        )
        self._store_with_embeddings(items)
        return items

    def induce_scaling(
        self,
        query: str,
        trajectories: list[dict],
        domain: str = "web",
    ) -> list[MemoryItem]:
        """Extract memory items by comparing multiple trajectories and store them."""
        self._require_llm()
        self._llm_limiter.wait()
        items = induce_scaling(
            llm=self._llm,
            query=query,
            trajectories=trajectories,
            domain=domain,
        )
        self._store_with_embeddings(items)
        return items

    def add(
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
        self._store_with_embeddings([item])
        return item

    def delete(self, item_id: str) -> None:
        """Delete a memory item by its ID."""
        self._storage.delete(item_id)

    def list(self) -> list[MemoryItem]:
        """List all stored memory items."""
        return self._storage.list_all()

    def count(self) -> int:
        """Return the total number of stored memory items."""
        return self._storage.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store_with_embeddings(self, items: list[MemoryItem]) -> None:
        """Compute embeddings and persist items."""
        if not items:
            return
        texts = [item.to_prompt_text() for item in items]
        self._embedding_limiter.wait()
        embeddings = self._embedding.embed(texts)

        if isinstance(self._storage, JsonlStorage):
            self._storage.add_batch(items)
            for item, emb in zip(items, embeddings):
                self._storage.store_embedding(item.id, emb)
        else:
            self._storage.add_batch(items, embeddings=embeddings)

    def _require_llm(self) -> None:
        if self._llm is None:
            raise ValueError(
                "An LLM client is required for induction. "
                "Pass llm_client= to MemoryBank constructor."
            )

    @staticmethod
    def _init_storage(backend: str, path: str) -> StorageBackend:
        if backend == "chroma":
            return ChromaStorage(storage_path=path)
        elif backend == "jsonl":
            return JsonlStorage(storage_path=path)
        raise ValueError(f"Unknown storage backend: {backend!r}. Use 'chroma' or 'jsonl'.")

    @staticmethod
    def _init_embedding(
        provider: Union[str, EmbeddingProvider],
        model: str | None,
    ) -> EmbeddingProvider:
        if isinstance(provider, EmbeddingProvider):
            return provider

        cls = _EMBEDDING_MAP.get(provider)
        if cls is None:
            raise ValueError(
                f"Unknown embedding provider: {provider!r}. "
                f"Use one of: {', '.join(_EMBEDDING_MAP)} or pass an EmbeddingProvider instance."
            )
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        elif provider == "gemini":
            kwargs["model"] = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
        elif provider == "openai":
            kwargs["model"] = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        return cls(**kwargs)
