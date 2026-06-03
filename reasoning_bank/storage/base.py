"""Abstract storage backend for ReasoningBank."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_bank.core.memory_item import MemoryItem


class StorageBackend(ABC):
    """All storage backends must implement this interface."""

    @abstractmethod
    async def add(self, item: MemoryItem) -> None: ...

    @abstractmethod
    async def add_batch(self, items: list[MemoryItem], *, embeddings: list[list[float]] | None = None) -> None: ...

    @abstractmethod
    async def retrieve(self, query_embedding: list[float], top_k: int) -> list[MemoryItem]: ...

    @abstractmethod
    async def delete(self, item_id: str) -> None: ...

    @abstractmethod
    async def list_all(self) -> list[MemoryItem]: ...

    @abstractmethod
    async def count(self) -> int: ...
