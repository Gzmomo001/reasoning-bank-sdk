"""Abstract storage backend for ReasoningBank."""

from __future__ import annotations

from abc import ABC, abstractmethod

from reasoning_bank.core.memory_item import MemoryItem


class StorageBackend(ABC):
    """All storage backends must implement this interface."""

    @abstractmethod
    def add(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def add_batch(self, items: list[MemoryItem]) -> None: ...

    @abstractmethod
    def retrieve(self, query_embedding: list[float], top_k: int) -> list[MemoryItem]: ...

    @abstractmethod
    def delete(self, task_id: str) -> None: ...

    @abstractmethod
    def list_all(self) -> list[MemoryItem]: ...

    @abstractmethod
    def count(self) -> int: ...
