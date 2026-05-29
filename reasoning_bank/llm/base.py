"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """All LLM clients must implement this interface."""

    @abstractmethod
    def chat(self, messages: list[dict], system: str | None = None) -> str: ...
