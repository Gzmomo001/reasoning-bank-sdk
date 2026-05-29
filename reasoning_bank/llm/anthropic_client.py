"""Anthropic Claude LLM client for ReasoningBank."""

from __future__ import annotations

import os

from reasoning_bank.llm.base import LLMClient


class AnthropicClient(LLMClient):
    """Anthropic Claude chat client."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._base_url = base_url or os.environ.get("LLM_API_BASE_URL", "") or None

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        from anthropic import Anthropic

        kwargs: dict = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = Anthropic(**kwargs)
        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
            "temperature": 0.7,
        }
        if system:
            create_kwargs["system"] = system

        message = client.messages.create(**create_kwargs)
        return message.content[0].text
