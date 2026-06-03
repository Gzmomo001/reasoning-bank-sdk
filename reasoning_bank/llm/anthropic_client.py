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
        from anthropic import AsyncAnthropic  # noqa: PLC0415

        self.model = model
        kwargs: dict = {}
        resolved_key = api_key or os.environ.get("LLM_API_KEY", "")
        resolved_url = base_url or os.environ.get("LLM_API_BASE_URL", "") or None
        if resolved_key:
            kwargs["api_key"] = resolved_key
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self._client = AsyncAnthropic(**kwargs)

    async def chat(self, messages: list[dict], system: str | None = None) -> str:
        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
            "temperature": 0.7,
        }
        if system:
            create_kwargs["system"] = system

        message = await self._client.messages.create(**create_kwargs)
        return message.content[0].text
