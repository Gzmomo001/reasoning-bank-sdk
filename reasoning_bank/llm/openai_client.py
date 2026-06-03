"""OpenAI LLM client for ReasoningBank."""

from __future__ import annotations

import os

from reasoning_bank.llm.base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat client."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI  # noqa: PLC0415

        self.model = model
        kwargs: dict = {}
        resolved_key = api_key or os.environ.get("LLM_API_KEY", "")
        resolved_url = base_url or os.environ.get("LLM_API_BASE_URL", "") or None
        if resolved_key:
            kwargs["api_key"] = resolved_key
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self._client = AsyncOpenAI(**kwargs)

    async def chat(self, messages: list[dict], system: str | None = None) -> str:
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=all_msgs,
            temperature=0.7,
        )
        return response.choices[0].message.content
