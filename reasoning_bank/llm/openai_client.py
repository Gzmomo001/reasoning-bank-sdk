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
        self.model = model
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._base_url = base_url or os.environ.get("LLM_API_BASE_URL", "") or None

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        from openai import OpenAI  # noqa: PLC0415

        kwargs: dict = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = OpenAI(**kwargs)
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)

        response = client.chat.completions.create(
            model=self.model,
            messages=all_msgs,
            temperature=0.7,
        )
        return response.choices[0].message.content
