"""Google Gemini LLM client for ReasoningBank."""

from __future__ import annotations

import os

from reasoning_bank.llm.base import LLMClient


class GeminiClient(LLMClient):
    """Google Gemini chat client via google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._client = self._init_client()

    def _init_client(self):
        from google import genai
        from google.genai.types import HttpOptions

        provider = os.environ.get("LLM_PROVIDER", "")
        if provider == "google_ai" and self._api_key:
            return genai.Client(api_key=self._api_key, http_options=HttpOptions(api_version="v1"))
        return genai.Client(vertexai=True, http_options=HttpOptions(api_version="v1"))

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        from google.genai.types import GenerateContentConfig

        # Extract user content from messages
        user_content = "\n".join(m.get("content", "") for m in messages)

        config = GenerateContentConfig(temperature=0.7)
        if system:
            config.system_instruction = system

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=config,
        )
        return response.text
