"""Embedding provider abstract base and implementations."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """All embedding providers must implement this interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def dimension(self) -> int: ...


class GeminiEmbedding(EmbeddingProvider):
    """Google Gemini embedding via google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 3072,
    ) -> None:
        self._model = model
        self._dim = output_dimensionality
        self._client = self._init_client()

    def _init_client(self):
        from google import genai
        from google.genai.types import HttpOptions

        api_key = os.environ.get("LLM_API_KEY", "")
        provider = os.environ.get("LLM_PROVIDER", "")
        if provider == "google_ai" and api_key:
            return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
        return genai.Client(vertexai=True, http_options=HttpOptions(api_version="v1"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig

        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dim,
            ),
        )
        return [item.values for item in response.embeddings]

    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding via openai SDK."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._base_url = base_url or os.environ.get("LLM_API_BASE_URL", "") or None
        self._dim = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        kwargs: dict = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = OpenAI(**kwargs)
        response = client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        return self._dim


class CustomEmbedding(EmbeddingProvider):
    """Wrap any callable as an embedding provider."""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]], dim: int = 1536) -> None:
        self._fn = fn
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def dimension(self) -> int:
        return self._dim
