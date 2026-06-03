"""Embedding provider abstract base and implementations."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """All embedding providers must implement this interface."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

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
        from google import genai  # noqa: PLC0415
        from google.genai.types import HttpOptions  # noqa: PLC0415

        api_key = os.environ.get("EMBEDDING_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
        use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")

        if not use_vertexai and api_key:
            return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
        return genai.Client(vertexai=True, http_options=HttpOptions(api_version="v1"))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig  # noqa: PLC0415

        response = await self._client.aio.models.embed_content(
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
        from openai import AsyncOpenAI  # noqa: PLC0415

        self._model = model
        kwargs: dict = {}
        resolved_key = api_key or os.environ.get("LLM_API_KEY", "")
        resolved_url = base_url or os.environ.get("LLM_API_BASE_URL", "") or None
        if resolved_key:
            kwargs["api_key"] = resolved_key
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self._client = AsyncOpenAI(**kwargs)
        self._dim = 1536

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        return self._dim


class CustomEmbedding(EmbeddingProvider):
    """Wrap any async callable as an embedding provider."""

    def __init__(self, fn: Callable[[list[str]], Awaitable[list[list[float]]]], dim: int = 1536) -> None:
        self._fn = fn
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._fn(texts)

    def dimension(self) -> int:
        return self._dim
