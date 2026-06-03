"""Shared fixtures for integration tests using real model APIs."""

import asyncio
import os

import pytest
from dotenv import load_dotenv

from reasoning_bank.core.rate_limiter import RateLimiter, get_llm_rpm

load_dotenv()

_llm_limiter = RateLimiter(get_llm_rpm())


def make_llm():
    """Create a real LLM client from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "")
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        pytest.skip("LLM_MODEL not set in environment")
    if provider == "anthropic":
        from reasoning_bank.llm.anthropic_client import AnthropicClient  # noqa: PLC0415

        return AnthropicClient(model=model)
    if provider in ("vertexai", "google_ai"):
        from reasoning_bank.llm.gemini_client import GeminiClient  # noqa: PLC0415

        return GeminiClient(model=model)
    from reasoning_bank.llm.openai_client import OpenAIClient  # noqa: PLC0415

    return OpenAIClient(model=model)


@pytest.fixture(scope="session")
def llm():
    return make_llm()


@pytest.fixture
async def bank(llm_with_retry, tmp_path):
    """MemoryBank with real embedding + real LLM (with retry) + ChromaDB in a temp directory."""
    from reasoning_bank import MemoryBank  # noqa: PLC0415

    return await MemoryBank.create(
        storage="chroma",
        storage_path=str(tmp_path / "memories"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "gemini"),
        embedding_model=os.environ.get("EMBEDDING_MODEL") or None,
        llm_client=llm_with_retry,
    )


@pytest.fixture(scope="session")
def llm_with_retry():
    """LLM client wrapper that retries on transient server errors (5xx) with rate limiting."""

    class _RetryLLM:
        def __init__(self, inner, max_retries=3, delay=5):
            self._inner = inner
            self._max_retries = max_retries
            self._delay = delay

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def chat(self, messages, system=None):
            last_err = None
            for attempt in range(self._max_retries):
                await _llm_limiter.wait()
                try:
                    return await self._inner.chat(messages, system=system)
                except Exception as e:
                    err_str = str(e)
                    if "500" in err_str or "502" in err_str or "503" in err_str or "429" in err_str:
                        last_err = e
                        if attempt < self._max_retries - 1:
                            await asyncio.sleep(self._delay)
                            continue
                    raise
            raise last_err  # type: ignore[misc]

    return _RetryLLM(make_llm())
