"""Token-bucket rate limiter — zero external dependencies."""

from __future__ import annotations

import asyncio
import os


class RateLimiter:
    """Token-bucket rate limiter for API calls.

    Args:
        rpm: Maximum requests per minute. 0 means unlimited.

    Usage::

        limiter = RateLimiter(rpm=15)
        await limiter.wait()  # blocks until a token is available
    """

    def __init__(self, rpm: float) -> None:
        self._rpm = rpm
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last_call: float | None = None

    async def wait(self) -> float:
        """Block until a request can be made. Returns seconds waited."""
        if self._rpm <= 0:
            return 0.0
        async with self._lock:
            now = _monotonic()
            if self._last_call is not None:
                elapsed = now - self._last_call
                if elapsed < self._min_interval:
                    sleep_time = self._min_interval - elapsed
                    await asyncio.sleep(sleep_time)
                    self._last_call = _monotonic()
                    return sleep_time
            self._last_call = now
            return 0.0


def _monotonic() -> float:
    """Return a monotonic clock value. Wrapped for testability."""
    import time  # noqa: PLC0415

    return time.monotonic()


def get_llm_rpm() -> float:
    """Read LLM_RPM from environment. Returns 0 (unlimited) if not set."""
    return float(os.environ.get("LLM_RPM", "0"))


def get_embedding_rpm() -> float:
    """Read EMBEDDING_RPM from environment. Returns 0 (unlimited) if not set."""
    return float(os.environ.get("EMBEDDING_RPM", "0"))
