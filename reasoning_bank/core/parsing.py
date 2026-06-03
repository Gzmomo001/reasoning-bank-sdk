"""Shared LLM output parsing for ReasoningBank induction."""

from __future__ import annotations

import re

# Matches <thinking>...</thinking> blocks (non-greedy, dotall).
# Non-greedy correctly handles multiple separate thinking blocks.
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)

# Matches an unclosed <thinking> that starts at the very beginning of the string.
# Anchored with \A so literal "<thinking>" in memory content is never stripped.
_THINKING_LEADING_UNCLOSED = re.compile(r"\A\s*<thinking>.*\Z", re.DOTALL | re.IGNORECASE)

# Matches orphan </thinking> closing tags left after partial nesting removal.
_THINKING_CLOSE_ORPHAN = re.compile(r"</thinking>", re.IGNORECASE)


def strip_thinking(raw: str) -> str:
    """Remove <thinking>...</thinking> blocks from LLM output.

    Handles:
      - Multiple separate thinking blocks (via non-greedy matching)
      - Unclosed thinking at the start of the output (via ``\\A`` anchor)
      - Orphan closing tags from nested structures
      - Literal ``<thinking>`` in content is preserved (not at string start)
    """
    # Iteratively remove closed <thinking>...</thinking> blocks
    result = raw
    for _ in range(5):
        prev = result
        result = _THINKING_RE.sub("", result)
        if result == prev:
            break
    # Clean up orphan </thinking> from partial nesting removal
    result = _THINKING_CLOSE_ORPHAN.sub("", result)
    # Handle unclosed <thinking> only when it leads the entire output
    result = _THINKING_LEADING_UNCLOSED.sub("", result)
    return result.strip()


def _split_by_memory_headers(cleaned: str) -> list[str] | None:
    """Split text by '# Memory Item' headers. Returns None if no headers found."""
    parts: list[str] = []
    current: list[str] = []
    found = False

    for line in cleaned.split("\n"):
        if line.strip().startswith("# Memory Item"):
            found = True
            if current:
                parts.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)

    if current:
        parts.append("\n".join(current).strip())

    return [p for p in parts if p] if found else None


def _split_by_title_headers(cleaned: str) -> list[str] | None:
    """Split text by '## Title' sub-headers. Returns None if none found."""
    title_parts: list[str] = []
    current: list[str] = []
    found = False
    in_item = False

    for line in cleaned.split("\n"):
        if line.strip().startswith("## Title"):
            found = True
            if in_item and current:
                title_parts.append("\n".join(current).strip())
            current = [line]
            in_item = True
        elif in_item:
            current.append(line)

    if in_item and current:
        title_parts.append("\n".join(current).strip())

    return [p for p in title_parts if p] if found else None


def _split_by_double_newline(cleaned: str) -> list[str]:
    """Last-resort split on double-newline boundaries."""
    return [p.strip() for p in cleaned.split("\n\n") if p.strip()]


def parse_memory_items(raw: str) -> list[str]:
    """Parse LLM output into individual memory item texts.

    Processing pipeline:
      1. Strip ``<thinking>...</thinking>`` blocks (reasoning, not output).
      2. Try structured parsing by ``# Memory Item`` headers.
      3. Fallback: split by ``## Title`` sub-headers.
      4. Last resort: split on double-newline ``\\n\\n``.
    """
    cleaned = strip_thinking(raw)

    return _split_by_memory_headers(cleaned) or _split_by_title_headers(cleaned) or _split_by_double_newline(cleaned)
