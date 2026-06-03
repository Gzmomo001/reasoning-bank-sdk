"""Single-trajectory memory induction for ReasoningBank."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.core.prompts import get_system_prompt

if TYPE_CHECKING:
    from reasoning_bank.llm.base import LLMClient

logger = logging.getLogger(__name__)


async def induce(
    llm: LLMClient,
    query: str,
    trajectory: str,
    status: str,
    domain: str = "web",
) -> list[MemoryItem]:
    """Extract memory items from a single agent trajectory.

    Args:
        llm: LLM client for generating memory items.
        query: The user query / task objective.
        trajectory: Formatted trajectory text (think/action pairs).
        status: "success" or "fail".
        domain: "web", "coding", or "general".

    Returns:
        List of MemoryItem objects (already formatted, not yet stored).
    """
    system_prompt = get_system_prompt(status, domain)

    # Format the trajectory as the user message
    user_msg = f"**Query:** {query}\n\n**Trajectory:**\n{trajectory}"

    raw = await llm.chat(
        messages=[{"role": "user", "content": user_msg}],
        system=system_prompt,
    )

    memory_texts = _parse_memory_items(raw)

    items = []
    for _i, text in enumerate(memory_texts):
        item = MemoryItem(
            query=query,
            status=status,
            domain=domain,
            memory_items=[text],
        )
        items.append(item)

    if not items:
        logger.warning("No memory items extracted for query: %s", query)

    return items


def _parse_memory_items(raw: str) -> list[str]:
    """Parse LLM output into individual memory item texts.

    Splits on double-newline boundaries between '# Memory Item' headers.
    Falls back to splitting on '\\n\\n' if no headers found.
    """
    # Try structured parsing by '# Memory Item' markers
    parts = []
    current: list[str] = []
    found_header = False

    for line in raw.split("\n"):
        if line.strip().startswith("# Memory Item"):
            found_header = True
            if current:
                parts.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)

    if current:
        parts.append("\n".join(current).strip())

    if found_header:
        return [p for p in parts if p]

    # Fallback: split on double newline
    return [p.strip() for p in raw.split("\n\n") if p.strip()]
