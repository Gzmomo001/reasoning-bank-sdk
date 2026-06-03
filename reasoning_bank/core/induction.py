"""Single-trajectory memory induction for ReasoningBank."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.core.parsing import parse_memory_items
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

    memory_texts = parse_memory_items(raw)

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
