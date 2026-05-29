"""Multi-trajectory parallel scaling induction for ReasoningBank."""

from __future__ import annotations

import logging

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.core.prompts import get_scaling_prompt
from reasoning_bank.llm.base import LLMClient

logger = logging.getLogger(__name__)


def induce_scaling(
    llm: LLMClient,
    task_id: str,
    query: str,
    trajectories: list[dict],
    domain: str = "web",
) -> list[MemoryItem]:
    """Extract memory items by comparing multiple trajectories.

    Args:
        llm: LLM client for generating memory items.
        task_id: Unique identifier for the task.
        query: The user query / task objective.
        trajectories: List of dicts with keys "trajectory" (str) and "status" ("success"/"fail").
        domain: "web", "coding", or "general".

    Returns:
        List of MemoryItem objects (already formatted, not yet stored).
    """
    system_prompt = get_scaling_prompt(domain)

    # Concatenate trajectories with index labels and status
    parts = [f"**Query:** {query}\n"]
    for i, traj in enumerate(trajectories):
        status = traj.get("status", "unknown")
        parts.append(f"**Trajectory {i + 1} (status: {status}):**\n{traj['trajectory']}\n")

    user_msg = "\n".join(parts)

    raw = llm.chat(
        messages=[{"role": "user", "content": user_msg}],
        system=system_prompt,
    )

    memory_texts = _parse_memory_items(raw)

    items = []
    for text in memory_texts:
        item = MemoryItem(
            task_id=task_id,
            query=query,
            status="mixed",
            domain=domain,
            memory_items=[text],
        )
        items.append(item)

    if not items:
        logger.warning("No memory items extracted from scaling induction for task %s", task_id)

    return items


def _parse_memory_items(raw: str) -> list[str]:
    """Parse LLM output into individual memory item texts."""
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

    return [p.strip() for p in raw.split("\n\n") if p.strip()]
