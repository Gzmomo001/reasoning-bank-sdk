"""MemoryItem data model for ReasoningBank."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MemoryItem:
    """A single memory entry extracted from an agent trajectory."""

    task_id: str
    query: str
    status: str  # "success" | "fail"
    domain: str  # "web" | "coding" | "general"
    memory_items: list[str]
    template_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt_text(self) -> str:
        """Format as injectable text for agent prompt."""
        return "\n\n".join(self.memory_items)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "query": self.query,
            "status": self.status,
            "domain": self.domain,
            "memory_items": self.memory_items,
            "template_id": self.template_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryItem:
        d = dict(data)
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
