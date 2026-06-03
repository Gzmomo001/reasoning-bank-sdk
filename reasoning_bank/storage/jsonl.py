"""JSONL storage backend for ReasoningBank — compatible with existing data format."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class JsonlStorage(StorageBackend):
    """Stores memories in JSONL files.

    Compatible with the existing WebArena JSONL format. Uses a separate
    embedding cache file for vector retrieval.
    """

    def __init__(self, storage_path: str = "./memories") -> None:
        self._dir = Path(storage_path)
        self._data_path = self._dir / "memories.jsonl"
        self._embed_path = self._dir / "embeddings.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)
        # Touch files if they don't exist
        for p in (self._data_path, self._embed_path):
            if not p.exists():
                p.touch()

    async def add(self, item: MemoryItem) -> None:
        import aiofiles  # noqa: PLC0415

        async with aiofiles.open(self._data_path, "a") as f:
            await f.write(json.dumps(item.to_dict()) + "\n")

    async def add_batch(self, items: list[MemoryItem]) -> None:
        import aiofiles  # noqa: PLC0415

        async with aiofiles.open(self._data_path, "a") as f:
            await f.writelines(json.dumps(item.to_dict()) + "\n" for item in items)

    async def retrieve(self, query_embedding: list[float], top_k: int) -> list[MemoryItem]:
        all_items = await self.list_all()
        if not all_items:
            return []

        # Load embeddings
        embeddings_map = await self._load_embeddings()
        scored: list[tuple[float, MemoryItem]] = []
        for item in all_items:
            emb = embeddings_map.get(item.id)
            if emb is None:
                continue
            sim = self._cosine_similarity(query_embedding, emb)
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def delete(self, item_id: str) -> None:
        import aiofiles  # noqa: PLC0415

        items = await self.list_all()
        remaining = [item for item in items if item.id != item_id]
        async with aiofiles.open(self._data_path, "w") as f:
            await f.writelines(json.dumps(item.to_dict()) + "\n" for item in remaining)

    async def list_all(self) -> list[MemoryItem]:
        import aiofiles  # noqa: PLC0415

        items: list[MemoryItem] = []
        if not self._data_path.exists():
            return items
        async with aiofiles.open(self._data_path) as f:
            async for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    items.append(MemoryItem.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return items

    async def count(self) -> int:
        return len(await self.list_all())

    async def store_embedding(self, item_id: str, embedding: list[float]) -> None:
        import aiofiles  # noqa: PLC0415

        record = {"id": item_id, "embedding": embedding}
        async with aiofiles.open(self._embed_path, "a") as f:
            await f.write(json.dumps(record) + "\n")

    async def _load_embeddings(self) -> dict[str, list[float]]:
        import aiofiles  # noqa: PLC0415

        result: dict[str, list[float]] = {}
        if not self._embed_path.exists():
            return result
        async with aiofiles.open(self._embed_path) as f:
            async for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    result[obj["id"]] = obj["embedding"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
