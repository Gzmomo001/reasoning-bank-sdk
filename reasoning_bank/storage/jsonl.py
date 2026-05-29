"""JSONL storage backend for ReasoningBank — compatible with existing data format."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Sequence

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class JsonlStorage(StorageBackend):
    """Stores memories in JSONL files.

    Compatible with the existing WebArena JSONL format. Uses a separate
    embedding cache file for vector retrieval.
    """

    def __init__(self, storage_path: str = "./memories") -> None:
        self._data_path = os.path.join(storage_path, "memories.jsonl")
        self._embed_path = os.path.join(storage_path, "embeddings.jsonl")
        os.makedirs(storage_path, exist_ok=True)
        # Touch files if they don't exist
        for p in (self._data_path, self._embed_path):
            if not os.path.exists(p):
                open(p, "w").close()

    def add(self, item: MemoryItem) -> None:
        with open(self._data_path, "a") as f:
            f.write(json.dumps(item.to_dict()) + "\n")

    def add_batch(self, items: list[MemoryItem]) -> None:
        with open(self._data_path, "a") as f:
            for item in items:
                f.write(json.dumps(item.to_dict()) + "\n")

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[MemoryItem]:
        all_items = self.list_all()
        if not all_items:
            return []

        # Load embeddings
        embeddings_map = self._load_embeddings()
        scored: list[tuple[float, MemoryItem]] = []
        for item in all_items:
            emb = embeddings_map.get(item.task_id)
            if emb is None:
                continue
            sim = self._cosine_similarity(query_embedding, emb)
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def delete(self, task_id: str) -> None:
        items = self.list_all()
        remaining = [item for item in items if item.task_id != task_id]
        with open(self._data_path, "w") as f:
            for item in remaining:
                f.write(json.dumps(item.to_dict()) + "\n")

    def list_all(self) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        if not os.path.exists(self._data_path):
            return items
        with open(self._data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(MemoryItem.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return items

    def count(self) -> int:
        return len(self.list_all())

    def store_embedding(self, task_id: str, embedding: list[float]) -> None:
        record = {"id": task_id, "embedding": embedding}
        with open(self._embed_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load_embeddings(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        if not os.path.exists(self._embed_path):
            return result
        with open(self._embed_path, "r") as f:
            for line in f:
                line = line.strip()
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
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
