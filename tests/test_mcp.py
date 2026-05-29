"""Tests for MCP server tools."""

import json
import tempfile

from reasoning_bank import MemoryBank
from reasoning_bank.core.embedding import CustomEmbedding
from reasoning_bank_mcp import server


def _dummy_embed(texts):
    return [[1.0] * 8 for _ in texts]


def _setup_bank(tmp: str) -> MemoryBank:
    bank = MemoryBank(
        storage="jsonl",
        storage_path=tmp,
        embedding_provider=CustomEmbedding(_dummy_embed, dim=8),
    )
    server._bank_instance = bank
    return bank


def test_add_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        _setup_bank(tmp)

        result = server.reasoning_bank_add("t1", "q1", ["m1"])
        data = json.loads(result)
        assert data["ok"]

        result = server.reasoning_bank_count()
        data = json.loads(result)
        assert data["count"] == 1


def test_list():
    with tempfile.TemporaryDirectory() as tmp:
        _setup_bank(tmp)

        server.reasoning_bank_add("t1", "q1", ["m1"])
        server.reasoning_bank_add("t2", "q2", ["m2"])

        result = server.reasoning_bank_list()
        data = json.loads(result)
        assert len(data) == 2


def test_delete():
    with tempfile.TemporaryDirectory() as tmp:
        _setup_bank(tmp)

        server.reasoning_bank_add("t1", "q1", ["m1"])
        result = server.reasoning_bank_delete("t1")
        data = json.loads(result)
        assert data["ok"]

        result = server.reasoning_bank_count()
        assert json.loads(result)["count"] == 0


def test_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        _setup_bank(tmp)

        server.reasoning_bank_add("t1", "fix login", ["use button"])
        result = server.reasoning_bank_retrieve("fix login", top_k=1)
        data = json.loads(result)
        assert len(data) >= 1
