"""Tests for MCP server tools — real embedding + real LLM + ChromaDB."""

import json

import pytest

from reasoning_bank_mcp import server


@pytest.fixture(autouse=True)
def setup_mcp_bank(bank):
    server._bank_instance = bank
    yield
    server._bank_instance = None


def test_add_and_count():
    result = server.reasoning_bank_add("t1", "q1", ["m1"])
    assert json.loads(result)["ok"]
    assert json.loads(server.reasoning_bank_count())["count"] == 1


def test_list():
    server.reasoning_bank_add("t1", "q1", ["m1"])
    server.reasoning_bank_add("t2", "q2", ["m2"])
    assert len(json.loads(server.reasoning_bank_list())) == 2


def test_delete():
    server.reasoning_bank_add("t1", "q1", ["m1"])
    assert json.loads(server.reasoning_bank_delete("t1"))["ok"]
    assert json.loads(server.reasoning_bank_count())["count"] == 0


def test_retrieve():
    server.reasoning_bank_add("t1", "fix login", ["use button"])
    data = json.loads(server.reasoning_bank_retrieve("fix login", top_k=1))
    assert len(data) >= 1
