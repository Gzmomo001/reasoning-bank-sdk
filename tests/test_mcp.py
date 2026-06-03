"""Tests for MCP server tools — real embedding + real LLM + ChromaDB."""

import json

import pytest

from reasoning_bank_mcp import server


@pytest.fixture(autouse=True)
async def setup_mcp_bank(bank):
    server._bank_instance = bank
    yield
    server._bank_instance = None


async def test_add_and_count():
    result = await server.reasoning_bank_add("t1", ["m1"])
    assert json.loads(result)["ok"]
    assert json.loads(await server.reasoning_bank_count())["count"] == 1


async def test_list():
    await server.reasoning_bank_add("t1", ["m1"], "q1")
    await server.reasoning_bank_add("t2", ["m2"], "q2")
    assert len(json.loads(await server.reasoning_bank_list())) == 2


async def test_delete():
    result = await server.reasoning_bank_add("t1", ["m1"])
    item_id = json.loads(result)["id"]
    assert json.loads(await server.reasoning_bank_delete(item_id))["ok"]
    assert json.loads(await server.reasoning_bank_count())["count"] == 0


async def test_retrieve():
    await server.reasoning_bank_add("t1", ["use button"], "fix login")
    data = json.loads(await server.reasoning_bank_retrieve("fix login", top_k=1))
    assert len(data) >= 1
