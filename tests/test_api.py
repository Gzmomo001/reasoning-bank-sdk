"""Tests for FastAPI routes — real embedding + real LLM + ChromaDB."""

import httpx
import pytest

import reasoning_bank_api.routes as routes_mod
from reasoning_bank_api.app import app


@pytest.fixture
async def client(bank):
    routes_mod._bank = bank
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    routes_mod._bank = None


async def test_create_and_count(client):
    resp = await client.post(
        "/v1/memory/items",
        json={
            "query": "q1",
            "memory_items": ["m1"],
            "status": "success",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "data" in body
    assert body["data"]["id"]
    count_resp = await client.get("/v1/memory/items/count")
    assert count_resp.json()["data"]["count"] == 1


async def test_list(client):
    await client.post(
        "/v1/memory/items",
        json={
            "query": "q1",
            "memory_items": ["m1"],
        },
    )
    await client.post(
        "/v1/memory/items",
        json={
            "query": "q2",
            "memory_items": ["m2"],
        },
    )
    body = (await client.get("/v1/memory/items")).json()
    assert len(body["data"]) == 2
    assert body["meta"]["total"] == 2


async def test_delete(client):
    resp = await client.post(
        "/v1/memory/items",
        json={
            "query": "q1",
            "memory_items": ["m1"],
        },
    )
    item_id = resp.json()["data"]["id"]
    resp = await client.delete(f"/v1/memory/items/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] is True
    count_resp = await client.get("/v1/memory/items/count")
    assert count_resp.json()["data"]["count"] == 0


async def test_search_get(client):
    await client.post(
        "/v1/memory/items",
        json={
            "query": "fix login",
            "memory_items": ["use button"],
        },
    )
    resp = await client.get("/v1/memory/items/search", params={"query": "fix login", "top_k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert len(body["data"]) >= 1


async def test_search_post(client):
    await client.post(
        "/v1/memory/items",
        json={
            "query": "fix login",
            "memory_items": ["use button"],
        },
    )
    resp = await client.post("/v1/memory/items/search", json={"query": "fix login", "top_k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert len(body["data"]) >= 1


async def test_response_format(client):
    """Verify unified ApiResponse wrapper on all endpoints."""
    # POST /items → 201 with data
    resp = await client.post(
        "/v1/memory/items",
        json={
            "query": "q1",
            "memory_items": ["m1"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "data" in body
    assert "meta" in body
    item_id = body["data"]["id"]

    # GET /items → 200 with data + meta.total
    resp = await client.get("/v1/memory/items")
    body = resp.json()
    assert "data" in body
    assert body["meta"]["total"] >= 1

    # GET /items/count → 200 with data.count
    resp = await client.get("/v1/memory/items/count")
    body = resp.json()
    assert "data" in body
    assert isinstance(body["data"]["count"], int)

    # DELETE /items/{item_id} → 200 with data.deleted
    resp = await client.delete(f"/v1/memory/items/{item_id}")
    body = resp.json()
    assert "data" in body
    assert body["data"]["deleted"] is True
