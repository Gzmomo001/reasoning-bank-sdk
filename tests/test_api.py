"""Tests for FastAPI routes — real embedding + real LLM + ChromaDB."""

import pytest
from fastapi.testclient import TestClient

import reasoning_bank_api.routes as routes_mod
from reasoning_bank_api.app import app


@pytest.fixture
def client(bank):
    routes_mod._bank = bank
    with TestClient(app) as c:
        yield c
    routes_mod._bank = None


def test_create_and_count(client):
    resp = client.post(
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
    assert client.get("/v1/memory/items/count").json()["data"]["count"] == 1


def test_list(client):
    client.post(
        "/v1/memory/items",
        json={
            "query": "q1",
            "memory_items": ["m1"],
        },
    )
    client.post(
        "/v1/memory/items",
        json={
            "query": "q2",
            "memory_items": ["m2"],
        },
    )
    body = client.get("/v1/memory/items").json()
    assert len(body["data"]) == 2
    assert body["meta"]["total"] == 2


def test_delete(client):
    resp = client.post(
        "/v1/memory/items",
        json={
            "query": "q1",
            "memory_items": ["m1"],
        },
    )
    item_id = resp.json()["data"]["id"]
    resp = client.delete(f"/v1/memory/items/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] is True
    assert client.get("/v1/memory/items/count").json()["data"]["count"] == 0


def test_search_get(client):
    client.post(
        "/v1/memory/items",
        json={
            "query": "fix login",
            "memory_items": ["use button"],
        },
    )
    resp = client.get("/v1/memory/items/search", params={"query": "fix login", "top_k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert len(body["data"]) >= 1


def test_search_post(client):
    client.post(
        "/v1/memory/items",
        json={
            "query": "fix login",
            "memory_items": ["use button"],
        },
    )
    resp = client.post("/v1/memory/items/search", json={"query": "fix login", "top_k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert len(body["data"]) >= 1


def test_response_format(client):
    """Verify unified ApiResponse wrapper on all endpoints."""
    # POST /items → 201 with data
    resp = client.post(
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
    resp = client.get("/v1/memory/items")
    body = resp.json()
    assert "data" in body
    assert body["meta"]["total"] >= 1

    # GET /items/count → 200 with data.count
    resp = client.get("/v1/memory/items/count")
    body = resp.json()
    assert "data" in body
    assert isinstance(body["data"]["count"], int)

    # DELETE /items/{item_id} → 200 with data.deleted
    resp = client.delete(f"/v1/memory/items/{item_id}")
    body = resp.json()
    assert "data" in body
    assert body["data"]["deleted"] is True
