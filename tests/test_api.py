"""Tests for FastAPI routes — real embedding + real LLM + ChromaDB."""

import pytest
from fastapi.testclient import TestClient

from reasoning_bank_api.app import app
import reasoning_bank_api.routes as routes_mod


@pytest.fixture
def client(bank):
    routes_mod._bank = bank
    with TestClient(app) as c:
        yield c
    routes_mod._bank = None


def test_add_and_count(client):
    resp = client.post("/memory/add", json={
        "task_id": "t1", "query": "q1", "memory_items": ["m1"], "status": "success",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"]
    assert client.get("/memory/count").json()["count"] == 1


def test_list(client):
    client.post("/memory/add", json={
        "task_id": "t1", "query": "q1", "memory_items": ["m1"],
    })
    client.post("/memory/add", json={
        "task_id": "t2", "query": "q2", "memory_items": ["m2"],
    })
    data = client.get("/memory/list").json()
    assert len(data) == 2


def test_delete(client):
    client.post("/memory/add", json={
        "task_id": "t1", "query": "q1", "memory_items": ["m1"],
    })
    resp = client.post("/memory/delete", json={"task_id": "t1"})
    assert resp.json()["ok"]
    assert client.get("/memory/count").json()["count"] == 0


def test_retrieve(client):
    client.post("/memory/add", json={
        "task_id": "t1", "query": "fix login", "memory_items": ["use button"],
    })
    resp = client.post("/memory/retrieve", json={"query": "fix login", "top_k": 1})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
