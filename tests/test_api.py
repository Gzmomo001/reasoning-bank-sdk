"""Tests for FastAPI routes."""

import tempfile
import os

import pytest
from fastapi.testclient import TestClient

from reasoning_bank_api.app import app
from reasoning_bank import MemoryBank
from reasoning_bank.core.embedding import CustomEmbedding


def _dummy_embed(texts):
    return [[1.0] * 8 for _ in texts]


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(
            storage="jsonl",
            storage_path=tmp,
            embedding_provider=CustomEmbedding(_dummy_embed, dim=8),
        )
        # Override the global bank in routes
        import reasoning_bank_api.routes as routes_mod
        routes_mod._bank = bank

        with TestClient(app) as c:
            yield c

        routes_mod._bank = None


def test_add_and_count(client):
    resp = client.post("/memory/add", json={
        "task_id": "t1",
        "query": "q1",
        "memory_items": ["m1"],
        "status": "success",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"]

    resp = client.get("/memory/count")
    assert resp.json()["count"] == 1


def test_list(client):
    client.post("/memory/add", json={
        "task_id": "t1", "query": "q1", "memory_items": ["m1"],
    })
    client.post("/memory/add", json={
        "task_id": "t2", "query": "q2", "memory_items": ["m2"],
    })

    resp = client.get("/memory/list")
    data = resp.json()
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
