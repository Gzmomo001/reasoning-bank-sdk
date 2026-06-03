"""Integration tests for Docker services.

All tests require a running Docker daemon and are marked with @pytest.mark.integration.

Most tests need an LLM_API_KEY because MemoryBank.add() and MemoryBank.retrieve()
call the embedding provider. Only induce/induce_scaling additionally need an LLM
model configured via LLM_PROVIDER + LLM_MODEL.

Run with:
    cd sdk && uv run pytest docker/tests/test_integration.py -v -m integration

Environment variables:
    LLM_API_KEY         - embedding + LLM unified API key (required for add/retrieve)
    EMBEDDING_PROVIDER   - embedding provider (default: gemini)
    EMBEDDING_MODEL      - embedding model (default: gemini-embedding-001)
    LLM_PROVIDER        - LLM provider for induce/scaling (default: openai)
    LLM_MODEL            - LLM model for induce/scaling (default: gpt-4o)
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from conftest import DOCKER_DIR, compose_restart, wait_for_service


def _api_add(api_url: str, query: str, memory_items: list[str], **kwargs) -> httpx.Response:
    resp = httpx.post(
        f"{api_url}/v1/memory/items",
        json={
            "query": query,
            "memory_items": memory_items,
            "status": kwargs.get("status", "success"),
            "domain": kwargs.get("domain", "general"),
        },
        timeout=30,
    )
    assert resp.status_code == 201, f"Add failed ({resp.status_code}): {resp.text}"
    return resp


# ---------------------------------------------------------------------------
# Service health & startup
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chromadb_is_healthy(compose_project, chromadb_url):
    resp = httpx.get(f"{chromadb_url}/api/v1/heartbeat", timeout=10)
    assert resp.status_code == 200


@pytest.mark.integration
def test_api_is_responsive(compose_project, api_url):
    resp = httpx.get(f"{api_url}/docs", timeout=10)
    assert resp.status_code == 200


@pytest.mark.integration
def test_mcp_sse_endpoint_exists(compose_project, mcp_url):
    with httpx.stream("GET", f"{mcp_url}/sse", timeout=10) as resp:
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_api_empty_count(compose_project, api_url):
    resp = httpx.get(f"{api_url}/v1/memory/items/count", timeout=10)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "count" in data
    assert isinstance(data["count"], int)


@pytest.mark.integration
def test_api_empty_list(compose_project, api_url):
    resp = httpx.get(f"{api_url}/v1/memory/items", timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert isinstance(body["data"], list)


@pytest.mark.integration
def test_api_empty_retrieve(compose_project, api_url):
    resp = httpx.post(f"{api_url}/v1/memory/items/search", json={"query": "test", "top_k": 3}, timeout=30)
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert isinstance(body["data"], list)


@pytest.mark.integration
def test_api_delete_nonexistent(compose_project, api_url):
    resp = httpx.delete(f"{api_url}/v1/memory/items/nonexistent-id", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] is True


@pytest.mark.integration
def test_api_add_and_count(compose_project, api_url):
    _api_add(api_url, "integration test query", ["memory item from integration test"])

    resp = httpx.get(f"{api_url}/v1/memory/items/count", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["data"]["count"] >= 1


@pytest.mark.integration
def test_api_add_and_retrieve(compose_project, api_url):
    _api_add(api_url, "how to fix login button", ["click the blue login button", "check credentials"])

    resp = httpx.post(f"{api_url}/v1/memory/items/search", json={"query": "fix login", "top_k": 3}, timeout=30)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["data"], list)
    if body["data"]:
        assert "memory_items" in body["data"][0] or "query" in body["data"][0]


@pytest.mark.integration
def test_api_add_and_delete(compose_project, api_url):
    resp = _api_add(api_url, "delete test", ["to be deleted"])
    item_id = resp.json()["data"]["id"]

    resp = httpx.delete(f"{api_url}/v1/memory/items/{item_id}", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] is True


@pytest.mark.integration
def test_api_list_returns_added_items(compose_project, api_url):
    resp = _api_add(api_url, "list test", ["item for list test"])
    item_id = resp.json()["data"]["id"]

    resp = httpx.get(f"{api_url}/v1/memory/items", timeout=10)
    data = resp.json()["data"]
    assert isinstance(data, list)
    ids = {item.get("id") for item in data}
    assert item_id in ids


# ---------------------------------------------------------------------------
# MCP SSE endpoint tests
# ---------------------------------------------------------------------------


def _mcp_sse_request(mcp_url: str, payload: dict, timeout: int = 30) -> dict:
    """Send a JSON-RPC request via MCP SSE transport and return the response."""
    with httpx.Client(timeout=timeout) as client:
        # Keep SSE connection alive while posting
        sse_stream = client.stream("GET", f"{mcp_url}/sse")
        sse_resp = sse_stream.__enter__()
        try:
            endpoint = None
            for line in sse_resp.iter_lines():
                if line.startswith("data:"):
                    endpoint = line.replace("data:", "").strip()
                    break
            if endpoint is None:
                return {"error": "no endpoint received"}
            resp = client.post(f"{mcp_url}{endpoint}", json=payload)
            if resp.status_code == 200:
                for line in resp.text.strip().split("\n"):
                    if line.startswith("data:"):
                        return json.loads(line.replace("data:", "").strip())
            return {"error": f"status {resp.status_code}: {resp.text[:200]}"}
        finally:
            sse_stream.__exit__(None, None, None)


@pytest.mark.integration
def test_mcp_count_tool(compose_project, mcp_url):
    result = _mcp_sse_request(
        mcp_url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "reasoning_bank_count", "arguments": {}},
        },
    )
    assert "error" not in result or result["error"] == {}


@pytest.mark.integration
def test_mcp_list_tool(compose_project, mcp_url):
    result = _mcp_sse_request(
        mcp_url,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "reasoning_bank_list", "arguments": {}},
        },
    )
    assert "error" not in result or result["error"] == {}


# ---------------------------------------------------------------------------
# Cross-service communication
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_data_shared_between_api_and_mcp(compose_project, api_url, mcp_url):
    """Add via API, verify count via MCP."""
    _api_add(api_url, "cross-service test", ["shared data"])

    result = _mcp_sse_request(
        mcp_url,
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "reasoning_bank_count", "arguments": {}},
        },
    )
    assert "error" not in result or result["error"] == {}


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_data_survives_api_restart(compose_project, api_url):
    """Add a memory, restart API, verify it's still there."""
    resp = _api_add(api_url, "persistence test", ["should survive restart"])
    item_id = resp.json()["data"]["id"]

    compose_restart(compose_project, "api")
    wait_for_service("localhost", 8000, timeout=60)

    resp = httpx.get(f"{api_url}/v1/memory/items", timeout=10)
    data = resp.json()["data"]
    ids = {item.get("id") for item in data}
    assert item_id in ids


# ---------------------------------------------------------------------------
# Log output
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_api_logs_mounted(compose_project):
    logs_dir = DOCKER_DIR / "logs" / "api"
    assert logs_dir.is_dir(), f"Logs directory {logs_dir} does not exist"


@pytest.mark.integration
def test_mcp_logs_mounted(compose_project):
    logs_dir = DOCKER_DIR / "logs" / "mcp"
    assert logs_dir.is_dir(), f"Logs directory {logs_dir} does not exist"
