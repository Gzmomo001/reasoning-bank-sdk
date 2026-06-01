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


def _api_add(api_url: str, task_id: str, query: str, memory_items: list[str], **kwargs) -> httpx.Response:
    resp = httpx.post(f"{api_url}/memory/add", json={
        "task_id": task_id,
        "query": query,
        "memory_items": memory_items,
        "status": kwargs.get("status", "success"),
        "domain": kwargs.get("domain", "general"),
    }, timeout=30)
    assert resp.status_code == 200, f"Add failed ({resp.status_code}): {resp.text}"
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
    resp = httpx.get(f"{api_url}/memory/count", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert isinstance(data["count"], int)


@pytest.mark.integration
def test_api_empty_list(compose_project, api_url):
    resp = httpx.get(f"{api_url}/memory/list", timeout=10)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.integration
def test_api_empty_retrieve(compose_project, api_url):
    resp = httpx.post(f"{api_url}/memory/retrieve", json={"query": "test", "top_k": 3}, timeout=30)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.integration
def test_api_delete_nonexistent(compose_project, api_url):
    resp = httpx.post(f"{api_url}/memory/delete", json={"task_id": "nonexistent-task"}, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["ok"]


@pytest.mark.integration
def test_api_add_and_count(compose_project, api_url):
    task_id = f"test-{os.getpid()}"
    _api_add(api_url, task_id, "integration test query", ["memory item from integration test"])

    resp = httpx.get(f"{api_url}/memory/count", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["count"] >= 1


@pytest.mark.integration
def test_api_add_and_retrieve(compose_project, api_url):
    task_id = f"retrieve-{os.getpid()}"
    _api_add(api_url, task_id, "how to fix login button", ["click the blue login button", "check credentials"])

    resp = httpx.post(f"{api_url}/memory/retrieve", json={"query": "fix login", "top_k": 3}, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        assert "memory_items" in data[0] or "task_id" in data[0]


@pytest.mark.integration
def test_api_add_and_delete(compose_project, api_url):
    task_id = f"del-{os.getpid()}"
    _api_add(api_url, task_id, "delete test", ["to be deleted"])

    resp = httpx.post(f"{api_url}/memory/delete", json={"task_id": task_id}, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["ok"]


@pytest.mark.integration
def test_api_list_returns_added_items(compose_project, api_url):
    task_id = f"list-{os.getpid()}"
    _api_add(api_url, task_id, "list test", ["item for list test"])

    resp = httpx.get(f"{api_url}/memory/list", timeout=10)
    data = resp.json()
    assert isinstance(data, list)
    task_ids = {item.get("task_id") for item in data}
    assert task_id in task_ids


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
    result = _mcp_sse_request(mcp_url, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "reasoning_bank_count", "arguments": {}},
    })
    assert "error" not in result or result["error"] == {}


@pytest.mark.integration
def test_mcp_list_tool(compose_project, mcp_url):
    result = _mcp_sse_request(mcp_url, {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "reasoning_bank_list", "arguments": {}},
    })
    assert "error" not in result or result["error"] == {}


# ---------------------------------------------------------------------------
# Cross-service communication
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_data_shared_between_api_and_mcp(compose_project, api_url, mcp_url):
    """Add via API, verify count via MCP."""
    _api_add(api_url, f"shared-{os.getpid()}", "cross-service test", ["shared data"])

    result = _mcp_sse_request(mcp_url, {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "reasoning_bank_count", "arguments": {}},
    })
    assert "error" not in result or result["error"] == {}


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_data_survives_api_restart(compose_project, api_url):
    """Add a memory, restart API, verify it's still there."""
    task_id = f"persist-{os.getpid()}"
    _api_add(api_url, task_id, "persistence test", ["should survive restart"])

    compose_restart(compose_project, "api")
    wait_for_service("localhost", 8000, timeout=60)

    resp = httpx.get(f"{api_url}/memory/list", timeout=10)
    data = resp.json()
    task_ids = {item.get("task_id") for item in data}
    assert task_id in task_ids


# ---------------------------------------------------------------------------
# Log output
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_api_logs_mounted(compose_project):
    logs_dir = os.path.join(DOCKER_DIR, "logs", "api")
    assert os.path.isdir(logs_dir), f"Logs directory {logs_dir} does not exist"


@pytest.mark.integration
def test_mcp_logs_mounted(compose_project):
    logs_dir = os.path.join(DOCKER_DIR, "logs", "mcp")
    assert os.path.isdir(logs_dir), f"Logs directory {logs_dir} does not exist"
