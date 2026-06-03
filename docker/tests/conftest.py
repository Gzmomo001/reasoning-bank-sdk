"""Pytest configuration and shared fixtures for Docker tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires running Docker daemon (deselect with '-m \"not integration\"')"
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
DOCKER_DIR = TESTS_DIR.parent
SDK_ROOT = DOCKER_DIR.parent
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
LOCAL_COMPOSE_FILE = DOCKER_DIR / "docker-compose.local.yml"
COMPOSE_FILES = ["-f", str(COMPOSE_FILE), "-f", str(LOCAL_COMPOSE_FILE)]


def _get_project_name() -> str:
    return f"rb-test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Docker Compose helpers
# ---------------------------------------------------------------------------


def compose_up(project: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-p", project, *COMPOSE_FILES, "up", "--wait", "--build", "-d"]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)  # noqa: PLW1510


def compose_down(project: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: PLW1510
        ["docker", "compose", "-p", project, *COMPOSE_FILES, "down", "--volumes", "--remove-orphans"],
        capture_output=True,
        text=True,
        timeout=60,
    )


def compose_restart(project: str, *services: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: PLW1510
        ["docker", "compose", "-p", project, *COMPOSE_FILES, "restart", *services],
        capture_output=True,
        text=True,
        timeout=120,
    )


def wait_for_service(host: str, port: int, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(min(2, deadline - time.monotonic()))
    msg = f"Service at {host}:{port} did not become reachable within {timeout}s"
    raise TimeoutError(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def compose_project():
    """Start all services via docker compose, tear down after session."""
    project = _get_project_name()
    try:
        result = compose_up(project)
        if result.returncode != 0:
            msg = f"docker compose up failed:\n{result.stderr}"
            raise RuntimeError(msg)

        # Wait for each service port
        wait_for_service("localhost", 8001, timeout=120)  # chromadb
        wait_for_service("localhost", 8000, timeout=120)  # api
        wait_for_service("localhost", 9000, timeout=120)  # mcp

        yield project
    finally:
        compose_down(project)


@pytest.fixture
def api_url() -> str:
    return "http://localhost:8000"


@pytest.fixture
def mcp_url() -> str:
    return "http://localhost:9000"


@pytest.fixture
def chromadb_url() -> str:
    return "http://localhost:8001"
