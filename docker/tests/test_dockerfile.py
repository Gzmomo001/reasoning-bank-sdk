"""Tests for Dockerfile correctness and best practices.

Static checks run without Docker.
Build tests require Docker and are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import subprocess

import pytest
from conftest import DOCKER_DIR, SDK_ROOT


def _read_dockerfile(name: str) -> str:
    path = DOCKER_DIR / name
    return path.read_text()


# ---------------------------------------------------------------------------
# Static checks (no Docker required)
# ---------------------------------------------------------------------------


class TestDockerfileAPI:
    @pytest.fixture
    def content(self) -> str:
        return _read_dockerfile("Dockerfile.api")

    def test_exists(self):
        assert (DOCKER_DIR / "Dockerfile.api").exists()

    def test_uses_multistage_build(self, content):
        from_count = content.count("FROM ")
        assert from_count >= 2, "Expected at least 2 FROM instructions for multi-stage build"
        assert "AS builder" in content

    def test_uses_nonroot_user(self, content):
        lines = content.split("\n")
        from_idx = [i for i, line in enumerate(lines) if line.startswith("FROM ") and "builder" not in line][-1]
        runtime = "\n".join(lines[from_idx:])
        assert "USER" in runtime

    def test_pins_python_version(self, content):
        assert "python:3.12" in content
        for line in content.split("\n"):
            if line.startswith("FROM python:"):
                assert ":latest" not in line, f"Python base image should pin version: {line}"

    def test_sets_python_optimizations(self, content):
        assert "PYTHONDONTWRITEBYTECODE=1" in content
        assert "PYTHONUNBUFFERED=1" in content

    def test_exposes_correct_port(self, content):
        assert "EXPOSE 8000" in content

    def test_copies_correct_source(self, content):
        assert "reasoning_bank/" in content
        assert "reasoning_bank_api/" in content

    def test_uses_uv_for_build(self, content):
        assert "uv" in content

    def test_builder_cleans_venv(self, content):
        assert "__pycache__" in content
        assert "*.dist-info" in content or ".dist-info" in content

    def test_runs_uvicorn(self, content):
        assert "uvicorn" in content
        assert "reasoning_bank_api.app:app" in content


class TestDockerfileMCP:
    @pytest.fixture
    def content(self) -> str:
        return _read_dockerfile("Dockerfile.mcp")

    def test_exists(self):
        assert (DOCKER_DIR / "Dockerfile.mcp").exists()

    def test_uses_multistage_build(self, content):
        from_count = content.count("FROM ")
        assert from_count >= 2
        assert "AS builder" in content

    def test_uses_nonroot_user(self, content):
        lines = content.split("\n")
        from_idx = [i for i, line in enumerate(lines) if line.startswith("FROM ") and "builder" not in line][-1]
        runtime = "\n".join(lines[from_idx:])
        assert "USER" in runtime

    def test_pins_python_version(self, content):
        assert "python:3.12" in content
        for line in content.split("\n"):
            if line.startswith("FROM python:"):
                assert ":latest" not in line, f"Python base image should pin version: {line}"

    def test_sets_python_optimizations(self, content):
        assert "PYTHONDONTWRITEBYTECODE=1" in content
        assert "PYTHONUNBUFFERED=1" in content

    def test_exposes_correct_port(self, content):
        assert "EXPOSE 9000" in content

    def test_copies_correct_source(self, content):
        assert "reasoning_bank/" in content
        assert "reasoning_bank_mcp/" in content

    def test_uses_uv_for_build(self, content):
        assert "uv" in content

    def test_builder_cleans_venv(self, content):
        assert "__pycache__" in content

    def test_runs_mcp_server(self, content):
        assert "reasoning_bank_mcp.server" in content
        assert "sse" in content
        assert "9000" in content


def test_dockerignore_exists():
    assert (SDK_ROOT / ".dockerignore").exists()


# ---------------------------------------------------------------------------
# Build tests (require Docker)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_build_api_image():
    result = subprocess.run(  # noqa: PLW1510
        ["docker", "build", "-f", "Dockerfile.api", "-t", "rb-test-api:latest", ".."],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(DOCKER_DIR),
    )
    assert result.returncode == 0, f"Build failed:\n{result.stderr}"


@pytest.mark.integration
def test_build_mcp_image():
    result = subprocess.run(  # noqa: PLW1510
        ["docker", "build", "-f", "Dockerfile.mcp", "-t", "rb-test-mcp:latest", ".."],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(DOCKER_DIR),
    )
    assert result.returncode == 0, f"Build failed:\n{result.stderr}"
