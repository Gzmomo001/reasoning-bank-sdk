"""Tests for docker-compose.yml configuration correctness."""

from __future__ import annotations

import yaml
from conftest import COMPOSE_FILE, LOCAL_COMPOSE_FILE


def _load_compose() -> dict:
    with COMPOSE_FILE.open() as f:
        return yaml.safe_load(f)


def _load_local_compose() -> dict:
    with LOCAL_COMPOSE_FILE.open() as f:
        return yaml.safe_load(f)


def test_compose_is_valid_yaml():
    data = _load_compose()
    assert isinstance(data, dict)
    assert "services" in data


def test_required_services_exist():
    data = _load_compose()
    assert set(data["services"].keys()) == {"chromadb", "api", "mcp"}


def test_chromadb_port_mapping():
    svc = _load_compose()["services"]["chromadb"]
    assert "8001:8000" in svc["ports"]


def test_api_port_mapping():
    svc = _load_compose()["services"]["api"]
    assert "8000:8000" in svc["ports"]


def test_mcp_port_mapping():
    svc = _load_compose()["services"]["mcp"]
    assert "9000:9000" in svc["ports"]


def test_network_definition():
    data = _load_compose()
    assert "reasoning-bank" in data["networks"]
    assert data["networks"]["reasoning-bank"]["driver"] == "bridge"


def test_all_services_use_custom_network():
    data = _load_compose()
    for name in ("chromadb", "api", "mcp"):
        svc = data["services"][name]
        assert "networks" in svc
        assert "reasoning-bank" in svc["networks"]


def test_api_depends_on_chromadb_healthy():
    svc = _load_compose()["services"]["api"]
    assert svc["depends_on"]["chromadb"]["condition"] == "service_healthy"


def test_mcp_depends_on_chromadb_healthy():
    svc = _load_compose()["services"]["mcp"]
    assert svc["depends_on"]["chromadb"]["condition"] == "service_healthy"


def test_chromadb_healthcheck_config():
    hc = _load_compose()["services"]["chromadb"]["healthcheck"]
    interval_s = int(hc["interval"].rstrip("s"))
    timeout_s = int(hc["timeout"].rstrip("s"))
    assert interval_s <= 10
    assert timeout_s <= 5
    assert hc["retries"] >= 5


def test_volume_mounts_exist():
    data = _load_compose()
    chroma_vols = data["services"]["chromadb"]["volumes"]
    assert any("chroma_data" in v for v in chroma_vols)
    assert any("config.yaml" in v for v in chroma_vols)

    api_vols = data["services"]["api"]["volumes"]
    assert any("logs" in v for v in api_vols)

    mcp_vols = data["services"]["mcp"]["volumes"]
    assert any("logs" in v for v in mcp_vols)


def test_environment_variables_passed():
    data = _load_compose()
    for svc_name in ("api", "mcp"):
        envs = data["services"][svc_name]["environment"]
        env_strs = [e if isinstance(e, str) else str(e) for e in envs]
        assert any("STORAGE=chroma" in s for s in env_strs)
        assert any("CHROMA_HOST=chromadb" in s for s in env_strs)
        assert any("CHROMA_PORT=8000" in s for s in env_strs)


def test_anonymized_telemetry_disabled():
    envs = _load_compose()["services"]["chromadb"]["environment"]
    assert "ANONYMIZED_TELEMETRY=FALSE" in envs


def test_log_rotation_configured():
    data = _load_compose()
    for svc_name in ("chromadb", "api", "mcp"):
        logging = data["services"][svc_name]["logging"]
        assert logging["options"]["max-size"] != ""
        assert logging["options"]["max-file"] != ""


def test_config_yaml_mounted():
    vols = _load_compose()["services"]["chromadb"]["volumes"]
    assert any("./config.yaml:/config.yaml" in v for v in vols)


# ---------------------------------------------------------------------------
# GHCR image tests (main compose file uses pre-built images)
# ---------------------------------------------------------------------------


def test_api_uses_ghcr_image():
    svc = _load_compose()["services"]["api"]
    assert "image" in svc
    assert "ghcr.io" in svc["image"]
    assert "reasoning-bank-sdk-api" in svc["image"]


def test_mcp_uses_ghcr_image():
    svc = _load_compose()["services"]["mcp"]
    assert "image" in svc
    assert "ghcr.io" in svc["image"]
    assert "reasoning-bank-sdk-mcp" in svc["image"]


# ---------------------------------------------------------------------------
# Local compose override tests
# ---------------------------------------------------------------------------


def test_local_compose_is_valid_yaml():
    data = _load_local_compose()
    assert isinstance(data, dict)
    assert "services" in data


def test_local_compose_has_api_and_mcp():
    data = _load_local_compose()
    assert "api" in data["services"]
    assert "mcp" in data["services"]


def test_local_build_context_points_to_parent():
    data = _load_local_compose()
    assert data["services"]["api"]["build"]["context"] == ".."
    assert data["services"]["mcp"]["build"]["context"] == ".."


def test_local_api_and_mcp_use_different_dockerfiles():
    data = _load_local_compose()
    assert "Dockerfile.api" in data["services"]["api"]["build"]["dockerfile"]
    assert "Dockerfile.mcp" in data["services"]["mcp"]["build"]["dockerfile"]
