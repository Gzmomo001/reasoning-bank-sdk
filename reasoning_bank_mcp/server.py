"""ReasoningBank MCP Server — exposes MemoryBank as MCP tools and resources."""

from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from reasoning_bank import MemoryBank
from reasoning_bank.logging_config import setup_logging
from reasoning_bank.llm.anthropic_client import AnthropicClient
from reasoning_bank.llm.gemini_client import GeminiClient
from reasoning_bank.llm.openai_client import OpenAIClient

if TYPE_CHECKING:
    from reasoning_bank.llm.base import LLMClient

mcp = FastMCP("ReasoningBank")


# ---------------------------------------------------------------------------
# Bank initialization from environment
# ---------------------------------------------------------------------------


def _get_llm() -> LLMClient | None:
    provider = os.environ.get("LLM_PROVIDER", "")
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        return None
    if provider == "anthropic":
        return AnthropicClient(model=model)
    if provider in ("vertexai", "google_ai"):
        return GeminiClient(model=model)
    return OpenAIClient(model=model)


_bank_instance: MemoryBank | None = None


def _get_or_create_bank() -> MemoryBank:
    global _bank_instance  # noqa: PLW0603
    if _bank_instance is None:
        _bank_instance = MemoryBank(
            storage=os.environ.get("STORAGE", "chroma"),
            storage_path=os.environ.get("STORAGE_PATH", "./memories"),
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "gemini"),
            embedding_model=os.environ.get("EMBEDDING_MODEL"),
            llm_client=_get_llm(),
        )
    return _bank_instance


def _bank() -> MemoryBank:
    return _get_or_create_bank()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def reasoning_bank_retrieve(query: str, top_k: int = 3) -> str:
    """Retrieve relevant memories from the ReasoningBank."""
    items = _bank().retrieve(query=query, top_k=top_k)
    return json.dumps([item.to_dict() for item in items], indent=2)


@mcp.tool()
def reasoning_bank_add(
    query: str,
    memory_items: list[str],
    status: str = "success",
    domain: str = "general",
) -> str:
    """Add a memory item directly to the ReasoningBank (no LLM induction)."""
    item = _bank().add(
        query=query,
        memory_items=memory_items,
        status=status,
        domain=domain,
    )
    return json.dumps({"ok": True, "id": item.id})


@mcp.tool()
def reasoning_bank_induce(
    query: str,
    trajectory: str,
    status: str,
    domain: str = "web",
) -> str:
    """Run full auto induction: extract memory items from a single trajectory using LLM."""
    items = _bank().induce(
        query=query,
        trajectory=trajectory,
        status=status,
        domain=domain,
    )
    return json.dumps([item.to_dict() for item in items], indent=2)


@mcp.tool()
def reasoning_bank_induce_scaling(
    query: str,
    trajectories: list[dict],
    domain: str = "web",
) -> str:
    """Run multi-trajectory contrast induction: compare trajectories and extract memory items."""
    items = _bank().induce_scaling(
        query=query,
        trajectories=trajectories,
        domain=domain,
    )
    return json.dumps([item.to_dict() for item in items], indent=2)


@mcp.tool()
def reasoning_bank_list() -> str:
    """List all memories stored in the ReasoningBank."""
    items = _bank().list()
    return json.dumps([item.to_dict() for item in items], indent=2)


@mcp.tool()
def reasoning_bank_delete(item_id: str) -> str:
    """Delete a memory item by its ID."""
    _bank().delete(item_id=item_id)
    return json.dumps({"ok": True})


@mcp.tool()
def reasoning_bank_count() -> str:
    """Return the total number of stored memories."""
    return json.dumps({"count": _bank().count()})


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@mcp.resource("reasoning-bank://stats")
def stats() -> str:
    """Return ReasoningBank statistics."""
    return json.dumps({"count": _bank().count()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ReasoningBank MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    setup_logging("mcp")

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
