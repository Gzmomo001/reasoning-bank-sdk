"""ReasoningBank Claude Code Hooks — calls Docker-deployed API server.

No SDK imports — pure HTTP calls to the ReasoningBank REST API.
Uses only Python stdlib (urllib, json).

Subcommands:
    retrieve  — called by UserPromptSubmit hook, retrieves relevant memories
    induce    — called by SessionEnd hook, induces new memories from transcript

Docker services required:
    docker compose -f docker/docker-compose.yml up -d

Usage:
    echo '<stdin_json>' | uv run python scripts/memory_hooks.py retrieve
    echo '<stdin_json>' | uv run python scripts/memory_hooks.py induce
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000/v1/memory"
LOG_FILE = Path(__file__).resolve().parent / "hooks_error.log"
MAX_TRAJECTORY_CHARS = 8000
RETRIEVE_TOP_K = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Append a timestamped message to the hook log file."""
    try:
        ts = datetime.now(UTC).isoformat()
        with LOG_FILE.open("a") as f:
            f.write(f"[{ts}] {msg}\n")
    except OSError:
        pass


def _api_post(path: str, body: dict) -> dict | None:
    """POST JSON to the API server, return parsed response or None on error."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        _log(f"API POST {path} failed: {e}")
        return None


def _extract_text(content) -> str:
    """Extract readable text from Claude message content (str or list[dict])."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = json.dumps(block.get("input", {}), ensure_ascii=False)[:200]
            parts.append(f"[Tool: {name}({inp})]")
        elif btype == "tool_result":
            rc = block.get("content", "")
            if isinstance(rc, list):
                txt = " ".join(b.get("text", "") for b in rc if isinstance(b, dict) and b.get("type") == "text")
            else:
                txt = str(rc)
            parts.append(f"[Tool Result: {txt[:200]}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Transcript parser
# ---------------------------------------------------------------------------


def _parse_line(entry: dict, query: str, turns: list[str]) -> str:
    """Process a single transcript entry, returns updated query."""
    msg_type = entry.get("type", "")
    content = entry.get("message", {}).get("content", "")
    text = _extract_text(content)
    if not text.strip():
        return query

    if msg_type == "user" and not query:
        query = text.strip()

    if msg_type == "user":
        turns.append(f"User: {text.strip()}")
    elif msg_type == "assistant":
        turns.append(f"Assistant: {text.strip()}")

    return query


def parse_transcript(transcript_path: str) -> tuple[str, str, str]:
    """Parse Claude Code transcript JSONL → (query, trajectory, status)."""
    path = Path(transcript_path)
    if not path.exists():
        return "", "", "success"

    raw_lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not raw_lines:
        return "", "", "success"

    query = ""
    turns: list[str] = []

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        query = _parse_line(entry, query, turns)

    trajectory = "\n\n".join(turns)
    if len(trajectory) > MAX_TRAJECTORY_CHARS:
        trajectory = trajectory[:MAX_TRAJECTORY_CHARS] + "\n\n[...truncated]"

    return query, trajectory, "success"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def retrieve() -> None:
    """Retrieve relevant memories via API and print to stdout."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    prompt = data.get("user_prompt") or data.get("prompt") or data.get("message") or ""
    if isinstance(prompt, dict):
        prompt = _extract_text(prompt.get("content", ""))
    if not isinstance(prompt, str) or not prompt.strip():
        return

    resp = _api_post("/items/search", {"query": prompt.strip(), "top_k": RETRIEVE_TOP_K})
    if not resp:
        return

    items = resp.get("data", [])
    if not items:
        return

    # Format as context for Claude
    parts = ["[ReasoningBank — relevant memories from previous sessions]\n"]
    for i, item in enumerate(items, 1):
        parts.append(f"### Memory {i}")
        parts.append("\n\n".join(item.get("memory_items", [])))
        parts.append("")

    sys.stdout.write("\n".join(parts))


def induce() -> None:
    """Induce new memories from transcript via API."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    transcript_path = data.get("transcript_path", "")
    if not transcript_path:
        return

    query, trajectory, status = parse_transcript(transcript_path)
    if not query or not trajectory:
        _log(f"induce skipped: empty query/trajectory (path={transcript_path})")
        return

    resp = _api_post(
        "/inductions",
        {
            "query": query,
            "trajectory": trajectory,
            "status": status,
            "domain": "general",
        },
    )

    if resp:
        count = resp.get("meta", {}).get("total", 0)
        _log(f"induced {count} memory items for query: {query[:80]}")
    else:
        _log(f"induce API call failed for query: {query[:80]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ReasoningBank Claude Code Hooks")
    parser.add_argument("command", choices=["retrieve", "induce"])
    args = parser.parse_args()

    if args.command == "retrieve":
        retrieve()
    elif args.command == "induce":
        induce()


if __name__ == "__main__":
    main()
