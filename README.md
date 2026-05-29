# ReasoningBank SDK

Persistent agent memory with induction, retrieval, and scaling. ReasoningBank lets AI agents learn from their trajectories — extracting, storing, and retrieving generalizable memory items across tasks.

## Features

- **Memory Induction** — Extract actionable insights from single or multiple agent trajectories using LLM
- **Semantic Retrieval** — Retrieve relevant memories via embedding-based similarity search
- **Scaling Induction** — Compare multiple trajectories (success vs. failure) to distill contrastive insights
- **Multiple Storage Backends** — ChromaDB (vector search) or JSONL (lightweight file-based)
- **Multiple LLM Providers** — OpenAI, Google Gemini, Anthropic, or custom clients
- **Multiple Embedding Providers** — OpenAI, Gemini, or custom embedding functions
- **REST API** — FastAPI server with full CRUD + induction endpoints
- **MCP Server** — Model Context Protocol server for integration with MCP-compatible clients

## Installation

```bash
uv sync
```

Requires Python >= 3.12.

## Quick Start

### Python SDK

```python
from reasoning_bank import MemoryBank
from reasoning_bank.llm.openai_client import OpenAIClient

bank = MemoryBank(
    storage="chroma",
    storage_path="./memories",
    embedding_provider="gemini",
    embedding_model="gemini-embedding-001",
    llm_client=OpenAIClient(api_key="sk-..."),
)

# Retrieve relevant memories
memories = bank.retrieve(query="how to fix login bug", top_k=3)

# Induce memories from a successful trajectory
items = bank.induce(
    task_id="task-456",
    query="Navigate to shopping cart",
    trajectory="...",
    status="success",
    domain="web",
)

# Induce memories by comparing multiple trajectories
items = bank.induce_scaling(
    task_id="task-789",
    query="Add item to wishlist",
    trajectories=[
        {"trajectory": "...", "status": "success"},
        {"trajectory": "...", "status": "fail"},
    ],
    domain="web",
)

# Directly add a memory item (no LLM needed)
item = bank.add(
    task_id="task-101",
    query="Search for products",
    memory_items=["Always use the search bar in the top navigation..."],
    status="success",
    domain="web",
)

# List, count, or delete
all_memories = bank.list()
total = bank.count()
bank.delete(task_id="task-456")
```

### REST API

```bash
# Set environment variables
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export LLM_API_KEY=sk-...
export EMBEDDING_PROVIDER=gemini
export STORAGE=chroma

# Start the server
uvicorn reasoning_bank_api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/memory/retrieve` | Retrieve relevant memories |
| POST | `/memory/add` | Add a memory item directly |
| POST | `/memory/delete` | Delete memories by task ID |
| POST | `/memory/induce` | Induce from a single trajectory |
| POST | `/memory/induce_scaling` | Induce from multiple trajectories |
| GET | `/memory/list` | List all memories |
| GET | `/memory/count` | Get total memory count |

### MCP Server

```bash
# Via entry point (stdio transport)
reasoning-bank-mcp

# Or with SSE transport
reasoning-bank-mcp --transport sse --port 9000
```

MCP tools exposed: `reasoning_bank_retrieve`, `reasoning_bank_add`, `reasoning_bank_induce`, `reasoning_bank_induce_scaling`, `reasoning_bank_list`, `reasoning_bank_delete`, `reasoning_bank_count`.

## Docker

```bash
# Start API server + ChromaDB
LLM_API_KEY=sk-... docker compose --profile api up

# Start MCP server + ChromaDB
LLM_API_KEY=sk-... docker compose --profile mcp up

# Start all services
LLM_API_KEY=sk-... docker compose --profile all up
```

## Configuration

Configuration is via environment variables. Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *(empty)* | **Required.** API key for LLM provider (needed for induction / scaling) |
| `LLM_PROVIDER` | `openai` | LLM provider (`openai`, `anthropic`, `vertexai`, `google_ai`) |
| `LLM_MODEL` | *(empty)* | LLM model name (e.g. `gpt-4o`, `gemini-2.0-flash`) |
| `LLM_API_BASE_URL` | *(empty)* | Custom base URL for OpenAI-compatible APIs |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `gemini` | Embedding provider (`gemini` or `openai`) |
| `EMBEDDING_MODEL` | *(provider default)* | Embedding model name (e.g. `gemini-embedding-001`) |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE` | `chroma` | Storage backend (`chroma` or `jsonl`) |
| `STORAGE_PATH` | `./memories` | Storage directory path |
| `CHROMA_HOST` | *(empty)* | ChromaDB host (leave empty for **embedded mode**) |
| `CHROMA_PORT` | *(empty)* | ChromaDB port (only needed when `CHROMA_HOST` is set) |

> **Note:** When `CHROMA_HOST` and `CHROMA_PORT` are not set, ChromaDB runs in **embedded mode** using `PersistentClient` — no separate database server is needed. Data is stored locally in the `STORAGE_PATH` directory. Only set these variables when connecting to a standalone ChromaDB server (e.g. in Docker deployments).

### Google Gemini / Vertex AI

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_GENAI_USE_VERTEXAI` | `False` | Set to `True` if using Vertex AI instead of Google AI Studio |

## Architecture

```
reasoning_bank/
  core/
    bank.py          # MemoryBank — main entry point
    memory_item.py   # MemoryItem data model
    induction.py     # Single-trajectory induction
    scaling.py       # Multi-trajectory scaling induction
    embedding.py     # Embedding providers (Gemini, OpenAI, Custom)
    prompts.py       # Domain-adapted prompt templates
  llm/
    base.py          # LLMClient abstract base
    openai_client.py # OpenAI LLM client
    gemini_client.py # Google Gemini LLM client
    anthropic_client.py # Anthropic LLM client
  storage/
    base.py          # StorageBackend abstract base
    chroma.py        # ChromaDB storage backend
    jsonl.py         # JSONL file-based storage backend
reasoning_bank_api/
  app.py             # FastAPI application factory
  routes.py          # HTTP route handlers
reasoning_bank_mcp/
  server.py          # MCP server (tools + resources)
```

## Domains

Induction prompts are domain-adapted for:

- **`web`** — Web navigation tasks
- **`coding`** — Code repository tasks
- **`general`** — General interactive environments

## Testing

```bash
uv run pytest
```

## License

MIT
