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

#### `POST /memory/retrieve`

Retrieve relevant memories via semantic similarity search.

```json
// Request
{
  "query": "how to fix login bug",
  "top_k": 3
}

// Response
[
  {
    "id": "a1b2c3...",
    "task_id": "task-456",
    "query": "fix login issue",
    "status": "success",
    "domain": "web",
    "memory_items": ["Check the session cookie expiration..."],
    "template_id": null,
    "created_at": "2025-01-01T00:00:00+00:00"
  }
]
```

#### `POST /memory/add`

Add a memory item directly (no LLM needed).

```json
// Request
{
  "task_id": "task-101",
  "query": "Search for products",
  "memory_items": ["Always use the search bar in the top navigation"],
  "status": "success",
  "domain": "general",
  "template_id": null
}

// Response
{ "ok": true, "id": "d4e5f6..." }
```

#### `POST /memory/delete`

Delete all memories for a given task ID.

```json
// Request
{ "task_id": "task-456" }

// Response
{ "ok": true }
```

#### `POST /memory/induce`

Run full auto induction — extract memory items from a single trajectory using LLM.

```json
// Request
{
  "task_id": "task-456",
  "query": "Navigate to shopping cart",
  "trajectory": "Step 1: Clicked on 'Cart' icon...",
  "status": "success",
  "domain": "web"
}

// Response
[
  {
    "id": "...",
    "task_id": "task-456",
    "query": "Navigate to shopping cart",
    "status": "success",
    "domain": "web",
    "memory_items": ["The cart icon is located in the top-right corner..."],
    "template_id": null,
    "created_at": "..."
  }
]
```

#### `POST /memory/induce_scaling`

Run multi-trajectory contrast induction — compare trajectories and extract contrastive insights.

```json
// Request
{
  "task_id": "task-789",
  "query": "Add item to wishlist",
  "trajectories": [
    { "trajectory": "Clicked wishlist button...", "status": "success" },
    { "trajectory": "Clicked add-to-cart instead...", "status": "fail" }
  ],
  "domain": "web"
}

// Response
[
  {
    "id": "...",
    "task_id": "task-789",
    "query": "Add item to wishlist",
    "status": "success",
    "domain": "web",
    "memory_items": ["The wishlist button is a heart icon, not the cart button..."],
    "template_id": null,
    "created_at": "..."
  }
]
```

#### `GET /memory/list`

List all stored memories.

```json
// Response
[
  {
    "id": "...",
    "task_id": "task-456",
    "query": "...",
    "status": "success",
    "domain": "web",
    "memory_items": ["..."],
    "template_id": null,
    "created_at": "..."
  }
]
```

#### `GET /memory/count`

Get the total number of stored memories.

```json
// Response
{ "count": 42 }
```

### MCP Server

```bash
# Via entry point (stdio transport)
reasoning-bank-mcp

# Or with SSE transport
reasoning-bank-mcp --transport sse --port 9000
```

#### MCP Tools

| Tool | Description |
|------|-------------|
| `reasoning_bank_retrieve` | Retrieve relevant memories via semantic similarity |
| `reasoning_bank_add` | Add a memory item directly (no LLM) |
| `reasoning_bank_induce` | Extract memory items from a single trajectory using LLM |
| `reasoning_bank_induce_scaling` | Compare multiple trajectories and extract contrastive insights |
| `reasoning_bank_list` | List all stored memories |
| `reasoning_bank_delete` | Delete all memories for a given task ID |
| `reasoning_bank_count` | Get total memory count |

#### MCP Resource

| URI | Description |
|-----|-------------|
| `reasoning-bank://stats` | Returns `{"count": N}` |

#### Tool Parameters

**`reasoning_bank_retrieve`**
```
query: string    — Search query
top_k: int       — Number of results (default: 3)
```

**`reasoning_bank_add`**
```
task_id: string        — Task identifier
query: string          — Task description
memory_items: string[] — Memory texts to store
status: string         — "success" | "fail" (default: "success")
domain: string         — "web" | "coding" | "general" (default: "general")
```

**`reasoning_bank_induce`**
```
task_id: string     — Task identifier
query: string       — Task description
trajectory: string  — Full agent trajectory text
status: string      — "success" | "fail"
domain: string      — "web" | "coding" | "general" (default: "web")
```

**`reasoning_bank_induce_scaling`**
```
task_id: string                       — Task identifier
query: string                         — Task description
trajectories: {trajectory, status}[]  — Multiple trajectory entries
domain: string                        — "web" | "coding" | "general" (default: "web")
```

**`reasoning_bank_delete`**
```
task_id: string — Task identifier to delete
```

#### Configure MCP for Claude Code

Add to your project's `.mcp.json` or global `~/.claude/mcp.json`:

**SSE transport (connect to Docker or remote server):**
```json
{
  "mcpServers": {
    "reasoning-bank": {
      "type": "sse",
      "url": "http://localhost:9000/sse"
    }
  }
}
```

**Stdio transport (local process):**
```json
{
  "mcpServers": {
    "reasoning-bank": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "reasoning-bank-mcp"],
      "env": {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o",
        "LLM_API_KEY": "sk-...",
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "STORAGE": "chroma",
        "STORAGE_PATH": "./memories"
      }
    }
  }
}
```

#### Configure MCP for OpenCode

Add to your project's `.opencode.json` or global `~/.config/opencode/opencode.json`:

**SSE transport:**
```json
{
  "mcp": {
    "reasoning-bank": {
      "type": "sse",
      "url": "http://localhost:9000/sse"
    }
  }
}
```

**Stdio transport:**
```json
{
  "mcp": {
    "reasoning-bank": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "reasoning-bank-mcp"],
      "env": {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o",
        "LLM_API_KEY": "sk-...",
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "STORAGE": "chroma",
        "STORAGE_PATH": "./memories"
      }
    }
  }
}
```

## Docker

```bash
# Copy .env and fill in API keys
cp .env.example .env

# Start all services (ChromaDB + API + MCP)
cd docker && docker compose up
```

- **ChromaDB**: `http://localhost:8001`
- **API**: `http://localhost:8000`
- **MCP (SSE)**: `http://localhost:9000`

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
