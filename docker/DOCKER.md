# ReasoningBank Docker 部署指南

## 架构概览

```
                    ┌──────────────────────────────────────────────────┐
                    │            reasoning-bank (bridge 网络)            │
                    │                                                  │
  宿主机:8000  ←──  │  api 服务 (FastAPI)     MCP 服务 (SSE)  ←── 宿主机:9000
                    │      │                        │                   │
                    │      └──────────┬─────────────┘                   │
                    │                 ↓                                  │
                    │         chromadb 服务                              │
                    │         宿主机:8001 → 内部:8000                    │
                    └──────────────────────────────────────────────────┘
```

三个服务协同工作：

| 服务 | 端口 | 说明 |
|------|------|------|
| **chromadb** | `8001` | 向量数据库，存储记忆的嵌入向量 |
| **api** | `8000` | REST API，提供记忆的 CRUD 和归纳操作 |
| **mcp** | `9000` | MCP 服务器，通过 SSE 暴露工具供 Agent 调用 |

## 快速启动

### 前置条件

- Docker + Docker Compose
- `LLM_API_KEY` — embedding 和 LLM 共用的 API 密钥（Gemini / OpenAI / Anthropic 均通过此变量传入）

### 1. 创建 `.env` 文件

```bash
cd sdk/docker
cp .env.example .env
# 编辑 .env，填入 API Key
```

`.env` 文件示例（使用 Gemini embedding）：

```env
# API Key（embedding 和 LLM 共用）
LLM_API_KEY=your-google-api-key

# Embedding 配置（默认 gemini）
# EMBEDDING_PROVIDER=gemini
# EMBEDDING_MODEL=gemini-embedding-001

# LLM 配置（induce/induce_scaling 需要）
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
# LLM_API_BASE_URL=                   # 可选，自定义 API 地址

# 日志级别
LOG_LEVEL=INFO
```

如需使用 OpenAI embedding：

```env
LLM_API_KEY=your-openai-api-key
EMBEDDING_PROVIDER=openai
```

### 2. 启动服务

```bash
docker compose up --build -d
```

### 3. 验证服务

```bash
# ChromaDB 健康检查
curl http://localhost:8001/api/v1/heartbeat

# API 文档
open http://localhost:8000/docs

# MCP SSE 端点
curl http://localhost:9000/sse
```

### 4. 查看日志

```bash
# 实时日志
docker compose logs -f

# 单个服务日志
docker compose logs -f api
docker compose logs -f mcp
docker compose logs -f chromadb

# 宿主机日志文件
ls logs/api/    # api.log
ls logs/mcp/     # mcp.log
```

## API 接口

所有 API 路由以 `/memory` 为前缀。完整文档见 `http://localhost:8000/docs`。

### 记忆操作（需要 Embedding，不需要 LLM）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/memory/add` | 直接添加记忆条目 |
| POST | `/memory/retrieve` | 语义检索 top-k 记忆 |
| POST | `/memory/delete` | 删除指定 task_id 的记忆 |
| GET | `/memory/list` | 列出所有记忆 |
| GET | `/memory/count` | 获取记忆总数 |

### 归纳操作（需要 LLM）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/memory/induce` | 从单条轨迹归纳记忆 |
| POST | `/memory/induce_scaling` | 多轨迹对比归纳 |

### 请求示例

**添加记忆：**

```bash
curl -X POST http://localhost:8000/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "query": "fix login bug",
    "memory_items": ["use button#login-btn", "wait 2s for animation"],
    "status": "success",
    "domain": "web"
  }'
```

**检索记忆：**

```bash
curl -X POST http://localhost:8000/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "how to login", "top_k": 3}'
```

**轨迹归纳（需要 LLM）：**

```bash
curl -X POST http://localhost:8000/memory/induce \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-002",
    "query": "navigate to checkout",
    "trajectory": "1. Click cart icon -> 2. Select items -> 3. Click checkout",
    "status": "success",
    "domain": "web"
  }'
```

## MCP 工具

MCP 服务器通过 SSE 传输（端口 9000），暴露以下工具：

| 工具名 | 说明 | 需要 LLM |
|--------|------|----------|
| `reasoning_bank_retrieve` | 语义检索记忆 | 否 |
| `reasoning_bank_add` | 直接添加记忆 | 否 |
| `reasoning_bank_list` | 列出所有记忆 | 否 |
| `reasoning_bank_delete` | 删除记忆 | 否 |
| `reasoning_bank_count` | 记忆计数 | 否 |
| `reasoning_bank_induce` | 单轨迹归纳 | 是 |
| `reasoning_bank_induce_scaling` | 多轨迹对比归纳 | 是 |

资源：`reasoning-bank://stats` 返回记忆统计。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STORAGE` | `chroma` | 存储后端 |
| `CHROMA_HOST` | `chromadb` | ChromaDB 主机（Docker 内部） |
| `CHROMA_PORT` | `8000` | ChromaDB 端口（Docker 内部） |
| `EMBEDDING_PROVIDER` | `gemini` | 嵌入模型提供商（`gemini` / `openai`） |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | 嵌入模型名称 |
| `LLM_PROVIDER` | `openai` | LLM 提供商（`openai` / `anthropic` / `vertexai` / `google_ai`） |
| `LLM_MODEL` | `gpt-4o` | LLM 模型名称 |
| `LLM_API_KEY` | — | API 密钥（embedding 和 LLM 共用） |
| `LLM_API_BASE_URL` | — | 自定义 LLM API 地址 |
| `GOOGLE_GENAI_USE_VERTEXAI` | `False` | 是否使用 Vertex AI |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_DIR` | `/app/logs` | 日志目录（容器内） |

## 数据持久化

| 路径 | 用途 |
|------|------|
| `./chroma_data/` | ChromaDB 向量数据 |
| `./logs/api/` | API 服务日志 |
| `./logs/mcp/` | MCP 服务日志 |

## 运维命令

```bash
# 停止服务
docker compose down

# 停止并清除数据卷
docker compose down -v

# 重启单个服务
docker compose restart api

# 重建镜像（代码更新后）
docker compose up --build -d

# 查看服务状态
docker compose ps
```

## 运行测试

```bash
cd sdk

# 仅静态测试（不需要 Docker）
uv run pytest docker/tests/ -m "not integration" -v

# 全部测试（包括 Docker 构建）
uv run pytest docker/tests/ -v

# 仅集成测试（需要 Docker 守护进程 + LLM_API_KEY）
uv run pytest docker/tests/ -m integration -v
```

**测试文件说明：**

| 文件 | 说明 | 需要 Docker |
|------|------|-------------|
| `test_compose_config.py` | docker-compose.yml 配置验证 | 否 |
| `test_dockerfile.py` | Dockerfile 最佳实践 + 构建测试 | 静态部分不需要 |
| `test_integration.py` | 端到端集成测试 | 是 |

**注意：** 集成测试占用端口 8000/8001/9000，运行期间请确保这些端口空闲。`add`/`retrieve` 操作会调用嵌入模型，需要 `LLM_API_KEY`。

## 常见问题

### Q: 如何切换 Embedding 提供商？

编辑 `.env` 文件：

```env
# 使用 OpenAI embedding
EMBEDDING_PROVIDER=openai
LLM_API_KEY=your-openai-api-key
```

### Q: induce 接口返回 400 错误

`induce` 和 `induce_scaling` 需要配置 LLM。确认 `.env` 中设置了 `LLM_PROVIDER` 和 `LLM_MODEL`，且 `LLM_API_KEY` 有对应提供商的密钥。

### Q: 如何重置 ChromaDB 数据？

```bash
docker compose down -v
rm -rf chroma_data
docker compose up --build -d
```

### Q: 端口被占用？

修改 `docker-compose.yml` 中的端口映射：

```yaml
services:
  api:
    ports:
      - "8080:8000"  # 将宿主机端口改为 8080
```

`CHROMA_HOST` 和 `CHROMA_PORT` 不受影响，因为它们使用 Docker 内部网络。
