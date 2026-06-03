#!/usr/bin/env bash
# Run MCP Inspector to test the ReasoningBank MCP Server via Streamable HTTP.
#
# Usage:
#   bash scripts/inspector.sh                              # connect to localhost:9000
#   MCP_SERVER_URL=http://host:9000/mcp bash scripts/inspector.sh
#
# Prerequisites:
#   - MCP Server running in Streamable HTTP mode
#     (uv run reasoning-bank-mcp --transport streamable-http --port 9000)
#   - pnpm installed

set -euo pipefail

MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:9000/mcp}"

echo "🔧 Starting MCP Inspector..."
echo "   MCP Server URL: ${MCP_SERVER_URL}"
echo "   Inspector UI will open at http://localhost:6274"
echo ""
echo "   In the UI: select Streamable HTTP transport, enter ${MCP_SERVER_URL}, click Connect"
echo ""

exec pnpm dlx @modelcontextprotocol/inspector
