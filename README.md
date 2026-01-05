# TOS Bridge MCP Server

MCP Server that bridges Claude Desktop/Code to VPS-hosted Token Optimization System (Qdrant + Neo4j).

## Features

- **Health Checks**: Monitor Qdrant and Neo4j connectivity
- **Pattern Sync**: Sync extracted patterns to both vector and graph databases
- **Dual Storage**: Write to Qdrant (semantic search) and Neo4j (relationships) simultaneously

## Installation

### Option 1: uvx (Recommended for MCP)

```bash
# Run directly without installation
uvx --from git+https://github.com/No-Smoke/tos-bridge tos-bridge
```

### Option 2: pip install from GitHub

```bash
pip install git+https://github.com/No-Smoke/tos-bridge.git
tos-bridge
```

### Option 3: Docker

```bash
docker run -i --rm \
  -e QDRANT_URL=http://your-qdrant:6333 \
  -e QDRANT_API_KEY=your-api-key \
  -e NEO4J_URI=bolt://your-neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=your-password \
  ghcr.io/no-smoke/tos-bridge:latest
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | Yes | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | Yes | - | Qdrant API key |
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | - | Neo4j password |

### Claude Desktop Configuration

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tos-bridge": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/No-Smoke/tos-bridge", "tos-bridge"],
      "env": {
        "QDRANT_URL": "http://your-vps:6333",
        "QDRANT_API_KEY": "your-api-key",
        "NEO4J_URI": "bolt://your-vps:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password"
      }
    }
  }
}
```

### Claude Code Configuration

```bash
claude mcp add --scope user tos-bridge \
  --env QDRANT_URL=http://your-vps:6333 \
  --env QDRANT_API_KEY=your-api-key \
  --env NEO4J_URI=bolt://your-vps:7687 \
  --env NEO4J_USER=neo4j \
  --env NEO4J_PASSWORD=your-password \
  -- uvx --from git+https://github.com/No-Smoke/tos-bridge tos-bridge
```

## Tools

### `check_tos_health`

Check health status of both Qdrant and Neo4j systems.

**Returns:**
- Overall status (healthy/degraded/error)
- Qdrant: latency, collection count, URL
- Neo4j: latency, node counts by label, relationship count

### `sync_to_tos`

Sync patterns to Qdrant and/or Neo4j.

**Parameters:**
- `patterns`: List of pattern dicts with `text`, `source`, `category`, `importance`
- `target`: "qdrant", "neo4j", or "both" (default: "both")
- `collection`: Qdrant collection name (default: "ebatt_pattern_library")

## Development

```bash
# Clone repo
git clone https://github.com/No-Smoke/tos-bridge.git
cd tos-bridge

# Install in development mode
pip install -e .

# Run server
tos-bridge
```

## License

MIT
