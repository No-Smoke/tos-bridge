# TOS Bridge MCP Server

MCP Server that bridges Claude Desktop/Code to VPS-hosted Token Optimization System (Qdrant + Neo4j).

## Features

- **Health Checks**: Monitor Qdrant and Neo4j connectivity
- **Pattern Sync**: Sync extracted patterns to both vector and graph databases
- **Graph-Enhanced Search**: Semantic search with Neo4j relationship boosting (GraphRAG)
- **Cross-Reference Storage**: Store documents in Qdrant with Neo4j entity relationships
- **Related Document Discovery**: Find related documents via graph traversal

## Installation

### Option 1: Local venv (Recommended for development)

```bash
git clone https://github.com/No-Smoke/tos-bridge.git
cd tos-bridge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Option 2: uvx (For MCP)

```bash
uvx --from git+https://github.com/No-Smoke/tos-bridge tos-bridge
```

### Option 3: Docker

```bash
docker run -i --rm \
  -e QDRANT_URL=http://your-qdrant:6333 \
  -e QDRANT_API_KEY=your-api-key \
  -e NEO4J_URI=bolt://your-neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=your-password \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
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
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama server for embeddings |
| `OLLAMA_EMBED_MODEL` | No | `mxbai-embed-large` | Embedding model (1024 dims) |

### Claude Desktop Configuration

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tos-bridge": {
      "command": "/path/to/tos-bridge/.venv/bin/python",
      "args": ["-m", "tos_bridge"],
      "env": {
        "QDRANT_URL": "http://your-vps:6333",
        "QDRANT_API_KEY": "your-api-key",
        "NEO4J_URI": "bolt://your-vps:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

## Tools

### `check_tos_health`

Check health status of both Qdrant and Neo4j systems.

**Returns:** Overall status, latency metrics, collection/node counts

### `sync_to_tos`

Sync patterns to Qdrant and/or Neo4j.

**Parameters:**
- `patterns`: List of pattern dicts with `text`, `source`, `category`, `importance`
- `target`: "qdrant", "neo4j", or "both" (default: "both")
- `collection`: Qdrant collection name

### `store_doc_with_graph` *(NEW)*

Store document in Qdrant with Neo4j graph cross-reference. Creates bidirectional links between vector embeddings and knowledge graph entities.

**Parameters:**
- `text`: Document content for embedding
- `collection`: Qdrant collection name
- `title`: Document title
- `path`: Optional file path
- `summary`: Optional brief summary
- `entities`: List of entities `[{name, type, importance}]`
- `relationships`: List of relationships `[{target, rel_type, context}]`

**Returns:** `qdrant_id`, `neo4j_id`, `entities_created`, `relationships_created`

### `search_with_graph` *(NEW)*

Graph-enhanced semantic search combining Qdrant vectors with Neo4j relationships. Discovers additional documents via shared entities and boosts scores for graph-connected results.

**Parameters:**
- `query`: Search query text
- `collection`: Qdrant collection to search
- `limit`: Maximum results (default: 10)
- `relationship_boost`: Score boost for connected docs (0.0-0.5, default: 0.2)
- `include_graph_context`: Include entity connections in results

**Returns:** Reranked results with `original_score`, `boosted_score`, `connected_entities`, `discovered_via_graph`

### `find_related_docs` *(NEW)*

Find documents related to a given document via Neo4j graph traversal through shared entities.

**Parameters:**
- `qdrant_id`: Source document's Qdrant UUID
- `max_depth`: Maximum traversal depth (1-3, default: 2)
- `limit`: Maximum related documents (default: 10)
- `include_paths`: Include relationship paths in results

**Returns:** Related documents with `distance`, `shared_entities`, `relationship_types`

## Neo4j Schema

The graph tools require these Neo4j indexes (created automatically or manually):

```cypher
CREATE INDEX doc_qdrant_id IF NOT EXISTS FOR (d:Document) ON (d.qdrant_id);
CREATE INDEX doc_qdrant_collection IF NOT EXISTS FOR (d:Document) ON (d.qdrant_collection);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Claude Desktop │────▶│   TOS-Bridge    │
│   or Claude.ai  │     │   MCP Server    │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │  Qdrant  │ │  Neo4j   │ │  Ollama  │
             │ (vectors)│ │ (graph)  │ │(embeddings)
             └──────────┘ └──────────┘ └──────────┘
```

## Development

```bash
# Clone and setup
git clone https://github.com/No-Smoke/tos-bridge.git
cd tos-bridge
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run server
python -m tos_bridge
```

## License

MIT
