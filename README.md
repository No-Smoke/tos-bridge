# TOS Bridge MCP Server v0.2.0

MCP Server bridging Claude Desktop/Code to VPS-hosted Qdrant + Neo4j. Provides
graph-enhanced semantic search (GraphRAG), dual-store writes, entity/relationship
CRUD, and cross-document discovery — all through a single MCP server.

## Tools (8)

### Infrastructure

**`check_tos_health`** — Monitor Qdrant and Neo4j connectivity, latency, node/collection counts.

**`sync_to_tos`** — Batch-sync patterns to Qdrant and/or Neo4j.
- `patterns`: List of `{text, source, category, importance}` dicts
- `target`: "qdrant", "neo4j", or "both" (default: "both")
- `collection`: Qdrant collection name

### Document Storage & Search (GraphRAG)

**`store_doc_with_graph`** — Store document in Qdrant with Neo4j graph cross-reference.
Creates a Document node in Neo4j with MENTIONS/REFERENCES edges to Entity nodes,
and upserts the embedded vector to Qdrant — all in one tool call. Handles both
named vectors (e.g. `dense`) and unnamed default vectors automatically.
- `text`: Document content for embedding
- `collection`: Qdrant collection name
- `title`: Document title
- `path`, `summary`, `metadata`: Optional enrichment
- `entities`: `[{name, type, importance}]` — creates Entity nodes linked via MENTIONS
- `relationships`: `[{target, rel_type, context}]` — creates additional typed edges
- Returns: `qdrant_id`, `neo4j_id`, entity/relationship counts

**`search_with_graph`** — Graph-enhanced semantic search (GraphRAG).
Embeds the query, searches Qdrant, then expands results via Neo4j shared-entity
traversal. Documents connected through the knowledge graph get score boosts.
- `query`, `collection`: Required
- `limit`: Max results (default 10)
- `relationship_boost`: Score boost for graph-connected docs (0.0–0.5, default 0.2)
- `include_graph_context`: Include entity connections in results
- Returns: Reranked results with original/boosted scores, `discovered_via_graph` flag

**`find_related_docs`** — Graph traversal from a known document.
Finds documents connected to a source document through shared Neo4j entities.
- `qdrant_id`: Source document's Qdrant UUID
- `max_depth`: Traversal depth 1–3 (default 2)
- `limit`: Max related docs (default 10)
- `include_paths`: Include relationship type paths
- Returns: Related documents with distance, shared entities, relationship types

### Entity & Relationship CRUD (v0.2.0)

**`create_or_update_entities`** — Create or update entities with observations.
Uses MERGE for deduplication: existing entities get new observations appended
(deduped), new entities are created. Replaces `neo4j-memory-remote:create_entities`
+ `add_observations`.
- `entities`: `[{name, type, observations: [...]}]`
- `check_existing`: If true (default), MERGE; if false, always CREATE
- Returns: created, updated, total counts

**`create_relationships`** — Create typed relationships between entities.
Uses MERGE safety — auto-creates missing entities as `concept` type.
Replaces `neo4j-memory-remote:create_relations`.
- `relationships`: `[{from_entity, to_entity, rel_type, context}]`
- Returns: created count

**`find_entities`** — Search entities by name (case-insensitive substring).
Returns entities with their observations, types, and which documents mention them.
Use for deduplication before creating new entities. Replaces
`neo4j-memory-remote:search_memories` + `find_memories_by_name`.
- `query`: Search string
- `entity_type`: Optional type filter
- `limit`: Max results (default 20)
- Returns: Matching entities with observations and document references

## Installation

```bash
git clone https://github.com/No-Smoke/tos-bridge.git
cd tos-bridge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | Yes | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | Yes | — | Qdrant API key |
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama embedding server |
| `OLLAMA_EMBED_MODEL` | No | `mxbai-embed-large` | Embedding model (1024 dims) |

## Claude Desktop Configuration

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
        "OLLAMA_URL": "http://nuc-ip:11434"
      }
    }
  }
}
```

## Neo4j Schema

Required indexes (create manually or let TOS-bridge auto-create):

```cypher
CREATE INDEX doc_qdrant_id IF NOT EXISTS FOR (d:Document) ON (d.qdrant_id);
CREATE INDEX doc_collection IF NOT EXISTS FOR (d:Document) ON (d.qdrant_collection);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
```

Node labels: `Document` (with `qdrant_id` cross-ref), `Entity` (with `observations` list).
Relationship types: `MENTIONS` (Document→Entity), `REFERENCES` (Document→Entity),
plus any custom types created via `create_relationships`.

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

## What TOS-bridge Does NOT Replace

These still require direct tool access:
- **qdrant-new**: Collection admin, delete_documents, hybrid_search, index_codebase, search_code
- **neo4j-mcp-remote**: Raw Cypher queries for complex multi-hop analysis
- **neo4j-memory-remote**: delete_entities, delete_observations, delete_relations, read_graph

## Changelog

### v0.2.0 (2026-03-14)
- **New tools**: `create_or_update_entities`, `create_relationships`, `find_entities`
- **Fixed**: `graph_tools.py` now handles named vectors via `_get_collection_vector_name`
- **Fixed**: `sync_to_tos` Qdrant path — was a placeholder, now embeds and upserts
- **Removed**: Redundant `graph_tools_1.py`
- **Chat-completion parity**: TOS-bridge can now serve as the primary memory tool for
  Claude's chat-completion workflow (Phase 2), replacing separate qdrant-new +
  neo4j-memory-remote calls

### v0.1.0 (2026-02-04)
- Initial release: health checks, pattern sync
- Graph-enhanced search (store_doc_with_graph, search_with_graph, find_related_docs)
- Circuit breakers for Qdrant, Neo4j, and Ollama
- Neo4j connection pooling with retry logic

## License

MIT
