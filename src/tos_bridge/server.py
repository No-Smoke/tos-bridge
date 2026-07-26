"""
TOS Bridge MCP Server

Bridges Claude Projects knowledge base to VPS-hosted TOS (Qdrant + Neo4j).
Enables pattern extraction from Projects and synchronization to remote memory systems.
"""

import os
import json
import uuid
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from .arg_recovery import split_leaked_envelope

# Import graph-enhanced tools
from .graph_tools import (
    store_document_with_graph,
    graph_enhanced_search,
    find_related_documents,
    manage_entities,
    manage_relationships,
    search_entities,
    neo4j_session,
    _run_cypher,
    _utcnow_iso,
    # Phase 3 additions
    ensure_constraints,
    run_cypher_tool,
    get_entities_tool,
    delete_entities_tool,
    delete_observations_tool,
    delete_relationships_tool,
    manage_collections_tool,
    create_payload_index_tool,
    delete_point_tool,
    update_payload_tool,
    hybrid_search_tool,
)
from .embedding import get_embedding, warmup_ollama, embed_cache_stats


def _pattern_key(text: str, source: str) -> str:
    """Deterministic hash key for a Pattern node.

    MERGE-ing on raw `text` causes any pattern with the same text from a
    different source to clobber each other's metadata. Hash (source||text)
    so same-text-different-source patterns stay distinct.
    """
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tos-bridge")


# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # FIXED: Added API key
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Pydantic models
class Pattern(BaseModel):
    """Extracted pattern from Project knowledge"""
    text: str = Field(description="Pattern text content")
    source: str = Field(description="Source document name")
    category: str = Field(default="general", description="Pattern category")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class TOSHealth(BaseModel):
    """TOS health check response"""
    status: str
    timestamp: str
    qdrant: Dict[str, Any]
    neo4j: Dict[str, Any]
    last_sync: Optional[str] = None


# Circuit Breaker for external services
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                logger.info(f"Circuit breaker half-open for {func.__name__}")
            else:
                raise Exception(f"Circuit breaker open for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info(f"Circuit breaker closed for {func.__name__}")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened for {func.__name__} after {self.failure_count} failures")

            raise e

# Initialize circuit breakers
qdrant_circuit_breaker = CircuitBreaker()
neo4j_circuit_breaker = CircuitBreaker()

# Initialize FastMCP server
mcp = FastMCP("tos-bridge")


# Client initialization helpers
def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with API key and circuit breaker protection"""
    def _create_client():
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return qdrant_circuit_breaker.call(_create_client)


def get_neo4j_driver():
    """Legacy sync method - deprecated, use neo4j_session() from graph_tools"""
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD environment variable required")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Pattern extraction happens conversationally - no API access to Projects knowledge base


@mcp.tool()
async def sync_to_tos(
    patterns: List[Dict[str, Any]],
    target: str = "both",
    collection: str = "ebatt_patterns_v2"
) -> Dict[str, Any]:
    """
    Sync extracted patterns to TOS (Qdrant and/or Neo4j).
    
    Args:
        patterns: List of pattern dicts with 'text', 'source', 'category', 'importance'
        target: "qdrant", "neo4j", or "both"
        collection: Qdrant collection name
        
    Returns:
        Sync status and counts
    """
    results: Dict[str, Any] = {"status": "success", "qdrant": None, "neo4j": None}

    # Qdrant write — any failure propagates (no silent error swallow).
    if target in ("qdrant", "both"):
        from .graph_tools import get_qdrant_client as gt_qdrant, _get_collection_vector_name
        qdrant_client = gt_qdrant()
        vector_name = _get_collection_vector_name(qdrant_client, collection)

        points = []
        for p in patterns:
            text = p.get("text", "")
            if not text:
                continue
            embedding = await get_embedding(text)
            source = p.get("source", "unknown")
            # Use uuid5 from the same hash so Qdrant + Neo4j stay in lockstep —
            # re-syncing the same (source, text) updates the same point.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, _pattern_key(text, source)))
            payload = {
                "title": source,
                "summary": text[:200],
                "source": source,
                "category": p.get("category", "general"),
                "importance": p.get("importance", 0.5),
                "synced_at": _utcnow_iso(),
            }
            point_vector = {vector_name: embedding} if vector_name else embedding
            points.append(PointStruct(id=point_id, vector=point_vector, payload=payload))

        if points:
            qdrant_client.upsert(collection_name=collection, points=points)

        results["qdrant"] = {
            "stored": len(points),
            "collection": collection,
            "timestamp": _utcnow_iso(),
        }

    # Neo4j write — wrap so Qdrant partial success can still be reported.
    # This is the ONE legitimate place where we capture an exception to a
    # status field instead of re-raising: it's a deliberate multi-store
    # partial-success contract, not silent swallowing.
    if target in ("neo4j", "both"):
        try:
            async with neo4j_session() as session:
                # MERGE on a deterministic hash so same-text-different-source
                # patterns don't clobber each other.
                payload_rows = [
                    {
                        "hash": _pattern_key(p.get("text", ""), p.get("source", "unknown")),
                        "text": p.get("text", ""),
                        "source": p.get("source", "unknown"),
                        "category": p.get("category", "general"),
                        "importance": p.get("importance", 0.5),
                        "synced_at": _utcnow_iso(),
                    }
                    for p in patterns
                    if p.get("text")
                ]
                rows = await _run_cypher(session, """
                    UNWIND $patterns AS pattern
                    MERGE (p:Pattern {hash: pattern.hash})
                    ON CREATE SET p.text = pattern.text,
                                  p.source = pattern.source,
                                  p.category = pattern.category,
                                  p.importance = pattern.importance,
                                  p.synced_at = pattern.synced_at,
                                  p.created_at = datetime()
                    ON MATCH SET p.importance = pattern.importance,
                                 p.category = pattern.category,
                                 p.synced_at = pattern.synced_at,
                                 p.updated_at = datetime()
                    RETURN count(p) AS touched
                """, {"patterns": payload_rows})

                touched = rows[0]["touched"] if rows else 0
                results["neo4j"] = {
                    "nodes_touched": touched,
                    "timestamp": _utcnow_iso(),
                }
                logger.info(f"Synced {touched} patterns to Neo4j")

        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}")
            results["status"] = "partial"
            results["neo4j"] = {
                "status": "error",
                "error": str(e),
                "timestamp": _utcnow_iso(),
            }

    return results


@mcp.tool()
async def check_tos_health() -> Dict[str, Any]:
    """
    Check health status of TOS systems (Qdrant + Neo4j).
    
    Returns:
        Health metrics including latency, counts, and status
    """
    health: Dict[str, Any] = {
        "status": "unknown",
        "timestamp": _utcnow_iso(),
        "qdrant": {"status": "unknown"},
        "neo4j": {"status": "unknown"},
    }

    # Qdrant probe — light-weight: only count collections, skip per-collection
    # walk (the old version made 119 sequential get_collection calls).
    try:
        qdrant_client = get_qdrant_client()
        t0 = time.perf_counter()
        collections = qdrant_client.get_collections()
        latency_ms = (time.perf_counter() - t0) * 1000

        health["qdrant"] = {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "collections": len(collections.collections),
            "url": QDRANT_URL,
        }
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        health["qdrant"] = {"status": "error", "error": str(e), "url": QDRANT_URL}

    # Neo4j probe — UNWIND labels(n) so multi-label nodes count under each of
    # their labels, plus an explicit unlabeled bucket.
    try:
        t0 = time.perf_counter()
        async with neo4j_session() as session:
            label_rows = await _run_cypher(session, """
                MATCH (n)
                WITH n, labels(n) AS lbls
                UNWIND CASE WHEN size(lbls) = 0 THEN ['_unlabeled'] ELSE lbls END AS lbl
                RETURN lbl AS label, count(*) AS node_count
            """)
            rel_rows = await _run_cypher(session, """
                MATCH ()-[r]->() RETURN count(r) AS rel_count
            """)
        latency_ms = (time.perf_counter() - t0) * 1000

        node_counts = {row["label"]: row["node_count"] for row in label_rows}
        rel_count = rel_rows[0]["rel_count"] if rel_rows else 0

        health["neo4j"] = {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "nodes": node_counts,
            "relationships": rel_count,
            "uri": NEO4J_URI,
            "connection_pool": "enabled",
        }
        logger.info("Neo4j health check passed - latency: %.2fms", latency_ms)

    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health["neo4j"] = {"status": "error", "error": str(e), "uri": NEO4J_URI}

    # Overall status
    qstat = health["qdrant"]["status"]
    nstat = health["neo4j"]["status"]
    if qstat == "healthy" and nstat == "healthy":
        health["status"] = "healthy"
    elif qstat == "error" and nstat == "error":
        health["status"] = "error"
    else:
        health["status"] = "degraded"

    return health


# ============================================================================
# Register Graph-Enhanced Tools
# ============================================================================

@mcp.tool()
async def store_doc_with_graph(
    text: str,
    collection: str = "",
    title: str = "",
    path: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Dict[str, Any] = {},
    entities: List[Dict[str, Any]] = [],
    relationships: List[Dict[str, Any]] = []
) -> Dict[str, Any]:
    """
    Store document in Qdrant with Neo4j graph cross-reference.

    Args:
        text: Document content for embedding
        collection: Qdrant collection name (always supply this)
        title: Document title (always supply this)
        path: Optional file path
        summary: Optional brief summary
        metadata: Additional metadata for Qdrant
        entities: List of entities [{name, type, importance}]
        relationships: List of relationships [{target, rel_type, context}]

    Returns:
        qdrant_id, neo4j_id, entities_created, relationships_created
    """
    supplied = {
        "collection": collection,
        "title": title,
        "path": path,
        "summary": summary,
        "metadata": metadata,
        "entities": entities,
        "relationships": relationships,
    }

    # `collection` and `title` carry defaults so that a call whose parameters
    # were folded into `text` by a malformed tool-call envelope still reaches
    # this function; client-side schema validation would otherwise reject it
    # before the envelope could be unpicked.
    if not collection or not title:
        text, recovered = split_leaked_envelope(text)
        if recovered:
            logger.warning(
                "Recovered parameters from a leaked tool-call envelope",
                extra={"event": "envelope_recovered", "params": sorted(recovered)},
            )
            for key, value in recovered.items():
                if not supplied.get(key):
                    supplied[key] = value

    missing = [k for k in ("collection", "title") if not supplied[k]]
    if missing:
        raise ToolError(
            f"store_doc_with_graph received no {' and no '.join(missing)}. "
            "Nothing was stored. This is a malformed tool call: the parameters "
            "were never transmitted, so re-send the call with every parameter "
            "closed by its own delimiter. Shorter text is less likely to trip it."
        )

    return await store_document_with_graph(
        text=text,
        collection=supplied["collection"],
        title=supplied["title"],
        path=supplied["path"],
        summary=supplied["summary"],
        metadata=supplied["metadata"],
        entities=supplied["entities"],
        relationships=supplied["relationships"],
    )


@mcp.tool()
async def search_with_graph(
    query: str,
    collection: str,
    limit: int = 10,
    relationship_boost: float = 0.2,
    include_graph_context: bool = True
) -> Dict[str, Any]:
    """
    Graph-enhanced semantic search combining Qdrant vectors with Neo4j relationships.
    
    Args:
        query: Search query text
        collection: Qdrant collection to search
        limit: Maximum results to return
        relationship_boost: Score boost for graph-connected docs (0.0-0.5)
        include_graph_context: Include entity connections in results
    
    Returns:
        Reranked results with graph context
    """
    return await graph_enhanced_search(
        query=query,
        collection=collection,
        limit=limit,
        relationship_boost=relationship_boost,
        include_graph_context=include_graph_context
    )


@mcp.tool()
async def find_related_docs(
    qdrant_id: str,
    max_depth: int = 2,
    limit: int = 10,
    include_paths: bool = True
) -> Dict[str, Any]:
    """
    Find documents related to a given document via Neo4j graph traversal.
    
    Args:
        qdrant_id: Source document's Qdrant UUID
        max_depth: Maximum traversal depth (1-3)
        limit: Maximum related documents
        include_paths: Include relationship paths in results
    
    Returns:
        Related documents with relationship context
    """
    return await find_related_documents(
        qdrant_id=qdrant_id,
        max_depth=max_depth,
        limit=limit,
        include_paths=include_paths
    )


# ============================================================================
# Entity & Relationship Management Tools (chat-completion parity)
# ============================================================================

@mcp.tool()
async def create_or_update_entities(
    entities: List[Dict[str, Any]],
    check_existing: bool = True
) -> Dict[str, Any]:
    """
    Create or update entities in Neo4j with observations.
    Replaces neo4j-memory-remote:create_entities + add_observations.
    
    Args:
        entities: List of entity dicts, each with:
            - name (str, required): Entity name
            - type (str): Entity type e.g. "project", "tool", "person", "concept"
            - observations (list[str]): Facts about this entity
        check_existing: If True, MERGE (dedup); if False, always CREATE
    
    Returns:
        Dict with created, updated, and total counts
    """
    return await manage_entities(
        entities=entities,
        check_existing=check_existing
    )


@mcp.tool()
async def create_relationships(
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create relationships between entities in Neo4j.
    Replaces neo4j-memory-remote:create_relations.
    
    Args:
        relationships: List of relationship dicts, each with:
            - from_entity (str, required): Source entity name
            - to_entity (str, required): Target entity name
            - rel_type (str): e.g. "USES", "DEPENDS_ON", "PART_OF", "RELATES_TO"
            - context (str): Optional description of the relationship
    
    Returns:
        Dict with created count
    """
    return await manage_relationships(
        relationships=relationships
    )


@mcp.tool()
async def find_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for entities in Neo4j by name substring match.
    Use for deduplication before creating new entities.
    Replaces neo4j-memory-remote:search_memories + find_memories_by_name.
    
    Args:
        query: Search string (case-insensitive substring match on name)
        entity_type: Optional filter by type (e.g. "project", "tool")
        limit: Maximum results (default 20)
    
    Returns:
        Matching entities with observations and document references
    """
    return await search_entities(
        query=query,
        entity_type=entity_type,
        limit=limit
    )


# ============================================================================
# Phase 3 tools — full coverage so the dedicated neo4j-mcp-remote and
# qdrant-new MCP servers can be retired.
# ============================================================================


@mcp.tool()
async def run_cypher(
    query: str,
    params: Dict[str, Any] = {},
    read_only: bool = True,
    limit: int = 100,
) -> Dict[str, Any]:
    """Execute arbitrary Cypher against the project's Neo4j.

    Defaults to read-only mode — write keywords (CREATE/MERGE/DELETE/SET/
    REMOVE/DROP/FOREACH/LOAD) are rejected unless `read_only=False` is set
    explicitly. Always use $param placeholders; never string-interpolate
    user input.

    Args:
        query: Cypher statement.
        params: Parameter map for $name placeholders.
        read_only: Reject write keywords if True (default).
        limit: Max rows returned (server-side cap, default 100).

    Returns:
        {rows, row_count, truncated, classification}
    """
    return await run_cypher_tool(query=query, params=params, read_only=read_only, limit=limit)


@mcp.tool()
async def get_entities(
    names: List[str],
    include_documents: bool = True,
) -> Dict[str, Any]:
    """Fetch entities by exact name (case-sensitive). Faster than search_entities
    when you already know which entities you want. Returns both found and missing
    names so you can detect deletions or typos.
    """
    return await get_entities_tool(names=names, include_documents=include_documents)


@mcp.tool()
async def delete_entities(
    names: List[str],
    detach: bool = True,
) -> Dict[str, Any]:
    """Delete entities by name.

    DETACH DELETE is the default: connected relationships are deleted too.
    Pass `detach=False` to fail-loud when relationships exist.
    """
    return await delete_entities_tool(names=names, detach=detach)


@mcp.tool()
async def delete_observations(
    name: str,
    observations: List[str],
) -> Dict[str, Any]:
    """Remove specific observation strings from an entity, preserving the rest.

    Idempotent: missing observations are no-ops. Use to retract a specific
    fact without losing everything else known about the entity.
    """
    return await delete_observations_tool(name=name, observations=observations)


@mcp.tool()
async def delete_relationships(
    from_entity: str,
    to_entity: str,
    rel_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete relationships between two entities.

    With no rel_type, deletes ALL relationships in the from→to direction.
    With a rel_type, only deletes that specific type. rel_type is whitelisted
    against ^[A-Z_][A-Z0-9_]+$ to block Cypher injection.
    """
    return await delete_relationships_tool(
        from_entity=from_entity, to_entity=to_entity, rel_type=rel_type
    )


@mcp.tool()
async def manage_collections(
    action: str,
    name: Optional[str] = None,
    vector_size: int = 1024,
    distance: str = "Cosine",
    on_disk: bool = False,
) -> Dict[str, Any]:
    """Manage Qdrant collections.

    Args:
        action: list | info | create | delete | recreate
        name: Collection name (required except for 'list').
        vector_size: For create/recreate. Default 1024 matches mxbai-embed-large.
        distance: One of Cosine, Euclid, Dot, Manhattan.
        on_disk: Store vectors on disk for large collections.
    """
    return await manage_collections_tool(
        action=action, name=name, vector_size=vector_size, distance=distance, on_disk=on_disk
    )


@mcp.tool()
async def create_payload_index(
    collection: str,
    field: str,
    field_schema: str = "keyword",
) -> Dict[str, Any]:
    """Create a Qdrant payload index on a collection field.

    Required for filtered queries to be fast. `field_schema` is one of:
    keyword, integer, float, bool, geo, text, datetime, uuid.
    """
    return await create_payload_index_tool(
        collection=collection, field=field, field_schema=field_schema
    )


@mcp.tool()
async def delete_point(
    collection: str,
    point_id: str,
) -> Dict[str, Any]:
    """Delete a single point from a Qdrant collection by id."""
    return await delete_point_tool(collection=collection, point_id=point_id)


@mcp.tool()
async def update_payload(
    collection: str,
    point_id: str,
    payload: Dict[str, Any],
    replace: bool = False,
) -> Dict[str, Any]:
    """Update or replace the payload of a Qdrant point.

    `replace=False` (default) merges into the existing payload; True overwrites
    the entire payload with `payload`.
    """
    return await update_payload_tool(
        collection=collection, point_id=point_id, payload=payload, replace=replace
    )


@mcp.tool()
async def hybrid_search(
    query: str,
    collection: str,
    limit: int = 10,
    title_filter: Optional[str] = None,
    payload_filter: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Dense semantic search with optional payload filtering.

    Practical alternative to BM25+vector hybrid (which requires sparse-vector
    collection config). Accepts a title substring filter and a payload-equality
    map applied server-side by Qdrant. For a pure vector query, omit both
    filter arguments.
    """
    return await hybrid_search_tool(
        query=query,
        collection=collection,
        limit=limit,
        title_filter=title_filter,
        payload_filter=payload_filter,
    )


@mcp.tool()
async def embed_cache_status() -> Dict[str, Any]:
    """Return embedding-cache statistics (size, hits, misses, hit rate).

    Useful for tuning OLLAMA_EMBED_CACHE_SIZE and confirming the cache is
    actually being hit on repeat queries.
    """
    return {
        "status": "success",
        "cache": embed_cache_stats(),
        "timestamp": _utcnow_iso(),
    }


if __name__ == "__main__":
    mcp.run()
