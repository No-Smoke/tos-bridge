"""
TOS Bridge MCP Server

Bridges Claude Projects knowledge base to VPS-hosted TOS (Qdrant + Neo4j).
Enables pattern extraction from Projects and synchronization to remote memory systems.
"""

import os
import json
import uuid
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

# Import graph-enhanced tools
from .graph_tools import (
    store_document_with_graph,
    graph_enhanced_search,
    find_related_documents,
    manage_entities,
    manage_relationships,
    search_entities,
    neo4j_session
)
from .embedding import get_embedding, warmup_ollama

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
    collection: str = "ebatt_pattern_library"
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
    results = {"status": "success", "qdrant": None, "neo4j": None}
    
    try:
        if target in ["qdrant", "both"]:
            from .graph_tools import get_qdrant_client as gt_qdrant, _get_collection_vector_name
            qdrant_client = gt_qdrant()
            vector_name = _get_collection_vector_name(qdrant_client, collection)

            points = []
            for p in patterns:
                text = p.get("text", "")
                if not text:
                    continue
                embedding = await get_embedding(text)
                point_id = str(uuid.uuid4())
                payload = {
                    "title": p.get("source", "pattern"),
                    "summary": text[:200],
                    "source": p.get("source", "unknown"),
                    "category": p.get("category", "general"),
                    "importance": p.get("importance", 0.5),
                    "synced_at": datetime.utcnow().isoformat()
                }
                point_vector = {vector_name: embedding} if vector_name else embedding
                points.append(PointStruct(id=point_id, vector=point_vector, payload=payload))

            if points:
                qdrant_client.upsert(collection_name=collection, points=points)

            results["qdrant"] = {
                "stored": len(points),
                "collection": collection,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if target in ["neo4j", "both"]:
            try:
                async with neo4j_session() as session:
                    # Create Pattern nodes and relationships
                    result = session.run("""
                        UNWIND $patterns AS pattern
                        MERGE (p:Pattern {text: pattern.text})
                        SET p.source = pattern.source,
                            p.category = pattern.category,
                            p.importance = pattern.importance,
                            p.synced_at = pattern.synced_at
                        RETURN count(p) as created
                    """, patterns=[
                        {
                            "text": p.get("text", ""),
                            "source": p.get("source", "unknown"),
                            "category": p.get("category", "general"),
                            "importance": p.get("importance", 0.5),
                            "synced_at": datetime.utcnow().isoformat()
                        }
                        for p in patterns
                    ])

                    created = result.single()["created"]
                    results["neo4j"] = {
                        "nodes_created": created,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    logger.info(f"Successfully synced {created} patterns to Neo4j")

            except Exception as e:
                logger.error(f"Neo4j sync failed: {e}")
                results["neo4j"] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return results
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@mcp.tool()
async def check_tos_health() -> Dict[str, Any]:
    """
    Check health status of TOS systems (Qdrant + Neo4j).
    
    Returns:
        Health metrics including latency, counts, and status
    """
    health = {
        "status": "unknown",
        "timestamp": datetime.utcnow().isoformat(),
        "qdrant": {"status": "unknown"},
        "neo4j": {"status": "unknown"}
    }
    
    # Check Qdrant
    try:
        qdrant_client = get_qdrant_client()
        start = datetime.utcnow()
        
        collections = qdrant_client.get_collections()
        latency = (datetime.utcnow() - start).total_seconds() * 1000
        
        # Get collection info
        collection_info = {}
        for col in collections.collections:
            info = qdrant_client.get_collection(col.name)
            collection_info[col.name] = {
                "points": info.points_count,
                "vectors": info.vectors_count if hasattr(info, 'vectors_count') else info.points_count
            }
        
        health["qdrant"] = {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "collections": len(collection_info),
            "url": QDRANT_URL
        }
    except Exception as e:
        health["qdrant"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check Neo4j
    try:
        start = datetime.utcnow()

        async with neo4j_session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as node_count
            """)

            latency = (datetime.utcnow() - start).total_seconds() * 1000

            node_counts = {}
            for record in result:
                label = record["label"] or "unlabeled"
                node_counts[label] = record["node_count"]

            # Get relationship counts
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as rel_count
            """)
            rel_count = rel_result.single()["rel_count"]

        health["neo4j"] = {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "nodes": node_counts,
            "relationships": rel_count,
            "uri": NEO4J_URI,
            "connection_pool": "enabled"
        }
        logger.info(f"Neo4j health check passed - latency: {latency:.2f}ms")

    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health["neo4j"] = {
            "status": "error",
            "error": str(e),
            "uri": NEO4J_URI
        }
    
    # Overall status
    if health["qdrant"]["status"] == "healthy" and health["neo4j"]["status"] == "healthy":
        health["status"] = "healthy"
    elif health["qdrant"]["status"] == "error" and health["neo4j"]["status"] == "error":
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
    collection: str,
    title: str,
    path: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Store document in Qdrant with Neo4j graph cross-reference.
    
    Args:
        text: Document content for embedding
        collection: Qdrant collection name
        title: Document title
        path: Optional file path
        summary: Optional brief summary
        metadata: Additional metadata for Qdrant
        entities: List of entities [{name, type, importance}]
        relationships: List of relationships [{target, rel_type, context}]
    
    Returns:
        qdrant_id, neo4j_id, entities_created, relationships_created
    """
    return await store_document_with_graph(
        text=text,
        collection=collection,
        title=title,
        path=path,
        summary=summary,
        metadata=metadata,
        entities=entities,
        relationships=relationships
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


if __name__ == "__main__":
    mcp.run()
