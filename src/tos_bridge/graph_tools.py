"""
Graph-enhanced search tools for TOS-Bridge.

Provides cross-referencing tools spanning Qdrant vectors and Neo4j knowledge graph.

Requires:
- Qdrant server with collections configured
- Neo4j with Document/Entity indexes
- Ollama for embeddings
"""
import os
import re
import uuid
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from .embedding import get_embedding

logger = logging.getLogger("tos-bridge.graph")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Global connection pool
_neo4j_driver = None
_last_health_check = 0
_health_check_interval = 60  # seconds

# Cypher safety
# Relationship types are interpolated into Cypher (Neo4j does not support
# parameterized rel types). Whitelist strictly: uppercase ASCII letters,
# digits, underscore; must start with a letter or underscore.
_REL_TYPE_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")


def _validate_rel_type(rel_type: str, default: str = "RELATES_TO") -> str:
    """Normalize and validate a Neo4j relationship type.

    Uppercases, replaces spaces with underscores, then enforces the whitelist.
    Falls back to `default` (also whitelisted) if validation fails.
    """
    if not rel_type:
        return default
    candidate = rel_type.strip().upper().replace(" ", "_").replace("-", "_")
    if _REL_TYPE_RE.match(candidate):
        return candidate
    logger.warning("Rejected rel_type %r (not in whitelist); using %s", rel_type, default)
    return default


# UUID5 namespace for deterministic document ids derived from path/title.
# Stable across runs so re-storing the same document updates in place.
_TOS_NAMESPACE = uuid.UUID("e3b0c442-98fc-1c14-9afb-f4c8996fb924")


def _stable_doc_id(path: Optional[str], title: str) -> str:
    """Deterministic Qdrant point id derived from path (preferred) or title.

    Re-storing the same path produces the same id, making Qdrant upserts and
    Neo4j MERGEs idempotent without external bookkeeping.
    """
    key = (path or "").strip()
    if not key:
        key = f"title:{title}"
    return str(uuid.uuid5(_TOS_NAMESPACE, key))


def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp using timezone-aware now() (datetime.utcnow is deprecated)."""
    return datetime.now(timezone.utc).isoformat()


async def _run_cypher(session, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run a Cypher query in a worker thread to avoid blocking the asyncio loop.

    The neo4j Python driver's .session()/.run() are synchronous and block on I/O;
    calling them directly serialises all concurrent MCP tool invocations. Wrapping
    in asyncio.to_thread restores concurrency while keeping the rest of the code
    sync-shaped for readability.

    Returns a list of records as dicts (eagerly consumed inside the thread).
    """
    def _work():
        result = session.run(query, params or {})
        return result.data()
    return await asyncio.to_thread(_work)


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with API key."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _get_collection_vector_name(qdrant_client: QdrantClient, collection: str) -> Optional[str]:
    """
    Detect whether a collection uses named vectors or an unnamed default vector.
    Returns the vector name (e.g. 'dense') for named vectors, or None for unnamed.
    """
    try:
        info = qdrant_client.get_collection(collection)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            # Named vectors — return first vector name (typically 'dense')
            return next(iter(vectors_config))
        else:
            # Unnamed default vector (VectorParams object)
            return None
    except Exception:
        return None


async def get_neo4j_driver_with_retry(max_retries: int = 3):
    """Return a healthy Neo4j driver, retrying with exponential backoff.

    A cached driver is reused while its last successful health check is fresher
    than `_health_check_interval`. On any failure (driver construction OR ping),
    the existing driver is torn down before the next attempt, so we never return
    a stale broken driver. Health-check pings run in a worker thread because
    the underlying neo4j driver API is synchronous.
    """
    global _neo4j_driver, _last_health_check

    current_time = time.time()

    # Hot path: cached driver is still inside the freshness window.
    if _neo4j_driver is not None and (current_time - _last_health_check) <= _health_check_interval:
        return _neo4j_driver

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            # Construct a fresh driver if we lack one (initial call or after teardown).
            if _neo4j_driver is None:
                if not NEO4J_PASSWORD:
                    raise ValueError("NEO4J_PASSWORD environment variable required")
                _neo4j_driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                    max_connection_lifetime=300,  # 5 minutes
                    max_connection_pool_size=10,  # raised in Phase 2
                    connection_timeout=20,
                )

            # Probe in a thread — session creation and run() are sync I/O.
            def _ping():
                assert _neo4j_driver is not None
                with _neo4j_driver.session() as session:
                    session.run("RETURN 1 AS test").single()
            await asyncio.to_thread(_ping)

            _last_health_check = current_time
            return _neo4j_driver

        except (ServiceUnavailable, TransientError, ValueError) as e:
            last_error = e
            # Always tear down a possibly-bad driver before the next attempt,
            # closing it off the event loop because close() can block.
            if _neo4j_driver is not None:
                try:
                    await asyncio.to_thread(_neo4j_driver.close)
                except Exception:
                    pass
                _neo4j_driver = None

            if attempt == max_retries - 1:
                logger.error("Neo4j connection failed after %d attempts: %s", max_retries, e)
                raise

            wait_time = 2 ** attempt
            logger.warning(
                "Neo4j connection failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, max_retries, wait_time, e,
            )
            await asyncio.sleep(wait_time)

    # Defensive — the loop above always raises on exhaustion.
    raise last_error if last_error else RuntimeError("Neo4j connection exhausted retries")


@asynccontextmanager
async def neo4j_session():
    """Async context manager for Neo4j sessions with auto-recovery."""
    driver = await get_neo4j_driver_with_retry()
    session = driver.session()
    try:
        yield session
    except Exception as e:
        # Reset connection on error
        global _neo4j_driver, _last_health_check
        _last_health_check = 0  # Force health check on next use
        raise e
    finally:
        session.close()


def get_neo4j_driver():
    """Legacy sync method - deprecated, use get_neo4j_driver_with_retry()."""
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD environment variable required")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ============================================================================
# Tool 1: store_document_with_graph
# ============================================================================

async def store_document_with_graph(
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
        summary: Optional brief summary (defaults to first 200 chars)
        metadata: Additional metadata for Qdrant
        entities: List of entities [{name, type, importance}]
        relationships: List of relationships [{target, rel_type, context}]
    
    Returns:
        Dict with qdrant_id, neo4j_id, entities_created, relationships_created
    """
    # 1. Generate embedding (may raise — let it propagate)
    embedding = await get_embedding(text)

    # 2. Compute a stable Qdrant point id from path/title so re-stores update
    #    the same record instead of creating duplicates.
    qdrant_id = _stable_doc_id(path, title)
    entity_names = [e.get("name") for e in (entities or [])]
    doc_summary = summary or text[:200]

    qdrant_metadata = {
        "title": title,
        "path": path or "",
        "summary": doc_summary,
        "neo4j_entities": entity_names,
        "synced_at": _utcnow_iso(),
        **(metadata or {}),
    }

    # 3. Upsert to Qdrant (idempotent because the id is deterministic)
    qdrant_client = get_qdrant_client()
    vector_name = _get_collection_vector_name(qdrant_client, collection)
    point_vector = {vector_name: embedding} if vector_name else embedding

    qdrant_client.upsert(
        collection_name=collection,
        points=[PointStruct(id=qdrant_id, vector=point_vector, payload=qdrant_metadata)],
    )

    # 4. MERGE Document node and create relationships in Neo4j
    async with neo4j_session() as session:
        doc_rows = await _run_cypher(session, """
            MERGE (d:Document {qdrant_id: $qdrant_id})
            ON CREATE SET
                d.name = $title,
                d.qdrant_collection = $collection,
                d.title = $title,
                d.path = $path,
                d.summary = $summary,
                d.created_at = datetime(),
                d.updated_at = datetime()
            ON MATCH SET
                d.title = $title,
                d.qdrant_collection = $collection,
                d.path = $path,
                d.summary = $summary,
                d.updated_at = datetime()
            RETURN elementId(d) AS neo4j_id
        """, {
            "qdrant_id": qdrant_id,
            "collection": collection,
            "title": title,
            "path": path or "",
            "summary": doc_summary,
        })
        if not doc_rows:
            raise RuntimeError(f"Neo4j MERGE failed to return Document row for {qdrant_id}")
        neo4j_id = doc_rows[0]["neo4j_id"]

        # MENTIONS relationships from entities (rel type is fixed, no injection risk)
        entities_created = 0
        for entity in (entities or []):
            entity_name = entity.get("name")
            if not entity_name:
                continue
            entity_type = entity.get("type", "concept")
            await _run_cypher(session, """
                MATCH (d:Document {qdrant_id: $qdrant_id})
                MERGE (e:Entity {name: $entity_name})
                ON CREATE SET e.type = $entity_type,
                              e.observations = [$initial_obs],
                              e.created_at = datetime()
                MERGE (d)-[r:MENTIONS]->(e)
                SET r.importance = $importance,
                    r.updated_at = datetime()
            """, {
                "qdrant_id": qdrant_id,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "importance": entity.get("importance", 0.5),
                "initial_obs": f"Referenced in document: {title}",
            })
            entities_created += 1

        # Additional rel types — validated against whitelist to block Cypher injection
        rels_created = 0
        for rel in (relationships or []):
            target = rel.get("target")
            if not target:
                continue
            rel_type = _validate_rel_type(rel.get("rel_type", "REFERENCES"), default="REFERENCES")
            await _run_cypher(session, f"""
                MATCH (d:Document {{qdrant_id: $qdrant_id}})
                MERGE (t:Entity {{name: $target}})
                ON CREATE SET t.type = 'concept',
                              t.observations = [$initial_obs],
                              t.created_at = datetime()
                MERGE (d)-[r:{rel_type}]->(t)
                SET r.context = $context,
                    r.updated_at = datetime()
            """, {
                "qdrant_id": qdrant_id,
                "target": target,
                "context": rel.get("context"),
                "initial_obs": f"Created via relationship from document: {title}",
            })
            rels_created += 1

    return {
        "status": "success",
        "qdrant_id": qdrant_id,
        "neo4j_id": neo4j_id,
        "collection": collection,
        "entities_created": entities_created,
        "relationships_created": rels_created,
        "timestamp": _utcnow_iso(),
    }


# ============================================================================
# Tool 2: graph_enhanced_search
# ============================================================================

async def graph_enhanced_search(
    query: str,
    collection: str,
    limit: int = 10,
    relationship_boost: float = 0.2,
    max_depth: int = 2,
    include_graph_context: bool = True
) -> Dict[str, Any]:
    """
    Graph-enhanced semantic search combining Qdrant vectors with Neo4j relationships.
    
    Process:
    1. Embed query and search Qdrant
    2. Get entity connections from Neo4j
    3. Discover additional docs via shared entities
    4. Rerank results with relationship boosts
    
    Args:
        query: Search query text
        collection: Qdrant collection to search
        limit: Maximum results to return
        relationship_boost: Score boost for graph-connected docs (0.0-0.5)
        max_depth: Graph traversal depth
        include_graph_context: Include entity connections in results
    
    Returns:
        Reranked results with graph context
    """
    # 1. Generate query embedding (may raise — let it propagate)
    query_embedding = await get_embedding(query)

    # 2. Search Qdrant — pull 2x limit for reranking headroom
    qdrant_client = get_qdrant_client()
    vector_name = _get_collection_vector_name(qdrant_client, collection)

    search_kwargs: Dict[str, Any] = {
        "collection_name": collection,
        "query": query_embedding,
        "limit": limit * 2,
        "with_payload": True,
    }
    if vector_name:
        search_kwargs["using"] = vector_name

    search_results = qdrant_client.query_points(**search_kwargs).points

    if not search_results:
        return {"status": "success", "results": [], "total": 0, "timestamp": _utcnow_iso()}

    # 3. Build results map seeded by Qdrant scores
    results_map: Dict[str, Dict[str, Any]] = {}
    for hit in search_results:
        results_map[str(hit.id)] = {
            "qdrant_id": str(hit.id),
            "score": hit.score,
            "boosted_score": hit.score,
            "payload": hit.payload,
            "graph_connections": [],
            "discovered_via_graph": False,
        }

    qdrant_ids = list(results_map.keys())

    # 4. Pull graph context from Neo4j (off-loop via _run_cypher)
    async with neo4j_session() as session:
        entity_rows = await _run_cypher(session, """
            MATCH (d:Document)-[r:MENTIONS|REFERENCES]->(e:Entity)
            WHERE d.qdrant_id IN $qdrant_ids
            RETURN d.qdrant_id AS doc_id,
                   collect(DISTINCT e.name) AS entities
        """, {"qdrant_ids": qdrant_ids})

        doc_entities: Dict[str, List[str]] = {}
        all_entities: set = set()
        for record in entity_rows:
            doc_entities[record["doc_id"]] = record["entities"]
            all_entities.update(record["entities"])

        if all_entities:
            expanded_rows = await _run_cypher(session, """
                MATCH (d:Document)-[r:MENTIONS|REFERENCES]->(e:Entity)
                WHERE e.name IN $entity_names
                AND d.qdrant_collection = $collection
                AND NOT d.qdrant_id IN $exclude_ids
                WITH d, collect(DISTINCT e.name) AS shared_entities,
                     count(DISTINCT e) AS entity_count
                ORDER BY entity_count DESC
                LIMIT $limit
                RETURN d.qdrant_id AS qdrant_id,
                       d.title AS title,
                       d.summary AS summary,
                       shared_entities,
                       entity_count
            """, {
                "entity_names": list(all_entities),
                "collection": collection,
                "exclude_ids": qdrant_ids,
                "limit": limit,
            })

            for record in expanded_rows:
                doc_id = record["qdrant_id"]
                entity_count = record["entity_count"]
                graph_score = min(0.9, 0.5 + (entity_count * 0.1))
                results_map[doc_id] = {
                    "qdrant_id": doc_id,
                    "score": graph_score,
                    "boosted_score": graph_score + relationship_boost,
                    "payload": {"title": record["title"], "summary": record["summary"]},
                    "graph_connections": record["shared_entities"],
                    "discovered_via_graph": True,
                }

        # Apply relationship boost to original Qdrant-discovered results
        for doc_id, entities in doc_entities.items():
            if doc_id in results_map:
                connection_boost = min(relationship_boost, len(entities) * 0.05)
                results_map[doc_id]["boosted_score"] += connection_boost
                results_map[doc_id]["graph_connections"] = entities

    # 5. Rerank by boosted score
    sorted_results = sorted(
        results_map.values(),
        key=lambda x: x["boosted_score"],
        reverse=True,
    )[:limit]

    # 6. Format output
    formatted_results: List[Dict[str, Any]] = []
    for r in sorted_results:
        result: Dict[str, Any] = {
            "qdrant_id": r["qdrant_id"],
            "original_score": round(r["score"], 4),
            "boosted_score": round(r["boosted_score"], 4),
            "title": (r["payload"] or {}).get("title", ""),
            "summary": (r["payload"] or {}).get("summary", ""),
            "discovered_via_graph": r["discovered_via_graph"],
        }
        if include_graph_context and r["graph_connections"]:
            result["connected_entities"] = r["graph_connections"]
        formatted_results.append(result)

    return {
        "status": "success",
        "query": query,
        "collection": collection,
        "results": formatted_results,
        "total": len(formatted_results),
        "graph_expanded": any(r["discovered_via_graph"] for r in formatted_results),
        "timestamp": _utcnow_iso(),
    }


# ============================================================================
# Tool 3: find_related_documents
# ============================================================================

async def find_related_documents(
    qdrant_id: str,
    relationship_types: Optional[List[str]] = None,
    max_depth: int = 2,
    limit: int = 10,
    include_paths: bool = True
) -> Dict[str, Any]:
    """
    Find documents related to a given document via Neo4j graph traversal.
    
    Args:
        qdrant_id: Source document's Qdrant UUID
        relationship_types: Filter by types (default: all)
        max_depth: Maximum traversal depth (1-3)
        limit: Maximum related documents
        include_paths: Include relationship paths in results
    
    Returns:
        Related documents with relationship context
    """
    # Validate max_depth — interpolated into Cypher, must be a sane small int
    if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 5:
        raise ValueError(f"max_depth must be an int in [1, 5], got {max_depth!r}")

    related_docs: List[Dict[str, Any]] = []

    async with neo4j_session() as session:
        # Verify source document exists; raise if not found so MCP surfaces a real error
        source_rows = await _run_cypher(session, """
            MATCH (d:Document {qdrant_id: $qdrant_id})
            RETURN d.title AS title, d.qdrant_collection AS collection
        """, {"qdrant_id": qdrant_id})
        if not source_rows:
            raise ValueError(f"Document not found: {qdrant_id}")
        source_info = {
            "title": source_rows[0]["title"],
            "collection": source_rows[0]["collection"],
        }

        # Graph traversal — depth bound is interpolated (parameterised depth not
        # supported in Neo4j path patterns), validated above.
        traversal_query = f"""
            MATCH path = (source:Document {{qdrant_id: $qdrant_id}})
                  -[:MENTIONS|REFERENCES*1..{max_depth}]-(related:Document)
            WHERE source <> related
            WITH related, path,
                 [node IN nodes(path) WHERE node:Entity | node.name] AS shared_entities,
                 [rel IN relationships(path) | type(rel)] AS rel_types
            RETURN DISTINCT
                related.qdrant_id AS qdrant_id,
                related.title AS title,
                related.summary AS summary,
                related.qdrant_collection AS collection,
                related.path AS path,
                shared_entities,
                rel_types,
                length(path) AS distance
            ORDER BY distance, size(shared_entities) DESC
            LIMIT $limit
        """
        traversal_rows = await _run_cypher(session, traversal_query, {
            "qdrant_id": qdrant_id,
            "limit": limit,
        })

        for record in traversal_rows:
            doc: Dict[str, Any] = {
                "qdrant_id": record["qdrant_id"],
                "title": record["title"],
                "summary": record["summary"],
                "collection": record["collection"],
                "path": record["path"],
                "distance": record["distance"],
                "shared_entities": record["shared_entities"],
            }
            if include_paths:
                doc["relationship_types"] = record["rel_types"]
            related_docs.append(doc)

    return {
        "status": "success",
        "source": {"qdrant_id": qdrant_id, **source_info},
        "related_documents": related_docs,
        "total": len(related_docs),
        "max_depth": max_depth,
        "timestamp": _utcnow_iso(),
    }



# ============================================================================
# Tool 4: manage_entities (create/update entities with observations)
# ============================================================================

async def manage_entities(
    entities: List[Dict[str, Any]],
    check_existing: bool = True
) -> Dict[str, Any]:
    """
    Create or update entities in Neo4j with observations.
    Replaces neo4j-memory-remote:create_entities + add_observations.
    
    Args:
        entities: List of entity dicts, each with:
            - name (str, required): Entity name
            - type (str): Entity type (default: "concept")
            - observations (list[str]): Facts about this entity
        check_existing: If True, MERGE instead of CREATE (dedup)
    
    Returns:
        Dict with created, updated, and total counts
    """
    created = 0
    updated = 0

    async with neo4j_session() as session:
        for entity in entities:
            name = entity.get("name")
            if not name:
                continue
            entity_type = entity.get("type", "concept")
            observations = entity.get("observations", [])

            if check_existing:
                # Detect create-vs-update by capturing whether the node already
                # existed BEFORE the SET clauses fire. CASE-on-equal-timestamps
                # is unreliable when ON CREATE and ON MATCH both run datetime()
                # in the same millisecond.
                rows = await _run_cypher(session, """
                    OPTIONAL MATCH (existing:Entity {name: $name})
                    WITH existing IS NOT NULL AS was_existing, $name AS name_param
                    MERGE (e:Entity {name: name_param})
                    ON CREATE SET
                        e.type = $type,
                        e.observations = $observations,
                        e.created_at = datetime(),
                        e.updated_at = datetime()
                    ON MATCH SET
                        e.type = CASE WHEN e.type = 'concept' THEN $type ELSE e.type END,
                        e.observations = e.observations + [obs IN $observations WHERE NOT obs IN e.observations],
                        e.updated_at = datetime()
                    RETURN was_existing
                """, {
                    "name": name,
                    "type": entity_type,
                    "observations": observations,
                })
                if not rows:
                    raise RuntimeError(f"Neo4j MERGE failed for entity {name!r}")
                if rows[0]["was_existing"]:
                    updated += 1
                else:
                    created += 1
            else:
                await _run_cypher(session, """
                    CREATE (e:Entity {
                        name: $name,
                        type: $type,
                        observations: $observations,
                        created_at: datetime(),
                        updated_at: datetime()
                    })
                """, {
                    "name": name,
                    "type": entity_type,
                    "observations": observations,
                })
                created += 1

    return {
        "status": "success",
        "created": created,
        "updated": updated,
        "total": created + updated,
        "timestamp": _utcnow_iso(),
    }



# ============================================================================
# Tool 5: manage_relationships (create relationships between entities)
# ============================================================================

async def manage_relationships(
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create relationships between entities in Neo4j.
    Replaces neo4j-memory-remote:create_relations.
    
    Args:
        relationships: List of relationship dicts, each with:
            - from_entity (str, required): Source entity name
            - to_entity (str, required): Target entity name  
            - rel_type (str): Relationship type (default: "RELATES_TO")
            - context (str): Optional context/description
    
    Returns:
        Dict with created count
    """
    created = 0

    async with neo4j_session() as session:
        for rel in relationships:
            from_name = rel.get("from_entity")
            to_name = rel.get("to_entity")
            if not from_name or not to_name:
                continue
            # Whitelist the rel type before interpolating into Cypher
            rel_type = _validate_rel_type(rel.get("rel_type", "RELATES_TO"), default="RELATES_TO")
            context = rel.get("context", "")

            await _run_cypher(session, f"""
                MERGE (a:Entity {{name: $from_name}})
                ON CREATE SET a.type = 'concept',
                              a.observations = [],
                              a.created_at = datetime(),
                              a.updated_at = datetime()
                MERGE (b:Entity {{name: $to_name}})
                ON CREATE SET b.type = 'concept',
                              b.observations = [],
                              b.created_at = datetime(),
                              b.updated_at = datetime()
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.context = $context,
                    r.updated_at = datetime()
            """, {
                "from_name": from_name,
                "to_name": to_name,
                "context": context,
            })
            created += 1

    return {
        "status": "success",
        "created": created,
        "timestamp": _utcnow_iso(),
    }



# ============================================================================
# Tool 6: search_entities (find existing entities for dedup)
# ============================================================================

async def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for entities in Neo4j by name (substring match).
    Replaces neo4j-memory-remote:search_memories and find_memories_by_name.
    
    Args:
        query: Search string (matched against entity name, case-insensitive)
        entity_type: Optional filter by entity type
        limit: Maximum results
    
    Returns:
        Dict with matching entities and their observations
    """
    entities: List[Dict[str, Any]] = []

    async with neo4j_session() as session:
        # Use NULL-or-equal pattern instead of conditional string interpolation —
        # the filter shape stays constant whether entity_type is supplied or not.
        rows = await _run_cypher(session, """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
              AND ($entity_type IS NULL OR e.type = $entity_type)
            OPTIONAL MATCH (e)<-[:MENTIONS|REFERENCES]-(d:Document)
            WITH e, collect(DISTINCT d.title) AS mentioned_in
            RETURN e.name AS name,
                   e.type AS type,
                   e.observations AS observations,
                   e.created_at AS created_at,
                   e.updated_at AS updated_at,
                   mentioned_in
            ORDER BY e.name
            LIMIT $limit
        """, {
            "query": query,
            "entity_type": entity_type,  # None routes through the IS NULL branch
            "limit": limit,
        })

        for record in rows:
            entities.append({
                "name": record["name"],
                "type": record["type"],
                "observations": record["observations"] or [],
                "created_at": str(record["created_at"]) if record["created_at"] else None,
                "updated_at": str(record["updated_at"]) if record["updated_at"] else None,
                "mentioned_in_documents": record["mentioned_in"],
            })

    return {
        "status": "success",
        "query": query,
        "results": entities,
        "total": len(entities),
        "timestamp": _utcnow_iso(),
    }
