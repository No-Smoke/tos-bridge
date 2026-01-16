"""
Graph-enhanced search tools for TOS-Bridge.

Provides 3 tools for cross-referencing Qdrant vectors with Neo4j knowledge graph:
1. store_document_with_graph - Store in both systems with bidirectional refs
2. graph_enhanced_search - Semantic search with graph reranking
3. find_related_documents - Graph traversal from document ID

Requires:
- Qdrant server with collections configured
- Neo4j with Document/Entity indexes
- Ollama for embeddings
"""
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from neo4j import GraphDatabase

from .embedding import get_embedding

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with API key."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_neo4j_driver():
    """Initialize Neo4j driver."""
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
    try:
        # 1. Generate embedding
        embedding = await get_embedding(text)
        
        # 2. Prepare Qdrant metadata
        qdrant_id = str(uuid.uuid4())
        entity_names = [e.get("name") for e in (entities or [])]
        doc_summary = summary or text[:200]
        
        qdrant_metadata = {
            "title": title,
            "path": path or "",
            "summary": doc_summary,
            "neo4j_entities": entity_names,
            "synced_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # 3. Store in Qdrant
        qdrant_client = get_qdrant_client()
        qdrant_client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=qdrant_id,
                    vector=embedding,
                    payload=qdrant_metadata
                )
            ]
        )
        
        # 4. Create Neo4j Document node and relationships
        driver = get_neo4j_driver()
        neo4j_result = {}
        
        with driver.session() as session:
            # Create Document node
            doc_result = session.run("""
                CREATE (d:Document {
                    qdrant_id: $qdrant_id,
                    qdrant_collection: $collection,
                    title: $title,
                    path: $path,
                    summary: $summary,
                    created_at: datetime(),
                    updated_at: datetime()
                })
                RETURN elementId(d) as neo4j_id
            """, {
                "qdrant_id": qdrant_id,
                "collection": collection,
                "title": title,
                "path": path or "",
                "summary": doc_summary
            })
            neo4j_result["neo4j_id"] = doc_result.single()["neo4j_id"]
            
            # Create entities and MENTIONS relationships
            entities_created = 0
            for entity in (entities or []):
                session.run("""
                    MATCH (d:Document {qdrant_id: $qdrant_id})
                    MERGE (e:Entity {name: $entity_name})
                    ON CREATE SET e.type = $entity_type, e.created_at = datetime()
                    MERGE (d)-[r:MENTIONS]->(e)
                    SET r.importance = $importance, r.created_at = datetime()
                """, {
                    "qdrant_id": qdrant_id,
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type", "concept"),
                    "importance": entity.get("importance", 0.5)
                })
                entities_created += 1
            
            # Create additional relationships
            rels_created = 0
            for rel in (relationships or []):
                rel_type = rel.get("rel_type", "REFERENCES").upper().replace(" ", "_")
                session.run(f"""
                    MATCH (d:Document {{qdrant_id: $qdrant_id}})
                    MERGE (t:Entity {{name: $target}})
                    ON CREATE SET t.created_at = datetime()
                    MERGE (d)-[r:{rel_type}]->(t)
                    SET r.context = $context, r.created_at = datetime()
                """, {
                    "qdrant_id": qdrant_id,
                    "target": rel.get("target"),
                    "context": rel.get("context")
                })
                rels_created += 1
        
        driver.close()
        
        return {
            "status": "success",
            "qdrant_id": qdrant_id,
            "neo4j_id": neo4j_result["neo4j_id"],
            "collection": collection,
            "entities_created": entities_created,
            "relationships_created": rels_created,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
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
    try:
        # 1. Generate query embedding
        query_embedding = await get_embedding(query)
        
        # 2. Search Qdrant - get 2x limit for reranking headroom
        qdrant_client = get_qdrant_client()
        search_results = qdrant_client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=limit * 2,
            with_payload=True
        ).points
        
        if not search_results:
            return {"status": "success", "results": [], "total": 0}
        
        # 3. Build results map with initial scores
        results_map = {}
        for hit in search_results:
            results_map[str(hit.id)] = {
                "qdrant_id": str(hit.id),
                "score": hit.score,
                "boosted_score": hit.score,
                "payload": hit.payload,
                "graph_connections": [],
                "discovered_via_graph": False
            }
        
        qdrant_ids = list(results_map.keys())
        
        # 4. Query Neo4j for graph relationships
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Find entities connected to our documents
            entity_result = session.run("""
                MATCH (d:Document)-[r:MENTIONS|REFERENCES]->(e:Entity)
                WHERE d.qdrant_id IN $qdrant_ids
                RETURN d.qdrant_id as doc_id, 
                       collect(DISTINCT e.name) as entities
            """, {"qdrant_ids": qdrant_ids})
            
            doc_entities = {}
            all_entities = set()
            for record in entity_result:
                doc_id = record["doc_id"]
                entities = record["entities"]
                doc_entities[doc_id] = entities
                all_entities.update(entities)
            
            # Find OTHER documents connected to same entities (graph expansion)
            if all_entities:
                expanded_result = session.run("""
                    MATCH (d:Document)-[r:MENTIONS|REFERENCES]->(e:Entity)
                    WHERE e.name IN $entity_names
                    AND d.qdrant_collection = $collection
                    AND NOT d.qdrant_id IN $exclude_ids
                    WITH d, collect(DISTINCT e.name) as shared_entities,
                         count(DISTINCT e) as entity_count
                    ORDER BY entity_count DESC
                    LIMIT $limit
                    RETURN d.qdrant_id as qdrant_id,
                           d.title as title,
                           d.summary as summary,
                           shared_entities,
                           entity_count
                """, {
                    "entity_names": list(all_entities),
                    "collection": collection,
                    "exclude_ids": qdrant_ids,
                    "limit": limit
                })
                
                # Add graph-discovered documents
                for record in expanded_result:
                    doc_id = record["qdrant_id"]
                    entity_count = record["entity_count"]
                    graph_score = min(0.9, 0.5 + (entity_count * 0.1))
                    
                    results_map[doc_id] = {
                        "qdrant_id": doc_id,
                        "score": graph_score,
                        "boosted_score": graph_score + relationship_boost,
                        "payload": {
                            "title": record["title"],
                            "summary": record["summary"]
                        },
                        "graph_connections": record["shared_entities"],
                        "discovered_via_graph": True
                    }
            
            # Apply relationship boost to original results
            for doc_id, entities in doc_entities.items():
                if doc_id in results_map:
                    connection_boost = min(relationship_boost, len(entities) * 0.05)
                    results_map[doc_id]["boosted_score"] += connection_boost
                    results_map[doc_id]["graph_connections"] = entities
        
        driver.close()
        
        # 5. Rerank by boosted score
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x["boosted_score"],
            reverse=True
        )[:limit]
        
        # 6. Format output
        formatted_results = []
        for r in sorted_results:
            result = {
                "qdrant_id": r["qdrant_id"],
                "original_score": round(r["score"], 4),
                "boosted_score": round(r["boosted_score"], 4),
                "title": r["payload"].get("title", ""),
                "summary": r["payload"].get("summary", ""),
                "discovered_via_graph": r["discovered_via_graph"]
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
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
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
    try:
        driver = get_neo4j_driver()
        related_docs = []
        
        with driver.session() as session:
            # Verify source document exists
            source_check = session.run("""
                MATCH (d:Document {qdrant_id: $qdrant_id})
                RETURN d.title as title, d.qdrant_collection as collection
            """, {"qdrant_id": qdrant_id})
            
            source_record = source_check.single()
            if not source_record:
                return {
                    "status": "error",
                    "error": f"Document not found: {qdrant_id}"
                }
            
            source_info = {
                "title": source_record["title"],
                "collection": source_record["collection"]
            }
            
            # Graph traversal - find related documents through shared entities
            traversal_query = f"""
                MATCH path = (source:Document {{qdrant_id: $qdrant_id}})
                      -[:MENTIONS|REFERENCES*1..{max_depth}]-(related:Document)
                WHERE source <> related
                WITH related, path,
                     [node in nodes(path) WHERE node:Entity | node.name] as shared_entities,
                     [rel in relationships(path) | type(rel)] as rel_types
                RETURN DISTINCT
                    related.qdrant_id as qdrant_id,
                    related.title as title,
                    related.summary as summary,
                    related.qdrant_collection as collection,
                    related.path as path,
                    shared_entities,
                    rel_types,
                    length(path) as distance
                ORDER BY distance, size(shared_entities) DESC
                LIMIT $limit
            """
            
            result = session.run(traversal_query, {
                "qdrant_id": qdrant_id,
                "limit": limit
            })
            
            for record in result:
                doc = {
                    "qdrant_id": record["qdrant_id"],
                    "title": record["title"],
                    "summary": record["summary"],
                    "collection": record["collection"],
                    "path": record["path"],
                    "distance": record["distance"],
                    "shared_entities": record["shared_entities"]
                }
                if include_paths:
                    doc["relationship_types"] = record["rel_types"]
                related_docs.append(doc)
        
        driver.close()
        
        return {
            "status": "success",
            "source": {
                "qdrant_id": qdrant_id,
                **source_info
            },
            "related_documents": related_docs,
            "total": len(related_docs),
            "max_depth": max_depth,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
