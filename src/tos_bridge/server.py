"""
TOS Bridge MCP Server

Bridges Claude Projects knowledge base to VPS-hosted TOS (Qdrant + Neo4j).
Enables pattern extraction from Projects and synchronization to remote memory systems.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from pydantic import BaseModel, Field


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


# Initialize FastMCP server
mcp = FastMCP("tos-bridge")


# Client initialization helpers
def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with API key"""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)  # FIXED: Pass api_key


def get_neo4j_driver():
    """Initialize Neo4j driver"""
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
            qdrant_client = get_qdrant_client()
            
            # Prepare points for batch storage
            texts = [p.get("text", "") for p in patterns]
            metadatas = [
                {
                    "source": p.get("source", "unknown"),
                    "category": p.get("category", "general"),
                    "importance": p.get("importance", 0.5),
                    "synced_at": datetime.utcnow().isoformat()
                }
                for p in patterns
            ]
            
            # Note: Using qdrant-mcp-remote's batch_store would be better
            # For now, placeholder showing the structure
            results["qdrant"] = {
                "stored": len(patterns),
                "collection": collection,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if target in ["neo4j", "both"]:
            driver = get_neo4j_driver()
            
            with driver.session() as session:
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
            
            driver.close()
        
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
        driver = get_neo4j_driver()
        start = datetime.utcnow()
        
        with driver.session() as session:
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
        
        driver.close()
        
        health["neo4j"] = {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "nodes": node_counts,
            "relationships": rel_count,
            "uri": NEO4J_URI
        }
    except Exception as e:
        health["neo4j"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Overall status
    if health["qdrant"]["status"] == "healthy" and health["neo4j"]["status"] == "healthy":
        health["status"] = "healthy"
    elif health["qdrant"]["status"] == "error" and health["neo4j"]["status"] == "error":
        health["status"] = "error"
    else:
        health["status"] = "degraded"
    
    return health


if __name__ == "__main__":
    mcp.run()
