"""In-process MCP Client smoke test for tos-bridge#1.

Calls `store_doc_with_graph` through fastmcp's in-process client transport
with structured `metadata` / `entities` / `relationships` payloads. The
backing storage call is mocked so the test does not require live Qdrant
or Neo4j.

Pre-fix, this call failed with three Pydantic validation errors before
ever reaching the backing function. Post-fix, it must succeed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from tos_bridge.server import mcp


@pytest.mark.asyncio
async def test_store_doc_with_graph_accepts_structured_args():
    fake_result = {
        "qdrant_id": "fake-qdrant-id",
        "neo4j_id": "fake-neo4j-id",
        "entities_created": 1,
        "relationships_created": 1,
    }

    with patch(
        "tos_bridge.server.store_document_with_graph",
        new=AsyncMock(return_value=fake_result),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "store_doc_with_graph",
                {
                    "text": "regression test for #1",
                    "collection": "fake_collection",
                    "title": "fix verification",
                    "metadata": {"category": "verification", "issue": 1},
                    "entities": [{"name": "tos-bridge fix", "type": "fix"}],
                    "relationships": [
                        {"target": "tos-bridge fix", "rel_type": "VERIFIES"}
                    ],
                },
            )

    assert result.data["qdrant_id"] == "fake-qdrant-id"
    assert result.data["entities_created"] == 1
    assert result.data["relationships_created"] == 1


@pytest.mark.asyncio
async def test_run_cypher_accepts_structured_params():
    fake_result = {"rows": [], "row_count": 0, "truncated": False, "classification": "read"}

    with patch(
        "tos_bridge.server.run_cypher_tool",
        new=AsyncMock(return_value=fake_result),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "run_cypher",
                {
                    "query": "MATCH (n {name: $name}) RETURN n LIMIT $top",
                    "params": {"name": "Alice", "top": 5},
                },
            )

    assert result.data["classification"] == "read"


@pytest.mark.asyncio
async def test_hybrid_search_accepts_structured_payload_filter():
    fake_result = {"results": [], "total": 0}

    with patch(
        "tos_bridge.server.hybrid_search_tool",
        new=AsyncMock(return_value=fake_result),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "hybrid_search",
                {
                    "query": "energy procurement",
                    "collection": "fake_collection",
                    "payload_filter": {"category": "session"},
                },
            )

    assert result.data["total"] == 0
