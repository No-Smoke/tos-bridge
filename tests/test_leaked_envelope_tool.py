"""Tool-level tests for the leaked-envelope guard on store_doc_with_graph.

The malformed call is rejected client-side unless `collection`/`title` are
absent from the schema's `required` list, so the guard has two halves: a
schema that lets the call reach the server, and server-side repair once it
does. Both are asserted here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from tos_bridge.server import mcp

CORRUPTED_TEXT = (
    "claude-opus-5 EXISTS: 5/25 USD per MTok."
    "</text>\n<collection>project_memory_v2</collection>\n"
    "<title>iwo-dbos Opus 5 retraction 2026-07-26</title>\n"
)

FAKE_RESULT = {
    "qdrant_id": "fake-qdrant-id",
    "neo4j_id": "fake-neo4j-id",
    "entities_created": 0,
    "relationships_created": 0,
}


async def _schema() -> dict:
    async with Client(mcp) as client:
        for tool in await client.list_tools():
            if tool.name == "store_doc_with_graph":
                return tool.inputSchema
    raise AssertionError("store_doc_with_graph not registered")


@pytest.mark.asyncio
async def test_collection_and_title_are_optional_but_still_typed_strings():
    schema = await _schema()

    for name in ("collection", "title"):
        prop = schema["properties"][name]
        assert prop.get("type") == "string", f"{name} lost its flat string type"
        assert "anyOf" not in prop, f"{name} regressed to an anyOf union (see #1)"
        assert name not in schema.get("required", []), (
            f"{name} must not be required, or the client rejects the "
            "malformed call before the server can repair it"
        )
    assert "text" in schema.get("required", [])


@pytest.mark.asyncio
async def test_leaked_envelope_is_repaired_before_storage():
    backing = AsyncMock(return_value=FAKE_RESULT)
    with patch("tos_bridge.server.store_document_with_graph", new=backing):
        async with Client(mcp) as client:
            await client.call_tool("store_doc_with_graph", {"text": CORRUPTED_TEXT})

    kwargs = backing.await_args.kwargs
    assert kwargs["collection"] == "project_memory_v2"
    assert kwargs["title"] == "iwo-dbos Opus 5 retraction 2026-07-26"
    assert kwargs["text"] == "claude-opus-5 EXISTS: 5/25 USD per MTok."


@pytest.mark.asyncio
async def test_caller_supplied_values_win_and_clean_text_is_untouched():
    backing = AsyncMock(return_value=FAKE_RESULT)
    with patch("tos_bridge.server.store_document_with_graph", new=backing):
        async with Client(mcp) as client:
            await client.call_tool(
                "store_doc_with_graph",
                {"text": "plain body", "collection": "c", "title": "t"},
            )

    kwargs = backing.await_args.kwargs
    assert kwargs == dict(
        kwargs, text="plain body", collection="c", title="t"
    )


@pytest.mark.asyncio
async def test_missing_args_without_an_envelope_name_the_real_cause():
    backing = AsyncMock(return_value=FAKE_RESULT)
    with patch("tos_bridge.server.store_document_with_graph", new=backing):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "store_doc_with_graph",
                {"text": "a body with no envelope at all"},
                raise_on_error=False,
            )

    assert result.is_error
    message = "".join(block.text for block in result.content)
    assert "collection" in message and "title" in message
    assert "malformed tool call" in message.lower()
    backing.assert_not_awaited()
