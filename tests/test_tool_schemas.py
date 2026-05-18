"""Regression tests for tos-bridge#1.

Optional[Dict[str, Any]] / Optional[List[Dict[str, Any]]] tool parameters
emitted a JSON schema wrapped in `anyOf: [<typed-shape>, {"type": "null"}]`.
The Claude Code MCP client couldn't introspect that as a structured shape
and fell back to JSON-string serialization, causing server-side Pydantic
validation to reject `type=str`.

The fix drops `Optional`/`= None` for these params in favor of plain
`Dict[str, Any] = {}` / `List[Dict[str, Any]] = []`, which emits a flat
typed schema (no `anyOf` wrapper) that matches the already-working tools
(`create_or_update_entities`, `create_relationships`).

Layer 1 (this file) asserts the on-wire MCP schema for every previously
broken parameter has a top-level `type` and does NOT contain a `null`
variant. Layer 2 (in-process Client call) lives in
test_in_process_client.py.
"""

from __future__ import annotations

import pytest

from tos_bridge.server import mcp


# tool_name -> param_name -> expected JSON Schema type
EXPECTED_TYPES = {
    "store_doc_with_graph": {
        "metadata": "object",
        "entities": "array",
        "relationships": "array",
    },
    "run_cypher": {"params": "object"},
    "hybrid_search": {"payload_filter": "object"},
}


def _schema_types(prop: dict) -> list[str]:
    """Return the JSON Schema `type` values declared by a property.

    Handles both flat (`{"type": "object"}`) and union
    (`{"anyOf": [{"type": "object"}, {"type": "null"}]}`) shapes.
    """
    if "type" in prop:
        return [prop["type"]]
    return [s.get("type") for s in prop.get("anyOf", []) if isinstance(s, dict)]


@pytest.mark.asyncio
async def test_previously_broken_params_emit_flat_typed_schema():
    """Every fixed parameter must have a top-level `type` (no anyOf wrapper).

    fastmcp 2.14.5 emits `anyOf: [<typed>, {"type": "null"}]` for
    `Optional[T]` params. The Claude Code MCP client treated that as
    untyped and serialized structured values as JSON strings — the actual
    bug behavior in #1. The fix removes `Optional`, which collapses the
    schema to a flat `{"type": "object"|"array"}`.
    """
    tools = await mcp.get_tools()

    for tool_name, params in EXPECTED_TYPES.items():
        assert tool_name in tools, f"tool {tool_name!r} not registered"
        props = tools[tool_name].parameters.get("properties", {})

        for param_name, expected_type in params.items():
            assert param_name in props, (
                f"{tool_name}.{param_name} missing from schema"
            )
            prop = props[param_name]

            assert "type" in prop, (
                f"{tool_name}.{param_name} schema is not flat-typed "
                f"(wrapped in anyOf?): {prop!r}"
            )
            assert prop["type"] == expected_type, (
                f"{tool_name}.{param_name} expected type={expected_type!r}, "
                f"got {prop['type']!r}"
            )
            assert "null" not in _schema_types(prop), (
                f"{tool_name}.{param_name} still allows null variant: {prop!r}"
            )
