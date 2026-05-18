"""Regression tests for the embedding-error hint.

When Ollama is unreachable, the bare `httpx.ConnectError("All connection
attempts failed")` is opaque — an operator has no idea what URL was tried
or how to fix it. Real bug: tos-bridge MCP processes inheriting a stale
env block from before `OLLAMA_URL` was added to `~/.claude.json` silently
fall back to `http://localhost:11434` and emit this useless error,
requiring 30+ minutes of /proc/<pid>/environ archaeology to diagnose.

The fix wraps the httpx call and re-raises `httpx.ConnectError` with a
message that surfaces:

1. The actual URL tried (which is the fingerprint of the env-drop bug).
2. A pointer to where `OLLAMA_URL` lives (MCP server env config).
3. A reminder that the MCP must be restarted after env edits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tos_bridge import embedding


@pytest.mark.asyncio
async def test_get_embedding_raw_wraps_connect_error_with_hint():
    """ConnectError must carry the URL and a remediation hint."""
    underlying = httpx.ConnectError("All connection attempts failed")

    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=underlying)):
        with pytest.raises(httpx.ConnectError) as exc_info:
            await embedding._get_embedding_raw("text", "mxbai-embed-large")

    msg = str(exc_info.value)
    assert embedding.OLLAMA_URL in msg, (
        f"error must include OLLAMA_URL ({embedding.OLLAMA_URL}) so the "
        f"operator can see which URL was tried; got: {msg}"
    )
    assert "OLLAMA_URL" in msg, "must mention the env var name as remediation"
    assert "MCP" in msg, "must hint that this is an MCP server env issue"
    assert exc_info.value.__cause__ is underlying, (
        "original httpx.ConnectError must be chained for stack inspection"
    )


@pytest.mark.asyncio
async def test_get_embeddings_batch_wraps_connect_error_with_hint():
    """Same hint must apply on the batch path the MCP also calls."""
    underlying = httpx.ConnectError("All connection attempts failed")

    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=underlying)):
        with pytest.raises(httpx.ConnectError) as exc_info:
            await embedding.get_embeddings_batch(["a", "b"])

    msg = str(exc_info.value)
    assert embedding.OLLAMA_URL in msg
    assert "OLLAMA_URL" in msg
    assert exc_info.value.__cause__ is underlying


def test_connection_error_hint_includes_all_pieces():
    """Pure unit test of the formatter — no httpx round-trip needed."""
    underlying = httpx.ConnectError("All connection attempts failed")
    hint = embedding._connection_error_hint(underlying)

    assert embedding.OLLAMA_URL in hint
    assert "OLLAMA_URL" in hint
    assert "localhost:11434" in hint, "must mention the silent-fallback URL"
    assert "restart" in hint.lower(), "must remind to restart MCP after env edits"
    assert "All connection attempts failed" in hint, (
        "must surface the underlying httpx error for full context"
    )
