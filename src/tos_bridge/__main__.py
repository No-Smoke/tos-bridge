"""Entry point for running tos-bridge as a module."""

import asyncio
import logging

from tos_bridge.server import mcp
from tos_bridge.embedding import warmup_ollama
from tos_bridge.graph_tools import ensure_constraints

logger = logging.getLogger("tos-bridge")


async def _startup() -> None:
    """Run all idempotent startup tasks concurrently.

    Both tasks tolerate failure of the remote (Ollama / Neo4j) — they log a
    warning and continue, so the MCP server still comes up even if a backend
    is briefly unavailable.
    """
    await asyncio.gather(
        warmup_ollama(),
        ensure_constraints(),
        return_exceptions=True,
    )


def main():
    """Run the TOS Bridge MCP server.

    Startup hooks:
    - Pre-warm Ollama (avoids 10-15s cold load on first real embedding).
    - Ensure Neo4j uniqueness constraints exist (entity_name_unique,
      document_qdrant_id_unique, pattern_hash_unique) so concurrent MERGE
      cannot create duplicates.
    """
    try:
        asyncio.run(_startup())
    except Exception as e:
        logger.warning(f"Startup hooks failed (non-fatal, continuing): {e}")

    mcp.run()


if __name__ == "__main__":
    main()
