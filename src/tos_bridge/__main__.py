"""Entry point for running tos-bridge as a module."""

import asyncio
import logging

from tos_bridge.server import mcp
from tos_bridge.embedding import warmup_ollama

logger = logging.getLogger("tos-bridge")

def main():
    """Run the TOS Bridge MCP server with Ollama warmup."""
    # Pre-warm Ollama embedding model before accepting requests
    try:
        asyncio.run(warmup_ollama())
    except Exception as e:
        logger.warning(f"Ollama warmup skipped: {e}")
    
    mcp.run()

if __name__ == "__main__":
    main()
