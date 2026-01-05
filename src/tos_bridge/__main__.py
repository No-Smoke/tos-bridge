"""Entry point for running tos-bridge as a module."""

from tos_bridge.server import mcp

def main():
    """Run the TOS Bridge MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
