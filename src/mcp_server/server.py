"""MCP Server entry point using official MCP SDK.

This module implements the MCP server using the official Python MCP SDK
with stdio transport. It ensures stdout only contains protocol messages
while all logs go to stderr.
"""

from __future__ import annotations

import asyncio
import sys
import os
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr() -> None:
    """Redirect all root logger handlers to stderr.

    MCP stdio transport reserves stdout for JSON-RPC messages.
    Any logging to stdout corrupts the protocol stream.
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # Replace any existing stream handlers that might point to stdout
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


def _preload_heavy_imports() -> None:
    """Eagerly import heavy third-party modules in the **main thread**.

    MCP SDK uses anyio + background threads for stdin/stdout I/O.
    When a tool handler runs ``asyncio.to_thread(fn)``, *fn* executes in
    a new worker thread.  If it tries to ``import chromadb`` (which
    transitively pulls in onnxruntime, numpy, sqlite3 C extensions …),
    that import can deadlock with the stdin-reader thread because both
    compete for Python's global *import lock*.

    Pre-importing here – before anyio spins up its I/O threads – avoids
    the deadlock entirely: subsequent ``import`` statements in worker
    threads simply hit ``sys.modules`` and return immediately.
    """
    # chromadb is the heaviest culprit (onnxruntime, numpy, …)
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401
    except ImportError:
        pass  # optional at install time

    # Internal modules that tools lazy-import inside asyncio.to_thread
    try:
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.dense_retriever  # noqa: F401
        import src.core.query_engine.sparse_retriever  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
        import src.ingestion.storage.bm25_indexer  # noqa: F401
        import src.libs.embedding.embedding_factory  # noqa: F401
        import src.libs.vector_store.vector_store_factory  # noqa: F401
    except ImportError:
        pass


async def run_stdio_server_async() -> int:
    """Run MCP server over stdio asynchronously.

    Returns:
        Exit code.
    """
    # Import here to avoid import errors if mcp not installed
    import mcp.server.stdio

    # Ensure ALL logging goes to stderr (stdout is reserved for JSON-RPC)
    _redirect_all_loggers_to_stderr()

    # Pre-load heavy deps in main thread to prevent import-lock deadlocks
    # when tool handlers later call asyncio.to_thread().
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    # Create server with protocol handler
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    # Run with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """Run MCP server over stdio (synchronous wrapper).

    Returns:
        Exit code.
    """
    return asyncio.run(run_stdio_server_async())


def create_starlette_app():
    """Create Starlette app for SSE transport."""
    import mcp.server.sse
    from starlette.applications import Starlette
    from starlette.routing import Route

    _preload_heavy_imports()
    
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    sse = mcp.server.sse.SseServerTransport("/messages")

    # Use classes to avoid Starlette's automatic wrapping of functions into Request/Response handlers.
    # Starlette's Route checks `inspect.isfunction` or `inspect.ismethod`. 
    # By using a class with `__call__`, we provide a raw ASGI application directly.
    
    class SSEHandler:
        async def __call__(self, scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )

    class MessagesHandler:
        async def __call__(self, scope, receive, send):
            await sse.handle_post_message(scope, receive, send)

    return Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=SSEHandler()),
            Route("/messages", endpoint=MessagesHandler(), methods=["POST"]),
        ],
    )


def run_sse_server(host: str = "0.0.0.0", port: int = 8000) -> int:
    """Run MCP server over SSE.

    Args:
        host: Bind host
        port: Bind port
    """
    import uvicorn
    
    # Configure logging
    logger = get_logger(log_level="INFO")
    logger.info(f"Starting MCP server (SSE transport) on {host}:{port}")
    
    app = create_starlette_app()
    uvicorn.run(app, host=host, port=port)
    return 0


def main() -> int:
    """Entry point for MCP server.
    
    Defaults to stdio unless --sse flag is provided.
    """
    if "--sse" in sys.argv:
        return run_sse_server()
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())