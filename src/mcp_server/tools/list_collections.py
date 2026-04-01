"""MCP Tool: list_collections

This tool provides collection listing capabilities through the MCP protocol.
It lists all available collections in the vector store with statistics.

Usage via MCP:
    Tool name: list_collections
    Input schema:
        - include_stats (boolean, optional): Include statistics for each collection
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = """List all available document collections in the knowledge base.

Returns information about each collection including:
- Collection name
- Document count (if include_stats=true)
- Collection metadata

Use this tool to discover available collections before querying.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "Whether to include statistics (document count) for each collection.",
            "default": True,
        },
    },
    "required": [],
}


@dataclass
class CollectionInfo:
    """Information about a single collection.
    
    Attributes:
        name: Collection name
        count: Number of documents/chunks in the collection (optional)
        metadata: Collection metadata dictionary
    """
    name: str
    count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {"name": self.name}
        if self.count is not None:
            result["count"] = self.count
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ListCollectionsConfig:
    """Configuration for list_collections tool.
    
    Attributes:
        persist_directory: Path to ChromaDB storage directory
        include_stats_default: Default value for include_stats parameter
    """
    persist_directory: str = "./data/db/chroma"
    include_stats_default: bool = True


class ListCollectionsTool:
    """MCP Tool for listing knowledge base collections.
    
    This class encapsulates the list_collections tool logic,
    querying the vector store to enumerate available collections.
    
    Design Principles:
    - Config-Driven: Paths from settings.yaml
    - Error Resilience: Graceful handling of missing directories
    - Observable: Logging for debugging
    
    Example:
        >>> tool = ListCollectionsTool(settings)
        >>> result = await tool.execute(include_stats=True)
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[ListCollectionsConfig] = None,
    ) -> None:
        """Initialize ListCollectionsTool.
        
        Args:
            settings: Application settings. If None, loaded from default path.
            config: Tool configuration. If None, derived from settings.
        """
        self._settings = settings
        self._config = config
        
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings
    
    @property
    def config(self) -> ListCollectionsConfig:
        """Get configuration, deriving from settings if necessary."""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
            
            self._config = ListCollectionsConfig(
                persist_directory=persist_dir
            )
        return self._config
    
    def _get_chroma_client(self) -> Any:
        """Get or create ChromaDB client.
        
        Returns:
            ChromaDB PersistentClient instance.
            
        Raises:
            ImportError: If chromadb is not installed.
            RuntimeError: If client creation fails.
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for list_collections. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            # Return client anyway - it will just have no collections
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e

    def _get_opensearch_client(self) -> Any:
        try:
            from opensearchpy import OpenSearch
        except ImportError:
            raise ImportError(
                "opensearch-py is required for list_collections with OpenSearch. "
                "Install it with: pip install opensearch-py"
            )

        vector_store = getattr(self.settings, "vector_store", None)
        opensearch_config = getattr(vector_store, "opensearch", None)
        if opensearch_config is None:
            raise RuntimeError("OpenSearch config not found in settings.vector_store.opensearch")

        hosts = getattr(opensearch_config, "hosts", None)
        if not hosts:
            host = getattr(opensearch_config, "host", "localhost")
            port = getattr(opensearch_config, "port", 9200)
            scheme = getattr(opensearch_config, "scheme", "http")
            hosts = [f"{scheme}://{host}:{port}"]

        username = getattr(opensearch_config, "username", None)
        password = getattr(opensearch_config, "password", None)
        use_ssl = getattr(opensearch_config, "use_ssl", False)
        verify_certs = getattr(opensearch_config, "verify_certs", False)

        return OpenSearch(
            hosts=hosts,
            http_auth=(username, password) if username or password else None,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
        )

    def _opensearch_hidden_prefixes(self) -> Tuple[str, ...]:
        """Prefixes that should never be exposed as MCP collections."""
        return (
            ".",
            "security-auditlog",
            "top_queries-",
        )

    def _is_opensearch_vector_collection(self, client: Any, index_name: str, base_index: str) -> bool:
        """Return True when index looks like a vector-document collection.

        We treat the configured base index as visible by default, then use mappings
        to validate that other indices are true vector collections.
        """
        if index_name == base_index:
            return True

        try:
            mappings = client.indices.get_mapping(index=index_name)
        except Exception as e:
            logger.debug(f"Failed to get mapping for index '{index_name}': {e}")
            return False

        # OpenSearch returns a dict keyed by index name.
        index_mapping = mappings.get(index_name) or next(iter(mappings.values()), {})
        properties = (
            index_mapping.get("mappings", {}).get("properties", {})
            if isinstance(index_mapping, dict)
            else {}
        )
        embedding_content = properties.get("embedding_content", {})
        return embedding_content.get("type") == "knn_vector"

    def _list_opensearch_collections(self, include_stats: bool = True) -> List[CollectionInfo]:
        try:
            client = self._get_opensearch_client()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get OpenSearch client: {e}")
            return []

        vector_store = getattr(self.settings, "vector_store", None)
        opensearch_config = getattr(vector_store, "opensearch", None)
        base_index = getattr(opensearch_config, "index_name", "base")
        hidden_prefixes = self._opensearch_hidden_prefixes()

        try:
            indices = client.cat.indices(format="json")
        except Exception as e:
            logger.error(f"Failed to list OpenSearch indices: {e}")
            return []

        collections_info: List[CollectionInfo] = []
        for item in indices:
            index_name = item.get("index", "")
            if not index_name or index_name.startswith(hidden_prefixes):
                continue
            if not self._is_opensearch_vector_collection(client, index_name, base_index):
                continue
            collections_info.append(CollectionInfo(name=index_name))

        if include_stats and collections_info:
            for info in collections_info:
                index_name = info.name
                try:
                    count = client.count(index=index_name).get("count")
                    info.count = int(count) if count is not None else None
                except Exception as e:
                    logger.warning(f"Failed to get count for index '{index_name}': {e}")

        return collections_info
    
    def list_collections(
        self,
        include_stats: bool = True
    ) -> List[CollectionInfo]:
        """List all available collections.
        
        Args:
            include_stats: Whether to include document counts.
            
        Returns:
            List of CollectionInfo objects.
        """
        # Keep config-only mode deterministic for unit tests and explicit Chroma usage.
        if self._settings is None and self._config is not None:
            provider = "chroma"
        else:
            provider = getattr(getattr(self.settings, "vector_store", None), "provider", "chroma")
            provider = str(provider).lower()
        if provider == "opensearch":
            return self._list_opensearch_collections(include_stats=include_stats)

        try:
            client = self._get_chroma_client()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get ChromaDB client: {e}")
            return []
        
        collections_info: List[CollectionInfo] = []
        
        try:
            # Get all collections from ChromaDB
            collections = client.list_collections()
            
            for collection in collections:
                info = CollectionInfo(
                    name=collection.name,
                    metadata=collection.metadata
                )
                
                if include_stats:
                    try:
                        info.count = collection.count()
                    except Exception as e:
                        logger.warning(
                            f"Failed to get count for collection '{collection.name}': {e}"
                        )
                        info.count = None
                
                collections_info.append(info)
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
        
        logger.info(f"Found {len(collections_info)} collections")
        return collections_info
    
    def format_response(
        self,
        collections: List[CollectionInfo]
    ) -> str:
        """Format collections list as a readable string.
        
        Args:
            collections: List of CollectionInfo objects.
            
        Returns:
            Formatted string suitable for MCP response.
        """
        if not collections:
            return "No collections found in the knowledge base."
        
        lines = [
            f"## Available Collections ({len(collections)} total)\n"
        ]
        
        for i, coll in enumerate(collections, 1):
            line = f"{i}. **{coll.name}**"
            
            if coll.count is not None:
                line += f" - {coll.count} documents"
            
            if coll.metadata:
                # Filter out internal metadata
                user_metadata = {
                    k: v for k, v in coll.metadata.items()
                    if not k.startswith('_') and not k.startswith('hnsw:')
                }
                if user_metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in user_metadata.items())
                    line += f" ({meta_str})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    async def execute(
        self,
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """Execute the list_collections tool.
        
        Args:
            include_stats: Whether to include statistics for each collection.
            
        Returns:
            CallToolResult with formatted collection list.
        """
        logger.info(f"Executing list_collections (include_stats={include_stats})")
        
        try:
            # Run blocking ChromaDB I/O in a thread to avoid blocking
            # the async event loop / MCP stdio transport
            collections = await asyncio.to_thread(
                self.list_collections, include_stats,
            )
            response_text = self.format_response(collections)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except Exception as e:
            logger.exception("Error executing list_collections")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error listing collections: {str(e)}",
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the list_collections tool with the protocol handler.
    
    This function is called by _register_default_tools() in protocol_handler.py
    to register this tool when the MCP server starts.
    
    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    tool = ListCollectionsTool()
    
    async def handler(
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """Handler function for MCP tool calls."""
        return await tool.execute(include_stats=include_stats)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
