"""MCP Tool: get_document_summary

This tool provides document summary retrieval capabilities through the MCP protocol.
It returns title, summary, and tags for a specific document identified by doc_id.

Usage via MCP:
    Tool name: get_document_summary
    Input schema:
        - doc_id (string, required): The document ID to retrieve summary for
        - collection (string, optional): Collection name to search in
"""

from __future__ import annotations

import asyncio
import logging
import re
from threading import RLock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = """Get summary and metadata for a specific document.

Returns structured information about a document including:
- Title (extracted or inferred from content)
- Summary (first chunk preview or metadata summary)
- Tags (document-level tags/categories)
- Source path
- Chunk count

Use this tool after list_collections to get details about specific documents.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "Document ID for summary lookup. Must be source_ref (e.g., 'doc_xxx') or doc_hash (64-hex). Do not pass source_path.",
        },
        "collection": {
            "type": "string",
            "description": "Collection name to search in. If not specified, searches the default collection.",
        },
    },
    "required": ["doc_id"],
}


@dataclass
class DocumentSummary:
    """Summary information for a document.
    
    Attributes:
        doc_id: Document identifier
        title: Document title (from metadata or inferred)
        summary: Brief summary or preview of document content
        tags: List of tags/categories associated with the document
        source_path: Original file path
        chunk_count: Number of chunks for this document
        metadata: Additional document metadata
    """
    doc_id: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source_path": self.source_path,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
        }


@dataclass
class GetDocumentSummaryConfig:
    """Configuration for get_document_summary tool.
    
    Attributes:
        persist_directory: Path to ChromaDB storage directory
        default_collection: Default collection name if not specified
        summary_max_length: Maximum characters for summary preview
    """
    persist_directory: str = "./data/db/chroma"
    default_collection: str = "knowledge_hub"
    summary_max_length: int = 500
    max_prefix_scan_chunks: int = 5000


class DocumentNotFoundError(Exception):
    """Raised when a document with the specified ID is not found."""
    
    def __init__(self, doc_id: str, collection: Optional[str] = None):
        self.doc_id = doc_id
        self.collection = collection
        message = f"Document '{doc_id}' not found"
        if collection:
            message += f" in collection '{collection}'"
        super().__init__(message)


class GetDocumentSummaryTool:
    """MCP Tool for retrieving document summaries.
    
    This class encapsulates the get_document_summary tool logic,
    querying the vector store to retrieve document metadata and content preview.
    
    Design Principles:
    - Config-Driven: Paths from settings.yaml
    - Error Resilience: Clear error messages for missing documents
    - Observable: Logging for debugging
    - Lazy Init: ChromaDB client created on first use
    
    Example:
        >>> tool = GetDocumentSummaryTool(settings)
        >>> result = await tool.execute(doc_id="doc_abc123")
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[GetDocumentSummaryConfig] = None,
    ) -> None:
        """Initialize GetDocumentSummaryTool.
        
        Args:
            settings: Application settings. If None, loaded from default path.
            config: Tool configuration. If None, derived from settings.
        """
        self._settings = settings
        self._config = config
        self._chroma_client = None
        self._vector_store = None
        self._init_lock = RLock()
        
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings
    
    @property
    def config(self) -> GetDocumentSummaryConfig:
        """Get configuration, deriving from settings if necessary."""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
                default_collection = getattr(
                    self.settings.vector_store,
                    'collection_name',
                    'knowledge_hub'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
                default_collection = 'knowledge_hub'
            
            self._config = GetDocumentSummaryConfig(
                persist_directory=persist_dir,
                default_collection=default_collection,
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
        with self._init_lock:
            if self._chroma_client is not None:
                return self._chroma_client
        
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError:
                raise ImportError(
                    "chromadb package is required for get_document_summary. "
                    "Install it with: pip install chromadb"
                )
        
            persist_path = Path(self.config.persist_directory).resolve()
        
            if not persist_path.exists():
                logger.warning(f"ChromaDB directory does not exist: {persist_path}")
                persist_path.mkdir(parents=True, exist_ok=True)
        
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    )
                )
                return self._chroma_client
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
                ) from e
    
    def _get_collection(self, collection_name: Optional[str] = None) -> Any:
        """Get ChromaDB collection.
        
        Args:
            collection_name: Collection name. Uses default if not specified.
            
        Returns:
            ChromaDB collection instance.
            
        Raises:
            ValueError: If collection does not exist.
        """
        client = self._get_chroma_client()
        name = collection_name or self.config.default_collection
        
        try:
            # Try to get existing collection
            collection = client.get_collection(name=name)
            return collection
        except Exception as e:
            raise ValueError(
                f"Collection '{name}' does not exist: {e}"
            ) from e

    def _get_vector_store(self) -> Any:
        """Get or create vector store instance from configured provider."""
        with self._init_lock:
            if self._vector_store is None:
                self._vector_store = VectorStoreFactory.create(self.settings)
            return self._vector_store

    def _normalize_doc_hash(self, doc_id: str) -> Optional[str]:
        """Normalize user-provided doc identifier to 64-hex doc_hash when possible."""
        if re.fullmatch(r"[0-9a-fA-F]{64}", doc_id or ""):
            return doc_id.lower()
        return None

    def _is_source_ref_doc_id(self, doc_id: str) -> bool:
        """Validate source_ref-style document IDs for controlled prefix fallback."""
        return re.fullmatch(r"doc_[0-9a-fA-F]{16}", doc_id or "") is not None

    def _find_document_chunks_opensearch(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find document chunks via vector store metadata queries (OpenSearch-compatible)."""
        vector_store = self._get_vector_store()
        target_collection = collection_name or self.config.default_collection

        # Try exact matches by most common document-level keys first.
        candidates: List[Dict[str, Any]] = [
            {"source_ref": doc_id},
            {"doc_hash": doc_id},
        ]

        chunks: List[Dict[str, Any]] = []
        for filt in candidates:
            try:
                rows = vector_store.get_by_metadata(filt, collection=target_collection)
            except Exception as e:
                logger.debug(f"OpenSearch metadata lookup failed for {filt}: {e}")
                continue
            if rows:
                chunks = rows
                break

        normalized_chunks: List[Dict[str, Any]] = []
        for row in chunks:
            normalized_chunks.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "metadata": row.get("metadata", {}) or {},
                }
            )
        return normalized_chunks
    
    def _find_document_chunks(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find all chunks belonging to a document.
        
        Searches for chunks where source_ref matches doc_id.
        Falls back to exact doc_hash metadata match when doc_id is a 64-hex hash.
        
        Args:
            doc_id: Document ID to search for.
            collection_name: Collection to search in.
            
        Returns:
            List of chunk data with metadata.
        """
        # Keep config-only mode deterministic for unit tests and explicit Chroma usage.
        if self._settings is None and self._config is not None:
            provider = "chroma"
        else:
            provider = getattr(getattr(self.settings, "vector_store", None), "provider", "chroma")
            provider = str(provider).lower()
        if provider == "opensearch":
            return self._find_document_chunks_opensearch(doc_id, collection_name)

        collection = self._get_collection(collection_name)
        
        # Strategy 1: Search by source_ref metadata
        # Chunks should have source_ref pointing to parent document
        try:
            results = collection.get(
                where={"source_ref": doc_id},
                include=["metadatas", "documents"]
            )
            
            if results and results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        'id': chunk_id,
                        'text': results['documents'][i] if results.get('documents') else '',
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {}
                    })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"source_ref search failed: {e}")
        
        # Strategy 2: Exact doc_hash match (safe and indexed metadata key)
        doc_hash = self._normalize_doc_hash(doc_id)
        if doc_hash:
            try:
                hash_results = collection.get(
                    where={"doc_hash": doc_hash},
                    include=["metadatas", "documents"]
                )

                if hash_results and hash_results.get('ids'):
                    chunks = []
                    for i, chunk_id in enumerate(hash_results['ids']):
                        chunks.append({
                            'id': chunk_id,
                            'text': hash_results['documents'][i] if hash_results.get('documents') else '',
                            'metadata': hash_results['metadatas'][i] if hash_results.get('metadatas') else {}
                        })
                    if chunks:
                        return chunks
            except Exception as e:
                logger.debug(f"doc_hash search failed: {e}")
        
        # Strategy 3 (controlled compatibility fallback):
        # For legacy datasets lacking source_ref/doc_hash metadata, we allow
        # strict prefix matching on chunk IDs only when collection size is
        # below a configurable safety threshold.
        if not self._is_source_ref_doc_id(doc_id):
            return []
        try:
            total = int(collection.count())
        except Exception:
            total = self.config.max_prefix_scan_chunks + 1
        if total > self.config.max_prefix_scan_chunks:
            logger.warning(
                "Skipping prefix fallback for doc_id=%s because collection size %d exceeds max_prefix_scan_chunks=%d",
                doc_id,
                total,
                self.config.max_prefix_scan_chunks,
            )
            return []
        try:
            all_results = collection.get(include=["metadatas", "documents"])
            if all_results and all_results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(all_results['ids']):
                    if isinstance(chunk_id, str) and chunk_id.startswith(f"{doc_id}_"):
                        chunks.append({
                            'id': chunk_id,
                            'text': all_results['documents'][i] if all_results.get('documents') else '',
                            'metadata': all_results['metadatas'][i] if all_results.get('metadatas') else {}
                        })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"Controlled prefix fallback failed: {e}")

        # No chunks found
        return []
    
    def get_document_summary(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> DocumentSummary:
        """Get summary for a specific document.
        
        Args:
            doc_id: Document ID to retrieve.
            collection: Collection name to search in.
            
        Returns:
            DocumentSummary with title, summary, tags, etc.
            
        Raises:
            DocumentNotFoundError: If document is not found.
        """
        chunks = self._find_document_chunks(doc_id, collection)
        
        if not chunks:
            raise DocumentNotFoundError(doc_id, collection)
        
        # Sort chunks by chunk_index if available
        chunks.sort(key=lambda c: c.get('metadata', {}).get('chunk_index', 0))
        
        # Extract document-level info from first chunk's metadata
        first_chunk = chunks[0]
        metadata = first_chunk.get('metadata', {})
        
        # Extract title
        title = self._extract_title(metadata, first_chunk.get('text', ''))
        
        # Extract or generate summary
        summary = self._extract_summary(chunks)
        
        # Extract tags
        tags = self._extract_tags(metadata)
        
        # Extract source path
        source_path = metadata.get('source_path', metadata.get('source', None))
        
        # Collect additional metadata (excluding internal fields)
        additional_metadata = self._filter_metadata(metadata)
        
        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            summary=summary,
            tags=tags,
            source_path=source_path,
            chunk_count=len(chunks),
            metadata=additional_metadata,
        )
    
    def _extract_title(self, metadata: Dict[str, Any], first_text: str) -> str:
        """Extract document title from metadata or content.
        
        Priority:
        1. metadata['title']
        2. First heading from content
        3. Filename from source_path
        4. "Untitled Document"
        
        Args:
            metadata: Chunk metadata.
            first_text: First chunk text content.
            
        Returns:
            Extracted or inferred title.
        """
        # Priority 1: Explicit title in metadata
        if metadata.get('title'):
            return str(metadata['title'])
        
        # Priority 2: First markdown heading
        if first_text:
            lines = first_text.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line.startswith('# '):
                    return line[2:].strip()
        
        # Priority 3: Filename from source_path
        source_path = metadata.get('source_path', metadata.get('source'))
        if source_path:
            filename = Path(source_path).stem
            # Convert snake_case/kebab-case to Title Case
            title = filename.replace('_', ' ').replace('-', ' ').title()
            return title
        
        # Priority 4: Default
        return "Untitled Document"
    
    def _extract_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Extract or generate document summary.
        
        Priority:
        1. metadata['summary'] from any chunk
        2. First N characters from first chunk text
        
        Args:
            chunks: List of document chunks.
            
        Returns:
            Summary text.
        """
        # Priority 1: Explicit summary in metadata
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('summary'):
                return str(metadata['summary'])
        
        # Priority 2: Preview from first chunk
        first_text = chunks[0].get('text', '') if chunks else ''
        if first_text:
            # Clean up and truncate
            summary = first_text.strip()
            
            # Skip markdown headers for preview
            lines = summary.split('\n')
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    content_lines.append(line)
                    if len(' '.join(content_lines)) > self.config.summary_max_length:
                        break
            
            summary = ' '.join(content_lines)
            
            if len(summary) > self.config.summary_max_length:
                summary = summary[:self.config.summary_max_length - 3] + "..."
            
            return summary if summary else "No content preview available."
        
        return "No summary available."
    
    def _extract_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract tags from metadata.
        
        Args:
            metadata: Chunk metadata.
            
        Returns:
            List of tags.
        """
        tags = []
        
        # Check for explicit tags field
        if 'tags' in metadata:
            tag_value = metadata['tags']
            if isinstance(tag_value, list):
                tags.extend(str(t) for t in tag_value)
            elif isinstance(tag_value, str):
                # Could be comma-separated
                tags.extend(t.strip() for t in tag_value.split(',') if t.strip())
        
        # Add doc_type as a tag if available
        if metadata.get('doc_type'):
            doc_type = str(metadata['doc_type']).upper()
            if doc_type not in tags:
                tags.append(doc_type)
        
        return tags
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metadata to exclude internal fields.
        
        Args:
            metadata: Raw metadata dict.
            
        Returns:
            Filtered metadata with user-relevant fields only.
        """
        # Fields to exclude from additional metadata
        exclude_fields = {
            'source_ref', 'chunk_index', 'start_offset', 'end_offset',
            '_placeholder', 'text', 'title', 'summary', 'tags',
            'source_path', 'source'
        }
        
        return {
            k: v for k, v in metadata.items()
            if k not in exclude_fields and not k.startswith('_')
        }
    
    def format_response(self, summary: DocumentSummary) -> str:
        """Format document summary as a readable string.
        
        Args:
            summary: DocumentSummary object.
            
        Returns:
            Formatted string suitable for MCP response.
        """
        lines = [
            f"## Document: {summary.title}",
            "",
            f"**Document ID:** `{summary.doc_id}`",
        ]
        
        if summary.source_path:
            lines.append(f"**Source:** {summary.source_path}")
        
        lines.append(f"**Chunks:** {summary.chunk_count}")
        
        if summary.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in summary.tags)
            lines.append(f"**Tags:** {tags_str}")
        
        lines.extend([
            "",
            "### Summary",
            "",
            summary.summary,
        ])
        
        if summary.metadata:
            lines.extend([
                "",
                "### Additional Metadata",
                "",
            ])
            for key, value in summary.metadata.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def format_error(self, error: Exception) -> str:
        """Format error as a readable string.
        
        Args:
            error: Exception that occurred.
            
        Returns:
            Formatted error message.
        """
        if isinstance(error, DocumentNotFoundError):
            return f"## Document Not Found\n\n{str(error)}\n\nPlease verify the document ID and collection name."
        elif isinstance(error, ValueError):
            return f"## Invalid Request\n\n{str(error)}"
        else:
            return f"## Error\n\nAn error occurred: {str(error)}"
    
    async def execute(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """Execute the get_document_summary tool.
        
        Args:
            doc_id: Document ID to retrieve summary for.
            collection: Optional collection name.
            
        Returns:
            CallToolResult with formatted document summary or error.
        """
        logger.info(f"Executing get_document_summary (doc_id={doc_id}, collection={collection})")
        
        try:
            # Run blocking ChromaDB I/O in a thread to avoid blocking
            # the async event loop / MCP stdio transport
            summary = await asyncio.to_thread(
                self.get_document_summary, doc_id, collection,
            )
            response_text = self.format_response(summary)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except DocumentNotFoundError as e:
            logger.warning(f"Document not found: {e}")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )
            
        except Exception as e:
            logger.exception("Error executing get_document_summary")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the get_document_summary tool with the protocol handler.
    
    This function is called by _register_default_tools() in protocol_handler.py
    to register this tool when the MCP server starts.
    
    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    tool = GetDocumentSummaryTool()
    
    async def handler(
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """Handler function for MCP tool calls."""
        return await tool.execute(doc_id=doc_id, collection=collection)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
