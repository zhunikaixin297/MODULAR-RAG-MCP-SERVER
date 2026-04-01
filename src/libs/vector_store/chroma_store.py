"""ChromaDB VectorStore implementation.

This module provides a concrete implementation of BaseVectorStore using ChromaDB,
a lightweight, open-source embedding database designed for local-first deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """ChromaDB implementation of VectorStore.
    
    This class provides local-first, persistent vector storage using ChromaDB.
    It supports upsert, query, and metadata filtering operations.
    
    Design Principles Applied:
    - Pluggable: Implements BaseVectorStore interface, swappable with other providers.
    - Config-Driven: All settings (persist_directory, collection_name) from settings.yaml.
    - Idempotent: upsert operations with same ID overwrite existing records.
    - Observable: Accepts optional TraceContext (reserved for Stage F).
    - Fail-Fast: Validates dependencies and configuration on initialization.
    
    Attributes:
        client: ChromaDB client instance.
        collection: ChromaDB collection for storing vectors.
        collection_name: Name of the collection.
        persist_directory: Directory path for persistent storage.
    
    Example:
        >>> settings = Settings.load('config/settings.yaml')
        >>> store = ChromaStore(settings=settings)
        >>> records = [
        ...     {
        ...         'id': 'doc1_chunk0',
        ...         'vector': [0.1, 0.2, 0.3],
        ...         'metadata': {'source': 'doc1.pdf'}
        ...     }
        ... ]
        >>> store.upsert(records)
        >>> results = store.query([0.1, 0.2, 0.3], top_k=5)
    """
    
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialize ChromaStore with configuration.
        
        Args:
            settings: Application settings containing vector_store configuration.
            **kwargs: Optional overrides for collection_name or persist_directory.
        
        Raises:
            ImportError: If chromadb package is not installed.
            ValueError: If required configuration is missing.
            RuntimeError: If ChromaDB client initialization fails.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for ChromaStore. "
                "Install it with: pip install chromadb"
            )
        
        # Extract configuration
        try:
            vector_store_config = settings.vector_store
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Please ensure 'vector_store' section exists in settings.yaml"
            ) from e
        
        # Default collection name (allow override for defaults only)
        self.default_collection = kwargs.get(
            'collection_name',
            getattr(vector_store_config, 'collection_name', 'base')
        )
        
        # Persist directory (allow override)
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
        self.persist_directory = resolve_path(persist_dir_str)
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Initializing ChromaStore: default_collection='%s', persist_directory='%s'",
            self.default_collection,
            self.persist_directory,
        )
        
        # Initialize ChromaDB client with persistent storage
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}"
            ) from e
        
        # Initialize collection cache (lazy per-collection)
        self._collections: Dict[str, Any] = {}

        # Prime the default collection to fail fast if misconfigured
        _ = self._get_collection(self.default_collection)
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update records in ChromaDB.
        
        Args:
            records: List of records to upsert. Each record must have:
                - 'id': Unique identifier (str)
                - 'vector': Embedding vector (List[float])
                - 'metadata': Optional metadata dict
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Provider-specific parameters (unused for Chroma).
        
        Raises:
            ValueError: If records list is empty or contains invalid entries.
            RuntimeError: If the upsert operation fails.
        """
        # Validate records
        self.validate_records(records)
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents = []  # ChromaDB requires documents field
        
        for record in records:
            ids.append(str(record['id']))
            embeddings.append(record['vector'])
            
            # Metadata: extract or default to empty dict
            metadata = record.get('metadata', {})
            # Ensure all metadata values are JSON-serializable
            # ChromaDB requires string, int, float, or bool values
            sanitized_metadata = self._sanitize_metadata(metadata)
            
            # ChromaDB requires non-empty metadata dict
            if not sanitized_metadata:
                sanitized_metadata = {'_placeholder': 'true'}
            
            metadatas.append(sanitized_metadata)
            
            # Document: use metadata.text if available, otherwise use id
            document = metadata.get('text', record['id'])
            documents.append(str(document))
        
        # Perform upsert (ChromaDB's add() is idempotent with same IDs)
        try:
            target_collection = self._get_collection(collection)
            target_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug(f"Successfully upserted {len(records)} records to ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records to ChromaDB: {e}"
            ) from e
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for similar vectors.
        
        Args:
            vector: Query vector (embedding) to search for.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (e.g., {'source': 'doc1.pdf'}).
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Provider-specific parameters (unused for Chroma).
        
        Returns:
            List of matching records, sorted by similarity (descending).
            Each record contains:
                - 'id': Record identifier
                - 'score': Similarity score (1.0 = identical, 0.0 = orthogonal)
                - 'metadata': Associated metadata
        
        Raises:
            ValueError: If vector is empty or top_k is invalid.
            RuntimeError: If the query operation fails.
        """
        # Validate query parameters
        self.validate_query_vector(vector, top_k)
        
        # Build ChromaDB where clause from filters
        where_clause = self._build_where_clause(filters) if filters else None
        
        # Perform query
        try:
            target_collection = self._get_collection(collection)
            results = target_collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query ChromaDB with top_k={top_k}: {e}"
            ) from e
        
        # Transform results to standard format
        # ChromaDB returns nested lists: [[id1, id2, ...]]
        output = []
        
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(ids)
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(ids)
            documents = results['documents'][0] if 'documents' in results else [''] * len(ids)
            
            for i, record_id in enumerate(ids):
                # Convert distance to similarity score
                # ChromaDB returns cosine distance (0=identical, 2=opposite)
                # Convert to similarity: score = 1 - (distance / 2)
                distance = distances[i]
                score = 1.0 - (distance / 2.0)
                
                output.append({
                    'id': record_id,
                    'score': max(0.0, score),  # Clamp to [0, 1]
                    'text': documents[i] if documents[i] else '',  # Include text from documents
                    'metadata': metadatas[i] if metadatas[i] else {}
                })
        
        logger.debug(f"Query returned {len(output)} results")
        return output
    
    def delete(
        self,
        ids: List[str],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Delete records from ChromaDB by IDs.
        
        Args:
            ids: List of record IDs to delete.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.
        
        Raises:
            ValueError: If ids list is empty.
            RuntimeError: If the delete operation fails.
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        try:
            target_collection = self._get_collection(collection)
            target_collection.delete(ids=[str(id_) for id_ in ids])
            logger.debug(f"Successfully deleted {len(ids)} records from ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {len(ids)} records from ChromaDB: {e}"
            ) from e
    
    def clear(
        self,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Clear all records from the ChromaDB collection.
        
        Args:
            collection_name: Optional collection name to clear. If None, clears current collection.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.
        
        Raises:
            RuntimeError: If the clear operation fails.
        """
        try:
            target_collection = collection or self.default_collection

            # Delete and recreate collection (most efficient way to clear in Chroma)
            self.client.delete_collection(name=target_collection)
            self._collections.pop(target_collection, None)
            self._get_collection(target_collection)
            logger.info(f"Successfully cleared collection '{target_collection}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to clear collection '{collection or self.default_collection}': {e}"
            ) from e

    def delete_by_metadata(
        self,
        filters: Dict[str, Any],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        """Delete records matching a metadata filter.

        Args:
            filters: Metadata key/value pairs to match
                (e.g. ``{"source_hash": "abc123"}``).
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.

        Returns:
            Number of records deleted.

        Raises:
            ValueError: If *filters* is empty.
            RuntimeError: If the operation fails.
        """
        if not filters:
            raise ValueError("filters cannot be empty")

        try:
            where = self._build_where_clause(filters)
            # Query matching IDs first
            target_collection = self._get_collection(collection)
            results = target_collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(f"delete_by_metadata: no records matched {filters}")
                return 0

            target_collection.delete(ids=matching_ids)
            logger.info(
                f"delete_by_metadata: deleted {len(matching_ids)} records "
                f"matching {filters}"
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filters}: {e}"
            ) from e

    def get_ids_by_metadata(
        self,
        filters: Dict[str, Any],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Retrieve record IDs from the vector store by metadata filters."""
        if not filters:
            raise ValueError("filters cannot be empty")

        try:
            where = self._build_where_clause(filters)
            target_collection = self._get_collection(collection)
            results = target_collection.get(where=where, include=[])
            return results.get("ids", [])
        except Exception as e:
            raise RuntimeError(
                f"Failed to get IDs by metadata {filters}: {e}"
            ) from e

    def count_by_metadata(
        self,
        filters: Dict[str, Any],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        """Count records in the vector store by metadata filters."""
        if not filters:
            raise ValueError("filters cannot be empty")

        try:
            where = self._build_where_clause(filters)
            target_collection = self._get_collection(collection)
            results = target_collection.get(where=where, include=[])
            return len(results.get("ids", []))
        except Exception as e:
            raise RuntimeError(
                f"Failed to count by metadata {filters}: {e}"
            ) from e

    def get_by_metadata(
        self,
        filters: Dict[str, Any],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve records from the vector store by metadata filters."""
        if not filters:
            raise ValueError("filters cannot be empty")

        try:
            where = self._build_where_clause(filters)
            target_collection = self._get_collection(collection)
            results = target_collection.get(
                where=where, include=["documents", "metadatas"]
            )
            
            output = []
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            
            for i, cid in enumerate(ids):
                output.append({
                    "id": cid,
                    "text": docs[i] if docs else "",
                    "metadata": metas[i] if metas else {},
                })
            return output
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by metadata {filters}: {e}"
            ) from e
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure ChromaDB compatibility.
        
        ChromaDB requires metadata values to be str, int, float, or bool.
        This method converts or filters out incompatible types.
        
        Args:
            metadata: Raw metadata dict.
        
        Returns:
            Sanitized metadata dict.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                # Skip None values
                continue
            elif isinstance(value, (list, tuple)):
                # Convert to comma-separated string
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                # Convert to string as fallback
                sanitized[key] = str(value)
        
        return sanitized
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters.
        
        Converts standard filter dict to ChromaDB's query format.
        
        Args:
            filters: Standard filter dict (e.g., {'source': 'doc1.pdf'}).
        
        Returns:
            ChromaDB where clause dict.
        
        Note:
            ChromaDB supports operators like $eq, $ne, $gt, $lt, $in, etc.
            For simplicity, we currently support only exact equality matches.
            Future enhancement: support complex filters.
        """
        # Simple implementation: exact equality matches only
        # For complex filters (e.g., {'score': {'$gt': 0.5}}), extend this method
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in ChromaDB operator format (e.g., {'$eq': 'value'})
                where[key] = value
            else:
                # Simple equality
                where[key] = value
        
        return where
    
    def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the current collection.
        
        Returns:
            Dict containing collection statistics:
                - count: Number of records in collection
                - name: Collection name
                - metadata: Collection metadata
        """
        target_collection = self._get_collection(collection)
        return {
            'count': target_collection.count(),
            'name': target_collection.name,
            'metadata': target_collection.metadata
        }

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 10,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("ChromaStore does not support keyword_search()")
    
    def get_by_ids(
        self,
        ids: List[str],
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve records by their IDs from ChromaDB.
        
        This method is used by SparseRetriever to fetch text and metadata
        for chunks that were matched by BM25 search.
        
        Args:
            ids: List of record IDs to retrieve.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters (unused for Chroma).
        
        Returns:
            List of records in the same order as input ids.
            Each record contains:
                - 'id': Record identifier
                - 'text': The stored text content
                - 'metadata': Associated metadata
            If an ID is not found, an empty dict is returned for that position.
        
        Raises:
            ValueError: If ids list is empty.
            RuntimeError: If the retrieval operation fails.
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        # Ensure all IDs are strings
        str_ids = [str(id_) for id_ in ids]
        
        try:
            # ChromaDB's get method retrieves records by IDs
            target_collection = self._get_collection(collection)
            results = target_collection.get(
                ids=str_ids,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by IDs from ChromaDB: {e}"
            ) from e
        
        # Build a mapping from ID to result for O(1) lookup
        id_to_result: Dict[str, Dict[str, Any]] = {}
        
        if results and results.get('ids'):
            result_ids = results['ids']
            documents = results.get('documents', [None] * len(result_ids))
            metadatas = results.get('metadatas', [{}] * len(result_ids))
            
            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    'id': record_id,
                    'text': documents[i] if documents and documents[i] else '',
                    'metadata': metadatas[i] if metadatas and metadatas[i] else {}
                }
        
        # Return results in the same order as input ids
        output = []
        for id_ in str_ids:
            if id_ in id_to_result:
                output.append(id_to_result[id_])
            else:
                # ID not found, return empty dict
                output.append({})
        
        logger.debug(f"Retrieved {len([r for r in output if r])} of {len(ids)} records by IDs")
        return output

    def _get_collection(self, collection: Optional[str]) -> Any:
        target = collection or self.default_collection
        if target in self._collections:
            return self._collections[target]
        try:
            coll = self.client.get_or_create_collection(
                name=target,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{target}': {e}"
            ) from e
        self._collections[target] = coll
        return coll
