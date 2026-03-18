"""
Vector Store Module.

This package contains vector store abstractions and implementations:
- Base vector store class
- Vector store factory
- Implementations (Chroma, etc.)
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# Auto-register ChromaStore provider
try:
    from src.libs.vector_store.chroma_store import ChromaStore
    VectorStoreFactory.register_provider('chroma', ChromaStore)
except ImportError:
    # ChromaDB not installed, skip registration
    pass
try:
    from src.libs.vector_store.opensearch_store import OpenSearchStore
    VectorStoreFactory.register_provider('opensearch', OpenSearchStore)
except ImportError:
    OpenSearchStore = None

__all__ = [
    'BaseVectorStore',
    'VectorStoreFactory',
    'ChromaStore',
    'OpenSearchStore',
]
