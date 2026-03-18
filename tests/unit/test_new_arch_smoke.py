import pytest
import os
from pathlib import Path
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.settings import load_settings

@pytest.mark.skipif(os.environ.get("HF_ENDPOINT") is None, reason="HF_ENDPOINT not set, might be slow")
def test_docling_semantic_flow():
    """Unit test for Docling -> Semantic Markdown flow."""
    settings = load_settings("config/settings.yaml")
    
    # 1. Loader
    loader = LoaderFactory.create(settings)
    pdf_path = "tests/fixtures/sample_documents/simple.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found: {pdf_path}")
        
    doc = loader.load(pdf_path)
    assert doc is not None
    assert len(doc.text) > 0
    
    # 2. Splitter
    splitter = SplitterFactory.create(settings)
    chunks = splitter.split_text(doc.text)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    print(f"\n[OK] Docling + Semantic Markdown: {len(chunks)} chunks generated.")

def test_opensearch_factory_init():
    """Verify OpenSearchStore can be initialized from factory."""
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    settings = load_settings("config/settings.yaml")
    
    # Mock settings to use opensearch if not default
    store = VectorStoreFactory.create(settings, collection_name="test_factory_init")
    assert store is not None
    from src.libs.vector_store.opensearch_store import OpenSearchStore
    assert isinstance(store, OpenSearchStore)
    
    # Test close
    import asyncio
    asyncio.run(store.close())
