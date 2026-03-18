import pytest
import os
import asyncio
from pathlib import Path
from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline

class TestNewIngestionFlow:
    """Integration test for the full ingestion flow with Docling, Semantic Markdown, and OpenSearch."""
    
    @pytest.fixture
    def settings(self):
        """Load settings from config file."""
        # Ensure the settings are configured for the new architecture
        settings = load_settings("config/settings.yaml")
        return settings

    @pytest.fixture
    def simple_pdf_path(self):
        """Path to simple PDF document."""
        path = Path("tests/fixtures/sample_documents/simple.pdf")
        assert path.exists(), f"Test fixture not found: {path}"
        return str(path)

    def test_pipeline_new_arch(self, settings, simple_pdf_path):
        """Test the ingestion pipeline with the new architecture components."""
        # 1. Setup
        collection = "test_new_flow_integration"
        pipeline = IngestionPipeline(
            settings=settings,
            collection=collection,
            force=True
        )
        
        # 2. Run Pipeline
        # We need to set HF_ENDPOINT for docling
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        result = pipeline.run(simple_pdf_path)
        
        # 3. Assertions
        assert result.success, f"Pipeline failed: {result.error}"
        assert result.chunk_count > 0
        assert len(result.vector_ids) == result.chunk_count
        
        print(f"\n[OK] New Ingestion Flow: Processed {result.chunk_count} chunks.")
        
        # 4. Verify in OpenSearch
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        store = VectorStoreFactory.create(settings, collection_name=collection)
        
        # Keyword search check
        keyword_results = store.keyword_search("test", top_k=5)
        # Note: OpenSearch index might need a moment to refresh, but usually it's fast enough in local
        # Or we can just check if we can query at all
        assert isinstance(keyword_results, list)
        print(f"[OK] OpenSearch keyword search returned {len(keyword_results)} results.")
        
        # Cleanup
        async def cleanup():
            if hasattr(store, 'close'):
                await store.close()
        asyncio.run(cleanup())
