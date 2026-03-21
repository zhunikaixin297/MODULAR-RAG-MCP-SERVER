"""Ingestion Pipeline orchestrator for the Modular RAG MCP Server.

This module implements the main pipeline that orchestrates the complete
document ingestion flow:
    1. File Integrity Check (SHA256 skip check)
    2. Document Loading (PDF → Document)
    3. Chunking (Document → Chunks)
    4. Transform (Refine + Enrich + Caption)
    5. Encoding (Dense + Sparse vectors)
    6. Storage (VectorStore + BM25 Index + ImageStorage)

Design Principles:
- Config-Driven: All components configured via settings.yaml
- Observable: Logs progress and stage completion
- Graceful Degradation: LLM failures don't block pipeline
- Idempotent: SHA256-based skip for unchanged files
"""

from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
import time
import copy
from concurrent.futures import ThreadPoolExecutor

from src.core.settings import Settings, load_settings, resolve_path
from src.core.types import Document, Chunk
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

# Libs layer imports
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# Ingestion layer imports
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner, inject_captions_into_text
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.image_storage import ImageStorage

logger = get_logger(__name__)


class PipelineResult:
    """Result of pipeline execution with detailed statistics.
    
    Attributes:
        success: Whether pipeline completed successfully
        file_path: Path to the processed file
        doc_id: Document ID (SHA256 hash)
        chunk_count: Number of chunks generated
        image_count: Number of images processed
        vector_ids: List of vector IDs stored
        error: Error message if pipeline failed
        stages: Dict of stage names to their individual results
    """
    
    def __init__(
        self,
        success: bool,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_count: int = 0,
        image_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
        stages: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.file_path = file_path
        self.doc_id = doc_id
        self.chunk_count = chunk_count
        self.image_count = image_count
        self.vector_ids = vector_ids or []
        self.error = error
        self.stages = stages or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids_count": len(self.vector_ids),
            "error": self.error,
            "stages": self.stages
        }


class IngestionPipeline:
    """Main pipeline orchestrator for document ingestion.
    
    This class coordinates all stages of the ingestion process:
    - File integrity checking for incremental processing
    - Document loading (PDF with image extraction)
    - Text chunking with configurable splitter
    - Chunk refinement (rule-based + LLM)
    - Metadata enrichment (rule-based + LLM)
    - Image captioning (Vision LLM)
    - Dense embedding (Azure text-embedding-ada-002)
    - Sparse encoding (BM25 term statistics)
    - Vector storage (ChromaDB)
    - BM25 index building
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("documents/report.pdf", collection="contracts")
        >>> print(f"Processed {result.chunk_count} chunks")
    """
    
    def __init__(
        self,
        settings: Settings,
        collection: Optional[str] = None,
        force: bool = False
    ):
        """Initialize pipeline with all components.
        
        Args:
            settings: Application settings from settings.yaml
            collection: Collection name for organizing documents
            force: If True, re-process even if file was previously processed
        """
        self.settings = settings
        self.collection = collection or getattr(settings.vector_store, "collection_name", "base")
        self.force = force
        
        # Initialize all components
        logger.info("Initializing Ingestion Pipeline components...")
        
        # Stage 1: File Integrity
        self.integrity_checker = SQLiteIntegrityChecker(db_path=str(resolve_path("data/db/ingestion_history.db")))
        logger.info("  ✓ FileIntegrityChecker initialized")
        
        # Stage 2: Loader
        self.loader = LoaderFactory.create(
            settings,
            extract_images=True,
            image_storage_dir=str(resolve_path(f"data/images/{self.collection}"))
        )
        logger.info("  ✓ Loader initialized")
        
        # Stage 3: Chunker
        self.chunker = DocumentChunker(settings)
        logger.info("  ✓ DocumentChunker initialized")
        
        # Stage 4: Transforms
        self.chunk_refiner = ChunkRefiner(settings)
        logger.info(f"  ✓ ChunkRefiner initialized (use_llm={self.chunk_refiner.use_llm})")
        
        self.metadata_enricher = MetadataEnricher(settings)
        logger.info(f"  ✓ MetadataEnricher initialized (use_llm={self.metadata_enricher.use_llm})")
        
        self.image_captioner = ImageCaptioner(settings)
        has_vision = self.image_captioner.llm is not None
        logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")
        
        # Stage 5: Encoders
        embedding = EmbeddingFactory.create(settings)
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100
        sparse_enabled = settings.ingestion.sparse_enabled if settings.ingestion else True
        bm25_enabled = settings.ingestion.bm25_enabled if settings.ingestion else True
        self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
        logger.info(f"  ✓ DenseEncoder initialized (provider={settings.embedding.provider})")
        
        self.sparse_encoder = SparseEncoder() if sparse_enabled else None
        logger.info(f"  ✓ SparseEncoder initialized (enabled={sparse_enabled})")
        
        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=batch_size,
            enable_sparse=sparse_enabled
        )
        logger.info(f"  ✓ BatchProcessor initialized (batch_size={batch_size})")
        
        # Stage 6: Storage
        self.vector_upserter = VectorUpserter(settings, collection_name=self.collection)
        logger.info(f"  ✓ VectorUpserter initialized (provider={settings.vector_store.provider}, collection={self.collection})")
        
        self.bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{self.collection}"))) if bm25_enabled else None
        logger.info(f"  ✓ BM25Indexer initialized (enabled={bm25_enabled})")
        
        self.image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images"))
        )
        logger.info("  ✓ ImageStorage initialized")
        
        logger.info("Pipeline initialization complete!")
    
    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        """Execute the full ingestion pipeline on a file.
        
        Args:
            file_path: Path to the file to process (e.g., PDF)
            trace: Optional trace context for observability
            on_progress: Optional callback ``(stage_name, current, total)``
                invoked when each pipeline stage completes.  *current* is
                the 1-based index of the completed stage; *total* is the
                number of stages (currently 6).
        
        Returns:
            PipelineResult with success status and statistics
        """
        file_path = Path(file_path)
        stages: Dict[str, Any] = {}
        _total_stages = 6

        def _notify(stage_name: str, step: int) -> None:
            if on_progress is not None:
                on_progress(stage_name, step, _total_stages)
        
        logger.info(f"=" * 60)
        logger.info(f"Starting Ingestion Pipeline for: {file_path}")
        logger.info(f"Collection: {self.collection}")
        logger.info(f"=" * 60)
        
        try:
            # ─────────────────────────────────────────────────────────────
            # Stage 1: File Integrity Check
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📋 Stage 1: File Integrity Check")
            _notify("integrity", 1)
            
            file_hash = self.integrity_checker.compute_sha256(str(file_path))
            logger.info(f"  File hash: {file_hash[:16]}...")
            
            if not self.force and self.integrity_checker.should_skip(file_hash, self.collection):
                logger.info(f"  ⏭️  File already processed, skipping (use force=True to reprocess)")
                return PipelineResult(
                    success=True,
                    file_path=str(file_path),
                    doc_id=file_hash,
                    stages={"integrity": {"skipped": True, "reason": "already_processed"}}
                )
            
            stages["integrity"] = {"file_hash": file_hash, "skipped": False}
            logger.info("  ✓ File needs processing")
            
            # ─────────────────────────────────────────────────────────────
            # Stage 2: Document Loading
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📄 Stage 2: Document Loading")
            _notify("load", 2)
            
            _t0 = time.monotonic()
            document = self.loader.load(str(file_path))
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            text_preview = document.text[:200].replace('\n', ' ') + "..." if len(document.text) > 200 else document.text
            image_count = len(document.metadata.get("images", []))
            
            logger.info(f"  Document ID: {document.id}")
            logger.info(f"  Text length: {len(document.text)} chars")
            logger.info(f"  Images extracted: {image_count}")
            logger.info(f"  Preview: {text_preview[:100]}...")
            
            stages["loading"] = {
                "doc_id": document.id,
                "text_length": len(document.text),
                "image_count": image_count
            }
            if trace is not None:
                trace.record_stage("load", {
                    "method": "markitdown",
                    "doc_id": document.id,
                    "text_length": len(document.text),
                    "image_count": image_count,
                    "text_preview": document.text,
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 3: Chunking
            # ─────────────────────────────────────────────────────────────
            logger.info("\n✂️  Stage 3: Document Chunking")
            _notify("split", 3)
            
            _t0 = time.monotonic()
            chunks = self.chunker.split_document(document)
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            logger.info(f"  Chunks generated: {len(chunks)}")
            if chunks:
                logger.info(f"  First chunk ID: {chunks[0].id}")
                logger.info(f"  First chunk preview: {chunks[0].text[:100]}...")
            
            stages["chunking"] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
            }
            if trace is not None:
                trace.record_stage("split", {
                    "method": "recursive",
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text": c.text,
                            "char_len": len(c.text),
                            "chunk_index": c.metadata.get("chunk_index", i),
                        }
                        for i, c in enumerate(chunks)
                    ],
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 4: Transform Pipeline
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔄 Stage 4: Transform Pipeline")
            _notify("transform", 4)
            
            _t0_transform = time.monotonic()
            # snapshot before refinement
            _pre_refine_texts = {c.id: c.text for c in chunks}
            logger.info("  4a/4b/4c. Chunk Refinement + Metadata Enrichment + Image Captioning (parallel)...")
            base_chunks = copy.deepcopy(chunks)
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_refine = executor.submit(self.chunk_refiner.transform, copy.deepcopy(base_chunks), None)
                future_meta = executor.submit(self.metadata_enricher.transform, copy.deepcopy(base_chunks), None)
                future_caption = executor.submit(self.image_captioner.transform, copy.deepcopy(base_chunks), None)
                refined_chunks = future_refine.result()
                meta_chunks = future_meta.result()
                caption_chunks = future_caption.result()

            refined_by_id = {c.id: c for c in refined_chunks}
            meta_by_id = {c.id: c for c in meta_chunks}
            caption_by_id = {c.id: c for c in caption_chunks}

            merged_chunks: List[Chunk] = []
            for base in chunks:
                refined_chunk = refined_by_id.get(base.id, base)
                meta_chunk = meta_by_id.get(base.id, base)
                caption_chunk = caption_by_id.get(base.id, base)

                merged_metadata = dict(refined_chunk.metadata)
                for key in ("title", "summary", "tags", "hypothetical_questions", "enriched_by", "enrich_error"):
                    if key in meta_chunk.metadata:
                        merged_metadata[key] = meta_chunk.metadata.get(key)
                image_captions = caption_chunk.metadata.get("image_captions", [])
                if image_captions:
                    merged_metadata["image_captions"] = image_captions
                merged_text = inject_captions_into_text(refined_chunk.text, image_captions)

                merged_chunks.append(
                    Chunk(
                        id=base.id,
                        text=merged_text,
                        metadata=merged_metadata,
                        start_offset=base.start_offset,
                        end_offset=base.end_offset,
                        source_ref=base.source_ref,
                    )
                )
            chunks = merged_chunks

            enriched_by_llm = sum(1 for c in chunks if c.metadata.get("enriched_by") == "llm")
            enriched_by_rule = sum(1 for c in chunks if c.metadata.get("enriched_by") == "rule")
            captioned = sum(1 for c in chunks if c.metadata.get("image_captions"))
            refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")
            refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")
            logger.info(f"      LLM refined: {refined_by_llm}, Rule refined: {refined_by_rule}")
            logger.info(f"      LLM enriched: {enriched_by_llm}, Rule enriched: {enriched_by_rule}")
            logger.info(f"      Chunks with captions: {captioned}")
            
            stages["transform"] = {
                "chunk_refiner": {"llm": refined_by_llm, "rule": refined_by_rule},
                "metadata_enricher": {"llm": enriched_by_llm, "rule": enriched_by_rule},
                "image_captioner": {"captioned_chunks": captioned}
            }
            _elapsed_transform = (time.monotonic() - _t0_transform) * 1000.0
            if trace is not None:
                trace.record_stage("transform", {
                    "method": "refine+enrich+caption",
                    "refined_by_llm": refined_by_llm,
                    "refined_by_rule": refined_by_rule,
                    "enriched_by_llm": enriched_by_llm,
                    "enriched_by_rule": enriched_by_rule,
                    "captioned_chunks": captioned,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text_before": _pre_refine_texts.get(c.id, ""),
                            "text_after": c.text,
                            "char_len": len(c.text),
                            "refined_by": c.metadata.get("refined_by", ""),
                            "enriched_by": c.metadata.get("enriched_by", ""),
                            "title": c.metadata.get("title", ""),
                            "tags": c.metadata.get("tags", []),
                            "summary": c.metadata.get("summary", ""),
                        }
                        for c in chunks
                    ],
                }, elapsed_ms=_elapsed_transform)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 5: Encoding
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔢 Stage 5: Encoding")
            _notify("embed", 5)
            
            # Process through BatchProcessor
            _t0 = time.monotonic()
            batch_result = self.batch_processor.process(chunks, trace)
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            dense_vectors = batch_result.dense_vectors
            sparse_stats = batch_result.sparse_stats
            summary_vectors = getattr(batch_result, "summary_vectors", [None for _ in chunks])
            question_vectors = getattr(batch_result, "hypothetical_vectors", [None for _ in chunks])

            logger.info(f"  Dense vectors: {len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")
            if sparse_stats:
                logger.info(f"  Sparse stats: {len(sparse_stats)} documents")
            
            stages["encoding"] = {
                "dense_vector_count": len(dense_vectors),
                "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                "sparse_doc_count": len(sparse_stats),
                "summary_vector_count": len([v for v in summary_vectors if v is not None]),
                "hypothetical_vector_count": len([v for v in question_vectors if v is not None])
            }
            if trace is not None:
                # Build per-chunk encoding details (both dense & sparse)
                chunk_details = []
                for idx, c in enumerate(chunks):
                    detail: dict = {
                        "chunk_id": c.id,
                        "char_len": len(c.text),
                    }
                    # Dense: vector dimension (same for all, but confirm per-chunk)
                    if idx < len(dense_vectors):
                        detail["dense_dim"] = len(dense_vectors[idx])
                    # Sparse: BM25 term stats
                    if sparse_stats and idx < len(sparse_stats):
                        ss = sparse_stats[idx]
                        detail["doc_length"] = ss.get("doc_length", 0)
                        detail["unique_terms"] = ss.get("unique_terms", 0)
                        # Top-10 terms by frequency for inspection
                        tf = ss.get("term_frequencies", {})
                        top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
                        detail["top_terms"] = [{"term": t, "freq": f} for t, f in top_terms]
                    chunk_details.append(detail)

                trace.record_stage("embed", {
                    "method": "batch_processor",
                    "dense_vector_count": len(dense_vectors),
                    "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                    "sparse_doc_count": len(sparse_stats),
                    "chunks": chunk_details,
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 6: Storage
            # ─────────────────────────────────────────────────────────────
            logger.info("\n💾 Stage 6: Storage")
            _notify("upsert", 6)
            
            # 6a: Vector Upsert
            logger.info("  6a. Vector Storage...")
            _t0_storage = time.monotonic()
            vector_ids = self.vector_upserter.upsert(
                chunks,
                dense_vectors,
                trace,
                extra_vectors={
                    "summary": summary_vectors,
                    "hypothetical_questions": question_vectors
                },
            )
            logger.info(f"      Stored {len(vector_ids)} vectors")

            if sparse_stats:
                for stat, vid in zip(sparse_stats, vector_ids):
                    stat["chunk_id"] = vid
            
            # 6b: BM25 Index
            if self.bm25_indexer is not None and sparse_stats:
                logger.info("  6b. BM25 Index...")
                self.bm25_indexer.add_documents(
                    sparse_stats,
                    collection=self.collection,
                    doc_id=document.id,
                    trace=trace,
                )
                logger.info(f"      Index built for {len(sparse_stats)} documents")
            
            # 6c: Register images in image storage index
            # Note: Images are already saved by PdfLoader, we just need to index them
            logger.info("  6c. Image Storage Index...")
            images = document.metadata.get("images", [])
            for img in images:
                img_path = Path(img["path"])
                if img_path.exists():
                    self.image_storage.register_image(
                        image_id=img["id"],
                        file_path=img_path,
                        collection=self.collection,
                        doc_hash=file_hash,
                        page_num=img.get("page", 0)
                    )
            logger.info(f"      Indexed {len(images)} images")
            
            stages["storage"] = {
                "vector_count": len(vector_ids),
                "bm25_docs": len(sparse_stats),
                "images_indexed": len(images)
            }
            _elapsed_storage = (time.monotonic() - _t0_storage) * 1000.0
            if trace is not None:
                # Per-chunk storage mapping: chunk_id → vector_id
                chunk_storage = [
                    {
                        "chunk_id": c.id,
                        "vector_id": vector_ids[i] if i < len(vector_ids) else "—",
                        "collection": self.collection,
                        "store": "ChromaDB",
                    }
                    for i, c in enumerate(chunks)
                ]
                # Image storage details
                image_storage_details = [
                    {
                        "image_id": img["id"],
                        "file_path": str(img["path"]),
                        "page": img.get("page", 0),
                        "doc_hash": file_hash,
                    }
                    for img in images
                ]
                trace.record_stage("upsert", {
                    "method": "vector_upserter",
                    "dense_store": {
                        "backend": "ChromaDB",
                        "collection": self.collection,
                        "count": len(vector_ids),
                        "path": "data/db/chroma/",
                    },
                    "sparse_store": {
                        "backend": "BM25",
                        "collection": self.collection,
                        "count": len(sparse_stats),
                        "path": f"data/db/bm25/{self.collection}/",
                    },
                    "image_store": {
                        "backend": "ImageStorage (JSON index)",
                        "count": len(images),
                        "images": image_storage_details,
                    },
                    "chunk_mapping": chunk_storage,
                }, elapsed_ms=_elapsed_storage)
            
            # ─────────────────────────────────────────────────────────────
            # Mark Success
            # ─────────────────────────────────────────────────────────────
            self.integrity_checker.mark_success(file_hash, str(file_path), self.collection)
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Vectors: {len(vector_ids)}")
            logger.info(f"   Images: {len(images)}")
            logger.info("=" * 60)
            
            return PipelineResult(
                success=True,
                file_path=str(file_path),
                doc_id=file_hash,
                chunk_count=len(chunks),
                image_count=len(images),
                vector_ids=vector_ids,
                stages=stages
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            self.integrity_checker.mark_failed(file_hash, str(file_path), str(e), self.collection)
            
            return PipelineResult(
                success=False,
                file_path=str(file_path),
                doc_id=file_hash if 'file_hash' in locals() else None,
                error=str(e),
                stages=stages
            )
    
    def close(self) -> None:
        """Clean up resources."""
        self.vector_upserter.close()
        self.image_storage.close()


def run_pipeline(
    file_path: str,
    settings_path: Optional[str] = None,
    collection: Optional[str] = None,
    force: bool = False
) -> PipelineResult:
    """Convenience function to run the pipeline.
    
    Args:
        file_path: Path to file to process
        settings_path: Path to settings.yaml (default: <repo>/config/settings.yaml)
        collection: Collection name
        force: Force reprocessing
    
    Returns:
        PipelineResult with execution details
    """
    settings = load_settings(settings_path)
    effective_collection = collection or getattr(settings.vector_store, "collection_name", "base")
    pipeline = IngestionPipeline(settings, collection=effective_collection, force=force)
    
    try:
        return pipeline.run(file_path)
    finally:
        pipeline.close()
