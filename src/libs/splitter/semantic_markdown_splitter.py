from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    MarkdownHeaderTextSplitter = None
    RecursiveCharacterTextSplitter = None

from src.libs.splitter.base_splitter import BaseSplitter


class SemanticMarkdownSplitter(BaseSplitter):
    """Semantic Markdown Splitter.
    
    This splitter first splits text based on Markdown headers to preserve
    semantic structure, then recursively splits over-length chunks by 
    character count while respecting Markdown-aware separators.
    
    Design Principles Applied:
    - Semantic-First: Prioritizes document structure over fixed lengths.
    - Recursive Fallback: Ensures all chunks meet size constraints.
    - Pure Text Input: Only takes text strings for better decoupling.
    """
    
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        headers_to_split_on: Optional[List[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SemanticMarkdownSplitter.
        
        Args:
            settings: Application settings.
            chunk_size: Optional override for chunk size.
            chunk_overlap: Optional override for overlap.
            headers_to_split_on: Optional list of (header_marker, header_name) tuples.
            **kwargs: Additional parameters.
        """
        if MarkdownHeaderTextSplitter is None or RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is required for SemanticMarkdownSplitter. "
                "Install it with: pip install langchain-text-splitters"
            )

        self.settings = settings
        splitter_config = getattr(settings, "splitter", None)
        if splitter_config is None:
            raise ValueError("Missing splitter configuration in settings")

        self.chunk_size = chunk_size if chunk_size is not None else splitter_config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else splitter_config.chunk_overlap

        # Validate configuration
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Split text into chunks semantically and then by size.
        
        Args:
            text: Input text string.
            trace: Optional trace context.
            **kwargs: Unused, kept for interface compatibility.
        
        Returns:
            List of text chunks.
        """
        self.validate_text(text)

        try:
            # Phase 1: Split by Markdown headers
            md_chunks = self._md_splitter.split_text(text)
        except Exception as e:
            raise RuntimeError(f"SemanticMarkdownSplitter markdown split failed: {e}") from e

        output: List[str] = []
        for md_chunk in md_chunks:
            content = md_chunk.page_content
            # Phase 2: If chunk is still too long, use recursive character splitting
            if len(content) <= self.chunk_size:
                output.append(content)
            else:
                try:
                    sub_chunks = self._recursive_splitter.split_text(content)
                    output.extend(sub_chunks if sub_chunks else [content])
                except Exception as e:
                    raise RuntimeError(f"SemanticMarkdownSplitter recursive split failed: {e}") from e

        if not output:
            output = [text]

        output = self._merge_heading_only_chunks(output)
        self.validate_chunks(output)
        return output

    def _merge_heading_only_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return chunks
        merged: List[str] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            if self._is_heading_only_chunk(current):
                j = i + 1
                combined = current
                while j < len(chunks) and self._is_heading_only_chunk(combined):
                    combined = f"{combined.rstrip()}\n\n{chunks[j].lstrip()}"
                    j += 1
                if j > i + 1 or (j == i + 1 and j < len(chunks)):
                    merged.append(combined)
                    i = j
                    continue
            merged.append(current)
            i += 1
        return merged

    def _is_heading_only_chunk(self, chunk: str) -> bool:
        stripped = chunk.strip()
        if not stripped:
            return False
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 2:
            return False
        heading_lines = [line for line in lines if line.startswith("#")]
        if not heading_lines:
            return False
        non_heading = [line for line in lines if not line.startswith("#")]
        if not non_heading:
            return True
        if len(non_heading) == 1 and len(non_heading[0]) <= 20:
            return True
        return False
