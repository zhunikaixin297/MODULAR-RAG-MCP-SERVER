from __future__ import annotations

import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set Hugging Face endpoint to mirror for faster model downloads in some regions
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from docling.document_converter import (
        DocumentConverter,
        InputFormat,
        PdfFormatOption,
        WordFormatOption,
    )
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
    from .docling_utils.vlm_enrichment_pipeline_options import VLMEnrichmentPipelineOptions
    from .docling_utils.vlm_enrichment_pipeline import VlmEnrichmentPipeline
    from .docling_utils.vlm_enrichment_pipeline_word import VlmEnrichmentWordPipeline
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from src.core.settings import resolve_path
from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class DoclingLoader(BaseLoader):
    def __init__(
        self,
        settings: Any,
        extract_images: bool = True,
        image_storage_dir: Optional[str | Path] = None,
        image_output_dir: Optional[str | Path] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "docling is required for DoclingLoader. Install it with: pip install docling"
            )

        logger.info("Initializing DoclingLoader...")
        self.settings = settings
        self.extract_images = extract_images
        loader = getattr(settings, "loader", None)
        docling = getattr(loader, "docling", None) if loader else None
        general = getattr(docling, "general", None) if docling else None
        
        default_image_dir = getattr(general, "image_output_dir", "./data/images/docling") if general else "./data/images/docling"
        resolved_image_dir = image_output_dir or image_storage_dir or default_image_dir
        self.image_output_dir = resolve_path(resolved_image_dir)
        self.timeout_seconds = timeout_seconds or (getattr(general, "timeout_seconds", 300) if general else 300)
        self.max_concurrent_docs = getattr(general, "max_concurrent_docs", 1) if general else 1
        
        logger.info("Building Docling Converter (this may take time for model loading)...")
        self._converter = self._build_converter()
        logger.info("DoclingLoader initialized successfully.")

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise ValueError(f"Unsupported file type for DoclingLoader: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        try:
            markdown, docling_doc, result = self._convert_with_timeout(path)
        except Exception as e:
            logger.error(f"Docling conversion failed for {path}: {e}", exc_info=True)
            raise RuntimeError(f"Docling parsing failed: {e}") from e

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": suffix.lstrip("."),
            "doc_hash": doc_hash,
        }

        title = self._extract_title(markdown)
        if title:
            metadata["title"] = title

        images_metadata: List[Dict[str, Any]] = []
        if self.extract_images:
            try:
                self.image_output_dir.mkdir(parents=True, exist_ok=True)
                images_metadata = self._export_images(docling_doc, doc_hash, result)
                markdown, images_metadata = self._replace_image_markdown(
                    markdown,
                    doc_hash,
                    images_metadata
                )
                
                # 移除之前的冗余 Caption 清理逻辑，因为我们将在 _build_converter 中禁用 Loader 阶段的 VLM
                
                markdown = self._append_image_placeholders_if_missing(markdown, images_metadata)
            except Exception as e:
                logger.error(f"Docling image extraction failed for {path}: {e}", exc_info=True)

        if images_metadata:
            metadata["images"] = images_metadata

        return Document(
            id=doc_id,
            text=markdown,
            metadata=metadata,
        )

    def _build_converter(self) -> DocumentConverter:
        loader = getattr(self.settings, "loader", None)
        docling = getattr(loader, "docling", None) if loader else None
        general = getattr(docling, "general", None) if docling else None
        
        options = VLMEnrichmentPipelineOptions()
        options.settings = self.settings  # 注入全局设置，供增强模型使用
        if general:
            options.images_scale = getattr(general, "images_scale", 1.0)
            options.generate_picture_images = getattr(general, "generate_picture_images", True)
            options.do_ocr = getattr(general, "do_ocr", False)
            options.do_formula_vlm_recognition = getattr(general, "do_formula_recognition", True)
            options.do_table_enrichment = getattr(general, "do_table_enrichment", True)
            # 根据用户要求，在 Loader 阶段禁用 VLM 描述生成
            options.do_pic_enrichment = False
            try:
                device_str = str(getattr(general, "accelerator_device", "CPU")).upper()
                device = AcceleratorDevice[device_str]
            except Exception:
                device = AcceleratorDevice.CPU
            options.accelerator_options = AcceleratorOptions(
                device=device,
                num_threads=getattr(general, "accelerator_num_threads", 4),
            )
        
        vlm = getattr(docling, "vlm", None) if docling else None
        if vlm:
            options.vlm_max_concurrency = getattr(vlm, "max_concurrency", 5)

        llm = getattr(docling, "llm", None) if docling else None
        if llm:
            options.llm_max_concurrency = getattr(llm, "max_concurrency", 5)

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmEnrichmentPipeline,
                    pipeline_options=options,
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=VlmEnrichmentWordPipeline,
                    pipeline_options=options,
                ),
            }
        )

    def _convert_with_timeout(self, path: Path) -> tuple[str, Any, Any]:
        def _blocking_convert() -> tuple[str, Any, Any]:
            result = self._converter.convert(path)
            doc = result.document
            markdown = doc.export_to_markdown()
            return markdown, doc, result

        with ThreadPoolExecutor(max_workers=self.max_concurrent_docs) as executor:
            future = executor.submit(_blocking_convert)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeout as e:
                raise TimeoutError(
                    f"Docling conversion timed out after {self.timeout_seconds}s for {path}"
                ) from e

    def _export_images(self, doc: Any, doc_hash: str, result: Any = None) -> List[Dict[str, Any]]:
        images_metadata: List[Dict[str, Any]] = []
        output_dir = self.image_output_dir / doc_hash
        output_dir.mkdir(parents=True, exist_ok=True)

        # 优先使用 ConversionResult.export_images (Docling 原生导出)
        target = result if result is not None else doc
        if hasattr(target, "export_images"):
            try:
                exported = target.export_images(output_dir)
                paths: List[Path] = []
                if isinstance(exported, dict):
                    for value in exported.values():
                        if isinstance(value, (str, Path)):
                            paths.append(Path(value))
                elif isinstance(exported, list):
                    for value in exported:
                        if isinstance(value, (str, Path)):
                            paths.append(Path(value))
                
                for idx, img_path in enumerate(paths, start=1):
                    if not img_path.is_absolute():
                        img_path = (output_dir / img_path).resolve()
                    image_id = f"{doc_hash[:8]}_{idx}"
                    images_metadata.append(
                        {
                            "id": image_id,
                            "path": str(img_path),
                        }
                    )
            except Exception as e:
                logger.error(f"Docling export_images failed: {e}", exc_info=True)
        if not images_metadata and hasattr(doc, "pictures"):
            images_metadata.extend(self._export_picture_items(doc, output_dir, doc_hash))
        return images_metadata

    def _export_picture_items(self, doc: Any, output_dir: Path, doc_hash: str) -> List[Dict[str, Any]]:
        images_metadata: List[Dict[str, Any]] = []
        try:
            from docling_core.types.doc import DescriptionAnnotation
            for idx, picture in enumerate(getattr(doc, "pictures", []), start=1):
                get_image = getattr(picture, "get_image", None)
                if not callable(get_image):
                    continue
                pil_image = get_image(doc)
                if pil_image is None:
                    continue
                
                # 提取 VLM 生成的描述
                caption = None
                if hasattr(picture, "annotations"):
                    for anno in picture.annotations:
                        if isinstance(anno, DescriptionAnnotation):
                            caption = anno.text
                            break

                image_id = f"{doc_hash[:8]}_{idx}"
                image_path = output_dir / f"{image_id}.png"
                pil_image.save(image_path, format="PNG")
                images_metadata.append({
                    "id": image_id, 
                    "path": str(image_path),
                    "caption": caption,
                    "description": caption  # 增加 description 字段以适配下游 ImageCaptioner
                })
        except Exception as e:
            logger.error(f"Docling picture-item export failed: {e}", exc_info=True)
        return images_metadata

    def _replace_image_markdown(
        self,
        markdown: str,
        doc_hash: str,
        images_metadata: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        # 同时匹配标准 Markdown 图片 ![]() 和 Docling 的占位符 <!-- image -->
        image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|<!--\s*image\s*-->")
        matches = list(image_pattern.finditer(markdown))
        if not matches:
            return markdown, images_metadata

        metadata_by_path = {img.get("path"): img for img in images_metadata}
        updated = markdown
        offset = 0
        generated_images: List[Dict[str, Any]] = []

        for index, match in enumerate(matches, start=1):
            raw_path = match.group(1).strip() if match.group(1) else None
            
            # 如果是 <!-- image -->，我们需要按顺序匹配 images_metadata
            if raw_path is None:
                if index <= len(images_metadata):
                    img_info = images_metadata[index - 1]
                else:
                    img_info = {}
            else:
                img_info = metadata_by_path.get(raw_path, {})
                
            image_id = img_info.get("id", f"{doc_hash[:8]}_{index}")
            placeholder = f"[IMAGE: {image_id}]"
            
            start, end = match.span()
            start += offset
            end += offset
            updated = updated[:start] + placeholder + updated[end:]
            offset += len(placeholder) - (end - start)

            if raw_path:
                image_path = self._resolve_image_path(raw_path, doc_hash)
            else:
                image_path = Path(img_info.get("path", ""))
                
            generated_images.append(
                {
                    "id": image_id,
                    "path": str(image_path),
                    "text_offset": start,
                    "text_length": len(placeholder),
                }
            )

        merged_images = {img["id"]: img for img in images_metadata}
        for img in generated_images:
            merged_images.setdefault(img["id"], img)

        return updated, list(merged_images.values())

    def _resolve_image_path(self, raw_path: str, doc_hash: str) -> Path:
        raw = Path(raw_path)
        if raw.is_absolute():
            return raw
        candidate = (self.image_output_dir / doc_hash / raw).resolve()
        if candidate.exists():
            return candidate
        return (Path.cwd() / raw).resolve()

    def _append_image_placeholders_if_missing(
        self,
        markdown: str,
        images_metadata: List[Dict[str, Any]],
    ) -> str:
        if not images_metadata:
            return markdown
        
        # 只追加那些在正文中没有出现过的图片占位符
        placeholders_list = []
        for img in images_metadata:
            tag = f"[IMAGE: {img['id']}]"
            if tag not in markdown:
                placeholders_list.append(tag)
        
        if not placeholders_list:
            return markdown
            
        placeholders = "\n".join(placeholders_list)
        if not markdown.endswith("\n"):
            markdown += "\n"
        return f"{markdown}\n{placeholders}\n"

    def _compute_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in lines[:10]:
            line = line.strip()
            if line:
                return line
        return None
