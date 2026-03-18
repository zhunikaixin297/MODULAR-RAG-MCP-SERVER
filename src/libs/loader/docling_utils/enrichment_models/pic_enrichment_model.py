# === 步骤 1: 导入所需库 ===
import logging
from io import BytesIO
from collections.abc import Iterable
from typing import Any

from PIL import Image  # 用于处理图像
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_vision_llm import ImageInput

# --- docling 核心库 ---
# docling_core 定义了文档的基本结构
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem,
    DescriptionAnnotation
)

from docling.models.base_model import BaseEnrichmentModel

from ..prompt.VLM_prompt import VLM_PROMPT
from ..vlm_enrichment_pipeline_options import VLMEnrichmentPipelineOptions

# === 日志配置 ===
log = logging.getLogger(__name__)

class VLMPictureEnrichmentModel(BaseEnrichmentModel):
    """
    一个 docling 增强模型，用于为 PictureItem 添加 VLM 生成的描述。
    使用 ThreadPoolExecutor 并行处理 VLM API 调用。
    """
    
    def __init__(self, options: VLMEnrichmentPipelineOptions):
        """
        初始化模型。

        Args:
            options: 包含 VLM 配置的 PipelineOptions。
        """
        super().__init__() 
        self.enabled = options.do_pic_enrichment
        self.settings = options.settings
        self.max_concurrency = options.vlm_max_concurrency
        
        # 兼容 docling 内部日志可能调用的属性
        self.model = "vlm-enricher"

        # 懒加载 Vision LLM 实例
        self._vision_llm = None

    @property
    def vision_llm(self):
        """获取 Vision LLM 实例 (延迟初始化)"""
        if self._vision_llm is None and self.settings:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(self.settings)
            except Exception as e:
                log.error(f"无法通过工厂创建 Vision LLM: {e}")
        return self._vision_llm

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        检查此模型是否应处理给定的元素。
        """
        return self.enabled and isinstance(element, PictureItem)

    def _call_vlm_for_description(self, pil_image: Image.Image, prompt: str=VLM_PROMPT) -> str:
        """
        使用 VLM  API 为给定的 PIL 图像生成描述。

        Args:
            pil_image: 待描述的 PIL.Image.Image 对象。
            prompt: 发送给 VLM 的提示词。

        Returns:
            VLM 返回的图像描述文本。
            
        Raises:
            Exception: 如果 API 调用失败。
        """
        try:
            # 1. 检查 Vision LLM 是否就绪
            if not self.vision_llm:
                return "[Vision LLM 未配置]"

            # 2. 预处理图像：RGBA -> RGB
            if pil_image.mode in ("RGBA", "LA", "P"):
                pil_image = pil_image.convert('RGB')
                
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            
            # 3. 通过统一工厂接口调用
            image_input = ImageInput(data=img_bytes, mime_type="image/jpeg")
            response = self.vision_llm.chat_with_image(
                text=prompt,
                image=image_input
            )
            
            description = response.content.strip()
            log.info("VLM 描述生成成功。")
            return description
            
        except Exception as e:
            log.warning(f"VLM 图像描述 API 失败: {e}")
            raise

    
    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        """
        处理一批元素 (由 docling 传入)，使用多线程并行处理 VLM Call。
        """
        # 1. 检查 VLM 是否被禁用
        if not self.enabled:
            yield from element_batch
            return

        # 2. 准备任务列表，保持原始顺序
        elements = list(element_batch)
        # 使用 self_ref 作为 key，因为 PictureItem 本身不可哈希
        futures_map = {}
        
        with ThreadPoolExecutor(
            max_workers=self.max_concurrency, 
            thread_name_prefix="VLM_Picture_Worker"
        ) as executor:
            
            # 3. 提交任务
            log.info(f"开始提交 VLM 任务，最大并发数: {self.max_concurrency}")
            for element in elements:
                if not self.is_processable(doc, element):
                    continue
                
                assert isinstance(element, PictureItem)
                
                try:
                    pil_img = element.get_image(doc)
                    if pil_img is None:
                        log.warning(f"获取 PictureItem {element.self_ref} 的图像失败 (返回 None)，跳过此项。")
                        continue

                    log.debug(f"正在提交 PictureItem {element.self_ref} 到 VLM 任务队列...")
                    future = executor.submit(
                        self._call_vlm_for_description,
                        pil_image=pil_img,
                        prompt=VLM_PROMPT,
                    )
                    futures_map[element.self_ref] = future
                    
                except Exception as e:
                    log.error(f"处理 PictureItem {element.self_ref} 时出错 (提交前): {e}")
            
            # 4. 按原始顺序产出结果
            for element in elements:
                if element.self_ref in futures_map:
                    future = futures_map[element.self_ref]
                    try:
                        description = future.result() 
                        log.info(f"VLM 任务成功 (PictureItem {element.self_ref})")
                        element.annotations.append(
                            DescriptionAnnotation(
                                provenance=self.model,
                                text=description
                            )
                        )
                    except Exception as e:
                        log.error(f"VLM 任务失败 (PictureItem {element.self_ref}): {e}")
                
                yield element
        
        log.info("VLM 批处理全部完成。")
