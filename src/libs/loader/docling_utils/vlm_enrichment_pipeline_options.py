from docling.datamodel.pipeline_options import PdfPipelineOptions
from typing import Optional, Any


class VLMEnrichmentPipelineOptions(PdfPipelineOptions):
    """
    自定义管道选项
    """
    # 是否启用 VLM 公式识别。
    do_formula_vlm_recognition: bool = True
    # 图片识别
    do_pic_enrichment: bool = True
    # 表格增强
    do_table_enrichment: bool = True

    # 注入全局 Settings 对象，以便工厂实例化 LLM
    settings: Optional[Any] = None

    # VLM API 调用的最大并发线程数 
    vlm_max_concurrency: int = 5
    # llm API 调用的最大并发线程数 
    llm_max_concurrency: int = 5
