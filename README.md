# Modular RAG MCP Server (Enterprise Edition)

> 一个企业级、高并发、可观测、可插拔的模块化 RAG（检索增强生成）服务框架。基于 MCP（Model Context Protocol）协议对外暴露知识检索接口，支持与 Claude Desktop、GitHub Copilot 等主流 AI 助手无缝集成。

***

## 📖 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [架构设计](#-架构设计)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [致谢与二次开发声明](#-致谢与二次开发声明)

***

## 🏗️ 项目概述

Modular RAG MCP Server 旨在解决企业私有数据与大语言模型（LLM）之间的知识壁垒。
它不仅仅是一个检索工具，而是一个端到端的知识流转管线。系统涵盖了从多模态文档解析（Docling）、语义切分（Semantic Chunking）、高并发数据摄取（OpenSearch Async Bulk），到多路召回（Hybrid Search）、TEI 模型重排（Rerank）的完整生命周期。

通过标准化的 **MCP (Model Context Protocol)**，本系统能以最低成本接入企业现有的 AI Agent 体系，实现真正的"一次部署，多端调用"。

***

## ✨ 核心特性

| 模块          | 企业级能力      | 技术实现                                                                             |
| ----------- | ---------- | -------------------------------------------------------------------------------- |
| **高精度数据解析** | 结构保留与多模态提取 | 集成 **Docling** 引擎，精准识别 PDF/Word 的标题层级、表格，并自动提取图片转为 Markdown 占位符。                 |
| **语义化切块**   | 保持文档逻辑完整性  | **Semantic Markdown Splitter** 基于文档的 `# H1 / ## H2` 层级进行语义切分，避免暴力截断破坏上下文。        |
| **高并发存储**   | 亿级文档吞吐能力   | 集成 **OpenSearch**，使用 `asyncio` + `async_bulk` 实现异步批量写入，配合 Semaphore 控制集群并发压力。    |
| **多路混合检索**  | 平衡查全率与查准率  | 支持 OpenSearch 下的多字段联合召回（正文/摘要/假设性问题），结合 **RRF (Reciprocal Rank Fusion)** 算法融合排序。 |
| **高性能重排**   | 毫秒级精准过滤    | 支持对接 **TEI (Text Embeddings Inference)** 驱动的 Cross-Encoder 模型，实现极低延迟的二次排序。       |
| **全链路可观测**  | 检索过程白盒化    | 内置双链路追踪体系（Ingestion & Query Trace），提供 Streamlit Dashboard 进行实时耗时分析与 Bad Case 定位。 |
| **双模传输协议**  | 适配不同部署环境   | MCP Server 同时支持 **Stdio**（本地零配置调用）与 **SSE**（远程 HTTP 流式调用，适合多租户与网关集成）。            |

***

## � 架构设计

### 全链路可插拔设计 (Pluggable Architecture)

系统严格遵循开闭原则，核心组件均定义了标准抽象接口，通过 `config/settings.yaml` 实现零代码热切换：

- **LLM**: OpenAI / Azure / DeepSeek / Ollama
- **Embedding**: OpenAI / BGE / Ollama
- **Vector Store**: OpenSearch (生产环境) / Chroma (本地开发)
- **Loader**: Docling (高精度) / PyMuPDF (极速)
- **Reranker**: TEI (高性能) / Cross-Encoder / LLM Rerank

### 智能摄取流水线 (Ingestion Pipeline)

数据摄取分为五个阶段，支持幂等更新（基于 SHA256与内容哈希）：

1. **Load**: Docling 高精度解析，生成带图片占位符的 Canonical Markdown。
2. **Split**: 基于 Markdown 结构的语义化切块。
3. **Transform**: LLM 驱动的 Chunk 修复、元数据增强（Title/Summary），以及 Vision LLM 驱动的 **Image Captioning**（实现搜文出图）。
4. **Embed**: 差量计算向量，避免重复调用 API。
5. **Upsert**: 异步高并发写入 OpenSearch。

***

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd Modular-RAG-MCP-Server

# 创建并激活虚拟环境 (可选)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安装项目依赖
pip install -e .
```

### 2. 启动基础设施 (Docker)

使用 `docker-compose.yml`一键启动 OpenSearch 和 TEI 重排服务：

```bash
docker compose up -d
```

### 3. 数据摄取与检索

在启动服务前，需要先将文档导入知识库：

```bash
# 导入单个文档或整个目录
python scripts/ingest.py --path ./test_data/ --collection my_kb

# (可选) 命令行直接测试检索效果
python scripts/query.py --query "什么是 RAG？" --collection my_kb
```

### 4. 启动服务

系统支持多种启动方式，满足本地开发与远程部署需求：

```bash
# 启动 MCP Server (Stdio 模式，供 Claude Desktop / Copilot 调用)
python src/mcp_server/server.py

# 启动 MCP Server (SSE 模式，默认端口 8000)
python src/mcp_server/server.py --sse

# 启动可视化管理后台 (Streamlit)
python scripts/start_dashboard.py
```

---

## ⚙️ 配置说明

系统核心配置位于 `config/settings.yaml`。首次使用时，可参考 `config/settings.yaml.example` 进行配置。

## 🤝 致谢与二次开发声明

本项目基于开源项目 **Modular-RAG-MCP-Server** 进行二次开发与企业级改造。

**在此基础上，进行了以下核心架构升级与扩展：**

1. **引入 Docling 高精度解析引擎**：替换了原有的 PyMuPDF，大幅提升了对复杂 PDF（尤其是多层级标题、表格）的结构识别能力，并完善了图片提取管线。
2. **实现 Semantic Markdown Splitter**：彻底改变了单纯依赖字符长度的递归切分逻辑，转为基于文档语义标题的智能切块。
3. **集成 OpenSearch 与并发优化**：替换了单机版 Chroma，实现了支持 `async_bulk` 与 Semaphore 并发控制的高吞吐向量与稀疏联合检索引擎。
4. **TEI 推理加速**：集成了 Hugging Face Text Embeddings Inference，将 Cross-Encoder 重排序的延迟降低至生产可用水平。
5. **MCP SSE 传输支持**：在原有的 Stdio 基础上增加了 Server-Sent Events (SSE) 传输支持，使系统具备了云端部署和跨网络调用的能力。

感谢原作者提供的优秀模块化骨架，使得本次企业级架构演进得以快速且优雅地落地。
