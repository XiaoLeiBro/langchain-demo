# RAG

基于 Python 的检索增强生成（Retrieval-Augmented Generation）项目，实现文档分片、向量化索引、语义召回与交叉编码器重排的完整 RAG 流程。

## 技术栈

- **向量模型**: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
- **重排模型**: [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- **向量数据库**: [ChromaDB](https://www.trychroma.com/)
- **包管理**: [uv](https://docs.astral.sh/uv/)

## 快速开始

### 前置条件

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 安装

```bash
git clone https://github.com/Xiaolei5035/python-rag.git
cd python-rag
uv sync
```

### 配置

在项目根目录创建 `.env` 文件，填写 API 密钥和 Base URL：

```env
OPENAI_API_KEY=your-api-key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 本项目使用 OpenAI 兼容协议，支持 DashScope 等提供兼容接口的服务。

### 运行

```bash
uv run python rag.py
```

## 流程

```
文档 → 分片 → 向量化 → ChromaDB 存储 → 语义召回 → 交叉编码器重排 → LLM 生成答案
```

1. **分片**: 将文档按段落切分为多个 chunk
2. **向量化**: 使用 BGE 模型将每个 chunk 转为向量
3. **索引**: 向量及原文存入 ChromaDB
4. **召回**: 查询时通过向量相似度检索 Top-K 候选
5. **重排**: 使用 Cross-Encoder 对召回结果精排
6. **生成**: 将重排后的上下文与大模型结合，生成最终答案
