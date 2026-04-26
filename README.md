# RAG

基于 Python 的检索增强生成（Retrieval-Augmented Generation）项目，实现文档分片、向量化索引、语义召回、交叉编码器重排与 LLM 生成的完整 RAG 流程。

## 项目结构

```
.
├── rag-demo.py            # RAG 主流程：索引 → 召回 → 重排 → 生成
├── cosine-similarity.py   # 余弦相似度原理演示（点积 / 模长）
├── doc.md                 # 示例文档（用于索引和检索）
├── pyproject.toml         # 项目依赖与配置
└── .env                   # API 密钥与 Base URL（需自行创建）
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 嵌入模型 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)（768 维） |
| 重排模型 | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |
| 向量数据库 | [ChromaDB](https://www.trychroma.com/)（内存模式） |
| LLM | qwen3.6-plus（通过 OpenAI 兼容协议调用） |
| 包管理 | [uv](https://docs.astral.sh/uv/) |

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

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 本项目使用 OpenAI 兼容协议，支持 DashScope 等提供兼容接口的服务。

### 运行

```bash
uv run python rag-demo.py
```

## RAG 流程

```mermaid
flowchart LR
    A[文档 doc.md] --> B[分片]
    B --> C[向量化 embedding]
    C --> D[(ChromaDB)]
    D --> E[语义召回 top-K]
    E --> F[Cross-Encoder 重排]
    F --> G[LLM 生成答案]
```

| 阶段 | 说明 |
|------|------|
| 分片 | 按双换行符将文档切分为多个 chunk |
| 向量化 | 使用 BGE 模型将每个 chunk 编码为 768 维向量 |
| 索引 | 向量及原文存入 ChromaDB（内存模式） |
| 召回 | 查询时通过向量相似度检索 Top-10 候选段落 |
| 重排 | Cross-Encoder 对召回结果重新打分，取 Top-3 |
| 生成 | 将重排后的上下文传入 LLM，生成最终答案 |

## 余弦相似度

`cosine-similarity.py` 从零实现了余弦相似度计算，展示向量检索的核心数学原理：

```
similarity = dot(A, B) / (norm(A) × norm(B))
```

可通过以下命令运行：

```bash
uv run python cosine-similarity.py
```
