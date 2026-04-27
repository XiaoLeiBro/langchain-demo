# LangChain Demo

基于 LangChain 的 Python 学习项目，包含 RAG（检索增强生成）、LLM 聊天模型、嵌入模型及余弦相似度原理演示。

## 项目结构

```
.
├── demo/
│   ├── llm-models.py                    # ChatOpenAI 聊天模型：简单调用 / 流式输出 / 多轮对话
│   ├── embedding-models.py              # HuggingFace 嵌入模型 + 余弦相似度检索
│   ├── universal-prompt.py              # 通用提示词模板
│   └── rag/
│       ├── rag.py                       # RAG 完整流程：索引 → 召回 → 重排 → 生成
│       ├── cosine-similarity/
│       │   └── cosine-similarity.py     # 余弦相似度从零实现
│       └── doc.md                       # 示例文档（用于 RAG 索引和检索）
├── pyproject.toml                       # 项目依赖与配置
├── uv.lock
└── .env                                 # API 密钥与 Base URL（需自行创建）
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 嵌入模型 | [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)（768 维） |
| 重排模型 | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |
| 向量数据库 | [ChromaDB](https://www.trychroma.com/)（内存模式） |
| LLM | qwen3.6-plus（阿里云百炼 OpenAI 兼容接口） |
| LLM 框架 | [LangChain](https://www.langchain.com/)（ChatOpenAI） |
| 嵌入框架 | [LangChain-HuggingFace](https://python.langchain.com/docs/integrations/text_embedding/huggingface/) |
| 包管理 | [uv](https://docs.astral.sh/uv/) |

## 快速开始

### 前置条件

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 安装

```bash
git clone https://github.com/XiaoLeiBro/langchain-demo.git
cd langchain-demo
uv sync
```

### 配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 本项目通过 OpenAI 兼容协议接入 LLM，支持阿里云百炼、DashScope 等服务。

### 运行

```bash
uv run demo/llm-models.py                              # LangChain 聊天模型
uv run demo/embedding-models.py                         # 嵌入模型 + 余弦相似度检索
uv run demo/rag/rag.py                                  # RAG 完整流程
uv run demo/rag/cosine-similarity/cosine-similarity.py  # 余弦相似度原理演示
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

## 嵌入模型

`demo/embedding-models.py` 使用 HuggingFaceEmbeddings 加载 BGE 中文模型，演示文本向量化与余弦相似度检索。

## 余弦相似度

`demo/rag/cosine-similarity/cosine-similarity.py` 从零实现了余弦相似度计算，展示向量检索的核心数学原理：

```
similarity = dot(A, B) / (norm(A) × norm(B))
```

## 提示词模板与链式调用

`demo/universal-prompt.py` 演示了 PromptTemplate 与 LCEL（LangChain Expression Language）链式调用：

- **PromptTemplate** — 使用 `{变量}` 占位符定义可复用的提示词模板，通过 `.format()` 注入参数
- **LCEL 链式调用** — 使用 `|` 管道符将 prompt 与 model 串联为链（chain），`chain.invoke()` 一次完成格式化+推理

```python
# 链式调用：prompt | model，一次调用完成格式化 + LLM 推理
chain = prompt_template | model
res = chain.invoke(input={"poet": "王维", "story": "唐诗"})
```

## LangChain 聊天模型

`demo/llm-models.py` 演示了 ChatOpenAI 的三种调用方式：

- **简单调用** `model.invoke()` — 一次性问答
- **流式输出** `model.stream()` — 打字机效果逐字输出
- **多轮对话** — 使用 `(role, content)` 元组构建 `system`/`human`/`ai` 消息列表
