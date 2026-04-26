"""
RAG (Retrieval-Augmented Generation) 应用
流程：文档分片 → 向量化 → 索引存储 → 召回 → 重排 → 生成答案
"""
from typing import List

import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 初始化配置
# ============================================================

# 加载 .env 环境变量
load_dotenv()

# OpenAI 兼容客户端（支持 DashScope / 其他兼容 OpenAI 协议的 API）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))

# 嵌入模型：BGE base 英文，768 维向量
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# 重排模型：BGE reranker，用于对召回结果精排
reranker_model = CrossEncoder('BAAI/bge-reranker-base')

# 向量数据库：内存级 Chroma
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="my_chunks")


# ============================================================
# 索引阶段：文档分片 → 向量化 → 存储
# ============================================================

def split_info_chunks(doc_file: str) -> List[str]:
    """按双换行符切分文档为段落块"""
    with open(doc_file, 'r', encoding='utf-8') as file:
        content = file.read()
    return [chunk for chunk in content.split('\n\n') if chunk.strip()]


def embed_chunk(text: str) -> List[float]:
    """将文本转换为向量表示"""
    return embedding_model.encode(text).tolist()


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """将分片和向量存入 Chroma"""
    ids = [str(i) for i in range(len(chunks))]
    chromadb_collection.add(documents=chunks, embeddings=embeddings, ids=ids)


# 执行索引
chunks = split_info_chunks('doc.md')
embeddings = [embed_chunk(chunk) for chunk in chunks]
save_embeddings(chunks, embeddings)



# ============================================================
# 查询阶段：召回 → 重排 → 生成
# ============================================================

def retrieve(query: str, top_k: int = 3) -> List[str]:
    """基于向量相似度从 Chroma 召回 top_k 个段落"""
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]


def rerank(query: str, retrieved_chunks: List[str], top_k: int = 3) -> List[str]:
    """使用 Cross-Encoder 对召回结果精排，返回 top_k"""
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = reranker_model.predict(pairs)
    chunk_with_score_list = list(zip(retrieved_chunks, scores))
    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in chunk_with_score_list[:top_k]]


def generate_answer(query: str, reranked_chunks: List[str]) -> str:
    """基于重排后的上下文生成回答"""
    context = "\n".join(f"- {chunk}" for chunk in reranked_chunks)
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[
            {"role": "system", "content": "你是一个AI助手。请根据提供的参考信息回答问题，直接给出答案，不要使用「根据提供的参考信息」「根据以上内容」等开头语。如果参考信息不足以回答问题，请如实说明。"},
            {"role": "user", "content": f"参考信息：\n{context}\n\n问题：{query}"},
        ],
    )
    return completion.choices[0].message.content


# ============================================================
# 执行示例
# ============================================================

if __name__ == "__main__":
    query = "天使兽最终的结局是什么？"

    # 召回：从向量库检索 10 个相关段落
    retrieved_chunks = retrieve(query, top_k=10)

    # 重排：用 Cross-Encoder 精排，取 top 3
    reranked_chunks = rerank(query, retrieved_chunks, top_k=3)

    # 生成：基于上下文输出答案
    answer = generate_answer(query, reranked_chunks)
    print(answer)
