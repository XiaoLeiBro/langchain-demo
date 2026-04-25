from typing import List

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb


# 提问前：分片 + 索引
def split_info_chunks(doc_file: str) -> List[str]:
    """
    处理分片逻辑
    :param doc_file: 文档名
    :return: 分片列表
    """
    with open(doc_file, 'r', encoding='utf-8') as file:
        content = file.read()
    return [chunk for chunk in content.split('\n\n') if chunk.strip()]


chunks = split_info_chunks('doc.md')
# for i, chunk in enumerate(chunks):
#     print(f"[{i} {chunk}\n]")

embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')


def embed_chunk(text: str) -> List[float]:
    """
    将文本转换为向量表示
    :param text: 要嵌入的文本
    :return: 向量(浮点数列表)
    """
    embedding = embedding_model.encode(text)
    return embedding.tolist()


# test_embedding = embed_chunk("测试内容")
# print(len(test_embedding))
# print(test_embedding)

# 获取所有分片的向量集合
embeddings = [embed_chunk(chunk) for chunk in chunks]

# print(len(embeddings))
# print(embeddings[0])

# 创建Chroma数据库
chromadb_client = chromadb.EphemeralClient()
# 创建集合
chromadb_collection = chromadb_client.get_or_create_collection(name="my_chunks")


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    保存向量集合
    :param chunks: 分片列表
    :param embeddings: 向量列表
    :return: None
    """
    # 创建id列表，每个id对应一个分片，id为分片索引，chromadb要求每个分片都要有一个id
    ids = [str(i) for i in range(len(chunks))]
    # 保存向量集合，id为分片索引，embedding为向量，chunk为分片内容
    chromadb_collection.add(documents=chunks, embeddings=embeddings, ids=ids)


# 保存向量集合
save_embeddings(chunks, embeddings)


# 提问后：召回 + 重排 + 生成

def retrieve(query: str, top_k: int = 3) -> List[str]:
    """
    召回
    :param query: 提问内容
    :param top_k: 召回数量
    :return: 召回结果
    """
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]


query = "陈满仓为什么哭了？"
# 召回
retrieved_chunks = retrieve(query, 10)


# for i, chunk in enumerate(retrieved_chunks):
#     print(f"[{i} {chunk}\n]")


# 重排
def rerank(query: str, retrieved_chunks: List[str], top_k: int = 3) -> List[str]:
    """
    重排
    :param query: 提问内容
    :param retrieved_chunks: 召回结果
    :param top_k: 重排数量
    :return: 重排结果
    """
    cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
    # 创建pair列表
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    # 获取重排分数
    scores = cross_encoder.predict(pairs)
    # 排序
    chunk_with_score_list = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
    # 获取重排结果
    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
    # 获取重排结果
    return [chunk for chunk, score in chunk_with_score_list[:top_k]]


reranked_chunks = rerank(query, retrieved_chunks, 3)
for i, chunk in enumerate(reranked_chunks):
    print(f"[{i} {chunk}\n]")
