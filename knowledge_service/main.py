import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI(title="Knowledge Retrieval Service")

print("Loading vector model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path="./kb_chroma_db")
collection = chroma_client.get_collection(name="mental_health_knowledge")
print(f"Knowledge base ready, {collection.count()} records.")

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

def preprocess_query(query: str) -> str:
    """将疑问句转为陈述语气"""
    query = query.replace("？", "").replace("?", "")
    query = query.replace("是什么", "是")
    query = query.replace("什么是", "是")
    query = query.replace("如何", "方法")
    query = query.replace("怎么", "方法")
    return query.strip()

@app.post("/v1/knowledge/retrieve")
async def retrieve_knowledge(req: RetrieveRequest):
    clean_query = preprocess_query(req.query)
    query_embedding = model.encode(clean_query).tolist()

    # 检索更多结果（top_k * 2），以便后续筛选
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k * 2, collection.count())
    )

    items = []
    if results["documents"] and results["documents"][0]:
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            # ChromaDB 默认返回的是欧氏距离的平方 (L2 squared)
            # 将其映射到 [0, 1] 区间的相似度分数
            # 这里使用 sigmoid 式映射，让分数分布更合理
            similarity = 1 / (1 + distance)  # 值域 (0, 1]，distance=0 时 similarity=1
            items.append({
                "content": doc,
                "source": "Mental Health Knowledge Base",
                "score": round(similarity, 4)
            })
        
        # 按相似度降序排序
        items.sort(key=lambda x: x["score"], reverse=True)
        # 只返回 top_k 个
        items = items[:req.top_k]

    return {
        "code": 200,
        "msg": "success",
        "data": {"results": items}
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "knowledge-retrieval",
        "doc_count": collection.count()
    }