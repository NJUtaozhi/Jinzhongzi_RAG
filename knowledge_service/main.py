import os
import logging
import time
import traceback
import re
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

# ===== 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Retrieval Service")

# ===== 初始化（整合远程同步逻辑）=====
try:
    logger.info("Loading vector model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    BASE_DIR = Path(__file__).parent.absolute()
    KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))
    
    collection = chroma_client.get_collection(name="mental_health_knowledge")
    title_collection = chroma_client.get_collection(name="mental_health_knowledge_titles")
    
    doc_count = collection.count()
    logger.info(f"Knowledge base ready, {doc_count} records.")
    
    # ===== 知识库自动同步（来自远程版本）=====
    try:
        KNOWLEDGE_FILE = BASE_DIR / "knowledge_data.txt"
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines and collection.count() != len(lines):
            logger.info(
                "Synchronizing knowledge base: %s -> %s records...",
                collection.count(),
                len(lines),
            )
            ids = [f"doc_{i+1}" for i in range(len(lines))]
            embeddings = [model.encode(text).tolist() for text in lines]
            collection.upsert(documents=lines, ids=ids, embeddings=embeddings)
            logger.info("Knowledge base synchronized: %s records.", collection.count())
        elif not lines:
            logger.warning("knowledge_data.txt is empty, no records added.")
    except FileNotFoundError:
        logger.error("knowledge_data.txt not found! Auto-sync skipped.")
    except Exception as e:
        logger.error("Knowledge synchronization failed: %s", str(e))
    
except Exception as e:
    logger.error(f"Service initialization failed: {str(e)}")
    logger.error(traceback.format_exc())
    model = None
    collection = None
    title_collection = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

def preprocess_query(query: str) -> str:
    """将疑问句转为陈述语气"""
    query = query.replace("？", "").replace("?", "")
    query = query.replace("是什么", "是")
    query = query.replace("什么是", "是")
    query = query.replace("如何", "方法")
    query = query.replace("怎么", "方法")
    return query.strip()

def extract_source(text: str) -> str:
    """
    从知识条目文本中提取【来源：xxx】标签。
    同时支持【】和[]两种括号格式（来自远程版本的改进）。
    """
    match = re.search(r'[【\[]来源：([^】\]]+)[】\]]', text)
    if match:
        return match.group(1).strip()
    return "Mental Health Knowledge Base"

def split_source(text: str) -> tuple[str, str]:
    """拆分正文与行尾来源标签，避免把标记重复显示给前端。"""
    source = extract_source(text)
    content = re.sub(r'\s*【来源：[^】]+】\s*$', '', text).strip()
    return content, source

@app.post("/v1/knowledge/retrieve")
async def retrieve_knowledge(req: RetrieveRequest):
    """
    语义检索：接收查询文本，返回最相关的知识片段。
    采用双路检索 + 结果融合：
    - 路径1：正文语义检索（权重 0.3）
    - 路径2：标题向量检索（权重 0.7）
    """
    start_time = time.perf_counter()
    logger.info(f"Query received: {req.query}, top_k: {req.top_k}")
    
    if model is None or collection is None:
        logger.error("Service not initialized properly")
        return JSONResponse(
            status_code=503,
            content={
                "code": 50300,
                "msg": "知识库服务未就绪，请检查模型或数据库是否正常加载",
                "data": None
            }
        )
    
    if not req.query or not req.query.strip():
        logger.warning("Empty query received")
        return JSONResponse(
            status_code=400,
            content={
                "code": 40001,
                "msg": "查询内容不能为空，请提供有效的问题或关键词",
                "data": None
            }
        )
    
    if req.top_k < 1:
        logger.warning(f"Invalid top_k: {req.top_k}, using default 5")
        req.top_k = 5
    
    try:
        clean_query = preprocess_query(req.query)
        logger.debug(f"Cleaned query: {clean_query}")
        
        query_embedding = model.encode(clean_query).tolist()
        
        # ===== 路径1：正文语义检索 =====
        content_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(req.top_k * 3, collection.count() or 1)
        )
        
        # ===== 路径2：标题向量检索 =====
        try:
            title_results = title_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(req.top_k * 3, title_collection.count() or 1)
            )
        except Exception as e:
            logger.error(f"Title query failed: {str(e)}")
            title_results = None
        
        # ===== 结果融合 =====
        fused = {}
        
        if content_results and content_results.get("documents") and content_results["documents"][0]:
            for doc, distance in zip(content_results["documents"][0], content_results["distances"][0]):
                key = doc
                similarity = 1 / (1 + distance)
                fused[key] = {
                    "doc": doc,
                    "score_content": similarity,
                    "score_title": 0.0
                }
        
        if title_results and title_results.get("documents") and title_results["documents"][0]:
            for title_doc, distance in zip(title_results["documents"][0], title_results["distances"][0]):
                similarity = 1 / (1 + distance)
                for key in list(fused.keys()):
                    if key.startswith(title_doc) or title_doc in key:
                        fused[key]["score_title"] = similarity
                        break
        
        # ===== 计算最终得分并组装返回结果 =====
        WEIGHT_CONTENT = 0.3
        WEIGHT_TITLE = 0.7
        
        items = []
        for key, data in fused.items():
            final_score = WEIGHT_CONTENT * data["score_content"] + WEIGHT_TITLE * data["score_title"]
            if data["score_title"] > 0.1:
                final_score = final_score * 1.05
            final_score = min(final_score, 1.0)
            
            # 使用 split_source 清理来源标签
            content_clean, source = split_source(data["doc"])
            
            items.append({
                "content": content_clean,
                "source": source,
                "score": round(final_score, 4)
            })
        
        items.sort(key=lambda x: x["score"], reverse=True)
        items = items[:req.top_k]
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"Query completed: {len(items)} hits, {elapsed:.3f}s")
        
        return {
            "code": 200,
            "msg": "success",
            "results": items
        }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "code": 50000,
                "msg": f"服务内部错误: {str(e)}",
                "data": None
            }
        )

@app.get("/health")
def health_check():
    """健康检查接口"""
    if collection is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "knowledge-retrieval",
                "doc_count": 0,
                "error": "知识库未初始化"
            }
        )
    return {
        "status": "ok",
        "service": "knowledge-retrieval",
        "doc_count": collection.count()
    }