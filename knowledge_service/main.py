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

# ===== 初始化（整合异常处理 + 自动构建）=====
try:
    logger.info("Loading vector model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # 使用绝对路径定位 ChromaDB 目录
    BASE_DIR = Path(__file__).parent.absolute()
    KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))
    collection = chroma_client.get_collection(name="mental_health_knowledge")
    
    doc_count = collection.count()
    logger.info(f"Knowledge base ready, {doc_count} records.")
    
    # ===== 采纳远程的自动构建逻辑 =====
    if collection.count() == 0:
        logger.info("Knowledge base is empty, auto-building...")
        try:
            KNOWLEDGE_FILE = BASE_DIR / "knowledge_data.txt"
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                ids = [f"doc_{i+1}" for i in range(len(lines))]
                embeddings = [model.encode(text).tolist() for text in lines]
                collection.add(documents=lines, ids=ids, embeddings=embeddings)
                logger.info(f"Auto-built: {collection.count()} records.")
            else:
                logger.warning("knowledge_data.txt is empty, no records added.")
        except FileNotFoundError:
            logger.error("knowledge_data.txt not found! Auto-build skipped.")
        except Exception as e:
            logger.error(f"Auto-build failed: {str(e)}")
    
except Exception as e:
    logger.error(f"Service initialization failed: {str(e)}")
    logger.error(traceback.format_exc())
    model = None
    collection = None

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

def extract_source(text: str) -> str:
    """
    从知识条目文本中提取【来源：xxx】标签。
    如果找到，返回来源名称；否则返回默认值。
    """
    match = re.search(r'【来源：([^】]+)】', text)
    if match:
        return match.group(1).strip()
    return "Mental Health Knowledge Base"

@app.post("/v1/knowledge/retrieve")
async def retrieve_knowledge(req: RetrieveRequest):
    """
    语义检索：接收查询文本，返回最相关的知识片段。
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
        logger.warning(f"Invalid top_k: {req.top_k}, using default 3")
        req.top_k = 3
    
    try:
        clean_query = preprocess_query(req.query)
        logger.debug(f"Cleaned query: {clean_query}")
        
        try:
            query_embedding = model.encode(clean_query).tolist()
        except Exception as e:
            logger.error(f"Model encoding failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "code": 50001,
                    "msg": f"查询向量化失败，请检查输入内容是否有效: {str(e)}",
                    "data": None
                }
            )
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(req.top_k * 2, collection.count() or 1)
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "code": 50002,
                    "msg": f"知识库检索失败，请稍后重试: {str(e)}",
                    "data": None
                }
            )
        
        items = []
        if results.get("documents") and results["documents"][0]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                similarity = 1 / (1 + distance)
                source = extract_source(doc)
                items.append({
                    "content": doc,
                    "source": source,
                    "score": round(similarity, 4)
                })
            
            items.sort(key=lambda x: x["score"], reverse=True)
            items = items[:req.top_k]
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"Query completed: {len(items)} hits, {elapsed:.3f}s")
        
        return {
            "code": 200,
            "msg": "success",
            "data": {"results": items}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "code": 50000,
                "msg": f"服务内部错误，请稍后重试: {str(e)}",
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