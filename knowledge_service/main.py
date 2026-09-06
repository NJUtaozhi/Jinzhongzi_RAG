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

# ===== 初始化 =====
try:
    logger.info("Loading vector model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    BASE_DIR = Path(__file__).parent.absolute()
    KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))
    
    # 获取正文 collection
    collection = chroma_client.get_collection(name="mental_health_knowledge")
    # 获取标题 collection
    title_collection = chroma_client.get_collection(name="mental_health_knowledge_titles")
    
    doc_count = collection.count()
    logger.info(f"Knowledge base ready, {doc_count} records.")
    
except Exception as e:
    logger.error(f"Service initialization failed: {str(e)}")
    logger.error(traceback.format_exc())
    model = None
    collection = None
    title_collection = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5  # 默认返回 5 条

def preprocess_query(query: str) -> str:
    """将疑问句转为陈述语气"""
    query = query.replace("？", "").replace("?", "")
    query = query.replace("是什么", "是")
    query = query.replace("什么是", "是")
    query = query.replace("如何", "方法")
    query = query.replace("怎么", "方法")
    return query.strip()

def extract_source(text: str) -> str:
    """从知识条目文本中提取【来源：xxx】标签"""
    match = re.search(r'【来源：([^】]+)】', text)
    if match:
        return match.group(1).strip()
    return "Mental Health Knowledge Base"

def extract_title(text: str) -> str:
    """从知识条目文本中提取标题部分（第一个[...]中的内容）"""
    match = re.match(r'^(\[[^\]]+\])\s*', text)
    if match:
        return match.group(1)
    return ""

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
        
        # ===== 向量化查询 =====
        try:
            query_embedding = model.encode(clean_query).tolist()
        except Exception as e:
            logger.error(f"Model encoding failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "code": 50001,
                    "msg": f"查询向量化失败: {str(e)}",
                    "data": None
                }
            )
        
        # ===== 路径1：正文语义检索 =====
        try:
            # 检索更多结果（top_k * 3），用于后续融合
            content_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(req.top_k * 3, collection.count() or 1)
            )
        except Exception as e:
            logger.error(f"Content query failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "code": 50002,
                    "msg": f"知识库检索失败: {str(e)}",
                    "data": None
                }
            )
        
        # ===== 路径2：标题向量检索 =====
        try:
            title_results = title_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(req.top_k * 3, title_collection.count() or 1)
            )
        except Exception as e:
            logger.error(f"Title query failed: {str(e)}")
            # 标题检索失败时，仅使用正文检索结果
            title_results = None
        
        # ===== 结果融合 =====
        # 构建 id -> (doc, score_content, score_title) 的字典
        fused = {}
        
        # 处理正文检索结果
        if content_results and content_results.get("documents") and content_results["documents"][0]:
            for doc, distance in zip(content_results["documents"][0], content_results["distances"][0]):
                doc_id = doc  # 暂时用文档内容作为key，实际项目应该用更可靠的ID
                # 防止重复，用文档内容作为唯一标识（因为内容本身是唯一的）
                key = doc
                similarity = 1 / (1 + distance)
                fused[key] = {
                    "doc": doc,
                    "score_content": similarity,
                    "score_title": 0.0
                }
        
        # 处理标题检索结果
        if title_results and title_results.get("documents") and title_results["documents"][0]:
            # 标题检索的 document 是标题文本，需要用它去匹配正文
            for title_doc, distance in zip(title_results["documents"][0], title_results["distances"][0]):
                similarity = 1 / (1 + distance)
                # 用标题文本去查找对应的正文文档
                # 在正文检索中，文档内容包含了标题，所以可以通过标题前缀匹配
                found = False
                for key in list(fused.keys()):
                    if key.startswith(title_doc) or title_doc in key:
                        fused[key]["score_title"] = similarity
                        found = True
                        break
                if not found:
                    # 如果正文检索中没有，单独添加（这种情况很少发生）
                    # 此时需要获取完整文档，但这里简化处理
                    logger.warning(f"Title match not found in content results: {title_doc}")
        
        # ===== 计算最终得分 =====
        WEIGHT_CONTENT = 0.3
        WEIGHT_TITLE = 0.7
        
        items = []
        for key, data in fused.items():
            final_score = WEIGHT_CONTENT * data["score_content"] + WEIGHT_TITLE * data["score_title"]
            # 如果标题得分不为0，说明标题有匹配，可以给予一定的额外加成
            if data["score_title"] > 0.1:
                # 对标题匹配较好的结果，进一步轻微提权
                final_score = final_score * 1.05
            final_score = min(final_score, 1.0)  # 限制最大值
            
            items.append({
                "content": data["doc"],
                "source": extract_source(data["doc"]),
                "score": round(final_score, 4)
            })
        
        # 按最终得分降序排序
        items.sort(key=lambda x: x["score"], reverse=True)
        # 只返回 top_k 个
        items = items[:req.top_k]
        
        # ===== 记录成功日志 =====
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