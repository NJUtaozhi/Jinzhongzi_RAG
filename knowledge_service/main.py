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


def extract_tags(line: str):
    """提取条目开头的连续 [标签] 组，返回 tag1 和 tag2。"""
    match = re.match(r"^((?:\[[^\]\[]+\])+)", line)
    if not match:
        return "", ""
    tags = re.findall(r'\[([^\]]+)\]', match.group(1))
    tag1 = tags[0] if len(tags) > 0 else ""
    tag2 = tags[1] if len(tags) > 1 else ""
    return tag1, tag2


# ===== 初始化（整合远程同步逻辑 + 你的三标签架构）=====
try:
    logger.info("Loading vector model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    BASE_DIR = Path(__file__).parent.absolute()
    KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))

    # 使用 get_or_create_collection 避免服务启动崩溃
    collection = chroma_client.get_or_create_collection(name="mental_health_knowledge")
    tag1_collection = chroma_client.get_or_create_collection(name="mental_health_knowledge_tag1")
    tag2_collection = chroma_client.get_or_create_collection(name="mental_health_knowledge_tag2")

    doc_count = collection.count()
    logger.info(f"Knowledge base ready, {doc_count} records.")

    # ===== 知识库自动同步（主集合 + 标签一 + 标签二）=====
    try:
        KNOWLEDGE_FILE = BASE_DIR / "knowledge_data.txt"
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if lines:
            ids = [f"doc_{i+1}" for i in range(len(lines))]

            # 同步主集合
            if collection.count() != len(lines):
                logger.info(
                    "Synchronizing knowledge base: %s -> %s records...",
                    collection.count(),
                    len(lines),
                )
                embeddings = [model.encode(text).tolist() for text in lines]
                collection.upsert(documents=lines, ids=ids, embeddings=embeddings)
                logger.info("Knowledge base synchronized: %s records.", collection.count())

            # 提取标签
            tag1_list = []
            tag2_list = []
            for text in lines:
                t1, t2 = extract_tags(text)
                tag1_list.append(t1)
                tag2_list.append(t2)

            # 同步标签一集合
            unique_tag1 = list(dict.fromkeys([t for t in tag1_list if t]))  # 去重保序
            if tag1_collection.count() != len(unique_tag1):
                logger.info(
                    "Synchronizing tag1 collection: %s -> %s records...",
                    tag1_collection.count(),
                    len(unique_tag1),
                )
                tag1_ids = [f"tag1_{i+1}" for i in range(len(unique_tag1))]
                tag1_embeddings = [model.encode(t).tolist() for t in unique_tag1]
                tag1_collection.upsert(
                    documents=unique_tag1, ids=tag1_ids, embeddings=tag1_embeddings
                )
                logger.info("Tag1 collection synchronized: %s records.", tag1_collection.count())

            # 同步标签二集合
            unique_tag2 = list(dict.fromkeys([t for t in tag2_list if t]))
            if tag2_collection.count() != len(unique_tag2):
                logger.info(
                    "Synchronizing tag2 collection: %s -> %s records...",
                    tag2_collection.count(),
                    len(unique_tag2),
                )
                tag2_ids = [f"tag2_{i+1}" for i in range(len(unique_tag2))]
                tag2_embeddings = [model.encode(t).tolist() for t in unique_tag2]
                tag2_collection.upsert(
                    documents=unique_tag2, ids=tag2_ids, embeddings=tag2_embeddings
                )
                logger.info("Tag2 collection synchronized: %s records.", tag2_collection.count())
        else:
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
    tag1_collection = None
    tag2_collection = None


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
    """从知识条目文本中提取【来源：xxx】标签"""
    match = re.search(r'【来源：([^】]+)】', text)
    if match:
        return match.group(1).strip()
    return "Mental Health Knowledge Base"


# ===== 融合权重 =====
W_CONTENT = 0.3
W_TAG1 = 0.2
W_TAG2 = 0.5

# 标题（标签）匹配加成阈值
TAG_BONUS_THRESHOLD = 0.1
TAG_BONUS_FACTOR = 1.05

# 召回倍数
RECALL_MULTIPLIER = 6


@app.post("/v1/knowledge/retrieve")
async def retrieve_knowledge(req: RetrieveRequest):
    """
    语义检索：三路检索 + 加权融合。
    - 路径1：正文语义检索（权重 0.3）
    - 路径2：标签一向量检索（权重 0.2）
    - 路径3：标签二向量检索（权重 0.5）
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

        recall_n = req.top_k * RECALL_MULTIPLIER

        # ===== 路径1：正文语义检索 =====
        try:
            content_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(recall_n, collection.count() or 1),
                include=["documents", "distances", "metadatas"]
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

        # ===== 路径2：标签一向量检索 =====
        try:
            tag1_results = tag1_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(recall_n, tag1_collection.count() or 1),
                include=["documents", "distances"]
            )
        except Exception as e:
            logger.error(f"Tag1 query failed: {str(e)}")
            tag1_results = None

        # ===== 路径3：标签二向量检索 =====
        try:
            tag2_results = tag2_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(recall_n, tag2_collection.count() or 1),
                include=["documents", "distances"]
            )
        except Exception as e:
            logger.error(f"Tag2 query failed: {str(e)}")
            tag2_results = None

        # ===== 结果融合 =====
        fused = {}

        if content_results and content_results.get("documents") and content_results["documents"][0]:
            docs = content_results["documents"][0]
            distances = content_results["distances"][0]
            metas = content_results["metadatas"][0] if content_results.get("metadatas") else [{}] * len(docs)

            for doc, distance, meta in zip(docs, distances, metas):
                similarity = 1 / (1 + distance)
                fused[doc] = {
                    "doc": doc,
                    "tag1": meta.get("tag1", ""),
                    "tag2": meta.get("tag2", ""),
                    "score_content": similarity,
                    "score_tag1": 0.0,
                    "score_tag2": 0.0
                }

        if tag1_results and tag1_results.get("documents") and tag1_results["documents"][0]:
            tag1_docs = tag1_results["documents"][0]
            tag1_distances = tag1_results["distances"][0]
            for tag1_text, distance in zip(tag1_docs, tag1_distances):
                similarity = 1 / (1 + distance)
                for key, data in fused.items():
                    if data["tag1"] and data["tag1"] == tag1_text:
                        if similarity > data["score_tag1"]:
                            data["score_tag1"] = similarity

        if tag2_results and tag2_results.get("documents") and tag2_results["documents"][0]:
            tag2_docs = tag2_results["documents"][0]
            tag2_distances = tag2_results["distances"][0]
            for tag2_text, distance in zip(tag2_docs, tag2_distances):
                similarity = 1 / (1 + distance)
                for key, data in fused.items():
                    if data["tag2"] and data["tag2"] == tag2_text:
                        if similarity > data["score_tag2"]:
                            data["score_tag2"] = similarity

        # ===== 计算最终得分 =====
        items = []
        for key, data in fused.items():
            final_score = (
                W_CONTENT * data["score_content"]
                + W_TAG1 * data["score_tag1"]
                + W_TAG2 * data["score_tag2"]
            )

            if data["score_tag1"] > TAG_BONUS_THRESHOLD or data["score_tag2"] > TAG_BONUS_THRESHOLD:
                final_score = final_score * TAG_BONUS_FACTOR

            final_score = min(final_score, 1.0)

            items.append({
                "content": data["doc"],
                "tag1": data["tag1"],
                "tag2": data["tag2"],
                "source": extract_source(data["doc"]),
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
        "doc_count": collection.count(),
        "tag1_count": tag1_collection.count() if tag1_collection else 0,
        "tag2_count": tag2_collection.count() if tag2_collection else 0
    }