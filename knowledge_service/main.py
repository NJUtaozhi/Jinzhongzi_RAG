import os
import logging
import time
import traceback
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb


# ===== 路径与常量 =====
BASE_DIR = Path(__file__).parent.absolute()
KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
COLLECTION_NAME = "mental_health_knowledge"

TOPIC_VOCAB = [
    "焦虑", "抑郁", "社交焦虑", "学业", "人际", "恋爱", "拖延", "睡眠",
    "正念", "情绪", "完美主义", "心理韧性", "精神障碍", "职业倦怠",
    "校园支持", "心理健康", "危机干预",
]

CONTENT_TYPE_VOCAB = [
    "definition", "symptom", "cause", "assessment_tool", "scoring",
    "management_advice", "coping_strategy", "treatment", "crisis_rule",
    "referral_resource", "research_data", "misconception",
    "institutional_recommendation", "usage_note",
]

# 主题别名表（可扩展）
TOPIC_ALIASES = {
    "焦虑": ["焦虑", "紧张", "担心", "恐慌", "心慌", "惊恐"],
    "抑郁": ["抑郁", "低落", "没兴趣", "想哭", "开心不起来", "情绪低落"],
    "社交焦虑": ["社交焦虑", "社交恐惧", "社恐"],
    "学业": ["学业", "学习", "考试", "成绩", "考研", "作业"],
    "人际": ["人际", "社交", "室友", "朋友", "交往", "宿舍"],
    "恋爱": ["恋爱", "失恋", "分手", "感情", "伴侣", "依恋"],
    "拖延": ["拖延", "拖沓", "推迟", "磨蹭"],
    "睡眠": ["睡眠", "失眠", "睡不着", "早醒", "多梦", "入睡困难"],
    "正念": ["正念", "冥想", "觉察", "觉知"],
    "情绪": ["情绪", "内耗", "愤怒", "生气", "发火", "烦躁"],
    "完美主义": ["完美主义", "完美", "高标准"],
    "心理韧性": ["心理韧性", "韧性", "复原力", "抗压", "逆境"],
    "精神障碍": ["精神障碍", "PTSD", "强迫症", "双相", "厌食", "ADHD",
                 "多动症", "网瘾", "创伤后应激", "季节性情感障碍"],
    "职业倦怠": ["职业倦怠", "工作倦怠", "burnout"],
    "校园支持": ["校园支持", "心理咨询中心", "学校支持", "校园氛围"],
    "心理健康": ["心理健康", "心理亚健康", "心理保健"],
    "危机干预": ["危机", "自杀", "自伤", "想死", "不想活", "安全计划",
                 "紧急转介", "危机干预"],
}

# 内容类型关键词表
CONTENT_TYPE_KEYWORDS = {
    "definition": ["是什么", "什么是", "定义", "含义", "概念", "区别"],
    "symptom": ["症状", "表现", "会有哪些", "有哪些表现", "临床表现"],
    "cause": ["原因", "为什么", "诱因", "怎么产生", "风险因素"],
    "assessment_tool": ["量表", "测试", "评估工具", "条目", "题目"],
    "scoring": ["评分", "分数", "多少分", "分级", "阈值", "切分点", "标准分"],
    "management_advice": ["怎么办", "如何处理", "管理", "建议", "应对方案", "干预"],
    "coping_strategy": ["怎么缓解", "方法", "技巧", "应对", "放松"],
    "treatment": ["治疗", "药物", "疗法", "怎么治"],
    "crisis_rule": ["自杀", "自伤", "想死", "危机", "紧急", "危险"],
    "referral_resource": ["热线", "电话", "咨询室", "转介", "预约", "地址"],
    "research_data": ["研究", "数据", "统计", "流行病学", "灵敏度", "特异度"],
    "misconception": ["误区", "误解", "错误观念", "误区"],
    "institutional_recommendation": ["机构", "学校应该", "政策", "建议院校"],
    "usage_note": ["注意事项", "适用", "局限性", "适用人群"],
}

# 来源权威性权重
SOURCE_WEIGHT = {
    "clinical_guideline": 1.0,
    "original_instrument": 1.0,
    "peer_reviewed_paper": 0.8,
    "government_website": 0.7,
    "institutional_website": 0.6,
    "encyclopedia": 0.3,
    "unknown": 0.1,
}

# 评分权重
W_VECTOR = 0.6
W_SOURCE = 0.2
W_CONTENT_TYPE = 0.1
W_TOPIC = 0.1

# 危机条目分数下限
CRISIS_FLOOR = 0.95

# 召回倍数
RECALL_MULTIPLIER = 6

# ===== 日志 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")

app = FastAPI(title="Knowledge Retrieval Service")

# ===== 初始化 =====
try:
    logger.info("Loading vector model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    logger.info(f"Collection ready, {collection.count()} entries.")
except Exception as e:
    logger.error(f"Service init failed: {e}")
    logger.error(traceback.format_exc())
    model = None
    collection = None


# ===== 请求模型 =====
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    crisis_signal: bool = False  # 上游传入的危机信号


# ===== 工具函数 =====
def preprocess_query(query: str) -> str:
    """去标点、语气转换。"""
    query = query.replace("？", "").replace("?", "")
    query = query.replace("是什么", "是").replace("什么是", "是")
    return query.strip()


def detect_topics(query: str):
    """识别查询中的主题。返回主题列表（可能为空）。"""
    found = []
    for topic, aliases in TOPIC_ALIASES.items():
        for alias in aliases:
            if alias in query:
                found.append(topic)
                break
    # 若同时命中宽泛和具体主题，优先具体
    if "社交焦虑" in found and "焦虑" in found:
        found.remove("焦虑")
    if "职业倦怠" in found and "学业" in found:
        # 二者不冲突，保留
        pass
    return found


def detect_content_type(query: str):
    """识别查询的内容类型。返回单个 content_type 或 None。"""
    # 优先级：危机 > 转介 > 评分 > 评估工具 > 管理建议/应对 > 治疗 > 症状 > 原因 > 定义
    priority = [
        "crisis_rule", "referral_resource", "scoring", "assessment_tool",
        "management_advice", "coping_strategy", "treatment",
        "symptom", "cause", "definition", "research_data",
        "misconception", "institutional_recommendation", "usage_note",
    ]
    for ct in priority:
        for kw in CONTENT_TYPE_KEYWORDS.get(ct, []):
            if kw in query:
                return ct
    return None


def build_where(topics, content_type):
    """构造 ChromaDB where 条件。"""
    conditions = []

    if topics:
        if len(topics) == 1:
            conditions.append({f"topic_{topics[0]}": True})
        else:
            conditions.append({"$or": [{f"topic_{t}": True} for t in topics]})

    if content_type:
        conditions.append({"content_type": content_type})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def extract_entry_topics(meta: dict):
    """从元数据还原 topic 数组。"""
    return [t for t in TOPIC_VOCAB if meta.get(f"topic_{t}")]


def build_result_item(doc_id, doc, distance, meta,
                      detected_topics, detected_ct):
    """构造单条结果，含评分与分项。"""
    sim = 1 / (1 + distance)
    source_weight = SOURCE_WEIGHT.get(meta.get("source_type", "unknown"), 0.1)

    entry_topics = extract_entry_topics(meta)
    topic_match = 1.0 if (
        detected_topics and set(entry_topics) & set(detected_topics)
    ) else 0.0
    ct_match = 1.0 if (
        detected_ct and meta.get("content_type") == detected_ct
    ) else 0.0

    score = (
        W_VECTOR * sim
        + W_SOURCE * source_weight
        + W_CONTENT_TYPE * ct_match
        + W_TOPIC * topic_match
    )

    is_crisis = meta.get("content_type") == "crisis_rule"
    if is_crisis:
        score = max(score, CRISIS_FLOOR)

    score = min(score, 1.0)

    return {
        "id": doc_id,
        "text": doc,
        "topic": entry_topics,
        "content_type": meta.get("content_type"),
        "source": {
            "name": meta.get("source_name"),
            "type": meta.get("source_type"),
        },
        "score": round(score, 4),
        "score_breakdown": {
            "vector": round(sim, 4),
            "source": round(source_weight, 4),
            "content_type_match": ct_match,
            "topic_match": topic_match,
        },
        "is_crisis": is_crisis,
    }


# ===== 检索接口 =====
@app.post("/v1/knowledge/retrieve")
async def retrieve_knowledge(req: RetrieveRequest):
    start_time = time.perf_counter()

    if model is None or collection is None:
        return JSONResponse(
            status_code=503,
            content={"code": 50300, "msg": "知识库服务未就绪", "data": None},
        )

    if not req.query or not req.query.strip():
        return JSONResponse(
            status_code=400,
            content={"code": 40001, "msg": "查询内容不能为空", "data": None},
        )

    top_k = max(1, req.top_k)

    try:
        clean_query = preprocess_query(req.query)
        detected_topics = detect_topics(clean_query)
        detected_ct = detect_content_type(clean_query)

        logger.info(
            f"Query: {req.query} | topics={detected_topics} | ct={detected_ct} | "
            f"crisis_signal={req.crisis_signal}"
        )

        # 向量化
        query_embedding = model.encode(clean_query).tolist()
        recall_n = min(top_k * RECALL_MULTIPLIER, max(collection.count(), 1))

        # 分级过滤检索
        results = None
        used_filter = "none"

        where = build_where(detected_topics, detected_ct)
        if where is not None:
            try:
                r = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=recall_n,
                    where=where,
                    include=["documents", "distances", "metadatas"],
                )
                if r["documents"] and r["documents"][0]:
                    results = r
                    used_filter = "topic+ct" if detected_ct else "topic"
            except Exception as e:
                logger.warning(f"Filtered query failed: {e}")

        # 降级一：只按 topic 过滤
        if results is None and detected_topics:
            where2 = build_where(detected_topics, None)
            if where2 is not None:
                try:
                    r = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=recall_n,
                        where=where2,
                        include=["documents", "distances", "metadatas"],
                    )
                    if r["documents"] and r["documents"][0]:
                        results = r
                        used_filter = "topic-only"
                except Exception as e:
                    logger.warning(f"Topic-only query failed: {e}")

        # 降级二：全库检索
        if results is None:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=recall_n,
                include=["documents", "distances", "metadatas"],
            )
            used_filter = "full"

        logger.info(f"Used filter: {used_filter}")

        # 重排
        docs = results["documents"][0]
        distances = results["distances"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]

        items = [
            build_result_item(i, d, dist, m, detected_topics, detected_ct)
            for i, d, dist, m in zip(ids, docs, distances, metas)
        ]

        # 危机信号：强制补充所有 crisis_rule
        forced_crisis = []
        if req.crisis_signal:
            try:
                cr = collection.get(
                    where={"content_type": "crisis_rule"},
                    include=["documents", "metadatas"],
                )
                existing_ids = {item["id"] for item in items}
                for cid, cdoc, cmeta in zip(
                    cr["ids"], cr["documents"], cr["metadatas"]
                ):
                    if cid in existing_ids:
                        continue
                    forced_crisis.append({
                        "id": cid,
                        "text": cdoc,
                        "topic": extract_entry_topics(cmeta),
                        "content_type": cmeta.get("content_type"),
                        "source": {
                            "name": cmeta.get("source_name"),
                            "type": cmeta.get("source_type"),
                        },
                        "score": 1.0,
                        "is_crisis": True,
                        "forced": True,
                    })
            except Exception as e:
                logger.error(f"Crisis fetch failed: {e}")

        items.sort(key=lambda x: x["score"], reverse=True)
        items = forced_crisis + items[:top_k]

        elapsed = time.perf_counter() - start_time
        logger.info(f"Retrieved {len(items)} items in {elapsed:.3f}s")

        return {
            "code": 200,
            "msg": "success",
            "detected_topics": detected_topics,
            "detected_content_type": detected_ct,
            "used_filter": used_filter,
            "results": items,
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"code": 50000, "msg": f"服务内部错误: {e}", "data": None},
        )


# ===== 健康检查 =====
@app.get("/health")
def health_check():
    if collection is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "doc_count": 0},
        )
    return {
        "status": "ok",
        "service": "knowledge-retrieval",
        "doc_count": collection.count(),
    }