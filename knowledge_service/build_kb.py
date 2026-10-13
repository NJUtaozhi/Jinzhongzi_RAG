import os
import json
import logging
from pathlib import Path
from collections import Counter

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer


# ===== 路径与常量 =====
BASE_DIR = Path(__file__).parent.absolute()
KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
KNOWLEDGE_FILE = BASE_DIR / "knowledge_data.jsonl"
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

SOURCE_TYPE_VOCAB = [
    "clinical_guideline", "original_instrument", "peer_reviewed_paper",
    "government_website", "institutional_website", "encyclopedia", "unknown",
]

# ===== 日志 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_kb")


def load_jsonl(path: Path):
    """逐行读取 JSONL。返回 (entries, parse_errors)。"""
    entries = []
    errors = []
    if not path.exists():
        logger.error(f"Knowledge file not found: {path}")
        return entries, errors

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append((line_no, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append((line_no, str(e)))
    return entries, errors


def validate_entry(entry: dict):
    """校验必填字段与词表合法性。返回问题列表，空列表表示合法。"""
    problems = []

    for field in ["id", "topic", "content_type", "text", "source"]:
        if field not in entry:
            problems.append(f"missing field: {field}")
    if problems:
        return problems

    if not isinstance(entry["id"], str) or not entry["id"].strip():
        problems.append("id must be non-empty string")

    if not isinstance(entry["topic"], list) or not entry["topic"]:
        problems.append("topic must be non-empty array")
    else:
        for t in entry["topic"]:
            if t not in TOPIC_VOCAB:
                problems.append(f"unknown topic: {t}")

    if entry["content_type"] not in CONTENT_TYPE_VOCAB:
        problems.append(f"unknown content_type: {entry['content_type']}")

    if not isinstance(entry["text"], str) or not entry["text"].strip():
        problems.append("text must be non-empty string")

    src = entry.get("source")
    if not isinstance(src, dict):
        problems.append("source must be object")
    else:
        if not src.get("name"):
            problems.append("source.name missing")
        if src.get("type") not in SOURCE_TYPE_VOCAB:
            problems.append(f"unknown source.type: {src.get('type')}")

    return problems


def build_metadata(entry: dict) -> dict:
    """把 JSON 字段转换为 ChromaDB 支持的标量元数据。"""
    meta = {
        "content_type": entry["content_type"],
        "source_name": entry["source"]["name"],
        "source_type": entry["source"]["type"],
    }

    # topic 数组转布尔字段
    topic_set = set(entry["topic"])
    for t in TOPIC_VOCAB:
        meta[f"topic_{t}"] = t in topic_set

    # 可选字段：risk_level
    if "risk_level" in entry:
        meta["risk_level"] = entry["risk_level"]

    return meta


def main():
    logger.info("Loading vector model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    logger.info(f"Reading knowledge file: {KNOWLEDGE_FILE}")
    entries, parse_errors = load_jsonl(KNOWLEDGE_FILE)

    for line_no, err in parse_errors:
        logger.warning(f"Line {line_no} JSON parse error: {err}")

    valid_entries = []
    invalid_count = 0
    for line_no, entry in entries:
        problems = validate_entry(entry)
        if problems:
            invalid_count += 1
            eid = entry.get("id", "?") if isinstance(entry, dict) else "?"
            logger.warning(f"Line {line_no} id={eid} invalid: {problems}")
        else:
            valid_entries.append(entry)

    logger.info(
        f"Parsed {len(entries)} lines: {len(valid_entries)} valid, "
        f"{invalid_count} invalid, {len(parse_errors)} parse errors."
    )

    if not valid_entries:
        logger.error("No valid entries, abort.")
        return

    # 打开或创建集合
    client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Mental health knowledge base"},
        embedding_function=None,
    )

    # 去重：相同 id 保留最后一条
    dedup = {}
    for entry in valid_entries:
        if entry["id"] in dedup:
            logger.warning(f"Duplicate id, keeping last: {entry['id']}")
        dedup[entry["id"]] = entry
    valid_entries = list(dedup.values())

    # 向量化并准备批量写入
    ids, documents, embeddings, metadatas = [], [], [], []
    for entry in valid_entries:
        ids.append(entry["id"])
        documents.append(entry["text"])
        embeddings.append(model.encode(entry["text"]).tolist())
        metadatas.append(build_metadata(entry))

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(ids)} entries.")

    # 删除 JSONL 中已不存在的条目
    existing_ids = set(collection.get(include=[])["ids"])
    current_ids = set(ids)
    stale_ids = list(existing_ids - current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)
        logger.info(f"Deleted {len(stale_ids)} stale entries: {stale_ids}")

    # 统计
    ct_counter = Counter(e["content_type"] for e in valid_entries)
    st_counter = Counter(e["source"]["type"] for e in valid_entries)
    tp_counter = Counter()
    for e in valid_entries:
        for t in e["topic"]:
            tp_counter[t] += 1

    logger.info(f"Collection total: {collection.count()}")
    logger.info(f"content_type distribution: {dict(ct_counter)}")
    logger.info(f"source.type distribution: {dict(st_counter)}")
    logger.info(f"topic distribution: {dict(tp_counter)}")
    logger.info("Build complete.")


if __name__ == "__main__":
    main()