import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer

# 获取当前文件所在目录的绝对路径
BASE_DIR = Path(__file__).parent.absolute()
KB_CHROMA_PATH = BASE_DIR / "kb_chroma_db"
KNOWLEDGE_FILE = BASE_DIR / "knowledge_data.txt"

print("Loading vector model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

chroma_client = chromadb.PersistentClient(path=str(KB_CHROMA_PATH))

# 删除旧 collection（如果存在），确保重新创建
try:
    chroma_client.delete_collection("mental_health_knowledge")
    print("Existing collection deleted.")
except Exception as e:
    print(f"Collection not found or already deleted: {e}")

# 创建新的 collection
collection = chroma_client.create_collection(
    name="mental_health_knowledge",
    metadata={"description": "心理健康知识库"},
    embedding_function=None
)

# 读取知识语料（使用绝对路径）
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Read {len(lines)} knowledge entries.")

if len(lines) == 0:
    print("Error: knowledge_data.txt is empty or not found!")
    exit(1)

documents = []
ids = []
embeddings = []

for i, text in enumerate(lines):
    doc_id = f"doc_{i+1}"
    embedding = model.encode(text).tolist()
    documents.append(text)
    ids.append(doc_id)
    embeddings.append(embedding)

collection.add(
    documents=documents,
    ids=ids,
    embeddings=embeddings
)

print(f"Knowledge base built successfully! Total {len(documents)} entries.")