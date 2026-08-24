# 🧠 金种子 — 多模态心理健康分析系统

> 基于 **Agentic RAG** 的多模态心理健康分析系统，支持文本 + 面部表情综合分析。
> 面向大学生群体，通过面部动作单元（AU）分析 + 文本情绪检测，提供心理学知识驱动的个性化建议。
>
> Agent 决策接口 (:8003) 基于 LangChain + LangGraph 串联面部分析 (:8001) + 知识库检索 (:8002) + LLM 推理，
> 前端 (:8501) 提供完整聊天交互界面。

---

## 🚀 快速启动（Docker Compose 一键部署）

### 前提条件

- [Docker Desktop](https://www.docker.com/) 已安装并运行

### 启动步骤

```bash
# 1. 克隆项目
git clone https://github.com/NJUtaozhi/Jinzhongzi_RAG.git
cd Jinzhongzi_RAG

# 2. 配置大模型 API Key
#    编辑 .env 文件，填入你的 DeepSeek API Key
#    DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# 3. 一键启动所有服务
docker compose up -d --build

# 4. 浏览器访问
#    http://localhost:8501  （或配置 Nginx 后 http://服务器IP）
```

### 云服务器部署

```bash
# 服务器已部署: 146.56.204.132
# Nginx 统一入口: http://146.56.204.132
# 各服务端口: 8001 (Vision) / 8002 (Knowledge) / 8003 (Agent) / 8501 (Frontend)
```

---

## 📡 服务列表

| 服务 | 端口 | 说明 | API 文档 |
|------|------|------|----------|
| 🖥️ 前端界面 | `8501` | Streamlit 聊天界面，多轮对话 + 情绪曲线 | — |
| 🤖 Agent 决策 | `8003` | 核心决策，串联 Vision + Knowledge + LLM | `/docs` |
| 📚 知识库检索 | `8002` | 心理学知识语义检索（Chroma + sentence-transformers） | `/docs` |
| 👁️ 面部分析 | `8001` | 面部动作单元（AU）分析（FastAPI + OpenFace） | `/docs` |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────┐
│                  Nginx :80                    │
│            (反向代理 / 统一入口)               │
└──────┬──────────┬──────────┬─────────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Frontend │ │  Agent   │ │ Vision   │
│  :8501   │ │  :8003   │ │  :8001   │
│Streamlit │→│ FastAPI  │ │ OpenFace │
└──────────┘ └──┬───┬───┘ └──────────┘
                │   │
       ┌────────┘   └────────┐
       ▼                     ▼
┌──────────────┐    ┌──────────────┐
│   Vision     │    │  Knowledge   │
│    :8001     │    │    :8002     │
│ 面部 AU 分析  │    │ ChromaDB RAG │
└──────────────┘    └──────────────┘
```

**Agent 流水线：**
```
perceive → understand → retrieve → reason → safety → respond
  (感知)    (意图理解)  (知识检索)  (推理)   (安全校验)  (生成回复)
```

---

## 🏗️ 项目结构

```
金种子吴鑫涛/
├── docker-compose.yml          # Docker Compose 四服务编排
├── .env                        # 环境变量（API Key，已 gitignore）
├── .dockerignore               # Docker 构建排除
├── nginx.conf                  # Nginx 反向代理配置
├── README.md                   # 项目说明
├── API_DEVELOPMENT_GUIDE.md    # API 开发指南
├── api_spec.yaml               # OpenAPI 3.0 接口规范
├── main.py                     # Agent 服务入口 (FastAPI :8003)
├── requirements.txt            # Python 依赖
├── vision_service/             # 成员1：面部分析微服务
│   └── main.py
├── knowledge_service/          # 成员2：知识库检索服务
│   ├── main.py
│   ├── build_kb.py
│   └── knowledge_data.txt      # 46 条心理健康知识
├── orchestration/              # 成员3：LangGraph 编排引擎
│   ├── orchestrator.py         # 6 节点 ReAct 流水线
│   └── state.py                # 流程状态定义
├── services/                   # 成员3：下游服务客户端
│   ├── multimodal_client.py    # 调用 Vision :8001
│   └── rag_client.py           # 调用 Knowledge :8002
├── tools/                      # 成员3：Agent 工具集
│   └── counseling_tools.py     # 7 个心理辅导策略工具
├── factories/                  # 成员3：Agent 工厂
│   └── agent_factory.py
├── tests/                      # 测试
│   ├── test_pure.py            # 28 个纯逻辑测试
│   └── test_state_machine.py   # 33 个 Mock 状态机测试
├── frontend/                   # 成员4：Streamlit 前端
│   └── app.py
├── docker/                     # 成员5：Docker 构建配置
│   ├── vision/Dockerfile
│   ├── knowledge/Dockerfile
│   ├── agent/Dockerfile
│   └── frontend/Dockerfile
└── .github/workflows/
    └── test.yml                # CI：push 自动跑测试
```

---

## 📡 Agent API 端点

| 方法 | 路径 | 格式 | 说明 |
|------|------|------|------|
| GET | `/` | — | 健康检查 |
| GET | `/health` | — | 健康检查（含下游服务地址） |
| POST | `/chat` | JSON | 单轮心理健康辅导对话 |
| POST | `/chat/stream` | SSE | 流式对话 |
| POST | `/v1/agent/analyze` | multipart | 前端兼容路由（text + image） |

### `/chat` 请求示例

```json
POST /chat
{
  "query": "我最近总是失眠, 很焦虑",
  "session_id": "user-001",
  "history": [],
  "max_iterations": 8
}
```

### `/chat` 响应示例

```json
{
  "answer": "我理解你现在的感受...",
  "status": "completed",
  "intent": "seeking_emotional_support",
  "emotion": "anxiety",
  "react_trace": ["Thought: ...", "Action: ...", "Observation: ..."],
  "session_id": "user-001"
}
```

### `/v1/agent/analyze` 响应示例（前端兼容格式）

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "analysis": {
      "image_emotion": { "dominant_emotion": "happy", "au12_r_smile_intensity": 0, "au04_r_brow_lower": 0 },
      "text_sentiment": "positive",
      "text_keywords": []
    },
    "decision": "neutral",
    "reply": "你好呀，很高兴你愿意来这里聊聊...",
    "advice_source": "Agent 综合分析"
  }
}
```

---

## 🧪 运行测试

```bash
python tests/test_pure.py           # 28 个纯逻辑测试
python tests/test_state_machine.py  # 33 个 Mock 状态机测试
```

GitHub Actions CI 会在每次 push 到 `main` 分支时自动运行。

---

## 👥 团队分工

| 成员 | 职责 | 核心交付 |
|------|------|---------|
| **吴鑫涛**（组长） | 系统集成 + Docker + CI/CD + 部署 | docker-compose 一键部署、Nginx 反向代理、GitHub Actions |
| 高翊轩 | 多模态数据处理 | FastAPI + OpenFace 面部分析接口 (:8001) |
| 高天阔 | RAG 检索与知识库 | FastAPI + ChromaDB 语义检索接口 (:8002) |
| 单嵩然 | Agent 决策与工具调用 | LangChain + LangGraph 多模态决策接口 (:8003) |
| 胥庆阳 | 前端界面 | Streamlit 聊天界面 + 情绪曲线可视化 (:8501) |

---

## ⚠️ 注意事项

1. **OpenFace**：Docker 内运行需单独安装 OpenFace CLI 工具。若无 OpenFace，Vision 服务返回降级提示而非崩溃。
2. **API Key**：`.env` 中的 Key 已在 `.gitignore` 中排除，请勿提交到 Git。
3. **首次构建**：`docker compose up -d --build` 首次需下载基础镜像和模型文件，请耐心等待。
4. **端口冲突**：如端口被占用，修改 `docker-compose.yml` 中的 ports 映射。
5. **内存**：服务器建议 ≥ 4GB RAM（Knowledge 的 ChromaDB + sentence-transformers 约需 1.5GB）。

---

> 💡 从上传自拍 + 输入心情，到面部表情识别、知识库检索、AI 综合分析、情绪曲线可视化，全链路真实跑通。
