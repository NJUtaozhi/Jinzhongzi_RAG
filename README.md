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
#    编辑 .env 文件，填入你的 API Key 和 Base URL
#    （默认使用智谱 GLM-4，也可替换为 DeepSeek 等）

# 3. 一键启动所有服务
docker-compose up --build

# 4. 浏览器访问
#    http://localhost:8501
```

---

## 📡 服务列表

| 服务 | 端口 | 说明 | Swagger 文档 |
|------|------|------|-------------|
| 🖥️ 前端界面 | `8501` | Streamlit 聊天界面，支持多轮对话 + 情绪曲线 | — |
| 🤖 Agent 决策 | `8003` | 核心决策接口，串联 Vision + Knowledge + LLM | `http://localhost:8003/docs` |
| 📚 知识库检索 | `8002` | 心理学知识语义检索（Chroma + sentence-transformers） | `http://localhost:8002/docs` |
| 👁️ 面部分析 | `8001` | 面部动作单元（AU）分析（FastAPI + OpenFace） | `http://localhost:8001/docs` |

---

## 🏗️ 项目结构

```
金种子/
├── docker-compose.yml          # Docker Compose 一键编排
├── .env                        # 环境变量（API Key 等）
├── README.md                   # 项目说明
├── API_DEVELOPMENT_GUIDE.md    # API 开发指南
├── api_spec.yaml               # OpenAPI 3.0 接口规范（SSOT）
├── main.py                     # FastAPI 入口 (Agent API :8003)
├── requirements.txt            # Python 依赖
├── vision_service/             # 成员1：面部分析微服务
│   └── main.py
├── knowledge_service/          # 成员2：知识库检索服务
│   ├── main.py
│   ├── build_kb.py
│   └── knowledge_data.txt
├── factories/                  # 成员3：Agent 工厂
│   ├── agent_factory.py        # create_agent 工厂
│   └── __main__.py             # CLI: python -m factories
├── orchestration/              # 成员3：LangGraph 编排器
│   ├── orchestrator.py         # perceive→understand→retrieve→reason→safety→respond
│   ├── state.py                # 流程状态定义
│   └── __main__.py             # CLI: python -m orchestration
├── services/                   # 成员3：下游服务客户端
│   ├── multimodal_client.py    # Block 1 — 多模态情绪特征 (调用 :8001)
│   └── rag_client.py           # Block 2 — RAG 知识检索 (调用 :8002)
├── tools/                      # 成员3：工具集
│   ├── calculator.py           # 安全表达式计算
│   ├── tavily_search.py        # Tavily 网络搜索
│   └── counseling_tools.py     # 心理辅导工具集 (7个策略工具)
├── tests/                      # 测试
│   ├── test_pure.py            # 纯逻辑测试
│   └── test_state_machine.py   # Mock 状态机测试
├── frontend/                   # 成员4：Streamlit 前端
│   └── app.py
└── docker/                     # 成员5：Docker 构建配置
    ├── vision/
    ├── knowledge/
    ├── agent/
    └── frontend/
```

---

## 🔗 服务依赖关系

```
Frontend (:8501)
  └── Agent (:8003)
        ├── Vision (:8001) → OpenFace
        └── Knowledge (:8002) → Chroma DB
```

- **同一电脑**：容器间通过服务名互访（`http://vision:8001`、`http://knowledge:8002`、`http://agent:8003`）
- **不同电脑**：修改环境变量中的 IP 为局域网地址（如 `http://192.168.x.x:8001`）

---

## 🔧 本地开发（不使用 Docker）

如果不想用 Docker，也可以逐个启动服务：

```bash
# 终端1 — Vision Service
cd vision_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 终端2 — Knowledge Service
cd knowledge_service
python build_kb.py                          # 先构建知识库
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# 终端3 — Agent Service
uvicorn main:app --host 0.0.0.0 --port 8003 --reload

# 终端4 — Frontend
cd frontend
streamlit run app.py
```

---

## 📡 Agent API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/health` | 健康检查 (含下游服务地址) |
| POST | `/chat` | 单轮心理健康辅导对话 |
| GET | `/docs` | OpenAPI 文档 |

### 请求示例

```json
POST /chat
{
  "query": "我最近总是失眠, 很焦虑",
  "session_id": "user-001",
  "history": [],
  "max_iterations": 8
}
```

```json
{
  "answer": "我理解你现在的感受...",
  "status": "completed",
  "intent": "seeking_emotional_support",
  "emotion": "anxiety",
  "react_trace": ["Thought: ...", "Action: ...", "Observation: ..."]
}
```

---

## 🖥️ CLI 入口

```bash
# 简单 Agent 测试
python -m factories --query "1 + 2 等于多少？"

# 编排器单轮测试
python -m orchestration --query "如何缓解压力" --verbose
```

---

## 🧪 运行测试

```bash
python tests/test_pure.py           # 28 个纯逻辑测试
python tests/test_state_machine.py  # 33 个 Mock 状态机测试
```

---

## 📖 API 文档

- 接口规范与错误码：见 [API_DEVELOPMENT_GUIDE.md](./API_DEVELOPMENT_GUIDE.md)
- OpenAPI Schema 详情：见 `api_spec.yaml`

---

## 👥 团队分工

| 成员 | 职责 | 核心交付 |
|------|------|---------|
| **吴鑫涛**（组长） | 后端架构 + Docker 编排 + CI/CD | docker-compose 一键部署、GitHub 仓库管理 |
| 高翊轩 | 多模态数据处理 | FastAPI + OpenFace 面部分析接口 |
| 高天阔 | RAG 检索与知识库 | FastAPI + Chroma 语义检索接口 |
| 单嵩然 | Agent 决策与工具调用 | LangChain + LangGraph 多模态决策接口 |
| 胥庆阳 | 前端界面 | Streamlit 聊天界面 + 情绪曲线可视化 |

---

## ⚠️ 注意事项

1. **OpenFace 简化方案**：OpenFace 打包进 Docker 较复杂，建议在宿主机运行 Vision 服务，Docker 仅用于 Knowledge / Agent / Frontend。
2. **LLM API Key**：`.env` 中的 Key 请勿提交到 Git（已在 `.gitignore` 中排除）。
3. **首次构建**：Docker 首次构建需下载基础镜像，请耐心等待；后续构建会利用缓存加速。
4. **端口冲突**：如端口被占用，修改 `docker-compose.yml` 中的 ports 映射（如 `"18001:8001"`）。

---

> 💡 从上传自拍 + 输入心情，到面部表情识别、知识库检索、AI 综合分析、情绪曲线可视化，全链路真实跑通。
