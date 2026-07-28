# Jinzhongzi RAG — 心理健康辅导 Agent

基于 LangChain + LangGraph 的多模态心理健康辅导系统。

Agent 决策接口 (:8003) 串联面部分析 (:8001) + 知识库检索 (:8002) + LLM 推理，
前端 (:8501) 通过 `/chat` 调用本服务。

## 目录结构

```text
main.py                      # FastAPI 入口 (Agent API :8003)
requirements.txt             # Python 依赖
.env.example                 # 环境变量模板

factories/
  __init__.py
  __main__.py                # CLI: python -m factories
  agent_factory.py           # create_agent 工厂

orchestration/
  __init__.py
  __main__.py                # CLI: python -m orchestration
  orchestrator.py            # LangGraph 编排器 (perceive→understand→retrieve→reason→safety→respond)
  state.py                   # 流程状态定义

services/
  __init__.py
  multimodal_client.py       # Block 1 — 多模态情绪特征 (调用 :8001)
  rag_client.py              # Block 2 — RAG 知识检索 (调用 :8002)

tools/
  __init__.py                # 工具注册中心
  calculator.py              # 安全表达式计算
  tavily_search.py           # Tavily 网络搜索
  counseling_tools.py        # 心理辅导工具集 (7个策略工具)

tests/
  test_pure.py               # 纯逻辑测试 (无需 LLM/网络)
  test_state_machine.py      # Mock 状态机测试
```

## 服务拓扑

```
前端 (:8501) ──→ Agent API (:8003) ──→ DeepSeek LLM
                      │
                      ├──→ 面部分析 API (:8001)
                      └──→ 知识库 API (:8002)
```

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 编辑 .env, 填入 DEEPSEEK_API_KEY 和 TAVILY_API_KEY
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 开发模式 (热重载)
uvicorn main:app --host 0.0.0.0 --port 8003 --reload

# 生产模式
python main.py
```

### 4. 调用

```bash
curl -X POST http://localhost:8003/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我最近总是失眠, 很焦虑"}'
```

API 文档: `http://localhost:8003/docs`

## CLI 入口

```bash
# 简单 Agent 测试
python -m factories --query "1 + 2 等于多少？"

# 编排器单轮测试
python -m orchestration --query "如何缓解压力" --verbose
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/health` | 健康检查 (含下游服务地址) |
| POST | `/chat` | 单轮心理健康辅导对话 |
| GET | `/docs` | OpenAPI 文档 |

## 请求示例

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

## 运行测试

```bash
python tests/test_pure.py           # 28 个纯逻辑测试
python tests/test_state_machine.py  # 33 个 Mock 状态机测试
```
