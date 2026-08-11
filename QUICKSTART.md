# 快速开始

## 0. 环境配置

确保项目根目录存在 `.env`：

```bash
DEEPSEEK_API_KEY=xxx
TAVILY_API_KEY=xxx
```

## 1. 最小用法

```python
from factories import create_agent

agent = create_agent()
response = agent.invoke({"messages": [{"role": "user", "content": "1 + 2 等于多少？"}]})
print(response["messages"][-1].content)
```

## 2. 命令行运行（推荐）

```bash
python -m factories --query "Python 最新版本是什么？"
```

## 2.1 多智能体编排运行

```bash
python -m orchestration --query "请给出一个新项目的技术调研步骤"
```

## 3. 自定义 Agent

```python
from factories import create_agent

agent = create_agent(
    system_message="You are an expert Python assistant.",
  temperature=0.1,
)

response = agent.invoke({"messages": [{"role": "user", "content": "如何优化 Python 代码性能？"}]})
print(response["messages"][-1].content)
```

## 4. 项目结构

```text
factories/
  __init__.py
  __main__.py
  agent_factory.py

orchestration/
  __init__.py
  __main__.py
  orchestrator.py
  state.py

tools/
  __init__.py
  calculator.py
  tavily_search.py
```

## 5. 常用命令

```bash
pip install langchain langchain-deepseek langchain-core python-dotenv requests
python -m factories --query "2+3*4"
```

更多说明见 `REFACTOR_GUIDE.md`。
