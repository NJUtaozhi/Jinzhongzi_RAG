# LangChain 原生 Agent 项目化重构指南

## 目标

这次重构聚焦两个问题：
1. 清理根目录裸露脚本，避免入口分散。
2. 让工具实现回归 `tools` 包内，去除反向依赖。

## 重构后结构

```text
factories/
  __init__.py
  __main__.py            # 统一 CLI 入口
  agent_factory.py       # 返回 LangChain 原生 create_agent 实例

tools/
  __init__.py            # 工具注册中心
  calculator.py          # 计算工具实现
  tavily_search.py       # 搜索工具实现
```

## 删除的根目录脚本

- `main.py`
- `tooltestagent.py`
- `tools1.py`
- `searchtool.py`

这些能力已分别迁移到包内模块，不再需要根目录脚本承载。

## 核心职责

### tools/__init__.py
- 统一注册工具。
- 从包内模块导入 `calculator` 与 `TavilySearchResults`。

### factories/agent_factory.py
- 负责创建模型、Prompt 与 Tool Calling Agent。
- 直接返回 LangChain 原生 `create_agent` 生成的 Agent Runnable。

### factories/__main__.py
- 提供命令行入口。
- 用法：`python -m factories --query "..."`

## 使用方式

### Python API

```python
from factories import create_agent

agent = create_agent()
response = agent.invoke({"messages": [{"role": "user", "content": "1 + 2 等于多少？"}]})
print(response["messages"][-1].content)
```

### CLI

```bash
python -m factories --query "Python 最新版本是什么？"
```

## 环境变量

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 后续扩展建议

1. 新增工具时，优先在 `tools/` 下新增模块，再在 `tools/__init__.py` 注册。
2. 若继续项目化，可增加 `tests/` 并为原生 Agent 执行链与工具模块补单元测试。
3. 如需发布，可追加 `pyproject.toml` 并定义 console script。
