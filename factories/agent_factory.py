"""
Agent 工厂模块。

负责创建配置好的 LangChain 原生 Agent 实例。
"""
import os
from typing import Any

from langchain.agents import create_agent as lc_create_agent
from langchain_deepseek import ChatDeepSeek

# 可选: 加载 .env (不强制依赖 python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def create_agent(
    system_message: str = "You are a helpful React agentic assistant.",
    model: str = "deepseek-chat",
    temperature: float = 0.0,
) -> Any:
    """
    工厂函数：创建一个配置好的 LangChain 原生 Agent 实例。

    该函数负责：
    1. 从环境变量获取 DeepSeek API 密钥
    2. 初始化 ChatDeepSeek 模型
    3. 返回装配好的原生 Agent（Runnable）实例

    调用者无需关心 LLM 初始化细节，只需调用此工厂函数即可得到可用的 Agent。

    Args:
        system_message: 系统提示信息，默认为 "You are a helpful React agentic assistant."。
        model: 使用的 DeepSeek 模型名称，默认为 "deepseek-chat"。
        temperature: 模型温度参数，控制回答的随机性，默认为 0.0（确定性）。

    Returns:
        一个完全配置好的原生 Agent（Runnable）实例。

    Raises:
        ValueError: 如果 DEEPSEEK_API_KEY 环境变量未设置。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable not set. "
            "Please set it before creating an agent."
        )

    llm = ChatDeepSeek(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    agent = lc_create_agent(
        model=llm,
        tools=None,
        system_prompt=system_message,
        debug=False,
    )

    return agent


__all__ = ["create_agent"]
