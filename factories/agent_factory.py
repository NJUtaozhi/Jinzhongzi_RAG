"""
Agent 工厂模块。

负责创建配置好的 LangChain 原生 Agent 实例。
"""
import os
from typing import Any

from langchain.agents import create_agent as lc_create_agent
from langchain_deepseek import ChatDeepSeek

from tools import get_all_tools

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
    enable_tools: bool = True,
    strict_tools: bool = True,
) -> Any:
    """
    工厂函数：创建一个配置好的 LangChain 原生 Agent 实例。

    该函数负责：
    1. 从环境变量获取 DeepSeek API 密钥
    2. 初始化 ChatDeepSeek 模型
    3. 获取所有注册的工具
    4. 返回装配好的原生 Agent（Runnable）实例

    调用者无需关心 LLM 和工具的初始化细节，只需调用此工厂函数即可得到可用的 Agent。

    Args:
        system_message: 系统提示信息，传递给 Agent，默认为
            "You are a helpful React agentic assistant."。
        model: 使用的 DeepSeek 模型名称，默认为 "deepseek-chat"。
        temperature: 模型温度参数，控制回答的随机性，默认为 0.0（确定性）。
        enable_tools: 是否启用工具集，默认为 True。
        strict_tools: 启用工具时，若工具初始化失败是否抛错，默认为 True。

    Returns:
        一个完全配置好的原生 Agent（Runnable）实例。

    Raises:
        ValueError: 如果 DEEPSEEK_API_KEY 环境变量未设置。
        ValueError: 如果 TAVILY_API_KEY 环境变量未设置且 strict_tools=True。
    """
    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable not set. "
            "Please set it before creating an agent."
        )

    # 创建 DeepSeek 模型实例
    llm = ChatDeepSeek(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    tools = None
    if enable_tools:
        try:
            tools = get_all_tools()
        except ValueError:
            if strict_tools:
                raise
            tools = []

    agent = lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
        debug=False,
    )

    return agent


__all__ = ["create_agent"]
