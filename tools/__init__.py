"""工具注册中心."""

import os
from typing import List

from .counseling_tools import (
    COUNSELING_TOOLS,
    execute_counseling_tool,
    get_tools_schema_text,
)

# 可选: 加载 .env (不强制依赖 python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_all_tools() -> "List":
    """返回所有注册的 LangChain 工具列表.

    langchain 依赖延迟导入——避免在未安装 jsonpatch 等依赖时
    整个 tools 包不可用.

    Returns:
        包含 calculator（计算工具）和 TavilySearchResults（网络搜索工具）的列表.

    Raises:
        ValueError: 如果 TAVILY_API_KEY 环境变量未设置.
    """
    from langchain_core.tools import BaseTool
    from .calculator import calculator
    from .tavily_search import TavilySearchResults

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Please set it before initializing tools."
        )

    tools: List[BaseTool] = [
        calculator,
        TavilySearchResults(api_key=tavily_api_key, count=4),
    ]
    return tools


__all__ = [
    "get_all_tools",
    "COUNSELING_TOOLS",
    "execute_counseling_tool",
    "get_tools_schema_text",
]
