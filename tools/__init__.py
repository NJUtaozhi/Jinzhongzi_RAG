"""工具注册中心 — 心理辅导策略工具（纯文本生成，无外部 I/O）.

这些工具供 reason 节点在 ReAct 循环中调用, 是 prompt 模板 + 参数的封装,
不是外部 API——感知和检索的外部服务调用已在 perceive / retrieve 节点完成.
"""

from .counseling_tools import (
    COUNSELING_TOOLS,
    execute_counseling_tool,
    get_tools_schema_text,
)

__all__ = [
    "COUNSELING_TOOLS",
    "execute_counseling_tool",
    "get_tools_schema_text",
]
