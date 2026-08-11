"""Orchestration layer for multi-modal mental health counseling agent.

注意: CounselingOrchestrator 的导入是延迟的——它依赖 langgraph,
需要先安装 jsonpatch 等依赖.
"""


def get_orchestrator():
    """延迟导入 CounselingOrchestrator (需要 langgraph)."""
    from .orchestrator import CounselingOrchestrator
    return CounselingOrchestrator


# 纯 Python 模块——无外部依赖, 可以安全在顶层导入
from .state import CounselingState, default_state  # noqa: E402

__all__ = ["CounselingState", "default_state", "get_orchestrator"]
