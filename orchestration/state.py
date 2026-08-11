"""心理健康辅导 Agent 状态定义.

从原 WorkflowState（任务执行模型）重构为 CounselingState（对话感知模型）.

核心变化:
    - 去掉 plan / pending_steps / active_step / next_role（任务流水线遗产）
    - 新增 emotion_features / rag_context / safety_passed（多模态 + RAG + 安全门禁）
    - 新增 conversation_history 支持多轮对话
    - 保留 execution_log / react_trace（审计追踪, 框架无关）
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from services.multimodal_client import EmotionFeatures

# ── 状态枚举 ─────────────────────────────────────────────────────────────────

CounselingStatus = Literal[
    "perceiving",       # 调用 Block 1 多模态 API
    "understanding",    # LLM 意图识别 + 情绪分类
    "retrieving",       # 调用 Block 2 RAG 检索
    "reasoning",        # ReAct 式 LLM 决策 (选择辅导工具)
    "validating",       # 安全门禁检查
    "responding",       # 输出最终回应
    "completed",        # 本轮对话完成
    "failed",           # 出错终止
]

UserIntent = Literal[
    "seeking_emotional_support",   # 寻求情感支持
    "asking_knowledge",            # 询问心理健康知识
    "crisis_help",                 # 危机求助
    "casual_chat",                 # 纯寒暄
    "unclear",                     # 意图不明
]

# ── State ────────────────────────────────────────────────────────────────────


class CounselingState(TypedDict, total=False):
    """心理健康辅导 Agent 的完整状态.

    State 是图节点间的唯一数据通道——每个节点读取所需字段,
    写入产出字段, 返回部分更新 dict.
    """

    # ── 不可变会话标识 ──
    user_query: str          # 当前轮用户输入（保留原字段名, 兼容）
    session_id: str          # 会话 ID, 多轮对话追踪

    # ── 对话历史 ──
    conversation_history: List[Dict[str, str]]  # [{"role":"user/assistant","content":...}]

    # ── 多模态感知结果（Block 1）──
    #    注意: emotion_features 在 state 中以 dict 形式流转
    #    (LangGraph state 序列化要求), 消费节点按需反序列化.
    emotion_features: Dict[str, Any]    # EmotionFeatures.to_dict()
    emotion_label: str                  # "anxiety"|"depression"|"neutral"|...

    # ── RAG 检索结果（Block 2）──
    retrieved_docs: List[Dict[str, Any]]   # 原始检索结果列表
    rag_context: str                       # 拼接后的文本, 供 LLM 消费

    # ── Agent 决策中间产物 ──
    user_intent: str           # UserIntent 值
    agent_thought: str         # 当前 Thought (ReAct 推理链)
    agent_action: str          # 当前 Action (选中的辅导工具名)
    action_params: Dict[str, Any]   # 工具参数
    draft_response: str        # reason 节点生成的草稿回应

    # ── 安全门禁 ──
    safety_issues: List[str]   # ["检测到药物推荐", "包含诊断性断言", ...]
    safety_passed: bool

    # ── 最终输出 ──
    final_answer: str          # 安全校验后的最终回应

    # ── 生命周期 + 审计 ──
    status: CounselingStatus
    iteration: int
    max_iterations: int
    error: Optional[str]
    execution_log: List[str]
    react_trace: List[str]     # 保留原 ReAct 审计轨迹字段名


# ── 工厂函数 ─────────────────────────────────────────────────────────────────


def default_state(
    user_query: str,
    session_id: str = "",
    max_iterations: int = 8,
) -> CounselingState:
    """创建初始 CounselingState.

    Args:
        user_query: 用户输入文本.
        session_id: 会话标识, 用于多轮对话追踪.
        max_iterations: reason 节点 ReAct 循环最大迭代次数.
    """
    return {
        "user_query": user_query,
        "session_id": session_id,
        "conversation_history": [],
        "emotion_features": {},
        "emotion_label": "",
        "retrieved_docs": [],
        "rag_context": "",
        "user_intent": "",
        "agent_thought": "",
        "agent_action": "",
        "action_params": {},
        "draft_response": "",
        "safety_issues": [],
        "safety_passed": False,
        "final_answer": "",
        "status": "perceiving",
        "iteration": 0,
        "max_iterations": max_iterations,
        "error": None,
        "execution_log": [],
        "react_trace": [],
    }
