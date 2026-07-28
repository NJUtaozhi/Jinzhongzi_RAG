"""心理健康辅导内部工具集.

这些工具供 reason 节点在 ReAct 循环中调用, 是 prompt 模板 + 参数的封装,
**不是**外部 API——感知和检索的外部服务调用已在 perceive / retrieve 节点完成.

设计原则:
    - 每个工具接收结构化参数, 返回一个可供 LLM 进一步加工的半成品文本.
    - 工具只做策略选择和内容组织, 不做外部 I/O.
    - 所有工具返回 dict, 包含 ``action`` 和 ``content`` 两个字段,
      便于 reason 节点统一处理.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 工具注册
# ═══════════════════════════════════════════════════════════════════════════════

COUNSELING_TOOLS: Dict[str, Dict[str, Any]] = {
    "empathize_and_normalize": {
        "name": "empathize_and_normalize",
        "description": (
            "共情与正常化——先认可用户的情绪体验是合理的, 降低孤独感和自我批判. "
            "适用于用户表达痛苦、焦虑、压力等情绪, 需要先建立信任感时."
        ),
        "parameters": {
            "emotion_label": "用户当前情绪标签 (如 anxiety, depression, anger)",
            "user_concern": "用户核心困扰的简要概括",
        },
    },
    "deliver_cbt_technique": {
        "name": "deliver_cbt_technique",
        "description": (
            "提供 CBT（认知行为疗法）技巧——引导用户识别自动思维、检验认知扭曲、"
            "进行行为实验或记录思维记录表. 适用于用户有明确的负面思维模式."
        ),
        "parameters": {
            "technique": "具体 CBT 技巧: cognitive_restructuring | behavioral_activation | thought_record | exposure_ladder",
            "user_thought": "用户表达的具体负面思维",
            "rag_reference": "RAG 检索到的相关 CBT 方案片段 (可选)",
        },
    },
    "guide_breathing_exercise": {
        "name": "guide_breathing_exercise",
        "description": (
            "引导呼吸/正念练习——提供简短的呼吸训练或正念指导, "
            "适用于用户急性焦虑发作或需要即时放松技巧时."
        ),
        "parameters": {
            "exercise_type": "练习类型: box_breathing | 478_breathing | body_scan | five_senses",
            "duration_minutes": "建议时长 (分钟)",
        },
    },
    "crisis_intervention": {
        "name": "crisis_intervention",
        "description": (
            "危机干预——当检测到用户可能有自伤/自杀风险时, 提供紧急资源、"
            "安全计划, 并强烈建议寻求专业帮助. 这是最高优先级的工具."
        ),
        "parameters": {
            "risk_level": "风险等级: low | moderate | high | immediate",
            "crisis_signals": "检测到的危机信号描述",
        },
    },
    "ask_clarification": {
        "name": "ask_clarification",
        "description": (
            "追问澄清——当用户表述模糊或需要更多信息才能给出有效建议时, "
            "提出 1-2 个温和的追问问题."
        ),
        "parameters": {
            "missing_info": "需要澄清的具体信息",
            "suggested_questions": "建议的追问问题列表",
        },
    },
    "provide_psychoeducation": {
        "name": "provide_psychoeducation",
        "description": (
            "心理健康教育——用通俗语言解释心理学概念/机制, "
            "结合 RAG 检索知识, 帮助用户理解自己的状况."
        ),
        "parameters": {
            "topic": "教育主题",
            "rag_knowledge": "RAG 检索到的相关知识片段",
        },
    },
    "summarize_and_close": {
        "name": "summarize_and_close",
        "description": (
            "总结与收尾——在对话自然结束时总结本次交流要点, "
            "给出后续建议, 温和结束对话."
        ),
        "parameters": {
            "key_points": "本次对话中的关键发现和建议列表",
            "follow_up_suggestion": "后续行动建议",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 工具实现（纯文本生成, 无外部 I/O）
# ═══════════════════════════════════════════════════════════════════════════════

def execute_counseling_tool(
    tool_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """执行指定的辅导工具, 返回 action + content.

    Args:
        tool_name: 工具名称 (COUNSELING_TOOLS 的 key).
        params: 工具参数.

    Returns:
        {"action": tool_name, "content": "...", "params": {...}}
    """
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {
            "action": "unknown_tool",
            "content": f"未知工具 '{tool_name}', 可用: {list(COUNSELING_TOOLS)}",
            "params": params,
        }

    content = handler(params)
    return {"action": tool_name, "content": content, "params": params}


def get_tools_schema_text() -> str:
    """生成供 LLM system prompt 使用的工具描述文本."""
    lines = ["可用工具列表:\n"]
    for tool in COUNSELING_TOOLS.values():
        lines.append(f"- **{tool['name']}**: {tool['description']}")
        for param, desc in tool["parameters"].items():
            lines.append(f"    - {param}: {desc}")
        lines.append("")
    return "\n".join(lines)


# ── handlers ─────────────────────────────────────────────────────────────────

def _empathize_and_normalize(params: Dict[str, Any]) -> str:
    emotion = params.get("emotion_label", "困扰")
    concern = params.get("user_concern", "")
    return (
        f"[共情回应模板]\n"
        f"情绪类型: {emotion}\n"
        f"用户困扰: {concern}\n"
        f"策略: 先承认情绪的合理性 (\"很多人都会有类似的感受...\"), "
        f"再表达理解和支持, 避免急于给建议."
    )


def _deliver_cbt_technique(params: Dict[str, Any]) -> str:
    technique = params.get("technique", "cognitive_restructuring")
    user_thought = params.get("user_thought", "")
    rag_ref = params.get("rag_reference", "")

    technique_guides = {
        "cognitive_restructuring": (
            "认知重构步骤: 1) 识别自动思维 2) 寻找支持/反对证据 "
            "3) 生成替代性解释 4) 评估情绪变化"
        ),
        "behavioral_activation": (
            "行为激活: 1) 识别回避行为 2) 制定分级活动计划 "
            "3) 从最简单的活动开始 4) 记录愉悦感和成就感"
        ),
        "thought_record": (
            "思维记录表: 1) 情境描述 2) 自动思维 3) 情绪及强度 "
            "4) 支持证据 5) 反对证据 6) 平衡思维 7) 重新评估情绪"
        ),
        "exposure_ladder": (
            "暴露阶梯: 1) 确定恐惧情境 2) 按焦虑程度排序 "
            "3) 从最低等级开始逐级暴露 4) 每次记录焦虑评分"
        ),
    }
    guide = technique_guides.get(technique, technique_guides["cognitive_restructuring"])

    parts = [
        f"[CBT 技巧: {technique}]",
        f"用户思维: {user_thought}",
        f"指导内容:\n{guide}",
    ]
    if rag_ref:
        parts.append(f"\n参考资料:\n{rag_ref}")
    return "\n".join(parts)


def _guide_breathing_exercise(params: Dict[str, Any]) -> str:
    exercise_type = params.get("exercise_type", "box_breathing")
    duration = params.get("duration_minutes", 3)

    exercises = {
        "box_breathing": (
            "四方呼吸法:\n"
            "1. 吸气 4 秒 → 2. 屏息 4 秒 → 3. 呼气 4 秒 → 4. 屏息 4 秒\n"
            "重复循环, 持续 {duration} 分钟"
        ),
        "478_breathing": (
            "4-7-8 呼吸法:\n"
            "1. 用鼻子吸气 4 秒 → 2. 屏息 7 秒 → 3. 用嘴缓慢呼气 8 秒\n"
            "重复 4-5 轮"
        ),
        "body_scan": (
            "身体扫描:\n"
            "从脚趾开始, 逐步将注意力移到脚掌→脚踝→小腿→...→头顶, "
            "在每个部位停留 10-15 秒, 觉察身体感受而不评判."
        ),
        "five_senses": (
            "五感觉察:\n"
            "依次注意: 5 样看到的东西 → 4 样触摸到的东西 → "
            "3 样听到的声音 → 2 样闻到的气味 → 1 样尝到的味道"
        ),
    }
    guide = exercises.get(exercise_type, exercises["box_breathing"])
    return f"[正念/呼吸练习]\n类型: {exercise_type}\n时长: {duration} 分钟\n\n{guide.format(duration=duration)}"


def _crisis_intervention(params: Dict[str, Any]) -> str:
    risk_level = params.get("risk_level", "moderate")
    signals = params.get("crisis_signals", "")

    resources = {
        "low": "建议: 保持日常作息, 联系信任的朋友或家人, 关注情绪变化.",
        "moderate": (
            "建议: 预约心理咨询师进行专业评估, 避免独自应对, "
            "可拨打心理援助热线: 010-82951332 (24小时)"
        ),
        "high": (
            "⚠️ 强烈建议: 立即联系心理咨询师或精神科医生, "
            "不要独自承受. 心理援助热线: 010-82951332 (24小时)\n"
            "如情况紧急, 请直接拨打 120 或前往最近医院急诊科."
        ),
        "immediate": (
            "🚨 紧急: 您当前的情况需要立即获得专业干预.\n"
            "请立即拨打: 120 (急救) 或 110 (报警)\n"
            "全国24小时心理危机干预热线: 400-161-9995\n"
            "北京心理危机研究与干预中心: 010-82951332\n"
            "请不要独自一人, 立即联系身边可以信任的人."
        ),
    }

    return (
        f"[危机干预 - 风险等级: {risk_level}]\n"
        f"检测信号: {signals}\n\n"
        f"{resources.get(risk_level, resources['moderate'])}\n\n"
        f"重要提醒: 我是 AI 助手, 不能替代专业心理健康服务. "
        f"如果您正处于危机中, 请务必寻求专业帮助."
    )


def _ask_clarification(params: Dict[str, Any]) -> str:
    missing = params.get("missing_info", "")
    questions = params.get("suggested_questions", [])
    q_text = "\n".join(f"- {q}" for q in questions) if questions else "- (请根据上下文自行生成追问)"
    return (
        f"[追问澄清]\n"
        f"需要了解: {missing}\n"
        f"建议追问:\n{q_text}"
    )


def _provide_psychoeducation(params: Dict[str, Any]) -> str:
    topic = params.get("topic", "")
    rag_knowledge = params.get("rag_knowledge", "")
    parts = [f"[心理健康教育: {topic}]"]
    if rag_knowledge:
        parts.append(f"知识来源:\n{rag_knowledge}")
    parts.append(
        "指导原则: 用通俗比喻解释, 避免学术术语, "
        "强调这是常见现象, 许多人都有类似经历."
    )
    return "\n".join(parts)


def _summarize_and_close(params: Dict[str, Any]) -> str:
    key_points = params.get("key_points", [])
    follow_up = params.get("follow_up_suggestion", "")
    points_text = "\n".join(f"- {p}" for p in key_points) if key_points else "(待总结)"
    return (
        f"[对话总结]\n"
        f"关键发现:\n{points_text}\n\n"
        f"后续建议: {follow_up}\n\n"
        f"结束语: 肯定用户的努力和开放态度, 提醒可以随时回来继续交流."
    )


_TOOL_HANDLERS = {
    "empathize_and_normalize": _empathize_and_normalize,
    "deliver_cbt_technique": _deliver_cbt_technique,
    "guide_breathing_exercise": _guide_breathing_exercise,
    "crisis_intervention": _crisis_intervention,
    "ask_clarification": _ask_clarification,
    "provide_psychoeducation": _provide_psychoeducation,
    "summarize_and_close": _summarize_and_close,
}
