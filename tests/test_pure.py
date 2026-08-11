"""纯逻辑测试 —— 不依赖 langgraph, 无需 jsonpatch, 无需 API key.

运行:
    python tests/test_pure.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. State 定义
# ═══════════════════════════════════════════════════════════════════════════════

def test_default_state_fields():
    from orchestration.state import default_state
    s = default_state("test", session_id="s1")
    required = [
        "user_query", "session_id", "conversation_history",
        "emotion_features", "emotion_label",
        "retrieved_docs", "rag_context",
        "user_intent", "agent_thought", "agent_action", "action_params",
        "draft_response", "safety_issues", "safety_passed",
        "final_answer", "status", "iteration", "max_iterations",
        "error", "execution_log", "react_trace",
    ]
    for k in required:
        assert k in s, f"Missing: {k}"
    assert s["status"] == "perceiving"
    assert s["react_trace"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 辅导工具集
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_7_tools_registered():
    from tools.counseling_tools import COUNSELING_TOOLS
    expected = {
        "empathize_and_normalize", "deliver_cbt_technique",
        "guide_breathing_exercise", "crisis_intervention",
        "ask_clarification", "provide_psychoeducation",
        "summarize_and_close",
    }
    assert set(COUNSELING_TOOLS) == expected


def test_execute_valid_tool():
    from tools.counseling_tools import execute_counseling_tool
    r = execute_counseling_tool(
        "empathize_and_normalize",
        {"emotion_label": "anxiety", "user_concern": "失眠"},
    )
    assert r["action"] == "empathize_and_normalize"
    assert "anxiety" in r["content"]


def test_unknown_tool_graceful():
    from tools.counseling_tools import execute_counseling_tool
    r = execute_counseling_tool("nonexistent", {})
    assert r["action"] == "unknown_tool"


def test_crisis_intervention_has_hotline():
    from tools.counseling_tools import execute_counseling_tool
    # high 级别有 010 热线
    r = execute_counseling_tool(
        "crisis_intervention",
        {"risk_level": "high", "crisis_signals": "自杀念头"},
    )
    assert "010-82951332" in r["content"]

    # immediate 级别有 400 热线
    r2 = execute_counseling_tool(
        "crisis_intervention",
        {"risk_level": "immediate", "crisis_signals": "正在自伤"},
    )
    assert "400-161-9995" in r2["content"]


def test_all_cbt_techniques():
    from tools.counseling_tools import execute_counseling_tool
    for tech in ["cognitive_restructuring", "behavioral_activation",
                  "thought_record", "exposure_ladder"]:
        r = execute_counseling_tool(
            "deliver_cbt_technique",
            {"technique": tech, "user_thought": "我很失败"},
        )
        assert len(r["content"]) > 50, f"Empty: {tech}"


def test_all_breathing_exercises():
    from tools.counseling_tools import execute_counseling_tool
    for ex in ["box_breathing", "478_breathing", "body_scan", "five_senses"]:
        r = execute_counseling_tool(
            "guide_breathing_exercise",
            {"exercise_type": ex, "duration_minutes": 3},
        )
        assert len(r["content"]) > 30, f"Empty: {ex}"


def test_tools_schema_text():
    from tools.counseling_tools import get_tools_schema_text
    s = get_tools_schema_text()
    assert len(s) > 200
    for name in ["empathize_and_normalize", "crisis_intervention"]:
        assert name in s


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EmotionFeatures
# ═══════════════════════════════════════════════════════════════════════════════

def test_emotion_features_summary():
    from services.multimodal_client import EmotionFeatures
    ef = EmotionFeatures(valence=-0.7, arousal=0.8,
                         facial_expression="sad", text_sentiment="negative")
    s = ef.summary()
    assert "valence=-0.70" in s
    assert "facial=sad" in s


def test_emotion_features_empty():
    from services.multimodal_client import EmotionFeatures
    assert "无多模态数据" in EmotionFeatures().summary()


def test_emotion_features_json_serializable():
    from services.multimodal_client import EmotionFeatures
    d = EmotionFeatures(valence=-0.5,
                        text_emotion_labels=["anxiety"]).to_dict()
    json.dumps(d)  # 不能抛异常


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RAGContext
# ═══════════════════════════════════════════════════════════════════════════════

def test_rag_context_summary():
    from services.rag_client import RAGContext, RetrievalResult
    ctx = RAGContext(
        query="如何缓解焦虑",
        documents=[RetrievalResult(
            content="深呼吸有助于缓解焦虑",
            source="焦虑干预手册",
            relevance_score=0.9,
        )],
    )
    s = ctx.summary()
    assert "如何缓解焦虑" in s
    assert "1" in s  # docs=1


def test_format_empty_docs():
    from services.rag_client import RAGClient
    text = RAGClient._format_documents([])
    assert "未检索到" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 安全规则 (确定性, 不依赖 LLM)
# ═══════════════════════════════════════════════════════════════════════════════

# 为了避免 import langgraph 链, 直接复制规则常量和函数
_MEDICATION_BLACKLIST = [
    "舍曲林", "氟西汀", "帕罗西汀", "氟伏沙明", "西酞普兰", "艾司西酞普兰",
    "文拉法辛", "度洛西汀", "米氮平", "曲唑酮", "安非他酮",
    "阿普唑仑", "劳拉西泮", "氯硝西泮", "地西泮", "奥沙西泮",
    "唑吡坦", "佐匹克隆", "艾司佐匹克隆",
    "奥氮平", "喹硫平", "利培酮", "阿立哌唑",
    "碳酸锂", "丙戊酸钠", "拉莫三嗪",
    "赛乐特", "百忧解", "左洛复", "怡诺思", "欣百达", "瑞美隆",
]

_DIAGNOSIS_PATTERNS = [
    "你有抑郁症", "你有焦虑症", "你得了", "你患有", "你的病",
    "你这是典型", "你肯定是", "确诊", "临床诊断",
]

_METHOD_KEYWORDS = ["割腕", "跳楼", "上吊", "烧炭"]


def _rule_based_safety_check(text: str) -> List[str]:
    issues: List[str] = []
    for med in _MEDICATION_BLACKLIST:
        if med in text:
            issues.append(f"药物推荐: 提及 '{med}'")
    for pattern in _DIAGNOSIS_PATTERNS:
        if pattern in text:
            issues.append(f"诊断断言: 包含 '{pattern}'")
    for kw in _METHOD_KEYWORDS:
        if kw in text:
            issues.append(f"危险内容: 包含 '{kw}'")
    return issues


def test_safety_detects_medication():
    issues = _rule_based_safety_check("你可以试试吃舍曲林")
    assert len(issues) >= 1
    assert any("舍曲林" in i for i in issues)


def test_safety_detects_diagnosis():
    issues = _rule_based_safety_check("你有抑郁症, 需要治疗")
    assert len(issues) >= 1
    assert any("诊断断言" in i for i in issues)


def test_safety_detects_self_harm():
    issues = _rule_based_safety_check("割腕是一种方式")
    assert len(issues) >= 1
    assert any("割腕" in i for i in issues)


def test_safety_safe_text_passes():
    issues = _rule_based_safety_check(
        "我理解你的感受, 建议每天做几次深呼吸练习, 保持规律作息."
    )
    assert len(issues) == 0


def test_safety_multiple_issues():
    issues = _rule_based_safety_check(
        "你有焦虑症, 建议服用舍曲林, 临床诊断表明有效"
    )
    assert len(issues) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 输出解析器 (从 orchestrator 复制纯函数版本)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_understand_output(text: str) -> Dict[str, str]:
    result = {"intent": "unclear", "emotion": "neutral", "brief": ""}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("intent:"):
            result["intent"] = line.split(":", 1)[1].strip()
        elif line.startswith("emotion:"):
            result["emotion"] = line.split(":", 1)[1].strip()
        elif line.startswith("brief:"):
            result["brief"] = line.split(":", 1)[1].strip()
    return result


def _parse_reason_output(text: str) -> tuple:
    import re
    from tools.counseling_tools import COUNSELING_TOOLS

    thought = ""
    action = "empathize_and_normalize"
    params: Dict[str, Any] = {}

    m = re.search(r"Thought:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        thought = m.group(1).strip()

    m = re.search(r"Action:\s*(\S+)", text, re.IGNORECASE)
    if m:
        a = m.group(1).strip()
        if a in COUNSELING_TOOLS:
            action = a

    m = re.search(r"Params:\s*(\{.+?\})", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            params = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    if not params:
        params = {"emotion_label": "neutral", "user_concern": "(unknown)"}

    return thought, action, params


def _parse_safety_output(text: str) -> tuple:
    verdict = "safe"
    issues: List[str] = []
    suggestion = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("verdict:"):
            v = line.split(":", 1)[1].strip().lower()
            verdict = "safe" if v == "safe" else "unsafe"  # 精确匹配, "unsafe" 含子串 "safe"
        elif line.startswith("issues:"):
            iss = line.split(":", 1)[1].strip()
            if iss and iss.lower() != "none":
                issues = [i.strip() for i in iss.split(",") if i.strip()]
        elif line.startswith("suggestion:"):
            suggestion = line.split(":", 1)[1].strip()
    return verdict, issues, suggestion


def test_parse_understand_normal():
    r = _parse_understand_output(
        "intent: seeking_emotional_support\n"
        "emotion: anxiety, sadness\n"
        "brief: 用户失眠焦虑"
    )
    assert r["intent"] == "seeking_emotional_support"
    assert "anxiety" in r["emotion"]


def test_parse_understand_fallback():
    r = _parse_understand_output("乱码 ### 无格式")
    assert r["intent"] == "unclear"
    assert r["emotion"] == "neutral"


def test_parse_reason_with_params():
    t, a, p = _parse_reason_output(
        "Thought: 用户焦虑需要共情.\n"
        'Action: empathize_and_normalize\n'
        'Params: {"emotion_label": "anxiety", "user_concern": "失眠"}'
    )
    assert a == "empathize_and_normalize"
    assert p["emotion_label"] == "anxiety"


def test_parse_reason_unknown_action_fallback():
    t, a, p = _parse_reason_output(
        "Thought: test\nAction: prescribe_drugs\nParams: {}"
    )
    assert a == "empathize_and_normalize"  # 回退


def test_parse_safety_safe():
    v, issues, _ = _parse_safety_output(
        "verdict: safe\nissues: none\nsuggestion: none"
    )
    assert v == "safe"
    assert issues == []


def test_parse_safety_unsafe():
    v, issues, sug = _parse_safety_output(
        "verdict: unsafe\n"
        "issues: 药物推荐, 诊断断言\n"
        "suggestion: 去除药物名称和诊断"
    )
    assert v == "unsafe"
    assert len(issues) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 条件路由逻辑
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_INTENTS = [
    "seeking_emotional_support", "asking_knowledge",
    "crisis_help", "casual_chat", "unclear",
]


def test_retrieve_router_logic():
    """casual_chat 跳过 RAG, 其他意图走 retrieve."""
    # 模拟 _retrieve_router 逻辑
    for intent in _VALID_INTENTS:
        if intent == "casual_chat":
            assert _route_retrieve(intent) == "skip"
        else:
            assert _route_retrieve(intent) == "retrieve", \
                f"Intent '{intent}' should route to retrieve"


def _route_retrieve(intent: str) -> str:
    return "skip" if intent == "casual_chat" else "retrieve"


def test_safety_router_logic():
    assert _route_safety(True) == "respond"
    assert _route_safety(False) == "rewrite"


def _route_safety(passed: bool) -> str:
    return "respond" if passed else "rewrite"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 服务客户端构造 (不发起 HTTP)
# ═══════════════════════════════════════════════════════════════════════════════

def test_multimodal_client_creates():
    from services.multimodal_client import MultimodalClient
    c = MultimodalClient(base_url="localhost")
    assert c._base_url == "localhost"


def test_rag_client_creates():
    from services.rag_client import RAGClient
    c = RAGClient(base_url="localhost")
    assert c._base_url == "localhost"


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in dict(globals()).items()
             if k.startswith("test_")]

    passed = 0
    failed = 0
    errors = []

    for fn in tests:
        name = fn.__name__
        try:
            fn()
            passed += 1
            print(f"  \033[32m✓\033[0m {name}")
        except Exception:
            failed += 1
            err = f"  \033[31m✗\033[0m {name}\n{traceback.format_exc()}"
            errors.append(err)
            print(err)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    sys.exit(0 if failed == 0 else 1)
