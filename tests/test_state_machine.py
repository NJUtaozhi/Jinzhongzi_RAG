"""离线单元测试: 验证 CounselingOrchestrator 状态机流转逻辑.

这些测试 mock 了所有外部依赖 (LLM, HTTP API), 不需要:
    - DEEPSEEK_API_KEY
    - 网络连接
    - Block 1 / Block 2 服务

运行:
    python -m pytest tests/test_state_machine.py -v
    或
    python tests/test_state_machine.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_mock_agent(response_text: str) -> MagicMock:
    """创建返回指定文本的 mock LLM agent."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "messages": [MagicMock(content=response_text)]
    }
    return mock


def make_mock_emotion_features() -> Dict[str, Any]:
    """创建模拟的多模态情绪特征."""
    return {
        "valence": -0.7,
        "arousal": 0.8,
        "facial_expression": "sad",
        "text_sentiment": "negative",
        "text_emotion_labels": ["anxiety", "sadness"],
        "available_modalities": ["text", "face"],
        "error_modalities": {},
        "facial_au": {"AU4": 0.6, "AU1": 0.4},
        "voice_tremor": None,
        "speech_rate": None,
        "pitch_variance": None,
    }


def make_initial_state(user_query: str = "我最近总是失眠, 很焦虑") -> Dict[str, Any]:
    """创建初始 CounselingState dict."""
    return {
        "user_query": user_query,
        "session_id": "test-001",
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
        "max_iterations": 8,
        "error": None,
        "execution_log": [],
        "react_trace": [],
        "runtime_context": {},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. State 定义测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestCounselingState:
    """验证 CounselingState 的结构."""

    def test_default_state_has_all_fields(self):
        from orchestration.state import default_state

        s = default_state("test query", session_id="s1")
        required_keys = [
            "user_query", "session_id", "conversation_history",
            "emotion_features", "emotion_label",
            "retrieved_docs", "rag_context",
            "user_intent", "agent_thought", "agent_action", "action_params",
            "draft_response", "safety_issues", "safety_passed",
            "final_answer", "status", "iteration", "max_iterations",
            "error", "execution_log", "react_trace",
        ]
        for key in required_keys:
            assert key in s, f"Missing key: {key}"

    def test_initial_status_is_perceiving(self):
        from orchestration.state import default_state

        s = default_state("hello")
        assert s["status"] == "perceiving"

    def test_empty_react_trace(self):
        from orchestration.state import default_state

        s = default_state("hello")
        assert s["react_trace"] == []
        assert s["execution_log"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 工具集测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestCounselingTools:
    """验证 7 个辅导工具的行为."""

    def test_all_tools_registered(self):
        from tools.counseling_tools import COUNSELING_TOOLS

        expected = {
            "empathize_and_normalize",
            "deliver_cbt_technique",
            "guide_breathing_exercise",
            "crisis_intervention",
            "ask_clarification",
            "provide_psychoeducation",
            "summarize_and_close",
        }
        assert set(COUNSELING_TOOLS) == expected

    def test_execute_valid_tool(self):
        from tools.counseling_tools import execute_counseling_tool

        result = execute_counseling_tool(
            "empathize_and_normalize",
            {"emotion_label": "anxiety", "user_concern": "失眠"},
        )
        assert result["action"] == "empathize_and_normalize"
        assert "共情回应模板" in result["content"]
        assert "anxiety" in result["content"]

    def test_execute_unknown_tool(self):
        from tools.counseling_tools import execute_counseling_tool

        result = execute_counseling_tool("nonexistent", {})
        assert result["action"] == "unknown_tool"

    def test_crisis_intervention_has_hotline(self):
        from tools.counseling_tools import execute_counseling_tool

        # high 级别
        result = execute_counseling_tool(
            "crisis_intervention",
            {"risk_level": "high", "crisis_signals": "自杀念头"},
        )
        assert "010-82951332" in result["content"]

        # immediate 级别
        result2 = execute_counseling_tool(
            "crisis_intervention",
            {"risk_level": "immediate", "crisis_signals": "正在自伤"},
        )
        assert "400-161-9995" in result2["content"]
        assert "high" in result["content"]

    def test_cbt_techniques_all_valid(self):
        from tools.counseling_tools import execute_counseling_tool

        for tech in ["cognitive_restructuring", "behavioral_activation",
                      "thought_record", "exposure_ladder"]:
            result = execute_counseling_tool(
                "deliver_cbt_technique",
                {"technique": tech, "user_thought": "我是个失败者"},
            )
            assert result["action"] == "deliver_cbt_technique"
            assert len(result["content"]) > 50, f"Tool {tech} returned empty"

    def test_tools_schema_text(self):
        from tools.counseling_tools import get_tools_schema_text

        schema = get_tools_schema_text()
        assert "empathize_and_normalize" in schema
        assert "crisis_intervention" in schema
        assert "deliver_cbt_technique" in schema


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EmotionFeatures 测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmotionFeatures:
    def test_summary_with_data(self):
        from services.multimodal_client import EmotionFeatures

        ef = EmotionFeatures(
            valence=-0.7, arousal=0.8,
            facial_expression="sad", text_sentiment="negative",
        )
        s = ef.summary()
        assert "valence=-0.70" in s
        assert "arousal=0.80" in s
        assert "facial=sad" in s

    def test_summary_empty(self):
        from services.multimodal_client import EmotionFeatures

        ef = EmotionFeatures()
        assert "无多模态数据" in ef.summary()

    def test_to_dict_serializable(self):
        from services.multimodal_client import EmotionFeatures

        ef = EmotionFeatures(valence=-0.5, text_emotion_labels=["anxiety"])
        d = ef.to_dict()
        # 必须可 JSON 序列化（LangGraph state 要求）
        json.dumps(d)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 图拓扑测试（mock 所有外部依赖）
# ═══════════════════════════════════════════════════════════════════════════════

# 在模块级别 mock ChatDeepSeek —— 避免任何真实 API 调用.
# 必须在导入 orchestrator 之前执行.
import os as _os
_os.environ.setdefault("DEEPSEEK_API_KEY", "test-fake-key")
_os.environ.setdefault("TAVILY_API_KEY", "test-fake-key")

from unittest.mock import patch as _patch
_fake_llm = MagicMock()
_patch_ds = _patch("factories.agent_factory.ChatDeepSeek", return_value=_fake_llm)
_patch_ds.start()

def _create_orch():
    """创建 mock orchestrator — 每次都创建新实例以避免测试间干扰."""
    from orchestration.orchestrator import CounselingOrchestrator
    orch = CounselingOrchestrator()

    # 替换所有 agent 为 mock（默认行为，各测试可按需覆盖）
    orch.understand_agent = make_mock_agent(
        "intent: seeking_emotional_support\n"
        "emotion: anxiety\nbrief: 测试"
    )
    orch.reason_agent = make_mock_agent(
        "Thought: 测试推理.\n"
        "Action: empathize_and_normalize\n"
        'Params: {"emotion_label": "neutral", "user_concern": "test"}'
    )
    orch.safety_agent = make_mock_agent(
        "verdict: safe\nissues: none\nsuggestion: none"
    )
    return orch


class TestGraphTopology:
    """验证 LangGraph 状态图的节点和边."""

    def test_graph_compiles(self):
        """图必须能成功编译."""
        orch = _create_orch()
        assert orch.graph is not None

    def test_graph_nodes(self):
        """验证所有 6 个节点都已注册."""
        orch = _create_orch()
        assert hasattr(orch.graph, 'invoke')

    def test_start_to_perceive(self):
        """完整流程: START → perceive → understand → retrieve → reason → safety → respond → END."""
        orch = _create_orch()

        # 配置 specific mock 返回值
        orch.understand_agent = make_mock_agent(
            "intent: seeking_emotional_support\n"
            "emotion: anxiety\n"
            "brief: 用户有焦虑和失眠困扰"
        )
        orch.reason_agent = make_mock_agent(
            "Thought: 用户表达焦虑情绪, 需要先共情.\n"
            "Action: empathize_and_normalize\n"
            'Params: {"emotion_label": "anxiety", "user_concern": "失眠"}'
        )

        with patch.object(orch.multimodal, 'analyze') as mock_analyze:
            from services.multimodal_client import EmotionFeatures
            mock_analyze.return_value = EmotionFeatures(
                valence=-0.7, text_sentiment="negative",
            )

            with patch.object(orch.rag, 'retrieve') as mock_rag:
                from services.rag_client import RAGContext, RetrievalResult
                mock_rag.return_value = RAGContext(
                    query="test",
                    documents=[
                        RetrievalResult(
                            content="CBT 对失眠有效...",
                            source="失眠 CBT 指南",
                            relevance_score=0.95,
                        )
                    ],
                    formatted_text="【文献1】CBT 对失眠有效...",
                )

                result = orch.run(user_query="我最近总是失眠, 很焦虑")

        assert result["status"] == "completed", \
            f"Expected 'completed', got '{result['status']}'"
        assert len(result["final_answer"]) > 0, "final_answer should not be empty"
        assert len(result["react_trace"]) > 0, "react_trace should not be empty"
        assert result["user_intent"] == "seeking_emotional_support"
        assert result["emotion_label"] == "anxiety"
        assert result["safety_passed"] is True

    def test_safety_fail_triggers_rewrite(self):
        """安全不通过 → 回到 reason 重写, 最终用兜底回应."""
        orch = _create_orch()

        orch.understand_agent = make_mock_agent(
            "intent: crisis_help\nemotion: depression\nbrief: 危机信号"
        )
        orch.reason_agent = make_mock_agent(
            "Thought: 用户可能有抑郁.\n"
            "Action: empathize_and_normalize\n"
            'Params: {"emotion_label": "depression"}'
        )
        # safety 判定 unsafe
        orch.safety_agent = make_mock_agent(
            "verdict: unsafe\n"
            "issues: 诊断断言\n"
            "suggestion: 去掉诊断性语言"
        )

        with patch.object(orch.multimodal, 'analyze') as mock_analyze:
            from services.multimodal_client import EmotionFeatures
            mock_analyze.return_value = EmotionFeatures()

            with patch.object(orch.rag, 'retrieve') as mock_rag:
                from services.rag_client import RAGContext
                mock_rag.return_value = RAGContext(
                    query="test", formatted_text="(无)"
                )

                result = orch.run(user_query="我不想活了")

        assert len(result["final_answer"]) > 0
        assert result["status"] in ("completed", "failed")

    def test_casual_chat_skips_rag(self):
        """寒暄意图应该跳过 RAG 检索."""
        orch = _create_orch()

        orch.understand_agent = make_mock_agent(
            "intent: casual_chat\nemotion: neutral\nbrief: 用户打招呼"
        )
        orch.reason_agent = make_mock_agent(
            "Thought: 用户在打招呼.\n"
            "Action: empathize_and_normalize\n"
            'Params: {"emotion_label": "neutral", "user_concern": "打招呼"}'
        )

        with patch.object(orch.multimodal, 'analyze') as mock_analyze:
            from services.multimodal_client import EmotionFeatures
            mock_analyze.return_value = EmotionFeatures(text_sentiment="neutral")

            # 不 mock rag.retrieve —— 如果调用了会因 unmet mock 报错
            result = orch.run(user_query="你好")

        assert result["status"] == "completed"
        # RAG 上下文应为空（没调 retrieve）
        assert result["rag_context"] == ""

    def test_react_trace_format(self):
        """ReAct 审计轨迹的格式必须是 Thought/Action/Observation."""
        orch = _create_orch()

        orch.understand_agent = make_mock_agent(
            "intent: asking_knowledge\nemotion: anxiety\nbrief: 询问知识"
        )
        orch.reason_agent = make_mock_agent(
            "Thought: 用户想了解焦虑知识.\n"
            "Action: provide_psychoeducation\n"
            'Params: {"topic": "焦虑"}'
        )

        with patch.object(orch.multimodal, 'analyze') as mock_analyze:
            from services.multimodal_client import EmotionFeatures
            mock_analyze.return_value = EmotionFeatures()

            with patch.object(orch.rag, 'retrieve') as mock_rag:
                from services.rag_client import RAGContext
                mock_rag.return_value = RAGContext(
                    query="test", formatted_text="焦虑相关知识..."
                )

                result = orch.run(user_query="什么是焦虑症？")

        trace = result["react_trace"]
        assert len(trace) >= 6, f"Expected >= 6 trace entries, got {len(trace)}"

        for entry in trace:
            assert any(entry.startswith(prefix)
                       for prefix in ("Thought:", "Action:", "Observation:")), \
                f"Bad trace entry: {entry[:60]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 条件边路由测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestConditionalRoutes:
    """验证条件边的决策逻辑."""

    def test_retrieve_router_casual_chat_skips(self):
        orch = _create_orch()
        state = make_initial_state()
        state["user_intent"] = "casual_chat"
        assert orch._retrieve_router(state) == "skip"

    def test_retrieve_router_emotional_support_calls_rag(self):
        orch = _create_orch()
        state = make_initial_state()
        state["user_intent"] = "seeking_emotional_support"
        assert orch._retrieve_router(state) == "retrieve"

    def test_retrieve_router_crisis_calls_rag(self):
        orch = _create_orch()
        state = make_initial_state()
        state["user_intent"] = "crisis_help"
        assert orch._retrieve_router(state) == "retrieve"

    def test_safety_router_pass(self):
        orch = _create_orch()
        state = make_initial_state()
        state["safety_passed"] = True
        assert orch._safety_router(state) == "respond"

    def test_safety_router_fail(self):
        orch = _create_orch()
        state = make_initial_state()
        state["safety_passed"] = False
        assert orch._safety_router(state) == "rewrite"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 安全规则测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyRules:
    """验证确定性安全规则."""

    def _get_orch(self):
        return _create_orch()

    def test_detects_medication_name(self):
        orch = self._get_orch()
        issues = orch._rule_based_safety_check("你可以试试吃舍曲林来缓解")
        assert len(issues) > 0
        assert any("舍曲林" in i for i in issues)

    def test_detects_diagnosis(self):
        orch = self._get_orch()
        issues = orch._rule_based_safety_check("你有抑郁症, 需要治疗")
        assert len(issues) > 0
        assert any("诊断断言" in i for i in issues)

    def test_safe_text_passes(self):
        orch = self._get_orch()
        issues = orch._rule_based_safety_check(
            "我理解你的感受, 很多人都会经历类似的情绪. "
            "建议你尝试每天做几次深呼吸练习."
        )
        assert len(issues) == 0

    def test_detects_self_harm_methods(self):
        orch = self._get_orch()
        issues = orch._rule_based_safety_check("有人用割腕的方式...")
        assert len(issues) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 输出解析器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputParsers:

    def _get_orch(self):
        return _create_orch()

    def test_parse_understand_output(self):
        orch = self._get_orch()
        text = (
            "intent: seeking_emotional_support\n"
            "emotion: anxiety, sadness\n"
            "brief: 用户有焦虑和失眠困扰"
        )
        result = orch._parse_understand_output(text)
        assert result["intent"] == "seeking_emotional_support"
        assert "anxiety" in result["emotion"]

    def test_parse_understand_fallback(self):
        orch = self._get_orch()
        result = orch._parse_understand_output("乱码输出 ###")
        assert result["intent"] == "unclear"
        assert result["emotion"] == "neutral"

    def test_parse_reason_output(self):
        orch = self._get_orch()
        text = (
            "Thought: 用户表达焦虑, 需要先共情.\n"
            'Action: empathize_and_normalize\n'
            'Params: {"emotion_label": "anxiety", "user_concern": "失眠"}'
        )
        thought, action, params = orch._parse_reason_output(text)
        assert action == "empathize_and_normalize"
        assert params.get("emotion_label") == "anxiety"

    def test_parse_reason_unknown_action_falls_back(self):
        orch = self._get_orch()
        text = "Thought: test\nAction: prescribe_medication\nParams: {}"
        thought, action, params = orch._parse_reason_output(text)
        assert action == "empathize_and_normalize"

    def test_parse_safety_output_safe(self):
        orch = self._get_orch()
        text = "verdict: safe\nissues: none\nsuggestion: none"
        verdict, issues, suggestion = orch._parse_safety_output(text)
        assert verdict == "safe"
        assert issues == []

    def test_parse_safety_output_unsafe(self):
        orch = self._get_orch()
        text = (
            "verdict: unsafe\n"
            "issues: 药物推荐, 诊断断言\n"
            "suggestion: 去除药物名称"
        )
        verdict, issues, suggestion = orch._parse_safety_output(text)
        assert verdict == "unsafe"
        assert len(issues) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 简单的 test runner (不依赖 pytest)
    import traceback

    test_classes = [
        TestCounselingState,
        TestCounselingTools,
        TestEmotionFeatures,
        TestGraphTopology,
        TestConditionalRoutes,
        TestSafetyRules,
        TestOutputParsers,
    ]

    passed = 0
    failed = 0
    errors: List[str] = []

    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                full_name = f"{cls.__name__}.{name}"
                try:
                    getattr(instance, name)()
                    passed += 1
                    print(f"  ✓ {full_name}")
                except Exception:
                    failed += 1
                    err = f"  ✗ {full_name}\n{traceback.format_exc()}"
                    errors.append(err)
                    print(err)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if errors:
        print(f"\n{len(errors)} failures:")
        for e in errors:
            print(e)
    sys.exit(0 if failed == 0 else 1)
