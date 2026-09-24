"""任务 7.3 多轮记忆与上下文管理测试.

覆盖:
1. 历史截断 (truncate_history) —— 轮次上限 + 单条消息长度上限
2. 情绪摘要纯函数 (parse/render/fallback/prompt)
3. SessionStore —— 内存态 + 文件持久化 + 路径安全
4. 编排器集成 —— 多轮记忆接续 / reason 注入历史 / 长对话截断 / 摘要降级
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestration.memory import (
    DEFAULT_MAX_ROUNDS,
    SessionRecord,
    SessionStore,
    build_emotion_summary_prompt,
    fallback_emotion_summary,
    parse_emotion_summary,
    render_emotion_summary,
    truncate_history,
)


# ── 1. 历史截断 ───────────────────────────────────────────────────────────────

class TestTruncateHistory:
    def test_empty(self):
        assert truncate_history([]) == []
        assert truncate_history(None) == []

    def test_keeps_last_rounds(self):
        # 20 轮 = 40 条消息
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(40)
        ]
        result = truncate_history(history, max_rounds=5)
        assert len(result) == 10  # 5 轮 = 10 条
        assert result[0]["content"] == "m30"
        assert result[-1]["content"] == "m39"

    def test_truncates_long_message(self):
        result = truncate_history(
            [{"role": "user", "content": "x" * 1000}],
            max_rounds=1,
            max_chars=100,
        )
        assert len(result) == 1
        assert result[0]["content"].endswith("…")
        assert len(result[0]["content"]) == 101  # 100 字符 + 省略号

    def test_default_max_rounds(self):
        history = [{"role": "user", "content": f"m{i}"} for i in range(50)]
        assert len(truncate_history(history)) == DEFAULT_MAX_ROUNDS * 2


# ── 2. 情绪摘要纯函数 ─────────────────────────────────────────────────────────

class TestEmotionSummary:
    def test_parse_valid_json(self):
        parsed = parse_emotion_summary(
            '{"summary": "焦虑缓解", "focus": "失眠", "trend": "缓解", "assessment": "PHQ-9 12分"}'
        )
        assert parsed["summary"] == "焦虑缓解"
        assert parsed["focus"] == "失眠"
        assert parsed["trend"] == "缓解"
        assert parsed["assessment"] == "PHQ-9 12分"

    def test_parse_noisy_json(self):
        raw = (
            '好的，以下是摘要：\n'
            '{"summary": "情绪稳定", "focus": "学业压力", "trend": "稳定", "assessment": "null"}\n'
            '希望对你有帮助'
        )
        parsed = parse_emotion_summary(raw)
        assert parsed["summary"] == "情绪稳定"
        assert parsed["focus"] == "学业压力"
        assert parsed["assessment"] == ""  # "null" 被忽略

    def test_parse_garbage_fallback(self):
        parsed = parse_emotion_summary("这不是 JSON")
        assert parsed["summary"] == "这不是 JSON"

    def test_parse_empty(self):
        assert parse_emotion_summary("")["summary"] == ""

    def test_render(self):
        rendered = render_emotion_summary(
            {"summary": "焦虑", "focus": "失眠", "trend": "加重", "assessment": "GAD-7 10分"}
        )
        assert "情绪: 焦虑" in rendered
        assert "关注点: 失眠" in rendered
        assert "趋势: 加重" in rendered

    def test_fallback(self):
        s = fallback_emotion_summary(
            "anxiety",
            {"scale": "PHQ-9", "total_score": 12, "severity": "中度抑郁倾向"},
        )
        assert "情绪: anxiety" in s
        assert "PHQ-9" in s

    def test_build_prompt_includes_previous(self):
        prompt = build_emotion_summary_prompt(
            user_query="我失眠",
            emotion_label="anxiety",
            assessment={"scale": "GAD-7", "total_score": 10, "severity": "中度焦虑倾向"},
            final_answer="我理解你",
            previous_summary="情绪: 焦虑",
        )
        assert "我失眠" in prompt
        assert "情绪: 焦虑" in prompt
        assert "GAD-7" in prompt


# ── 3. SessionStore 内存态 ────────────────────────────────────────────────────

class TestSessionStoreMemory:
    def test_load_empty(self):
        rec = SessionStore().load("s1")
        assert rec.history == []
        assert rec.emotion_summary == ""
        assert rec.turn_count == 0

    def test_save_load_roundtrip(self):
        store = SessionStore()
        store.save("s1", SessionRecord(
            history=[{"role": "user", "content": "hi"}],
            emotion_summary="情绪: 焦虑",
            focus_trajectory=["情绪: 焦虑"],
            turn_count=2,
        ))
        rec = store.load("s1")
        assert rec.history[0]["content"] == "hi"
        assert rec.emotion_summary == "情绪: 焦虑"
        assert rec.turn_count == 2

    def test_clear(self):
        store = SessionStore()
        store.save("s1", SessionRecord(emotion_summary="x"))
        store.clear("s1")
        assert store.load("s1").emotion_summary == ""

    def test_load_returns_copy(self):
        store = SessionStore()
        store.save("s1", SessionRecord(history=[{"role": "user", "content": "a"}]))
        rec = store.load("s1")
        rec.history[0]["content"] = "mutated"
        assert store.load("s1").history[0]["content"] == "a"


# ── 4. SessionStore 文件持久化 ────────────────────────────────────────────────

class TestSessionStoreFile:
    def test_persist_across_instances(self, tmp_path):
        SessionStore(storage_dir=str(tmp_path)).save("s1", SessionRecord(
            history=[{"role": "user", "content": "hi"}],
            emotion_summary="情绪: 焦虑",
        ))
        rec = SessionStore(storage_dir=str(tmp_path)).load("s1")
        assert rec.history[0]["content"] == "hi"
        assert rec.emotion_summary == "情绪: 焦虑"

    def test_session_id_no_path_traversal(self, tmp_path):
        store = SessionStore(storage_dir=str(tmp_path))
        store.save("../../evil/../etc", SessionRecord(emotion_summary="x"))
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".json"

    def test_clear_removes_file(self, tmp_path):
        store = SessionStore(storage_dir=str(tmp_path))
        store.save("s1", SessionRecord(emotion_summary="x"))
        store.clear("s1")
        assert list(tmp_path.iterdir()) == []


# ── 5. 编排器集成（多轮记忆） ─────────────────────────────────────────────────

# 模块级 mock ChatDeepSeek，避免真实 API key / 网络
os.environ.setdefault("DEEPSEEK_API_KEY", "test-fake-key")
_fake_llm = MagicMock()
_patch_ds = patch("factories.agent_factory.ChatDeepSeek", return_value=_fake_llm)
_patch_ds.start()


def _make_agent(response_text):
    mock = MagicMock()
    mock.invoke.return_value = {"messages": [MagicMock(content=response_text)]}
    return mock


class _RecordingAgent:
    """记录每次 invoke 的 prompt，并按序返回 canned 响应。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def invoke(self, payload):
        prompt = payload["messages"][-1]["content"]
        self.prompts.append(prompt)
        idx = min(len(self.prompts) - 1, len(self._responses) - 1)
        return {"messages": [MagicMock(content=self._responses[idx])]}


class TestOrchestratorMemory:
    def _make_orch(self):
        from orchestration.orchestrator import CounselingOrchestrator

        orch = CounselingOrchestrator(session_store=SessionStore())
        orch.understand_agent = _make_agent(
            "intent: seeking_emotional_support\nemotion: anxiety\nbrief: 焦虑"
        )
        orch.reason_agent = _make_agent(
            "Thought: 用户焦虑.\nAction: empathize_and_normalize\n"
            'Params: {"emotion_label": "anxiety", "user_concern": "失眠"}'
        )
        orch.safety_agent = _make_agent("verdict: safe\nissues: none\nsuggestion: none")
        orch.summarize_agent = _make_agent(
            '{"summary": "焦虑略有缓解", "focus": "失眠", "trend": "缓解", "assessment": "无"}'
        )
        return orch

    def _mock_deps(self, orch):
        from services.multimodal_client import EmotionFeatures
        from services.rag_client import RAGContext

        orch.multimodal.analyze = MagicMock(return_value=EmotionFeatures())
        orch.rag.retrieve = MagicMock(
            return_value=RAGContext(query="test", formatted_text="(无)")
        )
        return orch

    def test_multiturn_remembers_history_and_summary(self):
        orch = self._mock_deps(self._make_orch())

        r1 = orch.run(user_query="我最近总是失眠", session_id="user-001")
        assert r1["status"] == "completed"
        assert len(r1["conversation_history"]) == 2  # user + assistant
        assert r1["emotion_summary"] != ""

        # 第二轮不传 history，仅凭 session_id 接续
        r2 = orch.run(user_query="昨晚又没睡着", session_id="user-001")
        assert r2["status"] == "completed"
        assert len(r2["conversation_history"]) == 4
        assert r2["conversation_history"][0]["content"] == "我最近总是失眠"
        assert r2["emotion_summary"] != ""
        assert len(r2["focus_trajectory"]) == 2

    def test_reason_node_receives_history(self):
        """验收核心：第二轮 reason 提示词应包含上一轮用户内容。"""
        orch = self._mock_deps(self._make_orch())
        reason_agent = _RecordingAgent([
            "Thought: t.\nAction: empathize_and_normalize\n"
            'Params: {"emotion_label": "anxiety"}',
        ])
        orch.reason_agent = reason_agent

        orch.run(user_query="我最近总是失眠", session_id="user-001")
        reason_agent.prompts.clear()

        orch.run(user_query="还是睡不着", session_id="user-001")

        joined = "\n".join(reason_agent.prompts)
        assert "我最近总是失眠" in joined

    def test_history_truncates_over_long_conversation(self):
        orch = self._mock_deps(self._make_orch())

        result = None
        for i in range(15):  # 15 轮
            result = orch.run(user_query=f"第{i}轮的问题", session_id="user-001")

        assert result["status"] == "completed"
        # 15 轮后历史被截断到 max_rounds（10 轮 = 20 条）
        assert len(result["conversation_history"]) <= DEFAULT_MAX_ROUNDS * 2
        assert len(result["conversation_history"]) == DEFAULT_MAX_ROUNDS * 2

    def test_summarize_failure_falls_back(self):
        orch = self._mock_deps(self._make_orch())
        orch.summarize_agent = MagicMock()
        orch.summarize_agent.invoke.side_effect = RuntimeError("boom")

        r = orch.run(user_query="我失眠", session_id="s-fallback")
        assert r["status"] == "completed"
        assert "情绪: anxiety" in r["emotion_summary"]

    def test_explicit_history_overrides_store(self):
        """请求显式携带 history 时以请求为准（兼容旧客户端回传）。"""
        orch = self._mock_deps(self._make_orch())

        orch.run(user_query="第一轮", session_id="user-001")

        # 传入全新的 history（无第一轮），应覆盖存储
        r = orch.run(
            user_query="新会话",
            session_id="user-001",
            conversation_history=[{"role": "user", "content": "你好"}],
        )
        assert r["status"] == "completed"
        assert r["conversation_history"][0]["content"] == "你好"
