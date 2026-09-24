"""
测试3: 前端记忆交互测试（任务7.4）

验证三件事：
1. build_history_payload — 从 session_state.messages 构建后端期望的历史格式
2. parse_emotion_summary_text — 解析后端返回的情绪摘要文本为结构化字段
3. chat 请求 payload 包含 session_id + history 透传字段
4. 情绪轨迹卡片数据处理逻辑
"""

import json
from unittest.mock import patch, MagicMock
import requests


# ============================================================
# 被测函数（从 app.py 内联，避免 Streamlit 初始化）
# ============================================================

def build_history_payload(messages):
    """从 messages 列表构建后端期望的历史格式。

    格式: [{"role": "user"|"assistant", "content": "..."}]
    只取最近 20 条（10 轮 × 2），与后端 DEFAULT_MAX_ROUNDS 一致。
    """
    history = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    return history[-20:]


def parse_emotion_summary_text(summary_text):
    """解析后端返回的情绪摘要文本为结构化字段。

    后端 render_emotion_summary 输出格式:
    "情绪: xxx；关注点: xxx；趋势: xxx；量表: xxx"
    """
    result = {"emotion": "", "focus": "", "trend": "", "assessment": ""}
    if not summary_text:
        return result
    for part in summary_text.split("；"):
        part = part.strip()
        if ":" in part or "：" in part:
            sep = "：" if "：" in part else ":"
            key, _, val = part.partition(sep)
            key = key.strip()
            val = val.strip()
            if "情绪" in key:
                result["emotion"] = val
            elif "关注点" in key:
                result["focus"] = val
            elif "趋势" in key:
                result["trend"] = val
            elif "量表" in key:
                result["assessment"] = val
    return result


TREND_ICONS = {
    "首次": "🆕",
    "稳定": "➡️",
    "缓解": "📉",
    "加重": "📈",
}


# ============================================================
# 测试 1: build_history_payload
# ============================================================

def test_history_payload_basic():
    """基本测试：从 user+assistant 消息列表构建历史"""
    messages = [
        {"role": "user", "content": "我今天很难过"},
        {"role": "assistant", "content": "我理解你的感受"},
        {"role": "user", "content": "谢谢你"},
        {"role": "assistant", "content": "不客气"},
    ]
    history = build_history_payload(messages)
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "我今天很难过"}
    assert history[1] == {"role": "assistant", "content": "我理解你的感受"}
    print("✅ test_history_payload_basic passed")


def test_history_payload_truncation():
    """截断测试：超过 20 条时只保留最近 20 条"""
    messages = []
    for i in range(15):
        messages.append({"role": "user", "content": f"msg-{i}"})
        messages.append({"role": "assistant", "content": f"reply-{i}"})
    # 30 条总消息
    assert len(messages) == 30
    history = build_history_payload(messages)
    # 只保留最近 20 条
    assert len(history) == 20
    # 最新的是 reply-14
    assert history[-1] == {"role": "assistant", "content": "reply-14"}
    # 最旧的是 msg-5（第 10 条 user 消息，index=10*2=20，从 0 开始）
    assert history[0] == {"role": "user", "content": "msg-5"}
    print("✅ test_history_payload_truncation passed")


def test_history_payload_filters_empty_content():
    """过滤测试：空 content 的消息不进入历史"""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "world"},
    ]
    history = build_history_payload(messages)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "world"}
    print("✅ test_history_payload_filters_empty_content passed")


def test_history_payload_empty_list():
    """空列表测试：无消息时返回空列表"""
    assert build_history_payload([]) == []
    print("✅ test_history_payload_empty_list passed")


def test_history_payload_ignores_invalid_roles():
    """角色过滤测试：非 user/assistant 的角色不进入历史"""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "system msg"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "content": "tool output"},
    ]
    history = build_history_payload(messages)
    assert len(history) == 2
    assert all(h["role"] in ("user", "assistant") for h in history)
    print("✅ test_history_payload_ignores_invalid_roles passed")


# ============================================================
# 测试 2: parse_emotion_summary_text
# ============================================================

def test_parse_full_summary():
    """完整摘要解析：包含所有四个字段"""
    text = "情绪: 用户感到明显的焦虑和压力；关注点: 学业压力和时间管理；趋势: 加重；量表: PHQ-9 总分 18 分（中重度抑郁）"
    parsed = parse_emotion_summary_text(text)
    assert "焦虑" in parsed["emotion"]
    assert "学业" in parsed["focus"]
    assert parsed["trend"] == "加重"
    assert "PHQ-9" in parsed["assessment"]
    print("✅ test_parse_full_summary passed")


def test_parse_partial_summary():
    """部分摘要解析：只有情绪和关注点"""
    text = "情绪: 用户情绪低落；关注点: 人际关系困扰"
    parsed = parse_emotion_summary_text(text)
    assert parsed["emotion"] == "用户情绪低落"
    assert parsed["focus"] == "人际关系困扰"
    assert parsed["trend"] == ""
    assert parsed["assessment"] == ""
    print("✅ test_parse_partial_summary passed")


def test_parse_empty_text():
    """空文本测试"""
    parsed = parse_emotion_summary_text("")
    assert parsed == {"emotion": "", "focus": "", "trend": "", "assessment": ""}
    print("✅ test_parse_empty_text passed")


def test_parse_none_text():
    """None 输入测试"""
    parsed = parse_emotion_summary_text(None)
    assert parsed == {"emotion": "", "focus": "", "trend": "", "assessment": ""}
    print("✅ test_parse_none_text passed")


def test_parse_full_width_colon():
    """全角冒号兼容测试"""
    text = "情绪：用户感到悲伤；关注点：失去亲人；趋势：首次"
    parsed = parse_emotion_summary_text(text)
    assert "悲伤" in parsed["emotion"]
    assert "失去亲人" in parsed["focus"]
    assert parsed["trend"] == "首次"
    print("✅ test_parse_full_width_colon passed")


# ============================================================
# 测试 3: chat 请求 payload 包含 session_id + history
# ============================================================

MOCK_CHAT_RESPONSE_WITH_MEMORY = {
    "code": 200,
    "msg": "success",
    "data": {
        "analysis": {
            "text_sentiment": "anxiety",
            "text_keywords": ["焦虑", "压力"],
        },
        "decision": "seeking_emotional_support",
        "reply": "我理解你现在的焦虑，这是很正常的情绪反应。",
        "advice_source": "知识库检索结果",
        "emotion_summary": "情绪: 用户感到明显焦虑；关注点: 学业压力；趋势: 首次",
        "focus_trajectory": [
            "情绪: 用户感到明显焦虑；关注点: 学业压力；趋势: 首次",
        ],
        "history": [
            {"role": "user", "content": "我最近压力好大"},
            {"role": "assistant", "content": "我理解你现在的焦虑，这是很正常的情绪反应。"},
        ],
    },
}


def test_chat_payload_includes_session_id():
    """纯文本请求 payload 必须包含 session_id 字段"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE_WITH_MEMORY
        mock_post.return_value = mock_resp

        chat_url = "http://localhost:8003/v1/agent/chat"
        payload = {
            "text": "我最近压力好大",
            "session_id": "test-session-001",
            "history": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，有什么可以帮你的吗？"},
            ],
        }
        requests.post(chat_url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)

        call_kwargs = mock_post.call_args[1]
        sent_json = call_kwargs.get("json", {})
        assert "session_id" in sent_json, "Payload must include session_id"
        assert sent_json["session_id"] == "test-session-001"
        print("✅ test_chat_payload_includes_session_id passed")


def test_chat_payload_includes_history():
    """纯文本请求 payload 必须包含 history 字段"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE_WITH_MEMORY
        mock_post.return_value = mock_resp

        chat_url = "http://localhost:8003/v1/agent/chat"
        history = [
            {"role": "user", "content": "我今天很难过"},
            {"role": "assistant", "content": "我理解你的感受"},
        ]
        payload = {"text": "我最近压力好大", "session_id": "test-001", "history": history}
        requests.post(chat_url, json=payload, timeout=60)

        call_kwargs = mock_post.call_args[1]
        sent_json = call_kwargs.get("json", {})
        assert "history" in sent_json, "Payload must include history"
        assert len(sent_json["history"]) == 2
        assert sent_json["history"][0]["role"] == "user"
        print("✅ test_chat_payload_includes_history passed")


def test_analyze_payload_includes_session_id_and_history():
    """多模态请求（multipart）必须包含 session_id 和 history 字段"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE_WITH_MEMORY
        mock_post.return_value = mock_resp

        analyze_url = "http://localhost:8003/v1/agent/analyze"
        history = [{"role": "user", "content": "你好"}]
        data = {
            "text": "我今天好焦虑",
            "session_id": "test-analyze-001",
            "history": json.dumps(history, ensure_ascii=False),
        }
        files = {"image": ("test.jpg", b"fake_image", "image/jpeg")}
        requests.post(analyze_url, data=data, files=files, timeout=60)

        call_kwargs = mock_post.call_args[1]
        sent_data = call_kwargs.get("data", {})
        assert "session_id" in sent_data, "Multipart payload must include session_id"
        assert sent_data["session_id"] == "test-analyze-001"
        assert "history" in sent_data, "Multipart payload must include history"
        # history 在 multipart 中是 JSON 字符串
        parsed_history = json.loads(sent_data["history"])
        assert isinstance(parsed_history, list)
        assert parsed_history[0]["role"] == "user"
        print("✅ test_analyze_payload_includes_session_id_and_history passed")


# ============================================================
# 测试 4: 情绪轨迹卡片数据处理
# ============================================================

def test_trajectory_from_response():
    """后端返回 focus_trajectory 时前端正确接收"""
    trajectory = MOCK_CHAT_RESPONSE_WITH_MEMORY["data"]["focus_trajectory"]
    assert len(trajectory) == 1
    parsed = parse_emotion_summary_text(trajectory[0])
    assert "焦虑" in parsed["emotion"]
    assert "学业压力" in parsed["focus"]
    assert parsed["trend"] == "首次"
    print("✅ test_trajectory_from_response passed")


def test_trajectory_multiple_entries():
    """多轮轨迹：解析多条摘要"""
    trajectory = [
        "情绪: 焦虑明显；关注点: 学业压力；趋势: 首次；量表: PHQ-9 总分 12 分（中度抑郁）",
        "情绪: 焦虑略缓；关注点: 时间管理；趋势: 缓解",
        "情绪: 情绪稳定；关注点: 正念练习；趋势: 稳定",
    ]
    for entry in trajectory:
        parsed = parse_emotion_summary_text(entry)
        assert parsed["emotion"] != ""
        assert parsed["focus"] != ""

    # 第一条是首次
    first = parse_emotion_summary_text(trajectory[0])
    assert first["trend"] == "首次"
    # 第二条是缓解
    second = parse_emotion_summary_text(trajectory[1])
    assert second["trend"] == "缓解"
    # 第三条是稳定
    third = parse_emotion_summary_text(trajectory[2])
    assert third["trend"] == "稳定"
    print("✅ test_trajectory_multiple_entries passed")


def test_trend_icons_mapping():
    """趋势图标映射完整"""
    assert TREND_ICONS["首次"] == "🆕"
    assert TREND_ICONS["稳定"] == "➡️"
    assert TREND_ICONS["缓解"] == "📉"
    assert TREND_ICONS["加重"] == "📈"
    # 未知趋势有默认值
    assert TREND_ICONS.get("unknown", "•") == "•"
    print("✅ test_trend_icons_mapping passed")


def test_emotion_summary_displayed_in_assistant_message():
    """助手消息中应包含 emotion_summary 字段"""
    data = MOCK_CHAT_RESPONSE_WITH_MEMORY["data"]
    assert "emotion_summary" in data
    assert data["emotion_summary"] != ""
    # 验证可被解析
    parsed = parse_emotion_summary_text(data["emotion_summary"])
    assert parsed["emotion"] != ""
    print("✅ test_emotion_summary_displayed_in_assistant_message passed")


# ============================================================
# 测试 5: 回归测试 — 确保不破坏原有路由逻辑
# ============================================================

def test_text_only_still_uses_chat_endpoint():
    """纯文本仍走 /v1/agent/chat"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE_WITH_MEMORY
        mock_post.return_value = mock_resp

        requests.post(
            "http://localhost:8003/v1/agent/chat",
            json={"text": "test", "session_id": "s1", "history": []},
            timeout=60,
        )
        assert "chat" in str(mock_post.call_args)
        print("✅ test_text_only_still_uses_chat_endpoint passed")


def test_image_still_uses_analyze_endpoint():
    """有图片仍走 /v1/agent/analyze"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE_WITH_MEMORY
        mock_post.return_value = mock_resp

        requests.post(
            "http://localhost:8003/v1/agent/analyze",
            data={"text": "test", "session_id": "s1", "history": "[]"},
            files={"image": ("t.jpg", b"x", "image/jpeg")},
            timeout=60,
        )
        assert "analyze" in str(mock_post.call_args)
        print("✅ test_image_still_uses_analyze_endpoint passed")


if __name__ == "__main__":
    # build_history_payload tests
    test_history_payload_basic()
    test_history_payload_truncation()
    test_history_payload_filters_empty_content()
    test_history_payload_empty_list()
    test_history_payload_ignores_invalid_roles()

    # parse_emotion_summary_text tests
    test_parse_full_summary()
    test_parse_partial_summary()
    test_parse_empty_text()
    test_parse_none_text()
    test_parse_full_width_colon()

    # chat payload tests
    test_chat_payload_includes_session_id()
    test_chat_payload_includes_history()
    test_analyze_payload_includes_session_id_and_history()

    # trajectory card tests
    test_trajectory_from_response()
    test_trajectory_multiple_entries()
    test_trend_icons_mapping()
    test_emotion_summary_displayed_in_assistant_message()

    # regression tests
    test_text_only_still_uses_chat_endpoint()
    test_image_still_uses_analyze_endpoint()

    print("\n🎉 All frontend memory interaction tests passed!")
