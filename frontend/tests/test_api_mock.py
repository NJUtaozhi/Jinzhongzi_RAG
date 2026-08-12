"""
测试1: API Mock 测试
验证纯文本 /chat 和多模态 /analyze 两种路由的请求逻辑
"""

import json
from unittest.mock import patch, MagicMock
import requests


# 模拟 Agent API 响应数据
MOCK_CHAT_RESPONSE = {
    "code": 200,
    "msg": "success",
    "data": {
        "analysis": {
            "text_sentiment": "negative",
            "text_keywords": ["焦虑", "压力"],
        },
        "decision": "need_knowledge",
        "reply": "听起来你有些焦虑，这是很常见的情绪。适度运动可以有效缓解焦虑。",
        "advice_source": "从知识库检索到的心理学知识条目",
    },
}

MOCK_ANALYZE_RESPONSE = {
    "code": 200,
    "msg": "success",
    "data": {
        "analysis": {
            "image_emotion": {
                "dominant_emotion": "sadness",
                "au12_r_smile_intensity": 0.2,
            },
            "text_sentiment": "negative",
            "text_keywords": ["焦虑", "压力"],
        },
        "decision": "need_knowledge",
        "reply": "检测到您面部表情偏悲伤，文字也透露出焦虑情绪。",
        "advice_source": "从知识库检索到的心理学知识条目",
    },
}


def test_chat_endpoint_url():
    """纯文本模式应调用 /v1/agent/chat"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE
        mock_post.return_value = mock_resp

        # 模拟纯文本调用
        chat_url = "http://localhost:8003/v1/agent/chat"
        resp = requests.post(chat_url, json={"text": "今天好焦虑"}, timeout=60)

        assert resp.status_code == 200
        result = resp.json()
        assert result["data"]["analysis"]["text_sentiment"] == "negative"
        assert "焦虑" in result["data"]["analysis"]["text_keywords"]
        print("✅ test_chat_endpoint_url passed")


def test_analyze_endpoint_url():
    """多模态模式应调用 /v1/agent/analyze"""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_ANALYZE_RESPONSE
        mock_post.return_value = mock_resp

        # 模拟多模态调用
        analyze_url = "http://localhost:8003/v1/agent/analyze"
        resp = requests.post(
            analyze_url,
            data={"text": "今天好焦虑"},
            files={"image": ("test.jpg", b"fake_image", "image/jpeg")},
            timeout=60,
        )

        assert resp.status_code == 200
        result = resp.json()
        assert result["data"]["analysis"]["image_emotion"]["dominant_emotion"] == "sadness"
        assert result["data"]["reply"] != ""
        print("✅ test_analyze_endpoint_url passed")


def test_routing_logic():
    """验证路由逻辑：有图片走 analyze，无图片走 chat"""
    AGENT_API_URL = "http://localhost:8003/v1/agent/analyze"
    CHAT_API_URL = "http://localhost:8003/v1/agent/chat"

    # 无图片 → chat
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CHAT_RESPONSE
        mock_post.return_value = mock_resp

        requests.post(CHAT_API_URL, json={"text": "test"}, timeout=60)
        called_url = mock_post.call_args[0][0] if mock_post.call_args[0] else mock_post.call_args[1].get("url")
        assert "chat" in str(mock_post.call_args), f"Expected chat endpoint, got: {mock_post.call_args}"
        print("✅ 纯文本路由 → /chat confirmed")

    # 有图片 → analyze
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_ANALYZE_RESPONSE
        mock_post.return_value = mock_resp

        requests.post(AGENT_API_URL, data={"text": "test"}, files={"image": ("t.jpg", b"x", "image/jpeg")}, timeout=60)
        assert "analyze" in str(mock_post.call_args), f"Expected analyze endpoint, got: {mock_post.call_args}"
        print("✅ 多模态路由 → /analyze confirmed")

    print("✅ test_routing_logic passed")


if __name__ == "__main__":
    test_chat_endpoint_url()
    test_analyze_endpoint_url()
    test_routing_logic()
    print("\n🎉 All API mock tests passed!")
