"""Week 6 compatibility-route integration tests."""

import json

from fastapi.testclient import TestClient

import main


class FakeOrchestrator:
    def run(self, **kwargs):
        assessment = kwargs.get("assessment")
        return {
            "emotion_features": {
                "facial_expression": "happiness",
                "facial_au": {"AU12_r": 2.6, "AU04_r": 0.3},
                "available_modalities": ["text", "face"],
                "error_modalities": {},
            },
            "emotion_label": "positive",
            "user_intent": "seeking_emotional_support",
            "retrieved_docs": [
                {"content": "知识", "source": "PHQ-9 原始验证研究", "score": 0.9}
            ],
            "final_answer": "量表结果已纳入分析。",
            "assessment": assessment or {},
            "crisis_risk": False,
            "crisis_reasons": [],
        }


def test_compatibility_route_returns_real_au_and_source(monkeypatch):
    monkeypatch.setattr(main, "get_orch", lambda: FakeOrchestrator())
    response = TestClient(main.app).post(
        "/v1/agent/analyze",
        data={
            "text": "今天状态还可以",
            "assessment": json.dumps(
                {"scale": "PHQ-9", "total_score": 12, "severity": "伪造分级", "item9_score": 0}
            ),
        },
        files={"image": ("face.jpg", b"fake", "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["analysis"]["image_emotion"]["au12_r_smile_intensity"] == 2.6
    assert data["analysis"]["image_emotion"]["dominant_emotion"] == "happiness"
    assert data["advice_source"] == "PHQ-9 原始验证研究"
    assert data["assessment"]["severity"] == "中度抑郁倾向"


def test_compatibility_route_rejects_invalid_assessment():
    response = TestClient(main.app).post(
        "/v1/agent/analyze",
        data={"text": "测试", "assessment": "not-json"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == 40002

