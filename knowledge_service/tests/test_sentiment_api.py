from fastapi import FastAPI
from fastapi.testclient import TestClient

from knowledge_service import sentiment_api


def make_client():
    app = FastAPI()
    app.include_router(sentiment_api.router)
    return TestClient(app)


def test_sentiment_endpoint_returns_model_result(monkeypatch):
    monkeypatch.setattr(
        sentiment_api.sentiment_model,
        "analyze",
        lambda text: {
            "sentiment": "happiness",
            "emotions": ["happiness"],
            "confidence": 0.93,
            "intensity": 0.93,
            "valence": 0.85,
            "scores": {"happiness": 0.93},
            "method": "transformer-model",
            "model": "test-model",
            "latency_ms": 10.0,
        },
    )
    response = make_client().post(
        "/v1/text/analyze-sentiment", json={"text": "今天很开心"}
    )
    assert response.status_code == 200
    assert response.json()["data"]["method"] == "transformer-model"
    assert response.json()["data"]["sentiment"] == "happiness"


def test_sentiment_endpoint_rejects_blank_text():
    response = make_client().post("/v1/text/analyze-sentiment", json={"text": "  "})
    assert response.status_code == 400
    assert response.json()["code"] == 40013

