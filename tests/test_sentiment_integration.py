from orchestration.orchestrator import CounselingOrchestrator
from orchestration.state import default_state
from services.multimodal_client import EmotionFeatures, MultimodalClient


class FakeAgent:
    def invoke(self, _payload):
        class Message:
            content = "intent: seeking_emotional_support\nemotion: happiness\nbrief: 用户需要支持"
        return {"messages": [Message()]}


def test_text_model_uses_dedicated_knowledge_base_url(monkeypatch):
    monkeypatch.setenv("TEXT_SENTIMENT_BASE_URL", "http://knowledge:8002")
    client = MultimodalClient(base_url="http://vision:8001")
    calls = []

    def fake_post(path, body, headers, base_url=None):
        calls.append((path, base_url))
        return {"data": {
            "sentiment": "fear", "emotions": ["fear"],
            "confidence": 0.91, "intensity": 0.91, "valence": -0.75,
            "scores": {"fear": 0.91}, "model": "test-model",
        }}

    monkeypatch.setattr(client, "_http_post", fake_post)
    features = client.analyze(text="我有点害怕")
    assert calls == [("/v1/text/analyze-sentiment", "http://knowledge:8002")]
    assert features.text_sentiment == "fear"
    assert features.text_confidence == 0.91
    assert features.valence == -0.75
    assert features.text_model == "test-model"


def test_understand_node_cannot_override_model_emotion():
    orchestrator = CounselingOrchestrator.__new__(CounselingOrchestrator)
    orchestrator.understand_agent = FakeAgent()
    state = default_state("我有点害怕", session_id="sentiment-test")
    state["emotion_features"] = EmotionFeatures(
        text_sentiment="fear", text_emotion_labels=["fear"]
    ).to_dict()
    update = orchestrator._understand_node(state)
    assert update["user_intent"] == "seeking_emotional_support"
    assert update["emotion_label"] == "fear"
    assert any("source=model" in item for item in update["execution_log"])
