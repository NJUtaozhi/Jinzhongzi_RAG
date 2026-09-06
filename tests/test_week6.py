"""Week 6 contract, assessment and crisis-rule tests."""

from pathlib import Path

import pytest

from assessment import (
    assess_crisis_risk,
    classify_score,
    crisis_referral_message,
    normalize_assessment,
    score_answers,
)
from services.multimodal_client import MultimodalClient
from orchestration.orchestrator import CounselingOrchestrator
from orchestration.state import default_state


def test_phq9_standard_boundaries():
    assert classify_score("PHQ-9", 4) == "无或极轻微抑郁倾向"
    assert classify_score("PHQ-9", 12) == "中度抑郁倾向"
    assert classify_score("PHQ-9", 15) == "中重度抑郁倾向"
    assert classify_score("PHQ-9", 27) == "重度抑郁倾向"


def test_gad7_standard_boundaries():
    assert classify_score("GAD-7", 9) == "轻度焦虑倾向"
    assert classify_score("GAD-7", 10) == "中度焦虑倾向"
    assert classify_score("GAD-7", 21) == "重度焦虑倾向"


def test_score_answers_records_phq9_item9():
    result = score_answers("PHQ-9", [0, 1, 2, 0, 1, 0, 2, 1, 1])
    assert result["total_score"] == 8
    assert result["item9_score"] == 1


@pytest.mark.parametrize(
    "assessment, expected_reason",
    [
        ({"scale": "PHQ-9", "total_score": 15, "item9_score": 0}, "总分"),
        ({"scale": "PHQ-9", "total_score": 3, "item9_score": 1}, "第 9 题"),
    ],
)
def test_phq9_crisis_thresholds(assessment, expected_reason):
    high_risk, reasons = assess_crisis_risk(assessment)
    assert high_risk is True
    assert expected_reason in " ".join(reasons)


def test_gad7_does_not_trigger_phq9_rule():
    assert assess_crisis_risk(
        {"scale": "GAD-7", "total_score": 21, "item9_score": None}
    ) == (False, [])


def test_crisis_message_contains_required_referral():
    message = crisis_referral_message()
    assert "12355" in message
    assert "400-161-9995" in message
    assert "就近医院" in message


def test_normalize_recomputes_untrusted_severity():
    normalized = normalize_assessment(
        {"scale": "PHQ-9", "total_score": 12, "severity": "无风险", "item9_score": 0}
    )
    assert normalized["severity"] == "中度抑郁倾向"


def test_multimodal_client_maps_real_vision_contract(tmp_path, monkeypatch):
    image = tmp_path / "face.jpg"
    image.write_bytes(b"fake")
    client = MultimodalClient()
    monkeypatch.setattr(
        client,
        "_http_post",
        lambda *args, **kwargs: {
            "code": 200,
            "msg": "success",
            "data": {
                "dominant_emotion": "happiness",
                "au_analysis": {"AU12_r": 2.4, "AU04_r": 0.2},
            },
        },
    )
    result = client._analyze_face(Path(image))
    assert result["expression"] == "happiness"
    assert result["action_units"]["AU12_r"] == 2.4


def test_safety_node_overrides_llm_draft_for_high_risk_assessment():
    orchestrator = CounselingOrchestrator.__new__(CounselingOrchestrator)
    state = default_state("我最近状态不好", session_id="risk-test")
    state.update({
        "assessment": {
            "scale": "PHQ-9",
            "total_score": 16,
            "severity": "中重度抑郁倾向",
            "item9_score": 1,
        },
        "crisis_risk": True,
        "crisis_reasons": ["PHQ-9 总分达到或超过 15", "PHQ-9 第 9 题提示风险"],
        "draft_response": "普通建议",
    })
    safety_update = orchestrator._safety_node(state)
    assert safety_update["safety_passed"] is True
    assert safety_update["status"] == "responding"
    assert "12355" in safety_update["draft_response"]
    assert "就近医院" in safety_update["draft_response"]

    state.update(safety_update)
    response_update = orchestrator._respond_node(state)
    assert "PHQ-9" in response_update["final_answer"]
    assert "16 分" in response_update["final_answer"]
    assert "400-161-9995" in response_update["final_answer"]
