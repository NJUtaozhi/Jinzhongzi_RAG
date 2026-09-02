from assessment_scoring import build_assessment


def test_phq9_score_and_level():
    assessment = build_assessment("PHQ-9", [1, 1, 2, 1, 1, 1, 2, 2, 1])
    assert assessment["total_score"] == 12
    assert assessment["severity"] == "中度抑郁倾向"
    assert assessment["item9_score"] == 1


def test_gad7_score_and_level():
    assessment = build_assessment("GAD-7", [2, 2, 2, 2, 2, 1, 1])
    assert assessment["total_score"] == 12
    assert assessment["severity"] == "中度焦虑倾向"
    assert assessment["item9_score"] is None

