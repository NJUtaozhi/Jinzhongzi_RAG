"""PHQ-9 / GAD-7 scoring and deterministic crisis assessment.

The module is deliberately dependency-free so the scoring and safety rules can
be shared by the API, orchestration graph, frontend tests and CI.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PHQ9 = "PHQ-9"
GAD7 = "GAD-7"

SCALE_LENGTHS = {PHQ9: 9, GAD7: 7}


def classify_score(scale: str, total_score: int) -> str:
    """Return the standard severity label for a validated total score."""
    if scale == PHQ9:
        thresholds = (
            (4, "无或极轻微抑郁倾向"),
            (9, "轻度抑郁倾向"),
            (14, "中度抑郁倾向"),
            (19, "中重度抑郁倾向"),
            (27, "重度抑郁倾向"),
        )
    elif scale == GAD7:
        thresholds = (
            (4, "无或极轻微焦虑倾向"),
            (9, "轻度焦虑倾向"),
            (14, "中度焦虑倾向"),
            (21, "重度焦虑倾向"),
        )
    else:
        raise ValueError(f"不支持的量表: {scale}")

    for upper, label in thresholds:
        if total_score <= upper:
            return label
    raise ValueError(f"{scale} 总分超出范围: {total_score}")


def score_answers(scale: str, answers: Sequence[int]) -> dict[str, Any]:
    """Validate item answers (0-3) and produce the assessment contract."""
    expected = SCALE_LENGTHS.get(scale)
    if expected is None:
        raise ValueError(f"不支持的量表: {scale}")
    if len(answers) != expected:
        raise ValueError(f"{scale} 需要 {expected} 道题，实际收到 {len(answers)} 道")
    clean_answers = [int(value) for value in answers]
    if any(value < 0 or value > 3 for value in clean_answers):
        raise ValueError("每道题分数必须在 0 到 3 之间")

    total = sum(clean_answers)
    return {
        "scale": scale,
        "total_score": total,
        "severity": classify_score(scale, total),
        "item9_score": clean_answers[8] if scale == PHQ9 else None,
    }


def normalize_assessment(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate an assessment supplied by a client and recompute severity."""
    if not value:
        return None
    scale = str(value.get("scale", "")).strip().upper()
    if scale not in SCALE_LENGTHS:
        raise ValueError("assessment.scale 必须为 PHQ-9 或 GAD-7")

    try:
        total = int(value.get("total_score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("assessment.total_score 必须为整数") from exc

    max_score = SCALE_LENGTHS[scale] * 3
    if not 0 <= total <= max_score:
        raise ValueError(f"{scale} 总分必须在 0 到 {max_score} 之间")

    item9 = value.get("item9_score")
    if scale == PHQ9:
        try:
            item9 = int(item9) if item9 is not None else 0
        except (TypeError, ValueError) as exc:
            raise ValueError("assessment.item9_score 必须为 0 到 3 的整数") from exc
        if not 0 <= item9 <= 3:
            raise ValueError("assessment.item9_score 必须在 0 到 3 之间")
    else:
        item9 = None

    return {
        "scale": scale,
        "total_score": total,
        "severity": classify_score(scale, total),
        "item9_score": item9,
    }


def assess_crisis_risk(assessment: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    """Apply the Week 6 deterministic PHQ-9 crisis thresholds."""
    normalized = normalize_assessment(assessment)
    if not normalized or normalized["scale"] != PHQ9:
        return False, []

    reasons: list[str] = []
    if normalized["total_score"] >= 15:
        reasons.append("PHQ-9 总分达到或超过 15")
    if (normalized.get("item9_score") or 0) >= 1:
        reasons.append("PHQ-9 第 9 题提示存在自伤或轻生意念")
    return bool(reasons), reasons


def crisis_referral_message() -> str:
    """Return the non-LLM crisis referral text required by Week 6."""
    return (
        "谢谢你愿意告诉我这些。你的量表结果提示目前可能存在较高风险，"
        "这不是诊断，但需要马上得到真人支持。请先不要独处，尽快联系一位你信任的家人、"
        "朋友、老师或辅导员，并远离可能伤害自己的物品。你可以拨打 12355 青少年服务台"
        "或 400-161-9995 心理援助热线；如果你已经有立即伤害自己的计划、工具或行动，"
        "请立即拨打 120/110，或由可信任的人陪同前往就近医院急诊、精神心理科就医。"
        "我愿意继续听你说，但此刻真人陪伴和专业帮助最重要。"
    )

