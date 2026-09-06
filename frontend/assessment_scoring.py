"""Pure frontend scoring helpers for PHQ-9 and GAD-7."""

from __future__ import annotations


def classify_assessment(scale: str, score: int) -> str:
    if scale == "PHQ-9":
        bounds = [
            (4, "无或极轻微抑郁倾向"), (9, "轻度抑郁倾向"),
            (14, "中度抑郁倾向"), (19, "中重度抑郁倾向"),
            (27, "重度抑郁倾向"),
        ]
    elif scale == "GAD-7":
        bounds = [
            (4, "无或极轻微焦虑倾向"), (9, "轻度焦虑倾向"),
            (14, "中度焦虑倾向"), (21, "重度焦虑倾向"),
        ]
    else:
        raise ValueError(f"不支持的量表: {scale}")
    return next(label for upper, label in bounds if score <= upper)


def build_assessment(scale: str, answers: list[int]) -> dict:
    expected = 9 if scale == "PHQ-9" else 7 if scale == "GAD-7" else 0
    if len(answers) != expected or any(value not in range(4) for value in answers):
        raise ValueError("量表答案数量或分值不合法")
    total = sum(answers)
    return {
        "scale": scale,
        "total_score": total,
        "severity": classify_assessment(scale, total),
        "item9_score": answers[8] if scale == "PHQ-9" else None,
    }

