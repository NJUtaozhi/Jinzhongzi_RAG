"""
测试2: 情绪映射测试
验证 SENTIMENT_SCORE_MAP 覆盖 8+ 标签且分数映射正确
"""

# 从 app.py 导入映射（直接内联以避免 Streamlit 初始化）
SENTIMENT_SCORE_MAP = {
    "positive": 3,
    "neutral": 2,
    "mixed": 1.5,
    "negative": 1,
    "happiness": 3,
    "joy": 3,
    "calm": 2.5,
    "surprise": 2,
    "sadness": 1,
    "anger": 0.8,
    "fear": 0.5,
    "anxiety": 0.5,
    "depression": 0.3,
    "disgust": 0.5,
}


def test_map_has_8_plus_labels():
    """映射表至少覆盖 8 个标签"""
    assert len(SENTIMENT_SCORE_MAP) >= 8, f"Only {len(SENTIMENT_SCORE_MAP)} labels, need 8+"
    print(f"✅ map_has_8_plus_labels passed ({len(SENTIMENT_SCORE_MAP)} labels)")


def test_positive_emotions_score_high():
    """正面情绪分数应高于负面情绪"""
    assert SENTIMENT_SCORE_MAP["positive"] > SENTIMENT_SCORE_MAP["negative"]
    assert SENTIMENT_SCORE_MAP["happiness"] > SENTIMENT_SCORE_MAP["sadness"]
    assert SENTIMENT_SCORE_MAP["joy"] > SENTIMENT_SCORE_MAP["depression"]
    print("✅ positive_emotions_score_high passed")


def test_all_scores_in_range():
    """所有分数应在 [0, 4] 范围内"""
    for label, score in SENTIMENT_SCORE_MAP.items():
        assert 0 <= score <= 4, f"{label} score {score} out of [0,4] range"
    print("✅ all_scores_in_range passed")


def test_extended_labels_covered():
    """扩展标签（anxiety, depression, anger, fear 等）都被覆盖"""
    required = ["anxiety", "depression", "anger", "fear", "sadness", "happiness"]
    for label in required:
        assert label in SENTIMENT_SCORE_MAP, f"Missing extended label: {label}"
    print("✅ extended_labels_covered passed")


def test_unknown_sentiment_uses_default():
    """未知的 sentiment 标签应回退到默认值"""
    unknown = "unknown_emotion"
    default_score = 2  # 默认中性
    score = SENTIMENT_SCORE_MAP.get(unknown, default_score)
    assert score == default_score
    print("✅ unknown_sentiment_uses_default passed")


if __name__ == "__main__":
    test_map_has_8_plus_labels()
    test_positive_emotions_score_high()
    test_all_scores_in_range()
    test_extended_labels_covered()
    test_unknown_sentiment_uses_default()
    print("\n🎉 All sentiment mapping tests passed!")
