from contextlib import nullcontext

import pytest

from knowledge_service.sentiment_model import (
    ChineseEmotionClassifier,
    normalize_label,
)


class FakeValues:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def sigmoid(_values):
        return FakeValues([0.05, 0.72, 0.08, 0.1, 0.61, 0.04])

    @staticmethod
    def softmax(_values, dim=-1):
        return FakeValues([0.05, 0.72, 0.08, 0.1, 0.01, 0.04])


class FakeConfig:
    id2label = {index: f"LABEL_{index}" for index in range(6)}


class FakeOutput:
    logits = [[0.0] * 6]


class FakeModel:
    config = FakeConfig()

    def __call__(self, **_kwargs):
        return FakeOutput()


class FakeTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {"input_ids": [[1, 2, 3]]}


def ready_classifier():
    classifier = ChineseEmotionClassifier(model_path="unused", threshold=0.35)
    classifier._model = FakeModel()
    classifier._tokenizer = FakeTokenizer()
    classifier._torch = FakeTorch()
    return classifier


@pytest.mark.parametrize(
    "raw,index,expected",
    [("LABEL_0", 0, "anger"), ("happy", 2, "happiness"), ("悲伤语调", 4, "sadness")],
)
def test_label_mapping_is_frontend_compatible(raw, index, expected):
    assert normalize_label(raw, index) == expected


def test_real_model_contract_contains_scores_and_metadata():
    result = ready_classifier().analyze("我最近很害怕，也有些难过")
    assert result["sentiment"] == "fear"
    assert result["emotions"] == ["fear", "sadness"]
    assert result["confidence"] == 0.72
    assert result["method"] == "transformer-model"
    assert result["model"]
    assert result["valence"] < 0


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        ready_classifier().analyze("   ")

