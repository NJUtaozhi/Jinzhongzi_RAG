"""Offline Chinese text-emotion inference for Task 7.1.

The production server has no access to Hugging Face. This module loads a model
only from ``SENTIMENT_MODEL_PATH`` by default. Downloads are an explicit local
preparation step handled by ``scripts/download_sentiment_model.py``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping


logger = logging.getLogger("knowledge_service.sentiment")

DEFAULT_MODEL_ID = "LXDaugh/chinese-6-emotion-model"
DEFAULT_MODEL_PATH = "/models/chinese-emotion"

LABEL_ALIASES = {
    "angry": "anger", "anger": "anger", "愤怒": "anger",
    "fear": "fear", "fearful": "fear", "害怕": "fear", "恐惧": "fear",
    "happy": "happiness", "happiness": "happiness", "joy": "happiness",
    "开心": "happiness", "快乐": "happiness",
    "neutral": "neutral", "平淡": "neutral", "中性": "neutral",
    "sad": "sadness", "sadness": "sadness", "悲伤": "sadness", "难过": "sadness",
    "surprise": "surprise", "surprised": "surprise", "惊讶": "surprise", "惊奇": "surprise",
    "disgust": "disgust", "disgusted": "disgust", "厌恶": "disgust",
    "concern": "anxiety", "concerned": "anxiety", "关切": "anxiety",
    "anxiety": "anxiety", "焦虑": "anxiety",
    "depression": "depression", "抑郁": "depression",
    "positive": "positive", "negative": "negative", "mixed": "mixed",
    "calm": "calm", "love": "joy", "gratitude": "positive",
}

# Documented index order for the selected checkpoint. Used only when its
# config exposes generic LABEL_n names.
SELECTED_MODEL_LABELS = ["anger", "fear", "happiness", "neutral", "sadness", "surprise"]

VALENCE = {
    "positive": 0.75, "happiness": 0.85, "joy": 0.85, "calm": 0.35,
    "neutral": 0.0, "mixed": 0.0, "surprise": 0.1,
    "sadness": -0.65, "anger": -0.7, "fear": -0.75,
    "anxiety": -0.7, "depression": -0.8, "disgust": -0.7, "negative": -0.65,
}


class SentimentModelUnavailable(RuntimeError):
    """Raised when the offline checkpoint cannot be loaded."""


def normalize_label(label: str, index: int | None = None) -> str:
    """Map a checkpoint label to the frontend's canonical emotion labels."""
    raw = str(label).strip().lower().replace(" tone", "").replace("_tone", "")
    if raw.startswith("label_") and index is not None and index < len(SELECTED_MODEL_LABELS):
        return SELECTED_MODEL_LABELS[index]
    for key, canonical in LABEL_ALIASES.items():
        if raw == key or key in raw:
            return canonical
    return "neutral"


class ChineseEmotionClassifier:
    """Thread-safe, lazily loaded CPU classifier."""

    def __init__(self, model_path: str | None = None, threshold: float | None = None):
        self.model_path = model_path or os.getenv("SENTIMENT_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.model_id = os.getenv("SENTIMENT_MODEL_ID", DEFAULT_MODEL_ID)
        self.threshold = float(threshold or os.getenv("SENTIMENT_THRESHOLD", "0.4"))
        self.max_length = int(os.getenv("SENTIMENT_MAX_LENGTH", "256"))
        self.multi_label = os.getenv("SENTIMENT_MULTI_LABEL", "true").lower() in {"1", "true", "yes"}
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._load_error = ""
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def load_error(self) -> str:
        return self._load_error

    def _load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                local_path = Path(self.model_path)
                allow_download = os.getenv("SENTIMENT_ALLOW_DOWNLOAD", "false").lower() in {"1", "true", "yes"}
                source = str(local_path) if local_path.exists() else self.model_id
                if not local_path.exists() and not allow_download:
                    raise FileNotFoundError(
                        f"离线模型目录不存在: {local_path}. "
                        "请先运行 scripts/download_sentiment_model.py 并上传模型目录。"
                    )
                local_only = not allow_download
                logger.info("Loading sentiment checkpoint from %s (local_only=%s)", source, local_only)
                self._tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
                self._model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=local_only)
                self._model.to("cpu")
                self._model.eval()
                torch.set_num_threads(max(1, int(os.getenv("SENTIMENT_CPU_THREADS", "2"))))
                self._torch = torch
                self._load_error = ""
                logger.info("Sentiment checkpoint loaded: %s", self.model_id)
            except Exception as exc:
                self._load_error = str(exc)
                self._tokenizer = None
                self._model = None
                logger.exception("Sentiment checkpoint failed to load")
                raise SentimentModelUnavailable(self._load_error) from exc

    def _id2label(self, size: int) -> List[str]:
        configured: Mapping[Any, Any] = getattr(self._model.config, "id2label", {}) or {}
        labels = []
        for index in range(size):
            raw = configured.get(index, configured.get(str(index), f"LABEL_{index}"))
            labels.append(normalize_label(str(raw), index))
        return labels

    def analyze(self, text: str) -> Dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("文本内容不能为空")
        self._load()

        started = time.perf_counter()
        encoded = self._tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits[0]
            probabilities = self._torch.sigmoid(logits) if self.multi_label else self._torch.softmax(logits, dim=-1)

        values = [float(value) for value in probabilities.cpu().tolist()]
        labels = self._id2label(len(values))
        merged_scores: Dict[str, float] = {}
        for label, score in zip(labels, values):
            merged_scores[label] = max(score, merged_scores.get(label, 0.0))

        ranked = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        primary, confidence = ranked[0]
        emotions = [label for label, score in ranked if score >= self.threshold][:3] or [primary]
        return {
            "sentiment": primary,
            "emotions": emotions,
            "confidence": round(confidence, 6),
            "intensity": round(confidence, 6),
            "valence": VALENCE.get(primary, 0.0),
            "scores": {label: round(score, 6) for label, score in ranked},
            "method": "transformer-model",
            "model": self.model_id,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


sentiment_model = ChineseEmotionClassifier()
