"""Block 1 — 多模态情绪特征提取客户端.

封装面部动作单元分析、语音情绪识别、文本情感分析等微服务调用,
将多个模态的结果合并为统一的情绪特征快照。

API 规范:
    - 面部分析: POST {base_url}/v1/vision/analyze-face
      multipart/form-data, field: image
    - 语音分析: POST {base_url}/v1/audio/analyze-prosody  (预留)
    - 文本分析: POST {base_url}/v1/text/analyze-sentiment (预留)

    实际路径由成员1的 FastAPI 路由决定, 通过环境变量覆盖默认值.

环境变量:
    MULTIMODAL_BASE_URL   — 多模态服务基地址 (默认 http://localhost:8001)
    MULTIMODAL_VISION_PATH — 面部分析路径 (默认 /v1/vision/analyze-face)
    MULTIMODAL_TEXT_PATH   — 文本分析路径 (默认 /v1/text/analyze-sentiment)
    MULTIMODAL_AUDIO_PATH  — 语音分析路径 (默认 /v1/audio/analyze-prosody)
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import time
from codecs import encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class EmotionFeatures:
    """合并后的多模态情绪特征快照."""

    valence: Optional[float] = None
    arousal: Optional[float] = None
    facial_au: Dict[str, float] = field(default_factory=dict)
    facial_expression: str = ""
    voice_tremor: Optional[float] = None
    speech_rate: Optional[float] = None
    pitch_variance: Optional[float] = None
    text_sentiment: str = ""
    text_emotion_labels: List[str] = field(default_factory=list)
    available_modalities: List[str] = field(default_factory=list)
    error_modalities: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "facial_au": self.facial_au,
            "facial_expression": self.facial_expression,
            "voice_tremor": self.voice_tremor,
            "speech_rate": self.speech_rate,
            "pitch_variance": self.pitch_variance,
            "text_sentiment": self.text_sentiment,
            "text_emotion_labels": self.text_emotion_labels,
            "available_modalities": self.available_modalities,
            "error_modalities": self.error_modalities,
        }

    def summary(self) -> str:
        parts = []
        if self.valence is not None:
            parts.append(f"valence={self.valence:.2f}")
        if self.arousal is not None:
            parts.append(f"arousal={self.arousal:.2f}")
        if self.facial_expression:
            parts.append(f"facial={self.facial_expression}")
        if self.voice_tremor is not None:
            parts.append(f"voice_tremor={self.voice_tremor:.2f}")
        if self.text_sentiment:
            parts.append(f"text_sentiment={self.text_sentiment}")
        if self.text_emotion_labels:
            parts.append(f"text_emotions={','.join(self.text_emotion_labels)}")
        return "; ".join(parts) if parts else "(无多模态数据)"


class MultimodalClient:
    """多模态特征提取客户端.

    Usage::

        client = MultimodalClient()
        features = client.analyze(
            text="我最近总是睡不着",
            image_path=Path("user_face.jpg"),
            audio_path=None,
        )
    """

    # 默认部署: 同机 localhost, 端口 8001
    _DEFAULT_BASE_URL = "http://localhost:8001"
    _DEFAULT_VISION_PATH = "/v1/vision/analyze-face"
    _DEFAULT_TEXT_PATH = "/v1/text/analyze-sentiment"
    _DEFAULT_AUDIO_PATH = "/v1/audio/analyze-prosody"

    _MAX_RETRIES = 3
    _RETRY_BACKOFF = 0.5  # 秒

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 15,
    ) -> None:
        self._base_url = base_url or os.getenv(
            "MULTIMODAL_BASE_URL", self._DEFAULT_BASE_URL
        )
        self._timeout = timeout

        self._vision_path = os.getenv(
            "MULTIMODAL_VISION_PATH", self._DEFAULT_VISION_PATH
        )
        self._text_path = os.getenv(
            "MULTIMODAL_TEXT_PATH", self._DEFAULT_TEXT_PATH
        )
        self._audio_path = os.getenv(
            "MULTIMODAL_AUDIO_PATH", self._DEFAULT_AUDIO_PATH
        )

    # ── public API ──────────────────────────────────────────────────────────

    def analyze(
        self,
        text: Optional[str] = None,
        image_path: Optional[Path] = None,
        audio_path: Optional[Path] = None,
    ) -> EmotionFeatures:
        features = EmotionFeatures()

        if text:
            features.available_modalities.append("text")
            try:
                text_result = self._analyze_text(text)
                features.text_sentiment = text_result.get("sentiment", "")
                features.text_emotion_labels = text_result.get("emotions", [])
                if features.text_sentiment == "positive":
                    features.valence = features.valence or 0.6
                elif features.text_sentiment == "negative":
                    features.valence = features.valence or -0.6
            except Exception as exc:
                features.error_modalities["text"] = str(exc)

        if image_path:
            features.available_modalities.append("face")
            try:
                face_result = self._analyze_face(image_path)
                features.facial_au = face_result.get("action_units", {})
                features.facial_expression = face_result.get("expression", "")
                features.valence = face_result.get("valence", features.valence)
                features.arousal = face_result.get("arousal", features.arousal)
            except Exception as exc:
                features.error_modalities["face"] = str(exc)

        if audio_path:
            features.available_modalities.append("voice")
            try:
                voice_result = self._analyze_voice(audio_path)
                features.voice_tremor = voice_result.get("tremor_index")
                features.speech_rate = voice_result.get("speech_rate")
                features.pitch_variance = voice_result.get("pitch_variance")
                if features.arousal is None:
                    features.arousal = voice_result.get("arousal")
            except Exception as exc:
                features.error_modalities["voice"] = str(exc)

        return features

    # ── HTTP helpers ───────────────────────────────────────────────────────

    def _http_post(self, path: str, body: bytes | str, headers: Dict[str, str]) -> Dict[str, Any]:
        """带重试的 HTTP POST."""
        parsed = urlparse(self._base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        last_exc = None
        for attempt in range(self._MAX_RETRIES):
            conn = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=self._timeout)
                if isinstance(body, str):
                    body = body.encode("utf-8")
                conn.request("POST", path, body, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                return json.loads(data)
            except (ConnectionRefusedError, ConnectionResetError,
                    OSError, http.client.HTTPException) as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_BACKOFF * (attempt + 1))
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

        raise ConnectionError(
            f"HTTP POST {path} 失败 (重试 {self._MAX_RETRIES} 次): {last_exc}"
        )

    # ── per-modality calls ────────────────────────────────────────────────

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        return self._http_post(
            self._text_path,
            json.dumps({"text": text}),
            {"Content-Type": "application/json"},
        )

    def _analyze_face(self, image_path: Path) -> Dict[str, Any]:
        boundary = "wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T"
        data_parts: List[bytes] = []

        filename = image_path.name
        file_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"

        data_parts.append(encode("--" + boundary))
        data_parts.append(
            encode(f'Content-Disposition: form-data; name=image; filename={filename}')
        )
        data_parts.append(encode(f"Content-Type: {file_type}"))
        data_parts.append(encode(""))

        with open(image_path, "rb") as f:
            data_parts.append(f.read())

        data_parts.append(encode("--" + boundary + "--"))
        data_parts.append(encode(""))
        body = b"\r\n".join(data_parts)

        return self._http_post(
            self._vision_path,
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def _analyze_voice(self, audio_path: Path) -> Dict[str, Any]:
        boundary = "audioBoundary12345"
        data_parts: List[bytes] = []

        filename = audio_path.name
        file_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"

        data_parts.append(encode("--" + boundary))
        data_parts.append(
            encode(f'Content-Disposition: form-data; name=audio; filename={filename}')
        )
        data_parts.append(encode(f"Content-Type: {file_type}"))
        data_parts.append(encode(""))

        with open(audio_path, "rb") as f:
            data_parts.append(f.read())

        data_parts.append(encode("--" + boundary + "--"))
        data_parts.append(encode(""))
        body = b"\r\n".join(data_parts)

        return self._http_post(
            self._audio_path,
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
