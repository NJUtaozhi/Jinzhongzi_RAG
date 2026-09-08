"""多轮记忆与上下文管理（任务 7.3）。

职责三件事：

1. **情绪摘要记忆** —— 每轮对话后由 LLM 生成一段会话情绪状态摘要
   （含量表分数、情绪标签、关注点），下一轮注入 understand / reason 提示词。
2. **历史截断** —— 历史不无限膨胀，按轮次上限 + 单条消息长度上限压缩。
3. **会话存储** —— 内存态为主，可通过 ``SESSION_STORE_DIR`` 环境变量启用
   文件级持久化（服务重启不丢）；时间紧可保持纯内存态。

字段格式约定（与成员4 前端对齐，见 main.py 兼容路由）：

    history:           [{"role": "user" | "assistant", "content": "..."}]
    session_id:        稳定会话标识（兼容路由用 user_id / session_id 透传）
    emotion_summary:   后端生成的会话情绪摘要（字符串）
    focus_trajectory:  关注点变化轨迹（历史摘要列表，供前端情绪轨迹卡片使用）
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_service.memory")

# ── 截断常量 ─────────────────────────────────────────────────────────────────

DEFAULT_MAX_ROUNDS = 10        # 保留最近 N 轮（1 轮 = user + assistant 两条消息）
DEFAULT_MAX_MESSAGE_CHARS = 500  # 单条消息最长字符数
MAX_TRAJECTORY = 20            # 关注点变化轨迹最多保留的摘要条数


# ═══════════════════════════════════════════════════════════════════════════════
# 情绪摘要 prompt
# ═══════════════════════════════════════════════════════════════════════════════

EMOTION_SUMMARY_SYSTEM = """\
你是心理健康辅导系统的会话情绪摘要模块。

根据本轮对话的关键信息，生成一段简洁的会话情绪状态摘要，供下一轮对话注入上下文。
只输出 JSON，不要输出任何其他文字：

{"summary": "<一句话概括用户当前情绪状态>", "focus": "<用户当前核心关注点/困扰>", "trend": "<与上一轮相比的情绪变化: 首次|稳定|缓解|加重>", "assessment": "<量表分数摘要，若无则写 null>"}"""


def _format_assessment(assessment: Optional[Dict[str, Any]]) -> str:
    """把量表结果渲染成简短文本（供摘要与 fallback 使用）。"""
    if not assessment:
        return "无"
    scale = assessment.get("scale", "")
    total = assessment.get("total_score", "")
    severity = assessment.get("severity", "")
    if scale and total != "":
        return f"{scale} 总分 {total} 分（{severity}）"
    return "无"


def build_emotion_summary_prompt(
    *,
    user_query: str,
    emotion_label: str,
    assessment: Optional[Dict[str, Any]],
    final_answer: str,
    previous_summary: str = "",
) -> str:
    """构造情绪摘要 LLM 提示词。"""
    return (
        f"上一轮情绪摘要: {previous_summary or '(无，首次对话)'}\n"
        f"本轮用户消息: {user_query}\n"
        f"本轮识别情绪标签: {emotion_label or '未知'}\n"
        f"本轮量表结果: {_format_assessment(assessment)}\n"
        f"本轮 AI 回复（截断）: {final_answer[:200]}\n"
        "\n请输出会话情绪摘要 JSON。"
    )


def parse_emotion_summary(raw: str) -> Dict[str, str]:
    """解析 LLM 输出的情绪摘要 JSON（容忍前后噪声，失败则降级）。"""
    result: Dict[str, str] = {
        "summary": "",
        "focus": "",
        "trend": "",
        "assessment": "",
    }
    text = (raw or "").strip()
    if not text:
        return result

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                for key in ("summary", "focus", "trend", "assessment"):
                    val = obj.get(key)
                    if (
                        isinstance(val, str)
                        and val.strip()
                        and val.strip().lower() != "null"
                    ):
                        result[key] = val.strip()
                if result["summary"]:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 降级：无法解析 JSON 时，把整段文本当 summary
    result["summary"] = text[:200]
    return result


def render_emotion_summary(parsed: Dict[str, str]) -> str:
    """把结构化摘要渲染成紧凑文本，用于下一轮注入提示词。"""
    parts = [f"情绪: {parsed.get('summary', '')}"]
    if parsed.get("focus"):
        parts.append(f"关注点: {parsed['focus']}")
    if parsed.get("trend"):
        parts.append(f"趋势: {parsed['trend']}")
    if parsed.get("assessment"):
        parts.append(f"量表: {parsed['assessment']}")
    return "；".join(parts)


def fallback_emotion_summary(
    emotion_label: str,
    assessment: Optional[Dict[str, Any]],
    previous_summary: str = "",
) -> str:
    """LLM 摘要失败时的确定性降级摘要。"""
    parts = [f"情绪: {emotion_label or '未知'}"]
    if assessment:
        parts.append(f"量表: {_format_assessment(assessment)}")
    if previous_summary:
        parts.append(f"承接: {previous_summary}")
    return "；".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 历史截断
# ═══════════════════════════════════════════════════════════════════════════════

def truncate_history(
    history: Optional[List[Dict[str, str]]],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> List[Dict[str, str]]:
    """截断对话历史：保留最近 max_rounds 轮，单条消息截断到 max_chars。

    采用"滑动窗口"策略——旧轮次的细节由情绪摘要（emotion_summary）覆盖，
    因此只保留最近若干轮的原文即可，历史不会无限膨胀。
    """
    if not history:
        return []

    max_messages = max(1, max_rounds) * 2  # 1 轮 = user + assistant
    recent = history[-max_messages:]

    truncated: List[Dict[str, str]] = []
    for msg in recent:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        truncated.append({"role": role, "content": content})
    return truncated


# ═══════════════════════════════════════════════════════════════════════════════
# 会话记录与存储
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionRecord:
    """单个会话的持久化记忆。"""

    history: List[Dict[str, str]] = field(default_factory=list)
    emotion_summary: str = ""
    focus_trajectory: List[str] = field(default_factory=list)
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "emotion_summary": self.emotion_summary,
            "focus_trajectory": self.focus_trajectory,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        return cls(
            history=list(data.get("history", [])),
            emotion_summary=str(data.get("emotion_summary", "")),
            focus_trajectory=list(data.get("focus_trajectory", [])),
            turn_count=int(data.get("turn_count", 0)),
        )


def _safe_filename(session_id: str) -> str:
    """把任意 session_id 映射为安全文件名（防路径穿越 + 非法字符）。"""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.json"


class SessionStore:
    """会话级存储：内存态为主，可选文件持久化。

    - ``storage_dir=None``（默认）：纯内存态，进程退出即丢。
    - ``storage_dir`` 指向可写目录：每条会话落一个 JSON 文件，重启不丢。
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self._storage_dir = storage_dir
        self.max_rounds = max_rounds
        self._lock = threading.Lock()
        self._memory: Dict[str, SessionRecord] = {}

        if storage_dir:
            Path(storage_dir).mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────

    def load(self, session_id: str) -> SessionRecord:
        if not session_id:
            return SessionRecord()
        with self._lock:
            record = self._memory.get(session_id)
            if record is not None:
                return self._copy(record)
            record = self._read_from_disk(session_id)
            if record is not None:
                self._memory[session_id] = record
                return self._copy(record)
            return SessionRecord()

    def save(self, session_id: str, record: SessionRecord) -> None:
        if not session_id:
            return
        with self._lock:
            self._memory[session_id] = self._copy(record)
            self._write_to_disk(session_id, record)

    def clear(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._memory.pop(session_id, None)
            self._delete_from_disk(session_id)

    # ── helpers ───────────────────────────────────────────────────────────

    def _copy(self, record: SessionRecord) -> SessionRecord:
        # 深拷贝，避免调用方 mutate 返回的历史污染存储中的记录
        return copy.deepcopy(record)

    def _disk_path(self, session_id: str) -> Optional[Path]:
        if not self._storage_dir:
            return None
        return Path(self._storage_dir) / _safe_filename(session_id)

    def _read_from_disk(self, session_id: str) -> Optional[SessionRecord]:
        path = self._disk_path(session_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return SessionRecord.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("session %s 读取失败: %s", session_id, exc)
        return None

    def _write_to_disk(self, session_id: str, record: SessionRecord) -> None:
        path = self._disk_path(session_id)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps(record.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("session %s 写入失败: %s", session_id, exc)

    def _delete_from_disk(self, session_id: str) -> None:
        path = self._disk_path(session_id)
        if path is None or not path.exists():
            return
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("session %s 删除失败: %s", session_id, exc)


def build_session_store(max_rounds: int = DEFAULT_MAX_ROUNDS) -> SessionStore:
    """按环境变量构造会话存储（无 SESSION_STORE_DIR 时纯内存）。"""
    storage_dir = os.getenv("SESSION_STORE_DIR")
    return SessionStore(storage_dir=storage_dir, max_rounds=max_rounds)
