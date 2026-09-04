"""心理健康辅导 Agent — FastAPI 服务入口.

部署到服务器 146.56.204.132:8003, 供前端页面 (8501) 调用.

启动:
    python main.py
    或
    uvicorn main:app --host 0.0.0.0 --port 8003

环境变量:
    AGENT_HOST — 监听地址 (默认 0.0.0.0)
    AGENT_PORT — 监听端口 (默认 8003)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from assessment import normalize_assessment

# 最优先加载 .env
load_dotenv()

# ── 日志 ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_service")

from orchestration import get_orchestrator  # noqa: E402
from orchestration.state import default_state  # noqa: E402

# ── Request / Response models ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """单轮对话请求."""

    query: str = Field(..., description="用户输入文本", min_length=1)
    session_id: str = Field(default="default", description="会话 ID (多轮对话追踪)")
    history: List[dict] = Field(
        default_factory=list,
        description="对话历史 [{\"role\":\"user/assistant\",\"content\":\"...\"}]",
    )
    image_path: Optional[str] = Field(default=None, description="可选: 面部图像文件路径")
    audio_path: Optional[str] = Field(default=None, description="可选: 语音文件路径")
    assessment: Optional[dict] = Field(
        default=None,
        description="可选: {scale, total_score, severity, item9_score}",
    )
    max_iterations: int = Field(default=8, ge=1, le=20, description="ReAct 最大迭代次数")


class ChatResponse(BaseModel):
    """单轮对话响应."""

    answer: str = Field(..., description="Agent 最终回复")
    status: str = Field(..., description="本轮状态: completed | failed")
    intent: str = Field(default="", description="识别的用户意图")
    emotion: str = Field(default="", description="识别的情绪标签")
    react_trace: List[str] = Field(default_factory=list, description="ReAct 审计轨迹")
    session_id: str = Field(default="", description="会话 ID")
    assessment: Optional[dict] = Field(default=None, description="标准化量表结果")
    crisis_risk: bool = Field(default=False, description="是否触发高风险转介")


class FrontendChatRequest(BaseModel):
    """前端纯文本对话请求 (兼容路由 /v1/agent/chat)."""

    text: str = Field(..., description="用户输入文本", min_length=1)
    user_id: str = Field(default=None, description="可选: 用户ID")
    assessment: Optional[dict] = Field(
        default=None,
        description="可选: {scale, total_score, severity, item9_score}",
    )


class HealthResponse(BaseModel):
    """健康检查响应."""

    status: str = "ok"
    version: str = "0.1.0"
    services: dict = Field(default_factory=lambda: {
        "multimodal": os.getenv("MULTIMODAL_BASE_URL", "http://localhost:8001"),
        "rag": os.getenv("RAG_BASE_URL", "http://localhost:8002"),
    })
    dependencies: dict = Field(default_factory=dict)


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="心理健康辅导 Agent API",
    description="多模态感知 → 意图理解 → RAG检索 → ReAct推理 → 安全校验 → 回应的完整链路",
    version="0.1.0",
)

# CORS — 允许前端 (8501) 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局编排器 (懒加载) ─────────────────────────────────────────────────────

_orchestrator = None


def dependency_status(base_url: str) -> str:
    try:
        with urlopen(f"{base_url.rstrip('/')}/health", timeout=2) as response:
            return "online" if 200 <= response.status < 300 else "offline"
    except Exception:
        return "offline"


def health_payload() -> HealthResponse:
    vision_url = os.getenv("MULTIMODAL_BASE_URL", "http://localhost:8001")
    rag_url = os.getenv("RAG_BASE_URL", "http://localhost:8002")
    return HealthResponse(
        services={"multimodal": vision_url, "rag": rag_url},
        dependencies={
            "vision_service": dependency_status(vision_url),
            "knowledge_service": dependency_status(rag_url),
        },
    )


def get_orch():
    global _orchestrator
    if _orchestrator is None:
        CounselingOrchestrator = get_orchestrator()
        _orchestrator = CounselingOrchestrator()
    return _orchestrator


def extract_advice_source(result: dict) -> str:
    sources = []
    for document in result.get("retrieved_docs", []) or []:
        source = str(document.get("source", "")).strip()
        if source and source not in sources:
            sources.append(source)
    return "；".join(sources) if sources else "未检索到知识来源"


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/", response_model=HealthResponse, tags=["系统"])
async def root():
    """根路径 — 健康检查."""
    return health_payload()


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查."""
    return health_payload()


@app.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(req: ChatRequest):
    """单轮心理健康辅导对话.

    完整链路: perceive → understand → retrieve → reason → safety → respond
    """
    t_start = time.time()
    try:
        orch = get_orch()
        result = orch.run(
            user_query=req.query,
            session_id=req.session_id,
            conversation_history=req.history if req.history else None,
            image_path=req.image_path,
            audio_path=req.audio_path,
            assessment=normalize_assessment(req.assessment),
            max_iterations=req.max_iterations,
        )

        elapsed = time.time() - t_start
        logger.info(
            "chat session=%s intent=%s emotion=%s elapsed=%.2fs trace_nodes=%d",
            req.session_id,
            result.get("user_intent", ""),
            result.get("emotion_label", ""),
            elapsed,
            len(result.get("react_trace", [])),
        )

        return ChatResponse(
            answer=result.get("final_answer", "(无回应)"),
            status=result.get("status", "failed"),
            intent=result.get("user_intent", ""),
            emotion=result.get("emotion_label", ""),
            react_trace=result.get("react_trace", []),
            session_id=req.session_id,
            assessment=result.get("assessment") or None,
            crisis_risk=bool(result.get("crisis_risk", False)),
        )
    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error(
            "chat session=%s error=%s elapsed=%.2fs",
            req.session_id, exc, elapsed,
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream", tags=["对话"])
async def chat_stream(req: ChatRequest):
    """流式对话 (SSE).

    使用 LangGraph astream 逐节点推送状态更新,
    respond 节点完成后将 final_answer 按句子分块推送.
    """
    async def event_generator():
        t_start = time.time()
        orch = get_orch()

        # 构建初始状态 (与 orch.run 保持一致)
        state = default_state(
            user_query=req.query,
            session_id=req.session_id,
            max_iterations=req.max_iterations,
        )
        if req.history:
            state["conversation_history"] = req.history
        state["runtime_context"] = {  # type: ignore[typeddict-unknown-key]
            "image_path": req.image_path,
            "audio_path": req.audio_path,
            "assessment": normalize_assessment(req.assessment),
        }

        try:
            async for chunk in orch.graph.astream(state, stream_mode="values"):
                status = chunk.get("status", "")
                trace = chunk.get("react_trace", [])
                # 只推送最近 2 条 ReAct 轨迹
                recent_trace = trace[-2:] if trace else []

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "status",
                            "status": status,
                            "trace": recent_trace,
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

                if status == "completed":
                    final = chunk.get("final_answer", "")
                    emotion = chunk.get("emotion_label", "")
                    intent = chunk.get("user_intent", "")

                    # 按句子分块推送最终回复
                    sentences = re.split(r"(?<=[。！？.!?])", final)
                    for sent in sentences:
                        stripped = sent.strip()
                        if stripped:
                            yield (
                                "data: "
                                + json.dumps(
                                    {"type": "reply_chunk", "content": stripped},
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )

                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "done",
                                "emotion": emotion,
                                "intent": intent,
                                "session_id": req.session_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                    elapsed = time.time() - t_start
                    logger.info(
                        "chat_stream session=%s intent=%s emotion=%s elapsed=%.2fs trace_nodes=%d",
                        req.session_id, intent, emotion, elapsed,
                        len(trace),
                    )
                    break

        except Exception as exc:
            elapsed = time.time() - t_start
            logger.error(
                "chat_stream session=%s error=%s elapsed=%.2fs",
                req.session_id, exc, elapsed,
            )
            yield (
                "data: "
                + json.dumps(
                    {"type": "error", "message": str(exc)},
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/agent/analyze", tags=["兼容"])
async def agent_analyze(
    text: str = Form(..., description="用户输入文本"),
    image: UploadFile = File(None, description="可选: 面部图片"),
    user_id: str = Form(None, description="可选: 用户ID"),
    assessment: str = Form(None, description="可选: JSON 格式 PHQ-9/GAD-7 结果"),
):
    """兼容成员4前端的 /v1/agent/analyze 路由.

    接收 multipart/form-data (text + image),
    转换为内部 /chat 格式调用, 返回前端期望的 {code, msg, data} 结构.
    """
    try:
        assessment_data = normalize_assessment(
            json.loads(assessment) if assessment else None
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={"code": 40002, "msg": f"量表数据格式错误: {exc}", "data": None},
        )

    # 处理图片: 保存到临时文件, 获取路径
    image_path = None
    if image and image.filename:
        temp_dir = tempfile.mkdtemp(prefix="agent_img_")
        try:
            img_bytes = await image.read()
            img_path = os.path.join(temp_dir, image.filename or "upload.jpg")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            image_path = img_path
        except Exception:
            pass

    t_start = time.time()
    try:
        orch = get_orch()
        result = orch.run(
            user_query=text,
            session_id=user_id or str(uuid.uuid4())[:8],
            conversation_history=None,
            image_path=image_path,
            audio_path=None,
            assessment=assessment_data,
            max_iterations=8,
        )

        elapsed = time.time() - t_start
        logger.info(
            "analyze user_id=%s intent=%s emotion=%s elapsed=%.2fs",
            user_id, result.get("user_intent", ""),
            result.get("emotion_label", ""), elapsed,
        )

        # 提取情绪信息
        emotion_features = result.get("emotion_features", {}) or {}
        facial_au = emotion_features.get("facial_au", {}) or {}
        face_errors = emotion_features.get("error_modalities", {}) or {}
        face_available = "face" in (emotion_features.get("available_modalities", []) or [])
        image_emotion_data = {
            "dominant_emotion": emotion_features.get("facial_expression") or "unknown",
            "au12_r_smile_intensity": float(facial_au.get("AU12_r", 0.0)),
            "au04_r_brow_lower": float(facial_au.get("AU04_r", 0.0)),
            "au_analysis": facial_au,
            "analysis_available": face_available and not bool(face_errors.get("face")),
        }
        if face_errors.get("face"):
            image_emotion_data["error"] = face_errors["face"]

        advice_source = extract_advice_source(result)

        # 组装前端期望格式
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "analysis": {
                    "image_emotion": image_emotion_data,
                    "text_sentiment": result.get("emotion_label", "neutral"),
                    "text_keywords": [],
                },
                "decision": result.get("user_intent", "unclear"),
                "reply": result.get("final_answer", ""),
                "advice_source": advice_source,
                "assessment": result.get("assessment") or None,
                "crisis_risk": bool(result.get("crisis_risk", False)),
                "crisis_reasons": result.get("crisis_reasons", []),
            },
        }
    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error(
            "analyze user_id=%s error=%s elapsed=%.2fs",
            user_id, exc, elapsed,
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": 50000,
                "msg": f"Agent 服务内部错误: {str(exc)}",
                "data": None,
            },
        )
    finally:
        # 清理临时图片
        if image_path and os.path.exists(os.path.dirname(image_path)):
            shutil.rmtree(os.path.dirname(image_path), ignore_errors=True)


@app.post("/v1/agent/chat", tags=["兼容"])
async def agent_chat(req: FrontendChatRequest):
    """兼容成员4前端的 /v1/agent/chat 路由 (纯文本 JSON).

    接收 {"text": "..."}，返回与 /v1/agent/analyze 一致的 {code, msg, data} 结构。
    """
    t_start = time.time()
    try:
        orch = get_orch()
        result = orch.run(
            user_query=req.text,
            session_id=req.user_id or str(uuid.uuid4())[:8],
            conversation_history=None,
            image_path=None,
            audio_path=None,
            assessment=normalize_assessment(req.assessment),
            max_iterations=8,
        )

        elapsed = time.time() - t_start
        logger.info(
            "chat_compat user_id=%s intent=%s emotion=%s elapsed=%.2fs",
            req.user_id, result.get("user_intent", ""),
            result.get("emotion_label", ""), elapsed,
        )

        return {
            "code": 200,
            "msg": "success",
            "data": {
                "analysis": {
                    "image_emotion": {},
                    "text_sentiment": result.get("emotion_label", "neutral"),
                    "text_keywords": [],
                },
                "decision": result.get("user_intent", "unclear"),
                "reply": result.get("final_answer", ""),
                "advice_source": extract_advice_source(result),
                "assessment": result.get("assessment") or None,
                "crisis_risk": bool(result.get("crisis_risk", False)),
                "crisis_reasons": result.get("crisis_reasons", []),
            },
        }
    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error(
            "chat_compat user_id=%s error=%s elapsed=%.2fs",
            req.user_id, exc, elapsed,
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": 50000,
                "msg": f"Agent 服务内部错误: {str(exc)}",
                "data": None,
            },
        )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "8003"))

    logger.info("心理健康辅导 Agent 启动中...")
    logger.info("  http://%s:%s", host, port)
    logger.info("  API 文档: http://%s:%s/docs", host, port)
    logger.info("  健康检查: http://%s:%s/health", host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")
