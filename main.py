"""心理健康辅导 Agent — FastAPI 服务入口.

部署到服务器 101.34.68.33:8003, 供前端页面 (8501) 调用.

启动:
    python main.py
    或
    uvicorn main:app --host 0.0.0.0 --port 8003

环境变量:
    AGENT_HOST — 监听地址 (默认 0.0.0.0)
    AGENT_PORT — 监听端口 (默认 8003)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import json
import shutil
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 最优先加载 .env
load_dotenv()

from orchestration import get_orchestrator  # noqa: E402

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
    max_iterations: int = Field(default=8, ge=1, le=20, description="ReAct 最大迭代次数")


class ChatResponse(BaseModel):
    """单轮对话响应."""

    answer: str = Field(..., description="Agent 最终回复")
    status: str = Field(..., description="本轮状态: completed | failed")
    intent: str = Field(default="", description="识别的用户意图")
    emotion: str = Field(default="", description="识别的情绪标签")
    react_trace: List[str] = Field(default_factory=list, description="ReAct 审计轨迹")
    session_id: str = Field(default="", description="会话 ID")


class HealthResponse(BaseModel):
    """健康检查响应."""

    status: str = "ok"
    version: str = "0.1.0"
    services: dict = Field(default_factory=lambda: {
        "multimodal": os.getenv("MULTIMODAL_BASE_URL", "http://localhost:8001"),
        "rag": os.getenv("RAG_BASE_URL", "http://localhost:8002"),
    })


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


def get_orch():
    global _orchestrator
    if _orchestrator is None:
        CounselingOrchestrator = get_orchestrator()
        _orchestrator = CounselingOrchestrator()
    return _orchestrator


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/", response_model=HealthResponse, tags=["系统"])
async def root():
    """根路径 — 健康检查."""
    return HealthResponse()


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查."""
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(req: ChatRequest):
    """单轮心理健康辅导对话.

    完整链路: perceive → understand → retrieve → reason → safety → respond
    """
    try:
        orch = get_orch()
        result = orch.run(
            user_query=req.query,
            session_id=req.session_id,
            conversation_history=req.history if req.history else None,
            image_path=req.image_path,
            audio_path=req.audio_path,
            max_iterations=req.max_iterations,
        )

        return ChatResponse(
            answer=result.get("final_answer", "(无回应)"),
            status=result.get("status", "failed"),
            intent=result.get("user_intent", ""),
            emotion=result.get("emotion_label", ""),
            react_trace=result.get("react_trace", []),
            session_id=req.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream", tags=["对话"])
async def chat_stream(req: ChatRequest):
    """流式对话 (SSE) — 暂未实现, 保留接口."""
    # TODO: 实现 SSE 流式输出
    raise HTTPException(status_code=501, detail="流式接口开发中")


@app.post("/v1/agent/analyze", tags=["兼容"])
async def agent_analyze(
    text: str = Form(..., description="用户输入文本"),
    image: UploadFile = File(None, description="可选: 面部图片"),
    user_id: str = Form(None, description="可选: 用户ID"),
):
    """兼容成员4前端的 /v1/agent/analyze 路由.

    接收 multipart/form-data (text + image),
    转换为内部 /chat 格式调用, 返回前端期望的 {code, msg, data} 结构.
    """
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

    try:
        orch = get_orch()
        result = orch.run(
            user_query=text,
            session_id=user_id or str(uuid.uuid4())[:8],
            conversation_history=None,
            image_path=image_path,
            audio_path=None,
            max_iterations=8,
        )

        # 提取情绪信息
        image_emotion_data = {
            "dominant_emotion": result.get("emotion_label", "unknown"),
            "au12_r_smile_intensity": 0,
            "au04_r_brow_lower": 0,
        }

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
                "decision": result.get("intent", "neutral"),
                "reply": result.get("final_answer", ""),
                "advice_source": "Agent 综合分析",
            },
        }
    except Exception as exc:
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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "8003"))

    print(f"🚀 心理健康辅导 Agent 启动中...")
    print(f"   📍 http://{host}:{port}")
    print(f"   📖 API 文档: http://{host}:{port}/docs")
    print(f"   🏥 健康检查: http://{host}:{port}/health")

    uvicorn.run(app, host=host, port=port, log_level="info")
