"""FastAPI router for the offline text-emotion model."""

import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from .sentiment_model import SentimentModelUnavailable, sentiment_model
except ImportError:  # uvicorn main:app inside /app
    from sentiment_model import SentimentModelUnavailable, sentiment_model


logger = logging.getLogger("knowledge_service.sentiment_api")
router = APIRouter()


class TextSentimentRequest(BaseModel):
    text: str


@router.post("/v1/text/analyze-sentiment")
async def analyze_text_sentiment(req: TextSentimentRequest):
    """Run the offline Chinese transformer emotion classifier."""
    if not req.text or not req.text.strip():
        return JSONResponse(
            status_code=400,
            content={"code": 40013, "msg": "文本内容不能为空", "data": None},
        )
    try:
        result = await run_in_threadpool(sentiment_model.analyze, req.text)
        return {"code": 200, "msg": "success", "data": result}
    except SentimentModelUnavailable as exc:
        logger.error("Sentiment model unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "code": 50310,
                "msg": "文本情绪模型未就绪，请检查离线模型目录。",
                "data": {"detail": str(exc)},
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"code": 40013, "msg": str(exc), "data": None},
        )
    except Exception:
        logger.exception("Text sentiment inference failed")
        return JSONResponse(
            status_code=500,
            content={"code": 50010, "msg": "文本情绪模型推理失败", "data": None},
        )

