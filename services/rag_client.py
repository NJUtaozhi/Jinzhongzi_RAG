"""Block 2 — RAG 知识库检索客户端.

封装心理健康知识库（文献/案例）的语义检索接口.

API 规范:
    POST {base_url}/v1/knowledge/retrieve
    Body: {"query": "感到焦虑和压力大时应该如何应对", "top_k": 5}

    实际路径由成员2的 FastAPI 路由决定, 通过环境变量覆盖默认值.

环境变量:
    RAG_BASE_URL       — 知识库服务基地址 (默认 http://localhost:8002)
    RAG_RETRIEVE_PATH  — 检索接口路径 (默认 /v1/knowledge/retrieve)
"""

from __future__ import annotations

import http.client
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class RetrievalResult:
    """单条检索结果."""

    content: str = ""
    source: str = ""
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGContext:
    """RAG 检索上下文."""

    query: str = ""
    documents: List[RetrievalResult] = field(default_factory=list)
    formatted_text: str = ""

    def summary(self) -> str:
        return (
            f"RAG query='{self.query[:60]}...' "
            f"docs={len(self.documents)}"
        )


class RAGClient:
    """心理健康知识库检索客户端.

    Usage::

        client = RAGClient()
        ctx = client.retrieve(query="如何缓解焦虑", top_k=5)
        print(ctx.formatted_text)
    """

    _DEFAULT_BASE_URL = "http://localhost:8002"
    _DEFAULT_RETRIEVE_PATH = "/v1/knowledge/retrieve"

    _MAX_RETRIES = 3
    _RETRY_BACKOFF = 0.5

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 15,
    ) -> None:
        self._base_url = base_url or os.getenv(
            "RAG_BASE_URL", self._DEFAULT_BASE_URL
        )
        self._timeout = timeout
        self._retrieve_path = os.getenv(
            "RAG_RETRIEVE_PATH", self._DEFAULT_RETRIEVE_PATH
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None,
    ) -> RAGContext:
        ctx = RAGContext(query=query)

        try:
            raw = self._call_retrieve_api(query, top_k, filter_tags)
            docs = self._parse_response(raw)
            ctx.documents = docs
            ctx.formatted_text = self._format_documents(docs)
        except Exception as exc:
            ctx.formatted_text = f"[RAG 检索失败: {exc}]"

        return ctx

    # ── HTTP ───────────────────────────────────────────────────────────────

    def _call_retrieve_api(
        self,
        query: str,
        top_k: int,
        filter_tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        payload = json.dumps({
            "query": query,
            "top_k": top_k,
            **(dict(tags=filter_tags) if filter_tags else {}),
        })
        return self._http_post(
            self._retrieve_path,
            payload,
            {"Content-Type": "application/json"},
        )

    def _http_post(self, path: str, body: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """带重试的 HTTP POST."""
        parsed = urlparse(self._base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        last_exc = None
        for attempt in range(self._MAX_RETRIES):
            conn = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=self._timeout)
                conn.request("POST", path, body.encode("utf-8"), headers)
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

    # ── parsing ────────────────────────────────────────────────────────────

    def _parse_response(self, raw: Dict[str, Any]) -> List[RetrievalResult]:
        # Knowledge 服务返回 {code, msg, data: {results: [...]}}
        data = raw.get("data", {})
        if isinstance(data, dict):
            items = data.get("results", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        docs: List[RetrievalResult] = []
        for item in items:
            docs.append(RetrievalResult(
                content=item.get("content", item.get("text", "")),
                source=item.get("source", item.get("title", "")),
                relevance_score=float(item.get("score", item.get("relevance", 0.0))),
                metadata={k: v for k, v in item.items()
                          if k not in ("content", "text", "source", "title",
                                       "score", "relevance")},
            ))
        return docs

    @staticmethod
    def _format_documents(docs: List[RetrievalResult]) -> str:
        if not docs:
            return "(未检索到相关知识)"

        parts: List[str] = []
        for i, doc in enumerate(docs, 1):
            source_info = f" (来源: {doc.source})" if doc.source else ""
            parts.append(
                f"【文献{i}】{source_info}\n{doc.content}\n"
                f"相关性: {doc.relevance_score:.2f}"
            )
        return "\n\n".join(parts)
