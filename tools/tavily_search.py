"""Tavily search tool implementation."""

from typing import Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="搜索的查询内容")


class TavilySearchResults(BaseTool):
    name: str = "tavily_web_search"
    description: str = "使用 Tavily API 进行网络搜索，可以用来查找实时信息或新闻"
    args_schema: Type[BaseModel] = TavilySearchInput

    _api_key: str = PrivateAttr()
    _count: int = PrivateAttr()
    _summary: bool = PrivateAttr()
    _freshness: str = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        count: int = 5,
        summary: bool = True,
        freshness: str = "noLimit",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._count = count
        self._summary = summary
        self._freshness = freshness

    def _run(
        self,
        *args,
        **kwargs,
    ) -> str:
        query = kwargs.get("query")
        if query is None and args:
            query = args[0]
        if not isinstance(query, str) or not query.strip():
            return "搜索失败: 缺少 query 参数"

        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "search_depth": "advanced" if self._summary else "basic",
            "include_answer": self._summary,
            "max_results": self._count,
            "topic": "general",
        }
        if self._freshness and self._freshness != "noLimit":
            payload["days"] = 7

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return f"未找到相关内容。\nAPI返回: {data}"

            output = ""
            for i, item in enumerate(results[: self._count]):
                title = item.get("title", "无标题")
                snippet = item.get("content", item.get("snippet", "无摘要"))
                result_url = item.get("url", "")
                output += f"{i + 1}. {title}\n{snippet}\n链接: {result_url}\n\n"

            return output.strip()

        except requests.exceptions.RequestException as e:
            return f"网络请求失败: {e}"
        except ValueError as e:
            return f"搜索失败: {e}"


__all__ = ["TavilySearchResults", "TavilySearchInput"]