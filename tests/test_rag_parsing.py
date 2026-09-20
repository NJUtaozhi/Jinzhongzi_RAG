"""RAG 客户端响应解析回归测试.

背景: 新版 Knowledge 服务返回 {code, msg, results: [...]}（results 在顶层），
旧版为 {code, msg, data: {results: [...]}}。解析必须同时兼容两者，
否则 Agent 检索静默拿到 0 条文档（advice_source 恒为"未检索到知识来源"）。
"""

from services.rag_client import RAGClient


def _client() -> RAGClient:
    return RAGClient(base_url="http://localhost:8002")


def test_parse_new_top_level_results_format():
    """新版格式: results 在顶层（知识库当前实际返回的形状）"""
    raw = {
        "code": 200,
        "msg": "success",
        "results": [
            {"content": "深呼吸有助于缓解焦虑", "source": "焦虑干预手册", "score": 0.9},
            {"content": "规律作息改善睡眠", "source": "睡眠指南", "score": 0.7},
        ],
    }
    docs = _client()._parse_response(raw)
    assert len(docs) == 2
    assert docs[0].source == "焦虑干预手册"
    assert docs[0].relevance_score == 0.9
    assert docs[1].content == "规律作息改善睡眠"


def test_parse_legacy_data_results_format():
    """旧版格式: results 嵌套在 data 下（保持向后兼容）"""
    raw = {
        "code": 200,
        "msg": "success",
        "data": {"results": [{"content": "x", "source": "y", "score": 0.5}]},
    }
    docs = _client()._parse_response(raw)
    assert len(docs) == 1
    assert docs[0].source == "y"


def test_parse_data_as_list_format():
    """data 直接是列表的变体"""
    raw = {"data": [{"content": "c", "score": 1.0}]}
    docs = _client()._parse_response(raw)
    assert len(docs) == 1
    assert docs[0].content == "c"


def test_parse_empty_and_malformed():
    assert _client()._parse_response({}) == []
    assert _client()._parse_response({"results": None, "data": None}) == []
    assert _client()._parse_response({"results": "not-a-list"}) == []
