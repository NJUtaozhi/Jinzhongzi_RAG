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


def test_parse_structured_source_object():
    """新版格式: source 为结构化对象 {"name": ..., "type": ...}，应取 name 字符串"""
    raw = {
        "code": 200,
        "msg": "success",
        "results": [
            {
                "text": "GAD-7 是自评工具",
                "source": {"name": "GAD-7量表", "type": "original_instrument"},
                "score": 0.8,
            },
        ],
    }
    docs = _client()._parse_response(raw)
    assert len(docs) == 1
    assert docs[0].source == "GAD-7量表"          # 不是 dict 原型串
    assert docs[0].content == "GAD-7 是自评工具"   # text 字段兜底


def test_parse_source_object_without_name():
    """结构化 source 缺 name 时应返回空串而不是抛异常/dict串"""
    raw = {"results": [{"text": "x", "source": {"type": "paper"}, "score": 0.5}]}
    assert _client()._parse_response(raw)[0].source == ""


def test_parse_source_legacy_string_kept():
    """旧版字符串 source 保持原样"""
    raw = {"results": [{"content": "x", "source": "焦虑干预手册", "score": 0.5}]}
    assert _client()._parse_response(raw)[0].source == "焦虑干预手册"
