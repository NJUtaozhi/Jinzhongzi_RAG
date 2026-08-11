# -*- coding: utf-8 -*-
import os
import sys
import pytest
from fastapi.testclient import TestClient

# 将父目录添加到 Python 路径，以便能导入 main 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# 创建测试客户端
client = TestClient(app)


class TestKnowledgeRetrieve:
    """知识检索接口测试套件（行为验证）"""

    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "knowledge-retrieval"
        assert data["doc_count"] > 0

    def test_retrieve_with_valid_query(self):
        """测试正常检索：输入有效查询，返回正确格式的结果"""
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "焦虑", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        
        # 验证响应格式（方案B）
        assert data["code"] == 200
        assert data["msg"] == "success"
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= 3
        
        # 验证每个结果项的字段完整性
        for item in data["results"]:
            assert "content" in item
            assert isinstance(item["content"], str)
            assert len(item["content"]) > 0
            assert "source" in item
            assert isinstance(item["source"], str)
            assert len(item["source"]) > 0
            assert "score" in item
            assert isinstance(item["score"], (int, float))
            assert 0 <= item["score"] <= 1

    def test_top_k_greater_than_total(self):
        """测试 top_k 大于知识库总数时，返回所有结果"""
        health_resp = client.get("/health")
        total = health_resp.json().get("doc_count", 100)
        
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "心理健康", "top_k": total + 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= total

    def test_empty_query(self):
        """测试空查询：返回标准错误格式"""
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "", "top_k": 3}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["code"] == 40001
        assert "查询内容不能为空" in data["msg"]
        assert data["data"] is None

    def test_query_with_question_mark(self):
        """
        测试带问号的查询：验证接口能正常处理并返回结果。
        不验证具体内容，只验证格式和状态码。
        """
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "什么是正念？", "top_k": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
        # 验证结果字段完整性（不验证内容）
        for item in data["results"]:
            assert "content" in item
            assert "source" in item
            assert "score" in item

    def test_invalid_top_k(self):
        """测试无效的 top_k（负数），应自动修正为默认值 3"""
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "焦虑", "top_k": -5}
        )
        assert response.status_code == 200
        data = response.json()
        # 服务应自动修正为默认值，返回最多 3 条
        assert len(data["results"]) <= 3

    def test_query_preprocessing(self):
        """
        测试查询预处理功能：接口能正常处理含'是什么'的问句。
        不验证具体内容，只验证返回格式正确。
        """
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "心流是什么？", "top_k": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
        # 验证每个结果项的字段完整性
        for item in data["results"]:
            assert "content" in item
            assert "source" in item
            assert "score" in item

    def test_source_extraction(self):
        """测试来源提取：验证 source 字段非空"""
        response = client.post(
            "/v1/knowledge/retrieve",
            json={"query": "焦虑", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        for item in data["results"]:
            assert item["source"] is not None
            assert len(item["source"]) > 0