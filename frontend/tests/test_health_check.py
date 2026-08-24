"""
测试3: 健康检查解析测试
验证健康检查能兼容 services 和 dependencies 两种返回格式
"""

from unittest.mock import patch, MagicMock
import json


def parse_health_response(health_json):
    """从健康检查 JSON 中提取服务状态，兼容 services 和 dependencies"""
    return health_json.get("services", health_json.get("dependencies", {}))


def test_health_with_services_key():
    """Agent 返回 services 键时能正确解析"""
    mock_health = {
        "status": "ok",
        "service": "agent-service",
        "services": {
            "vision_service": "online",
            "knowledge_service": "online",
        },
    }
    deps = parse_health_response(mock_health)
    assert deps.get("vision_service") == "online"
    assert deps.get("knowledge_service") == "online"
    print("✅ health_with_services_key passed")


def test_health_with_dependencies_key():
    """旧版返回 dependencies 键时也能正确解析（向后兼容）"""
    mock_health = {
        "status": "ok",
        "service": "agent-service",
        "dependencies": {
            "vision_service": "online",
            "knowledge_service": "offline",
        },
    }
    deps = parse_health_response(mock_health)
    assert deps.get("vision_service") == "online"
    assert deps.get("knowledge_service") == "offline"
    print("✅ health_with_dependencies_key passed")


def test_health_offline_status():
    """服务离线时能正确检测"""
    mock_health = {
        "status": "ok",
        "services": {
            "vision_service": "offline",
            "knowledge_service": "offline",
        },
    }
    deps = parse_health_response(mock_health)
    assert deps.get("vision_service") != "online"
    assert deps.get("knowledge_service") != "online"
    print("✅ health_offline_status passed")


def test_health_empty_response():
    """空返回时不会崩溃"""
    deps = parse_health_response({})
    assert deps == {}
    print("✅ health_empty_response passed")


if __name__ == "__main__":
    test_health_with_services_key()
    test_health_with_dependencies_key()
    test_health_offline_status()
    test_health_empty_response()
    print("\n🎉 All health check tests passed!")
