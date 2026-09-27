"""统一错误处理测试：中文可读、标准响应结构、后端有日志可定位。"""

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.handlers import register_exception_handlers


def _assert_standard_error(response, status_code: int, code: str, message: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert set(body) <= {"code", "message", "detail"}
    assert body["code"] == code
    assert body["message"] == message
    return body


def test_business_error_user_not_found(client: TestClient) -> None:
    body = _assert_standard_error(
        client.get("/api/v1/users/999"),
        status_code=404,
        code="USER_NOT_FOUND",
        message="用户不存在",
    )
    assert body["detail"] == "user_id=999"


def test_business_error_duplicate_username(client: TestClient) -> None:
    client.post("/api/v1/users", json={"username": "alice", "password": "secret123"})
    duplicate = client.post("/api/v1/users", json={"username": "alice", "password": "secret456"})
    body = _assert_standard_error(
        duplicate,
        status_code=409,
        code="USERNAME_ALREADY_EXISTS",
        message="用户名已存在",
    )
    assert "username='alice'" in body["detail"]


def test_validation_error_returns_chinese_message(client: TestClient) -> None:
    response = client.post("/api/v1/users", json={"username": "alice", "password": "123"})
    body = _assert_standard_error(
        response,
        status_code=422,
        code="PARAMETER_ERROR",
        message="请求参数校验失败",
    )
    assert body["detail"]


def test_unmatched_route_returns_standardized_404(client: TestClient) -> None:
    response = client.get("/api/v1/no-such-endpoint")
    body = _assert_standard_error(response, status_code=404, code="NOT_FOUND", message="请求的资源不存在")
    assert "path=/api/v1/no-such-endpoint" in body["detail"]


def test_unhandled_error_returns_500_and_logs_traceback(caplog) -> None:
    system_app = FastAPI()
    register_exception_handlers(system_app)

    @system_app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom-detail")

    with caplog.at_level(logging.ERROR):
        with TestClient(system_app, raise_server_exceptions=False) as system_client:
            response = system_client.get("/boom")

    body = _assert_standard_error(
        response,
        status_code=500,
        code="SYSTEM_ERROR",
        message="系统繁忙，请稍后重试",
    )
    assert "detail" not in body  # 响应不暴露内部细节

    # 后端日志必须保留原始异常信息，便于定位
    records = [record for record in caplog.records if record.name == "app.errors"]
    assert any(
        record.exc_info is not None
        and isinstance(record.exc_info[1], RuntimeError)
        and "boom-detail" in repr(record.exc_info[1])
        for record in records
    )
