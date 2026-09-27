"""日志体系测试：文件留痕、访问日志单条、级别分层、敏感信息不落日志。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings, settings
from app.core.logging import setup_logging


@pytest.fixture()
def isolated_root_logging() -> Iterator[None]:
    """隔离根日志器状态，避免用例之间互相影响。"""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            if handler not in saved_handlers:
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_setup_logging_writes_file_once(isolated_root_logging: None, tmp_path: Path) -> None:
    config = settings.model_copy(update={"log_dir": tmp_path, "log_file": "unit.log", "log_level": "INFO"})

    setup_logging(config)
    setup_logging(config)  # 重复调用（如 uvicorn --reload）：替换 handler 而不是叠加

    root = logging.getLogger()
    app_handlers = [handler for handler in root.handlers if getattr(handler, "_user_api_handler", False)]
    file_handlers = [handler for handler in app_handlers if isinstance(handler, RotatingFileHandler)]
    assert len(app_handlers) == 2  # 控制台 + 文件
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == tmp_path / "unit.log"

    logging.getLogger("app.demo").info("日志留痕内容")
    file_handlers[0].flush()

    content = (tmp_path / "unit.log").read_text(encoding="utf-8")
    assert "日志留痕内容" in content
    assert "app.demo" in content


def test_one_access_log_per_request_and_no_password(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.access"):
        response = client.post("/api/v1/users", json={"username": "alice", "password": "secret123"})

    assert response.status_code == 201
    lines = [record.getMessage() for record in caplog.records if record.name == "app.access"]
    assert len(lines) == 1  # 一个请求只记一条
    assert "method=POST" in lines[0]
    assert "path=/api/v1/users" in lines[0]
    assert "status=201" in lines[0]
    assert "secret123" not in caplog.text  # 不记录请求体，密码不会落日志


def test_docs_paths_are_not_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.access"):
        client.get("/openapi.json")

    assert [record for record in caplog.records if record.name == "app.access"] == []


def test_log_levels_by_category(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.errors"):
        client.get("/api/v1/users/999")  # 预期内业务异常 -> INFO
        client.post("/api/v1/users", json={"username": "alice", "password": "123"})  # 参数错误 -> WARNING

    levels = {record.getMessage().split()[0]: record.levelno for record in caplog.records if record.name == "app.errors"}
    assert levels["business"] == logging.INFO
    assert levels["param"] == logging.WARNING


def test_secrets_never_reach_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret_url = "mysql+pymysql://app:p%40ss%2Fword@127.0.0.1:3306/user_api?charset=utf8mb4"
    sample = Settings(mysql_url=secret_url)

    with caplog.at_level(logging.INFO):
        logging.getLogger("app.demo").info("settings=%r url=%s", sample, sample.mysql_url)

    assert "p%40ss" not in caplog.text
    assert "p@ss" not in caplog.text
    assert "**********" in caplog.text
    assert isinstance(sample.mysql_url, SecretStr)
