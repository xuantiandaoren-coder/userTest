"""冒烟测试：pytest + httpx 直连 ASGI 应用，覆盖健康检查与用户接口主链路。

运行：
    pytest tests/test_smoke.py -v

- 用 `httpx.ASGITransport` 在进程内调用应用，不需要启动服务、不依赖网络
- 数据库依赖被替换为内存 SQLite，不需要本地 MySQL
- 覆盖：服务存活 / 数据库连通性（含不可用时的 503）、用户增删改查、错误响应契约、密码不落明文
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import db_engine
from app.main import app
from app.services.user_service import pwd_context

USERS = "/api/v1/users"


@asynccontextmanager
async def use_engine(engine: Engine) -> AsyncIterator[None]:
    """临时替换健康检查使用的数据库引擎。"""
    app.dependency_overrides[db_engine] = lambda: engine
    try:
        yield
    finally:
        app.dependency_overrides.pop(db_engine, None)


@pytest.mark.anyio
async def test_health_ok(sqlite_engine: Engine, httpx_client: httpx.AsyncClient) -> None:
    """服务存活 + 数据库连通：200。"""
    async with use_engine(sqlite_engine):
        response = await httpx_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


@pytest.mark.anyio
async def test_health_returns_503_when_database_unreachable(httpx_client: httpx.AsyncClient) -> None:
    """数据库连不上：必须 503 而不是 500。"""
    broken_engine = create_engine("sqlite:////nonexistent-dir-for-smoke/db.sqlite")
    try:
        async with use_engine(broken_engine):
            response = await httpx_client.get("/health")
    finally:
        broken_engine.dispose()

    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "database": "unreachable"}


@pytest.mark.anyio
async def test_smoke_user_crud_flow(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    """用户增删改查主链路。"""
    created = await httpx_client.post(USERS, json={"username": "smoke", "password": "secret123"})
    assert created.status_code == 201
    user = created.json()
    assert {"id", "username"} <= set(user)  # 响应不泄露密码字段
    assert set(user) <= {"id", "username", "avatar", "avatar_url"}
    assert user["username"] == "smoke"

    listed = await httpx_client.get(USERS)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [user["id"]]

    detail = await httpx_client.get(f"{USERS}/{user['id']}")
    assert detail.status_code == 200
    assert detail.json() == user

    updated = await httpx_client.patch(f"{USERS}/{user['id']}", json={"username": "smoke2"})
    assert updated.status_code == 200
    assert updated.json() == {"id": user["id"], "username": "smoke2", "avatar": None, "avatar_url": None}

    deleted = await httpx_client.delete(f"{USERS}/{user['id']}")
    assert deleted.status_code == 204
    assert (await httpx_client.get(f"{USERS}/{user['id']}")).status_code == 404


@pytest.mark.anyio
async def test_smoke_error_response_contract(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    """错误响应保持 code/message/detail 契约。"""
    missing = await httpx_client.get(f"{USERS}/999")
    assert missing.status_code == 404
    assert missing.json()["code"] == "USER_NOT_FOUND"

    invalid = await httpx_client.post(USERS, json={"username": "a", "password": "1"})
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "PARAMETER_ERROR"

    assert (await httpx_client.post(USERS, json={"username": "dup", "password": "secret123"})).status_code == 201
    duplicate = await httpx_client.post(USERS, json={"username": "dup", "password": "secret456"})
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "USERNAME_ALREADY_EXISTS"


@pytest.mark.anyio
async def test_smoke_password_is_hashed(
    db_override: None, db_session: Session, httpx_client: httpx.AsyncClient
) -> None:
    """落库的是密码哈希，不是明文。"""
    response = await httpx_client.post(USERS, json={"username": "hashcheck", "password": "secret123"})

    assert response.status_code == 201
    row = db_session.get(User, response.json()["id"])
    assert row is not None
    assert row.password != "secret123"
    assert pwd_context.verify("secret123", row.password)


@pytest.mark.anyio
async def test_smoke_health_is_not_access_logged(
    sqlite_engine: Engine, caplog: pytest.LogCaptureFixture, httpx_client: httpx.AsyncClient
) -> None:
    """高频探针不写访问日志，避免刷日志。"""
    with caplog.at_level(logging.INFO, logger="app.access"):
        async with use_engine(sqlite_engine):
            await httpx_client.get("/health")

    assert [record for record in caplog.records if record.name == "app.access"] == []


def test_smoke_uses_real_httpx() -> None:
    """确保冒烟测试走的是 httpx（而不是环境里那个 httpx2 兼容包）。"""
    assert Path(httpx.__file__).parent.name == "httpx"
    assert hasattr(httpx, "ASGITransport")
