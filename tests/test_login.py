"""登录与 JWT 鉴权测试：令牌签发、受保护接口、刷新与自动重试。"""

from __future__ import annotations

import httpx
import jwt
import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.clients import RefreshableClient, RefreshTokenRejectedError
from app.core.config import settings
from app.core.tokens import TokenType, create_access_token, decode_token
from app.db.models import User
from app.services.user_service import pwd_context

REGISTER = "/auth/register"
LOGIN = "/auth/login"
REFRESH = "/auth/refresh"
ME = "/auth/me"

USERNAME = "login_user"
PASSWORD = "HZq7mK2p"


async def _register_and_login(client: httpx.AsyncClient) -> dict:
    """建一个用户并登录，返回令牌对。"""
    created = await client.post(REGISTER, json={"username": USERNAME, "password": PASSWORD})
    assert created.status_code == 201
    response = await client.post(LOGIN, json={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200
    return response.json()


@pytest.mark.anyio
async def test_login_returns_token_pair(db_override: None, db_session: Session, httpx_client: httpx.AsyncClient) -> None:
    tokens = await _register_and_login(httpx_client)

    assert set(tokens) == {"access_token", "refresh_token", "token_type", "expires_in"}
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == settings.jwt_access_token_expire_seconds

    row = db_session.get(User, decode_token(tokens["access_token"], expected_type=TokenType.ACCESS))
    assert row is not None and row.user_name == USERNAME
    # 两种令牌类型不同，不能混用
    assert decode_token(tokens["refresh_token"], expected_type=TokenType.REFRESH) == row.id


@pytest.mark.anyio
async def test_login_rejects_bad_credentials_without_enumerating_users(
    db_override: None, httpx_client: httpx.AsyncClient
) -> None:
    await _register_and_login(httpx_client)

    wrong_password = await httpx_client.post(LOGIN, json={"username": USERNAME, "password": "WrongPwd123"})
    unknown_user = await httpx_client.post(LOGIN, json={"username": "nobody", "password": PASSWORD})

    # 密码错误与用户不存在必须给出完全一致的 code / message，避免被枚举用户名
    for response in (wrong_password, unknown_user):
        assert response.status_code == 401
        assert response.json()["code"] == "INVALID_CREDENTIALS"
        assert response.json()["message"] == "用户名或密码错误"


@pytest.mark.anyio
async def test_login_is_rate_limited(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    assert (await httpx_client.post(REGISTER, json={"username": USERNAME, "password": PASSWORD})).status_code == 201

    for _ in range(settings.login_rate_limit):
        assert (await httpx_client.post(LOGIN, json={"username": USERNAME, "password": PASSWORD})).status_code == 200

    blocked = await httpx_client.post(LOGIN, json={"username": USERNAME, "password": PASSWORD})
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "RATE_LIMITED"


@pytest.mark.anyio
async def test_me_requires_valid_access_token(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    tokens = await _register_and_login(httpx_client)

    no_token = await httpx_client.get(ME)
    assert no_token.status_code == 401
    assert no_token.json()["code"] == "TOKEN_INVALID"

    ok = await httpx_client.get(ME, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert ok.status_code == 200
    assert ok.json()["username"] == USERNAME

    # 刷新令牌不能当访问令牌用（type 校验）
    wrong_type = await httpx_client.get(ME, headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert wrong_type.status_code == 401
    assert wrong_type.json()["code"] == "TOKEN_INVALID"


@pytest.mark.anyio
async def test_me_reports_expired_access_token(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    tokens = await _register_and_login(httpx_client)
    user_id = decode_token(tokens["access_token"], expected_type=TokenType.ACCESS)

    expired = create_access_token(user_id, expires_in=-1)  # 已过期
    response = await httpx_client.get(ME, headers={"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_EXPIRED"  # 客户端据此触发自动刷新


@pytest.mark.anyio
async def test_token_signed_with_other_secret_is_rejected(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    tokens = await _register_and_login(httpx_client)
    user_id = decode_token(tokens["access_token"], expected_type=TokenType.ACCESS)

    forged = jwt.encode(
        {"sub": str(user_id), "type": "access", "exp": 9999999999},
        "attacker-secret-key-0123456789abcdef",
        algorithm="HS256",
    )
    response = await httpx_client.get(ME, headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_INVALID"


@pytest.mark.anyio
async def test_refresh_issues_new_access_token(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    tokens = await _register_and_login(httpx_client)

    refreshed = await httpx_client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})

    assert refreshed.status_code == 200
    new_access = refreshed.json()["access_token"]
    me = await httpx_client.get(ME, headers={"Authorization": f"Bearer {new_access}"})
    assert me.status_code == 200
    assert me.json()["username"] == USERNAME

    # 访问令牌不能用来刷新
    rejected = await httpx_client.post(REFRESH, json={"refresh_token": tokens["access_token"]})
    assert rejected.status_code == 401
    assert rejected.json()["code"] == "TOKEN_INVALID"


@pytest.mark.anyio
async def test_auto_refresh_retries_original_request(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    """核心场景：access 过期 -> 自动刷新 -> 重放原请求，调用方无感知。"""
    tokens = await _register_and_login(httpx_client)
    user_id = decode_token(tokens["access_token"], expected_type=TokenType.ACCESS)

    api = RefreshableClient(httpx_client)
    api.set_tokens(access_token=create_access_token(user_id, expires_in=-1), refresh_token=tokens["refresh_token"])

    response = await api.get(ME)  # 第一次 401 TOKEN_EXPIRED，客户端自动刷新后重试

    assert response.status_code == 200
    assert response.json()["username"] == USERNAME
    refreshed_token = api.access_token
    assert refreshed_token is not None
    assert decode_token(refreshed_token, expected_type=TokenType.ACCESS) == user_id
    # 新的访问令牌确实生效：再请求一次不需要再刷新
    assert (await api.get(ME)).status_code == 200


@pytest.mark.anyio
async def test_auto_refresh_raises_when_refresh_token_is_invalid(
    db_override: None, httpx_client: httpx.AsyncClient
) -> None:
    """刷新令牌也不可用时，明确报错让调用方重新登录，而不是无限重试。"""
    api = RefreshableClient(httpx_client)
    api.set_tokens(
        access_token=create_access_token(1, expires_in=-1),  # 过期 -> 触发刷新
        refresh_token="not-a-refresh-token",                 # 刷新令牌也无效
    )

    with pytest.raises(RefreshTokenRejectedError):
        await api.get(ME)

