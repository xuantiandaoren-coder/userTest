"""注册接口测试：密码强度校验、bcrypt 哈希、用户名查重与限流。"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.rate_limit import FixedWindowRateLimiter, register_limiter
from app.db.models import User
from app.services.user_service import pwd_context

REGISTER = "/auth/register"
REGISTER_VERSIONED = "/api/v1/auth/register"

# 合规密码：字母 + 数字、不在黑名单、不含测试用户名
STRONG_PASSWORD = "HZq7mK2p"


def test_register_hashes_password(client: TestClient, db_session: Session) -> None:
    response = client.post(REGISTER, json={"username": "newbie", "password": STRONG_PASSWORD})

    assert response.status_code == 201
    body = response.json()
    assert {"id", "username"} <= set(body)  # 响应不泄露密码字段
    assert set(body) <= {"id", "username", "avatar", "avatar_url"}

    row = db_session.get(User, body["id"])
    assert row is not None
    assert row.user_name == "newbie"
    assert row.password != STRONG_PASSWORD  # 明文不入库
    assert pwd_context.verify(STRONG_PASSWORD, row.password)  # 落库的是 bcrypt 哈希


def test_register_is_available_under_versioned_prefix(client: TestClient) -> None:
    response = client.post(REGISTER_VERSIONED, json={"username": "newbie", "password": STRONG_PASSWORD})

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("username", "password", "reason"),
    [
        ("newbie", "abc123", "命中弱口令黑名单"),
        ("newbie", "qwerty123", "命中弱口令黑名单"),
        ("newbie", "abcdefgh", "缺少数字"),
        ("newbie", "3141592653", "缺少字母"),
        ("alice", "alice2026x", "包含用户名"),
    ],
)
def test_register_rejects_weak_password(client: TestClient, username: str, password: str, reason: str) -> None:
    response = client.post(REGISTER, json={"username": username, "password": password})

    assert response.status_code == 422, reason
    body = response.json()
    assert body["code"] == "PARAMETER_ERROR"
    assert body["detail"]  # 具体原因回给调用方，便于前端提示


def test_register_duplicate_username_returns_conflict(client: TestClient) -> None:
    assert client.post(REGISTER, json={"username": "dup", "password": STRONG_PASSWORD}).status_code == 201

    duplicate = client.post(REGISTER, json={"username": "dup", "password": STRONG_PASSWORD.replace("7", "9")})

    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "USERNAME_ALREADY_EXISTS"


def test_register_is_rate_limited_per_ip(client: TestClient) -> None:
    for index in range(5):  # 配额 5 次/分钟
        created = client.post(REGISTER, json={"username": f"limited{index}", "password": STRONG_PASSWORD})
        assert created.status_code == 201

    blocked = client.post(REGISTER, json={"username": "limited9", "password": STRONG_PASSWORD})

    assert blocked.status_code == 429
    assert blocked.json()["code"] == "RATE_LIMITED"
    assert "limit=5/60s" in blocked.json()["detail"]


def test_register_not_blocked_after_window(client: TestClient) -> None:
    """窗口过期后配额恢复：直接把窗口起点往前推，避免用例里 sleep。"""
    register_limiter.reset()
    for index in range(5):
        assert client.post(REGISTER, json={"username": f"win{index}", "password": STRONG_PASSWORD}).status_code == 201
    assert client.post(REGISTER, json={"username": "win9", "password": STRONG_PASSWORD}).status_code == 429

    register_limiter.expire_windows()

    assert client.post(REGISTER, json={"username": "win10", "password": STRONG_PASSWORD}).status_code == 201


def test_rest_of_api_is_not_rate_limited(client: TestClient) -> None:
    """限流只作用于注册接口，其他接口不受影响。"""
    for _ in range(6):
        assert client.get("/api/v1/users").status_code == 200


def test_fixed_window_limiter_counts_per_key() -> None:
    """固定窗口计数按 key（IP）隔离，超限后窗口内不再放行。"""
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.hit("203.0.113.7", now=0) is True
    assert limiter.hit("203.0.113.7", now=1) is True
    assert limiter.hit("203.0.113.7", now=2) is False  # 同一 IP 超出配额
    assert limiter.hit("198.51.100.9", now=2) is True  # 其他 IP 不受影响
