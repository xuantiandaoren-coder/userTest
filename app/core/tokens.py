"""JWT 令牌：签发与校验（access / refresh 两类）。

- 密钥、算法、有效期全部来自配置层（环境变量 / `.env`），代码里不写死任何密钥
- access token 短命，用于访问受保护接口；refresh token 长命，只用于换新的 access token
- 过期与无效分成两个错误码，客户端据此决定“自动刷新”还是“重新登录”
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

import jwt

from app.core.config import settings
from app.core.exceptions import BusinessError


class TokenType(str, Enum):
    """令牌类型，写进 payload 的 type 字段，防止两种令牌混用。"""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenExpiredError(BusinessError):
    """业务异常：令牌过期（客户端据此触发自动刷新）。"""

    code = "TOKEN_EXPIRED"
    http_status = 401
    message = "登录凭证已过期，请刷新或重新登录"


class TokenInvalidError(BusinessError):
    """业务异常：令牌无效（签名不对、类型不符、用户不存在等）。"""

    code = "TOKEN_INVALID"
    http_status = 401
    message = "登录凭证无效，请重新登录"


def create_access_token(user_id: int, *, expires_in: int | None = None) -> str:
    """签发访问令牌。"""
    ttl = settings.jwt_access_token_expire_seconds if expires_in is None else expires_in
    return _encode(user_id, TokenType.ACCESS, ttl)


def create_refresh_token(user_id: int, *, expires_in: int | None = None) -> str:
    """签发刷新令牌。"""
    ttl = settings.jwt_refresh_token_expire_seconds if expires_in is None else expires_in
    return _encode(user_id, TokenType.REFRESH, ttl)


def decode_token(token: str, *, expected_type: TokenType) -> int:
    """校验令牌并返回 user_id；过期 / 无效分别抛对应业务异常。"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            # 固定允许的算法，避免 alg 混淆攻击（如把 HS256 换成 none）
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError() from exc
    except jwt.PyJWTError as exc:
        raise TokenInvalidError() from exc

    if payload.get("type") != expected_type.value:
        raise TokenInvalidError(detail=f"expected_type={expected_type.value} actual_type={payload.get('type')}")

    try:
        return int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenInvalidError(detail="invalid subject") from exc


def _encode(user_id: int, token_type: TokenType, expires_in: int) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user_id),  # JWT 规范要求 sub 为字符串
        "type": token_type.value,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    return jwt.encode(payload, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)
