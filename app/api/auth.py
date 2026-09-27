"""认证路由：注册、登录、刷新令牌与受保护的当前用户接口。

密码校验、bcrypt 哈希等业务规则复用 UserService，路由层只做参数校验、令牌签发与限流。
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import CurrentUserDep, ServiceDep
from app.core.config import settings
from app.core.rate_limit import login_limiter, rate_limit, register_limiter
from app.core.tokens import TokenInvalidError, TokenType, create_access_token, create_refresh_token, decode_token
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenPair
from app.schemas.user import UserPublic
from app.services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(register_limiter))],
    summary="用户注册（密码强度校验 + 限流）",
)
def register(payload: RegisterRequest, service: ServiceDep) -> UserPublic:
    """注册新用户：密码经 bcrypt 哈希后落库，明文不入库。"""
    return service.create_user(payload)


@router.post(
    "/login",
    response_model=TokenPair,
    dependencies=[Depends(rate_limit(login_limiter))],
    summary="登录（签发 access / refresh 令牌）",
)
def login(payload: LoginRequest, service: ServiceDep) -> TokenPair:
    """校验用户名密码，成功后签发访问令牌与刷新令牌。"""
    user = service.authenticate(payload.username, payload.password)
    return _issue_tokens(user.id)


@router.post("/refresh", response_model=TokenPair, summary="用刷新令牌换取新的访问令牌")
def refresh(payload: RefreshRequest, service: ServiceDep) -> TokenPair:
    """校验刷新令牌的类型与有效性，签发新的令牌对。"""
    user_id = decode_token(payload.refresh_token, expected_type=TokenType.REFRESH)
    if not service.user_exists(user_id):
        raise TokenInvalidError(detail=f"user_id={user_id}")
    return _issue_tokens(user_id)


@router.get("/me", response_model=UserPublic, summary="当前登录用户（受保护接口）")
def me(current_user: CurrentUserDep, service: ServiceDep) -> UserPublic:
    """受 `Authorization: Bearer <access token>` 保护的接口示例。"""
    return service.to_public(current_user)


def _issue_tokens(user_id: int) -> TokenPair:
    """签发令牌对；有效期来自配置层。"""
    return TokenPair(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
        expires_in=settings.jwt_access_token_expire_seconds,
    )
