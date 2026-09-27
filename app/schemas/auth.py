"""校验层：认证相关请求模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.security import validate_password_strength
from app.schemas.user import UserCreate


class RegisterRequest(UserCreate):
    """注册请求：字段与新增用户一致，额外做密码强度校验。"""

    @model_validator(mode="after")
    def _validate_password(self) -> RegisterRequest:
        """用户名与密码都在手，做“黑名单 / 字母数字 / 不含用户名”校验。"""
        validate_password_strength(self.password, self.username)
        return self


class LoginRequest(BaseModel):
    """登录请求体。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=2, max_length=20, description="用户名")
    password: str = Field(min_length=6, max_length=72, description="密码明文")


class RefreshRequest(BaseModel):
    """刷新令牌请求体。"""

    refresh_token: str = Field(min_length=1, description="登录时下发的 refresh token")


class TokenPair(BaseModel):
    """登录 / 刷新成功后的令牌对。"""

    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="访问令牌有效期（秒）")
