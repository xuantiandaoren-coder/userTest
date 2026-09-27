"""校验层：用户相关的 Pydantic v2 模型。"""

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    """新增用户请求体：password 为明文，仅在业务层用于生成哈希。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=2, max_length=20, description="用户名")
    password: str = Field(min_length=6, max_length=72, description="密码明文")


class UserUpdate(BaseModel):
    """部分更新用户请求体：所有字段均可选。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str | None = Field(default=None, min_length=2, max_length=20, description="新用户名")
    password: str | None = Field(default=None, min_length=6, max_length=72, description="新密码明文")


class UserPublic(BaseModel):
    """对外响应模型：只暴露 id / username / avatar / avatar_url，绝不返回密码相关字段。

    - avatar：存储键（SeaweedFS 对象键），需要自己拼地址时用
    - avatar_url：可直接访问的完整地址（CDN 地址或预签名 URL），前端 `<img src>` 直接用
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    avatar: str | None = None
    avatar_url: str | None = None
