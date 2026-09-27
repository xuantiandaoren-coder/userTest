"""校验层：通用响应模型（错误响应 + 分页壳）。"""

from typing import Generic, TypeVar

from pydantic import BaseModel, computed_field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """统一错误响应：code 供程序判断，message 面向用户（中文），detail 用于定位。"""

    code: str
    message: str
    detail: str | None = None


class Page(BaseModel, Generic[T]):
    """分页响应：items 为当前页数据，其余字段供前端渲染分页器。"""

    items: list[T]
    total: int              # 满足条件的总条数
    page: int               # 当前页码（从 1 开始）
    page_size: int          # 每页条数

    @computed_field
    @property
    def pages(self) -> int:
        """总页数（至少 1，便于前端直接渲染）。"""
        return max(1, -(-self.total // self.page_size))
