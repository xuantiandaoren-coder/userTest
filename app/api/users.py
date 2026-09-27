"""路由层：用户信息增删改查 HTTP 接口。

接口只负责参数透传与响应，业务异常由全局处理器
(app/core/handlers.py) 统一转换为 code/message/detail 响应。
"""

from fastapi import APIRouter, status

from app.api.deps import ServiceDep
from app.schemas.user import UserCreate, UserPublic, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, service: ServiceDep) -> UserPublic:
    """新增用户。"""
    return service.create_user(payload)


@router.get("", response_model=list[UserPublic])
def list_users(service: ServiceDep) -> list[UserPublic]:
    """查询全部用户。"""
    return service.list_users()


@router.get("/{user_id}", response_model=UserPublic)
def get_user(user_id: int, service: ServiceDep) -> UserPublic:
    """按 id 查询单个用户。"""
    return service.get_user(user_id)


@router.patch("/{user_id}", response_model=UserPublic)
def update_user(user_id: int, payload: UserUpdate, service: ServiceDep) -> UserPublic:
    """部分更新用户（用户名/密码），未提交的字段保持不变。"""
    return service.update_user(user_id, payload)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, service: ServiceDep) -> None:
    """删除用户。"""
    service.delete_user(user_id)
