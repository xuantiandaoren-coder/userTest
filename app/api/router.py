"""路由汇总：把所有模块路由挂载到统一的路由器上。"""

from fastapi import APIRouter

from app.api import users

api_router = APIRouter()
api_router.include_router(users.router)
