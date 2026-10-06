"""FastAPI 应用入口：创建应用实例并挂载各层路由。"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.files import router as files_router
from app.api.health import router as health_router
from app.api.interviews import router as interviews_router
from app.api.learning_workflows import router as learning_workflows_router
from app.api.prompt import router as prompt_router
from app.api.router import api_router
from app.api.sessions import router as sessions_router
from app.core.config import settings
from app.core.handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import register_request_logging
from app.core.redis_client import get_redis_client, reset_redis_client
from app.db.prompt_template_repository import PromptTemplateRepository
from app.db.session import dispose_engine
from app.db.session import get_session_factory
from app.prompts.prompt_template_manager import PromptTemplateManager
from app.services.resource_cleanup import run_cleanup_scheduler

logger = logging.getLogger("app.main")


def warm_prompt_cache() -> int:
    """启动加载：把生效的提示词模板灌进 Redis（任何失败都只记日志，不影响启动）。"""
    session = None
    try:
        session = get_session_factory()()
        manager = PromptTemplateManager(PromptTemplateRepository(session), get_redis_client())
        return manager.warm_cache()
    except Exception as exc:
        logger.warning("提示词模板预热失败（服务继续启动，读路径回退数据库）：%r", exc)
        return 0
    finally:
        if session is not None:
            session.close()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """启动过期资源清理定时器；关闭时取消定时器并释放数据库连接池。"""
    cleanup_task = asyncio.create_task(run_cleanup_scheduler())
    # 预热模板缓存放到后台线程：数据库 / Redis 不可用时只影响预热，不拖慢（或卡住）启动
    warm_task = (
        asyncio.create_task(asyncio.to_thread(warm_prompt_cache))
        if settings.prompt_cache_warmup
        else None
    )
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        if warm_task is not None:
            warm_task.cancel()
            with suppress(asyncio.CancelledError):
                await warm_task
        reset_redis_client()
        dispose_engine()


def create_app() -> FastAPI:
    setup_logging()
    application = FastAPI(
        title="User API",
        description="基于 FastAPI + SQLAlchemy 2.0 + MySQL 的用户信息增删改查接口",
        version="0.1.0",
        lifespan=lifespan,
        # 生产环境默认关闭 /docs /redoc /openapi.json，避免暴露内部接口清单
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    register_exception_handlers(application)
    register_request_logging(application)
    application.include_router(api_router, prefix="/api/v1")
    # 探针接口不带版本前缀：GET /health
    application.include_router(health_router)
    # 注册接口：POST /auth/register（同时兼容带版本前缀的 /api/v1/auth/register）
    application.include_router(auth_router)
    application.include_router(auth_router, prefix="/api/v1", include_in_schema=False)
    # 文件上传：POST /files/upload（同样兼容 /api/v1/files/upload）
    application.include_router(files_router)
    application.include_router(files_router, prefix="/api/v1", include_in_schema=False)
    # 会话 / 消息 / 面试：/sessions、/interviews（同样兼容 /api/v1 前缀）
    application.include_router(sessions_router)
    application.include_router(sessions_router, prefix="/api/v1", include_in_schema=False)
    application.include_router(interviews_router)
    application.include_router(interviews_router, prefix="/api/v1", include_in_schema=False)
    # 提示词模板版本管理：/prompt/templates、/prompt/rollback、/prompt/config/agents
    application.include_router(prompt_router)
    application.include_router(prompt_router, prefix="/api/v1", include_in_schema=False)
    # 学习测评工作流：/learning-workflows/start、/learning-workflows/{run_id}/submit、GET /learning-workflows/{run_id}
    application.include_router(learning_workflows_router)
    application.include_router(learning_workflows_router, prefix="/api/v1", include_in_schema=False)

    @application.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        info = {"message": "User API is running"}
        if settings.docs_enabled:
            info["docs"] = "/docs"
        return info

    return application


app = create_app()
