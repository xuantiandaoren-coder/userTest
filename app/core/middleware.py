"""中间件：统一记录访问日志。

约定：每个请求只记一条日志，只包含方法、路径、状态码、耗时、客户端地址；
不记录请求体与查询串，避免密码等敏感信息落进日志文件。
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response

from app.core.config import settings

logger = logging.getLogger("app.access")

# 文档、静态资源、健康探针等非业务请求不记访问日志，避免噪音
_QUIET_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/favicon.ico", "/health"})


def register_request_logging(app: FastAPI) -> None:
    """按配置注册访问日志中间件（log_access=false 时整体关闭）。"""
    if not settings.log_access:
        return

    @app.middleware("http")
    async def log_request(request: Request, call_next) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 异常堆栈由全局异常处理器记录，这里只留一条访问痕迹，避免堆栈重复打印
            _log(request, status_code=500, start=start)
            raise
        _log(request, status_code=response.status_code, start=start)
        return response


def _log(request: Request, *, status_code: int, start: float) -> None:
    path = request.url.path
    if path in _QUIET_PATHS:
        return
    cost_ms = (time.perf_counter() - start) * 1000
    client = request.client.host if request.client else "-"
    logger.info(
        "method=%s path=%s status=%s cost=%.1fms client=%s",
        request.method,
        path,
        status_code,
        cost_ms,
        client,
    )
