"""健康检查：探测服务存活与数据库连通性。

- `200`：服务存活，且数据库可连接（执行 `SELECT 1` 成功）
- `503`：数据库不可用或未配置

响应只给状态，具体原因写日志，避免对外暴露内部细节（与全局异常处理策略一致）。
探针会被高频调用，因此不记访问日志，失败时只留一条 WARNING、不打堆栈，避免刷爆日志。
"""

from __future__ import annotations

import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import Engine, text

from app.db.session import db_engine
from app.schemas.health import HealthReport

logger = logging.getLogger("app.health")

router = APIRouter(tags=["health"])

# 兜底掩码连接串中的账号密码，异常信息进日志前先脱敏
_CREDENTIALS_PATTERN = re.compile(r"://[^:@/\s]+:[^@/\s]+@")


@router.get(
    "/health",
    response_model=HealthReport,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthReport, "description": "数据库不可用"}},
    summary="健康检查（服务存活 + 数据库连通性）",
)
def health(
    response: Response,
    engine: Annotated[Engine | None, Depends(db_engine)],
) -> HealthReport:
    """数据库不可用时返回 503。"""
    if engine is None:
        return _unhealthy(response, "MYSQL_URL 未配置")

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - 健康检查需要兜住所有连接异常
        return _unhealthy(response, f"{type(exc).__name__}: {exc}")

    return HealthReport(status="ok", database="ok")


def _unhealthy(response: Response, reason: str) -> HealthReport:
    """统一降级：响应给状态，原因脱敏后只写日志。"""
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    logger.warning("health check failed: %s", _CREDENTIALS_PATTERN.sub("://***@", reason))
    return HealthReport(status="unhealthy", database="unreachable")
