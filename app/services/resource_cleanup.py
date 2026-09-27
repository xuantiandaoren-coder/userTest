"""过期资源清理：先删 SeaweedFS 对象，再删 MySQL 元数据。

由 app/main.py 在启动时拉起常驻协程，每天 03:00 执行一次。
先删对象的顺序保证：即使中途失败，元数据仍在，第二天会重试，不会留下无主对象。
"""

from __future__ import annotations

import logging
from datetime import datetime

from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.scheduler import run_daily
from app.core.seaweedfs import SeaweedFSClient, get_seaweedfs_client
from app.db.resource_repository import ResourceRepository
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)


async def cleanup_expired_resources(storage: SeaweedFSClient | None = None) -> int:
    """清理已到期资源，返回成功删除的条数。

    单条失败只记日志并跳过（保留元数据，次日重试），不影响其余资源。
    """
    storage = storage or get_seaweedfs_client()
    session = get_session_factory()()
    try:
        repository = ResourceRepository(session)
        expired = list(repository.list_expired(datetime.now()))
        deleted = 0
        for resource in expired:
            try:
                await run_in_threadpool(storage.delete_object, resource.storage_path)
            except Exception:
                logger.exception("删除 SeaweedFS 对象失败，保留元数据待重试：%s", resource.storage_path)
                continue
            repository.delete(resource)
            deleted += 1
        session.commit()
        logger.info("过期资源清理完成：到期 %d 条，已删除 %d 条", len(expired), deleted)
        return deleted
    finally:
        session.close()


async def run_cleanup_scheduler() -> None:
    """常驻协程：每天 03:00 跑一次过期清理。"""
    await run_daily(settings.resource_cleanup_hour, settings.resource_cleanup_minute, cleanup_expired_resources)
