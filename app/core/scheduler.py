"""轻量定时器：不引入额外调度依赖，按「每天固定时刻」循环执行任务。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def seconds_until(hour: int, minute: int, *, now: datetime | None = None) -> float:
    """距下一个 hour:minute 还有多少秒（已过今天的点则算到明天）。"""
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def run_daily(hour: int, minute: int, job: Callable[[], Awaitable[None]]) -> None:
    """常驻协程：每天 hour:minute 执行一次 job。

    job 抛异常时只记日志、不退出循环，避免一次失败让定时任务永久停摆；
    协程被取消（应用关闭）时正常向上抛出。
    """
    while True:
        await asyncio.sleep(seconds_until(hour, minute))
        try:
            await job()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("定时任务执行失败")
