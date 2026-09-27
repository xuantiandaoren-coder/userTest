"""固定窗口限流：按维度计数，用于注册这类高成本接口。

固定窗口：为每个 key 记录「窗口起点 + 计数」，计数超过上限即拒绝，
窗口过期后重新计数。比滑动窗口简单，代价是窗口边界处可能瞬时放过 2 倍请求。

单进程内存实现，够用且无额外依赖；多 worker / 多副本部署时需要换成 Redis 等
共享存储，否则每个进程各算一份配额。

IP 取 `request.client.host`。部署在反向代理后应让 uvicorn 信任代理头
（`--proxy-headers --forwarded-allow-ips=...`），否则所有请求会共用代理 IP 的配额。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock

from fastapi import Request

from app.core.config import settings
from app.core.exceptions import BusinessError


class RateLimitedError(BusinessError):
    """业务异常：触发限流。"""

    code = "RATE_LIMITED"
    http_status = 429
    message = "操作过于频繁，请稍后再试"


class FixedWindowRateLimiter:
    """固定窗口计数器。"""

    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = Lock()

    def hit(self, key: str, now: float | None = None) -> bool:
        """记一次访问：未超限返回 True，超限返回 False。"""
        current = time.monotonic() if now is None else now
        with self._lock:
            window_start, count = self._hits.get(key, (current, 0))
            if current - window_start >= self.window_seconds:
                window_start, count = current, 0
            count += 1
            self._hits[key] = (window_start, count)

            # 防止 key 无限增长：超量时清掉已过期窗口
            if len(self._hits) > self.max_keys:
                self._hits = {k: v for k, v in self._hits.items() if current - v[0] < self.window_seconds}
            return count <= self.limit

    def reset(self) -> None:
        """清空计数（测试用）。"""
        with self._lock:
            self._hits.clear()

    def expire_windows(self, seconds: float | None = None) -> None:
        """把已有窗口整体前移，便于测试“窗口过期后配额恢复”，避免用例里 sleep。"""
        offset = self.window_seconds if seconds is None else seconds
        with self._lock:
            self._hits = {key: (start - offset, count) for key, (start, count) in self._hits.items()}


# 注册接口配额：默认 5 次 / 分钟 / IP，可通过环境变量调整
register_limiter = FixedWindowRateLimiter(settings.register_rate_limit, settings.register_rate_window)

# 登录接口配额：默认 10 次 / 分钟 / IP，用于缓解口令爆破
login_limiter = FixedWindowRateLimiter(settings.login_rate_limit, settings.login_rate_window)


def rate_limit(limiter: FixedWindowRateLimiter) -> Callable[[Request], None]:
    """生成限流依赖；同一个 limiter 实例的各路由共享计数。"""

    def dependency(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.hit(client_ip):
            raise RateLimitedError(detail=f"ip={client_ip} limit={limiter.limit}/{limiter.window_seconds:g}s")

    return dependency
