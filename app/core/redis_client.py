"""Redis 客户端（提示词模板缓存用）。

设计要点：
- **惰性连接**：`redis.Redis` 在第一条命令时才真正建连，导入 / 构造本模块不会阻塞启动；
- **失败可降级**：Redis 不可用时只记 WARNING 并返回 None，业务自动回退查库，
  缓存不属于「必须可用」的依赖，不能因为它让聊天接口整体不可用；
- **超时短**：socket 超时默认 2 秒，避免 Redis 挂掉时把请求拖慢。

提示词模板缓存在 Redis 中是一个 Hash：key = `settings.prompt_cache_key`
（默认 `prompt:templates:active`），field = `{agent}:{scene}`（公共模板为 `__common__:{scene}`）。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Protocol

import redis

from app.core.config import settings

logger = logging.getLogger("app.redis")


class RedisLike(Protocol):
    """本模块用到的 Redis 命令子集，便于测试注入内存实现。"""

    def ping(self) -> Any: ...

    def hget(self, name: str, key: str) -> Any: ...

    def hgetall(self, name: str) -> Any: ...

    def hset(self, name: str, key: str | None = None, value: str | None = None, mapping: dict[str, str] | None = None) -> Any: ...

    def hdel(self, name: str, *keys: str) -> Any: ...

    def delete(self, *names: str) -> Any: ...

    def expire(self, name: str, time: int) -> Any: ...


def build_redis_client() -> redis.Redis | None:
    """按配置创建 Redis 客户端；未启用返回 None（调用方据此直接走数据库）。"""
    if not settings.redis_enabled:
        logger.info("Redis 未启用（REDIS_ENABLED=false），提示词模板直接从数据库读取")
        return None

    url = settings.redis_url.get_secret_value().strip()
    kwargs: dict[str, Any] = {
        "decode_responses": True,          # 模板是文本，避免到处 decode
        "socket_connect_timeout": settings.redis_socket_timeout,
        "socket_timeout": settings.redis_socket_timeout,
    }
    if url:
        return redis.Redis.from_url(url, **kwargs)

    password = settings.redis_password.get_secret_value()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=password or None,
        **kwargs,
    )


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis | None:
    """进程级单例（连接池由 redis-py 内部维护）。"""
    return build_redis_client()


def reset_redis_client() -> None:
    """丢弃单例并关闭连接池（应用关闭 / 测试切换时调用）。"""
    if get_redis_client.cache_info().currsize:
        client = get_redis_client()
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - 关闭失败不影响退出
                logger.warning("关闭 Redis 连接池失败", exc_info=True)
        get_redis_client.cache_clear()


def redis_healthy(client: RedisLike | None) -> bool:
    """探活：Redis 不可用只记日志，不抛异常。"""
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception as exc:
        logger.warning("Redis 不可用，提示词模板回退数据库：%r", exc)
        return False
