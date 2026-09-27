"""数据库层：引擎与会话管理。

引擎和会话工厂都惰性创建：导入本模块时不会连接数据库，
因此本地未配置 MYSQL_URL 时应用仍可正常启动，测试也能替换为 SQLite。
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """创建全局引擎（连接池 + 连接探活）。"""
    url = settings.sqlalchemy_url
    kwargs: dict[str, object] = {
        "echo": settings.db_echo,
        "pool_pre_ping": True,  # 使用前探活，避免连接被 MySQL 断开后报错
        "pool_recycle": settings.db_pool_recycle,
    }
    if not url.startswith("sqlite"):
        # SQLite（仅用于本地冒烟/测试）不支持连接池尺寸参数
        kwargs["pool_size"] = settings.db_pool_size
        kwargs["max_overflow"] = settings.db_max_overflow
        # 连接超时：数据库不可达时快速失败，健康检查不会被拖住
        kwargs["connect_args"] = {"connect_timeout": settings.db_connect_timeout}
    return create_engine(url, **kwargs)


def db_engine() -> Engine | None:
    """FastAPI 依赖：返回数据库引擎，连接串未配置时返回 None。

    健康检查据此把“未配置”和“连不上”都降级成 503，而不是 500。
    """
    try:
        return get_engine()
    except RuntimeError:
        return None


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """创建会话工厂；expire_on_commit=False 让提交后的对象仍可读取属性。"""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个会话，正常结束提交，异常回滚。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """释放连接池（应用关闭时调用）。"""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
        get_engine.cache_clear()
