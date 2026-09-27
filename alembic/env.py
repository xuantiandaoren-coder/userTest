"""Alembic 运行环境：连接串取自应用配置，元数据取自 ORM 模型。

连接串默认来自 app.core.config.settings（.env / 环境变量 MYSQL_URL），
alembic.ini 的 sqlalchemy.url 留空即可；显式填写时以 ini 为准。

注意：这里刻意不使用 config.set_main_option() / engine_from_config()。
alembic.ini 底层是 ConfigParser，默认会对 % 做插值，而 URL 编码后的密码
（如 @ -> %40、# -> %23）恰好含 %，走这套 API 会直接抛
"invalid interpolation syntax"。因此连接串一律 raw 读取后交给 SQLAlchemy 解析。
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

import app.db.models  # noqa: F401  必须导入模型，autogenerate 才能感知表结构
from app.core.config import settings
from app.db.base import Base

config = context.config

# disable_existing_loggers=False：避免迁移过程把应用自身的日志配置关掉
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    """取连接串：ini 中显式配置优先（raw 读取，% 不被插值），否则用应用配置。"""
    ini_url = config.file_config.get(config.config_ini_section, "sqlalchemy.url", fallback="", raw=True)
    if ini_url.strip():
        return ini_url.strip()
    return settings.sqlalchemy_url


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不真正连接数据库。"""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库并执行迁移。"""
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # 列类型变更也能被 autogenerate 检测到
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
