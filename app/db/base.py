"""数据库层：SQLAlchemy 2.0 声明式基类。"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# 统一约束命名规则，保证 Alembic 自动生成的迁移脚本命名稳定、可回滚
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """所有 ORM 模型的基类，metadata 供 Alembic autogenerate 使用。"""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
