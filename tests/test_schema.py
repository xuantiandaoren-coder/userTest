"""数据建模测试：主键、唯一索引、创建时间自动填充，以及迁移与模型的一致性。"""

import io
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import User

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_defines_expected_columns() -> None:
    columns = User.__table__.c

    assert sorted(columns.keys()) == sorted(["id", "userName", "password", "avatar", "create_time"])

    assert columns.id.primary_key is True
    assert columns.id.autoincrement is True

    assert columns.userName.nullable is False

    assert columns.avatar.nullable is True  # 头像非必填
    assert columns.create_time.server_default is not None  # 由数据库自动填充当前时间


def test_username_has_unique_index() -> None:
    unique_indexes = [index for index in User.__table__.indexes if index.unique]

    assert len(unique_indexes) == 1
    index = unique_indexes[0]
    assert index.name == "uk_user_userName"
    assert [column.name for column in index.columns] == ["userName"]


def test_username_unique_constraint_enforced(sqlite_engine: Engine) -> None:
    """唯一索引在数据库层真实生效：绕过接口直接插入重复用户名应失败。"""
    with Session(sqlite_engine) as session:
        session.add_all(
            [
                User(user_name="frank", password="hash-1"),
                User(user_name="frank", password="hash-2"),
            ]
        )
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
        else:  # pragma: no cover - 只有索引缺失时才会走到
            raise AssertionError("userName 唯一索引未生效")


def test_create_time_is_filled_by_database(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        user = User(user_name="gina", password="hash")
        session.add(user)
        session.commit()
        assert user.create_time is not None


def test_migrations_match_models(tmp_path: Path) -> None:
    """在临时库上跑完迁移后，模型与数据库结构不应存在差异（防止手写迁移漏改）。"""
    url = f"sqlite:///{tmp_path / 'migrate.db'}"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("prepend_sys_path", str(PROJECT_ROOT))
    config.set_main_option("sqlalchemy.url", url)

    command.upgrade(config, "head")

    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            assert compare_metadata(context, Base.metadata) == []
        inspector = inspect(engine)
        assert "user" in inspector.get_table_names()
    finally:
        engine.dispose()


def test_alembic_env_accepts_percent_encoded_password(monkeypatch, tmp_path: Path) -> None:
    """密码含 @ 等特殊字符时按 URL 编码写入，Alembic 不能因 ConfigParser 插值而报错。"""
    from app.core.config import settings

    monkeypatch.setattr(
        settings,
        "mysql_url",
        SecretStr("mysql+pymysql://app:p%40ss%2Fword@127.0.0.1:3306/user_api?charset=utf8mb4"),
    )

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("prepend_sys_path", str(PROJECT_ROOT))
    buffer = io.StringIO()
    config.output_buffer = buffer

    command.upgrade(config, "head", sql=True)  # 离线模式生成 SQL，不连接数据库

    sql = buffer.getvalue()
    assert "CREATE TABLE user" in sql
    assert "uk_user_userName" in sql
