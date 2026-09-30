"""资源元数据测试：唯一索引去重、过期时间、每日清理（先删对象再删元数据）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core import scheduler
from app.db.models import Resource
from app.db.resource_repository import ResourceRepository
from app.services import resource_cleanup
from tests.conftest import FakeSeaweedFSClient


def _resource(*, user_id: int, file_hash: str, path: str, expire_time: datetime | None) -> Resource:
    return Resource(
        resource_type=0,
        storage_scene=0,
        upload_purpose=0,
        file_name="note.txt",
        file_hash=file_hash,
        storage_path=path,
        user_id=user_id,
        expire_time=expire_time,
    )


def test_unique_index_covers_hash_and_user() -> None:
    unique_indexes = [index for index in Resource.__table__.indexes if index.unique]

    assert len(unique_indexes) == 1
    index = unique_indexes[0]
    assert index.name == "uk_file_hash_user_id"
    assert [column.name for column in index.columns] == ["file_hash", "user_id"]


def test_same_hash_for_same_user_is_rejected(sqlite_engine: Engine) -> None:
    """数据库兜底：同一用户同一 MD5 只能有一条元数据。"""
    with Session(sqlite_engine) as session:
        session.add_all(
            [
                _resource(user_id=1, file_hash="a" * 32, path="1/a.txt", expire_time=None),
                _resource(user_id=1, file_hash="a" * 32, path="1/a.txt", expire_time=None),
            ]
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_same_hash_for_different_users_is_allowed(sqlite_engine: Engine) -> None:
    """去重是用户级的：不同用户上传同样内容各存一份。"""
    with Session(sqlite_engine) as session:
        session.add_all(
            [
                _resource(user_id=1, file_hash="b" * 32, path="1/b.txt", expire_time=None),
                _resource(user_id=2, file_hash="b" * 32, path="2/b.txt", expire_time=None),
            ]
        )
        session.commit()

        assert session.query(Resource).count() == 2


def test_expired_resources_are_scanned_by_expire_time(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        repository = ResourceRepository(session)
        session.add_all(
            [
                _resource(user_id=1, file_hash="c" * 32, path="1/old", expire_time=datetime.now() - timedelta(hours=1)),
                _resource(user_id=1, file_hash="d" * 32, path="1/new", expire_time=datetime.now() + timedelta(days=1)),
                _resource(user_id=1, file_hash="e" * 32, path="1/never", expire_time=None),
            ]
        )
        session.commit()

        expired = repository.list_expired(datetime.now())

        assert [row.storage_path for row in expired] == ["1/old"]


@pytest.mark.anyio
async def test_cleanup_deletes_object_then_metadata(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(resource_cleanup, "get_session_factory", lambda: factory)
    storage = FakeSeaweedFSClient()
    storage.objects["1/old"] = b"expired"
    storage.objects["1/new"] = b"fresh"

    with factory() as session:
        session.add_all(
            [
                _resource(
                    user_id=1,
                    file_hash="c" * 32,
                    path="1/old",
                    expire_time=datetime.now() - timedelta(minutes=1),
                ),
                _resource(
                    user_id=1,
                    file_hash="d" * 32,
                    path="1/new",
                    expire_time=datetime.now() + timedelta(days=1),
                ),
            ]
        )
        session.commit()

    deleted = await resource_cleanup.cleanup_expired_resources(storage)

    assert deleted == 1
    assert "1/old" not in storage.objects  # 对象已删
    assert "1/new" in storage.objects
    with factory() as session:
        assert [row.storage_path for row in session.query(Resource).all()] == ["1/new"]


@pytest.mark.anyio
async def test_cleanup_keeps_metadata_when_object_delete_fails(
    sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对象没删成功就保留元数据，第二天重试，避免留下无主对象。"""
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(resource_cleanup, "get_session_factory", lambda: factory)
    storage = FakeSeaweedFSClient()
    storage.objects["1/stuck"] = b"expired"
    storage.fail_delete.add("1/stuck")

    with factory() as session:
        session.add(
            _resource(
                user_id=1,
                file_hash="f" * 32,
                path="1/stuck",
                expire_time=datetime.now() - timedelta(minutes=1),
            )
        )
        session.commit()

    deleted = await resource_cleanup.cleanup_expired_resources(storage)

    assert deleted == 0
    with factory() as session:
        assert session.query(Resource).count() == 1  # 元数据仍在，等待重试


def test_scheduler_waits_until_next_daily_run() -> None:
    now = datetime(2026, 9, 15, 1, 0, 0)

    assert scheduler.seconds_until(3, 0, now=now) == 2 * 3600  # 凌晨 1 点 -> 今天 3 点
    assert scheduler.seconds_until(3, 0, now=datetime(2026, 9, 15, 4, 0, 0)) == 23 * 3600  # 已过点 -> 明天 3 点


def test_resources_table_exists_after_migration(tmp_path) -> None:
    """迁移脚本建出的表结构里有 resources 与唯一索引。"""
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    from sqlalchemy import create_engine

    project_root = Path(__file__).resolve().parents[1]
    url = f"sqlite:///{tmp_path / 'resources.db'}"
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("prepend_sys_path", str(project_root))
    config.set_main_option("sqlalchemy.url", url)

    command.upgrade(config, "head")

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        assert "resources" in inspector.get_table_names()
        index_names = {index["name"] for index in inspector.get_indexes("resources")}
        assert "uk_file_hash_user_id" in index_names
        columns = {column["name"]: column for column in inspector.get_columns("resources")}
        assert "doc_category" in columns
        assert columns["doc_category"]["nullable"] is True  # 文档分类可空，且仅文件类型有意义
    finally:
        engine.dispose()
