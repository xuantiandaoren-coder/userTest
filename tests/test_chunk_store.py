"""分块原文存储：knowledge_chunks 写入、覆盖语义、与 Qdrant 的 UUID 关联。

用例走内存 SQLite（conftest 的 sqlite_engine / db_session），不连 MySQL。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunk, Resource, User
from app.rag import chunk_store
from app.rag.chunk_store import save_chunks


def _user(session: Session, name: str = "chunkstore") -> User:
    user = User(user_name=name, password="x" * 60)
    session.add(user)
    session.flush()
    return user


def _resource(session: Session, *, user_id: int, file_name: str = "resume.pdf", file_hash: str = "a" * 32) -> Resource:
    resource = Resource(
        resource_type=0,
        storage_scene=0,
        upload_purpose=0,
        file_name=file_name,
        file_hash=file_hash,
        storage_path=f"{user_id}/{file_hash}.pdf",
        user_id=user_id,
    )
    session.add(resource)
    session.flush()
    return resource


def _chunks(session: Session, resource_id: int) -> list[KnowledgeChunk]:
    stmt = select(KnowledgeChunk).where(KnowledgeChunk.resource_id == resource_id).order_by(KnowledgeChunk.chunk_index)
    return list(session.scalars(stmt).all())


@pytest.fixture()
def stored(db_session: Session) -> Iterator[dict[str, int]]:
    """先落一个用户 + 一个资源行，save_chunks 需要它们做外键与覆盖定位。"""
    user = _user(db_session)
    resource = _resource(db_session, user_id=user.id)
    yield {"user_id": user.id, "resource_id": resource.id}


def test_save_chunks_persists_text_and_meta(db_session: Session, stored: dict[str, int]) -> None:
    """分块原文、序号、字符数、vector_id 一起落库。"""
    saved = save_chunks(
        db_session,
        resource_id=stored["resource_id"],
        user_id=stored["user_id"],
        file_name="resume.pdf",
        texts=["第一块内容", "第二块内容更长一些"],
        vector_ids=["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"],
    )

    assert saved == 2
    rows = _chunks(db_session, stored["resource_id"])
    assert [row.text for row in rows] == ["第一块内容", "第二块内容更长一些"]
    assert [row.chunk_index for row in rows] == [0, 1]
    assert [row.char_count for row in rows] == [len("第一块内容"), len("第二块内容更长一些")]
    assert [row.vector_id for row in rows] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    assert all(row.created_at for row in rows)  # 数据库侧默认值填 Unix 秒


def test_save_chunks_keeps_vector_id_in_sync_with_point_id(db_session: Session, stored: dict[str, int]) -> None:
    """vector_id 原样落库：它就是 Qdrant point id，两边必须同值。"""
    point_ids = ["point-a", "point-b"]

    save_chunks(
        db_session,
        resource_id=stored["resource_id"],
        user_id=stored["user_id"],
        file_name="resume.pdf",
        texts=["甲", "乙"],
        vector_ids=point_ids,
    )

    assert [row.vector_id for row in _chunks(db_session, stored["resource_id"])] == point_ids


def test_save_chunks_overwrites_same_user_file_only(db_session: Session, stored: dict[str, int]) -> None:
    """同名文件重传覆盖旧分块；同用户其它文件、其它用户的同名文件都不受影响。"""
    user_id = stored["user_id"]
    other_user = _user(db_session, "chunkstore-other")
    other_resource = _resource(db_session, user_id=other_user.id)
    note_resource = _resource(db_session, user_id=user_id, file_name="note.txt", file_hash="b" * 32)

    save_chunks(
        db_session,
        resource_id=stored["resource_id"],
        user_id=user_id,
        file_name="resume.pdf",
        texts=["旧简历内容"],
        vector_ids=["old-1"],
    )
    save_chunks(
        db_session,
        resource_id=note_resource.id,
        user_id=user_id,
        file_name="note.txt",
        texts=["笔记内容"],
        vector_ids=["note-1"],
    )
    save_chunks(
        db_session,
        resource_id=other_resource.id,
        user_id=other_user.id,
        file_name="resume.pdf",
        texts=["别人的简历"],
        vector_ids=["other-1"],
    )

    # 同名重传：新资源行 + 新 vector_id，旧资源行的分块被删掉
    new_resource = _resource(db_session, user_id=user_id, file_hash="c" * 32)
    save_chunks(
        db_session,
        resource_id=new_resource.id,
        user_id=user_id,
        file_name="resume.pdf",
        texts=["新简历内容"],
        vector_ids=["new-1"],
    )

    assert _chunks(db_session, stored["resource_id"]) == []
    assert [row.text for row in _chunks(db_session, new_resource.id)] == ["新简历内容"]
    assert [row.text for row in _chunks(db_session, note_resource.id)] == ["笔记内容"]
    assert [row.text for row in _chunks(db_session, other_resource.id)] == ["别人的简历"]


def test_save_chunks_is_idempotent_for_same_input(db_session: Session, stored: dict[str, int]) -> None:
    """同一份分块重复保存两次不会翻倍（覆盖语义下 delete 先执行）。"""
    for _ in range(2):
        save_chunks(
            db_session,
            resource_id=stored["resource_id"],
            user_id=stored["user_id"],
            file_name="resume.pdf",
            texts=["甲", "乙"],
            vector_ids=["v1", "v2"],
        )

    assert len(_chunks(db_session, stored["resource_id"])) == 2


def test_save_chunks_empty_texts_is_noop(db_session: Session, stored: dict[str, int]) -> None:
    """没有分块时不写库也不报错（空文本文件场景）。"""
    assert save_chunks(
        db_session,
        resource_id=stored["resource_id"],
        user_id=stored["user_id"],
        file_name="resume.pdf",
        texts=[],
        vector_ids=[],
    ) == 0
    assert _chunks(db_session, stored["resource_id"]) == []


def test_save_chunks_rejects_mismatched_lengths(db_session: Session, stored: dict[str, int]) -> None:
    """分块数与向量数不一致直接报错，避免写入错位的 text / vector_id。"""
    with pytest.raises(ValueError, match="不一致"):
        save_chunks(
            db_session,
            resource_id=stored["resource_id"],
            user_id=stored["user_id"],
            file_name="resume.pdf",
            texts=["甲", "乙"],
            vector_ids=["only-one"],
        )


def test_save_chunks_failure_rolls_back_only_chunks(db_session: Session) -> None:
    """写分块失败只回滚分块（savepoint）：会话仍可用，调用方已 flush 的行不受牵连。"""
    user = _user(db_session, "chunkstore-keep")
    first = _resource(db_session, user_id=user.id, file_name="a.pdf", file_hash="d" * 32)
    second = _resource(db_session, user_id=user.id, file_name="b.pdf", file_hash="e" * 32)

    save_chunks(
        db_session,
        resource_id=first.id,
        user_id=user.id,
        file_name="a.pdf",
        texts=["先写入的分块"],
        vector_ids=["dup-vector-id"],
    )

    with pytest.raises(IntegrityError):  # 与上面那条分块撞 vector_id 唯一索引
        save_chunks(
            db_session,
            resource_id=second.id,
            user_id=user.id,
            file_name="b.pdf",
            texts=["后写入的分块"],
            vector_ids=["dup-vector-id"],
        )

    # 会话仍可用；先写入的分块和调用方 flush 的资源行都还在，失败的那次没有留下半截数据
    assert [row.text for row in _chunks(db_session, first.id)] == ["先写入的分块"]
    assert _chunks(db_session, second.id) == []
    assert db_session.scalars(select(Resource).where(Resource.id == second.id)).first() is not None


def test_chunk_store_module_exports_read_and_write() -> None:
    """对外只暴露写（save_chunks）与两个读入口（按 id 取原文 / 原文带资源）。"""
    assert chunk_store.__all__ == ["save_chunks", "fetch_chunks_by_vector_ids", "fetch_chunks_with_resources"]
