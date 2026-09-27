"""会话 / 消息 / 面试记录三张表的建模测试：列定义、索引、外键与默认值。"""

from typing import Any

import pytest
from sqlalchemy import Engine, Table
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models
from app.db.models import ChatMessage, ChatSession, Interview, User


def _index_columns(table: Table, name: str) -> list[str]:
    index = next(index for index in table.indexes if index.name == name)
    return [column.name for column in index.columns]


def _fks(table: Table) -> dict[tuple[str, ...], tuple[str, ...]]:
    return {
        tuple(column.name for column in constraint.columns): tuple(
            element.target_fullname for element in constraint.elements
        )
        for constraint in table.foreign_key_constraints
    }


def test_sessions_columns() -> None:
    columns = ChatSession.__table__.c

    assert sorted(columns.keys()) == sorted(["id", "user_id", "session_model", "title", "created_at"])
    assert columns.id.primary_key is True
    assert columns.user_id.nullable is False
    assert columns.session_model.nullable is False
    assert columns.title.type.length == 255
    assert columns.created_at.server_default is not None  # 由数据库填充 Unix 秒


def test_chat_messages_columns() -> None:
    columns = ChatMessage.__table__.c

    assert sorted(columns.keys()) == sorted(
        [
            "id",
            "user_id",
            "session_id",
            "select_model",
            "request_id",
            "request_text",
            "response_text",
            "request_segments",
            "response_segments",
            "file_extracted_text",
            "created_at",
        ]
    )
    assert columns.request_id.type.length == 64
    assert columns.request_text.nullable is False
    assert columns.response_text.nullable is False
    assert columns.request_segments.nullable is True  # 只存附件段，无附件为空
    assert columns.response_segments.nullable is True
    assert columns.file_extracted_text.nullable is True  # 只有带附件的消息才有提取文本
    assert columns.created_at.server_default is not None


def test_interviews_columns() -> None:
    columns = Interview.__table__.c

    assert sorted(columns.keys()) == sorted(
        [
            "id",
            "session_id",
            "user_id",
            "message_id",
            "qa_object",
            "interview_duration",
            "status",
            "created_at",
            "updated_at",
        ]
    )
    assert columns.user_id.nullable is False  # 面试记录必须能按 user_id 隔离
    assert columns.message_id.unique is True  # 入口消息唯一
    assert columns.qa_object.nullable is False
    assert columns.interview_duration.nullable is False
    assert columns.status.nullable is False
    assert columns.created_at.server_default is not None
    assert columns.updated_at.server_default is not None


def test_foreign_keys() -> None:
    assert _fks(ChatSession.__table__) == {("user_id",): ("user.id",)}
    assert _fks(ChatMessage.__table__) == {("user_id",): ("user.id",), ("session_id",): ("sessions.id",)}
    assert _fks(Interview.__table__) == {
        ("session_id",): ("sessions.id",),
        ("user_id",): ("user.id",),
        ("message_id",): ("chat_messages.id",),
    }


def test_indexes() -> None:
    assert _index_columns(ChatSession.__table__, "ix_sessions_user_id") == ["user_id"]

    assert _index_columns(ChatMessage.__table__, "ix_chat_messages_session_id_created_at") == [
        "session_id",
        "created_at",
    ]
    assert _index_columns(ChatMessage.__table__, "ix_chat_messages_request_id") == ["request_id"]

    assert _index_columns(Interview.__table__, "ix_interviews_session_id_message_id") == [
        "session_id",
        "message_id",
    ]
    assert _index_columns(Interview.__table__, "ix_interviews_status") == ["status"]
    assert _index_columns(Interview.__table__, "ix_interviews_user_id") == ["user_id"]


def _add_user_and_session(db: Session) -> tuple[User, ChatSession]:
    user = User(user_name="ivy", password="hash")
    db.add(user)
    db.flush()
    chat_session = ChatSession(user_id=user.id, session_model=1, title="模拟面试")
    db.add(chat_session)
    db.flush()
    return user, chat_session


def test_defaults_and_json_round_trip(sqlite_engine: Engine) -> None:
    qa_object: list[dict[str, Any]] = [
        {"id": "0f1e2d3c", "question": "讲讲索引", "answer": "…", "created_at": 1717171717}
    ]

    with Session(sqlite_engine) as db:
        user, chat_session = _add_user_and_session(db)
        message = ChatMessage(
            user_id=user.id,
            session_id=chat_session.id,
            select_model=4,
            request_id="req-0001",
            request_text="开始模拟面试",
            response_text="第一题：…",
        )
        db.add(message)
        db.flush()
        interview = Interview(
            session_id=chat_session.id, user_id=user.id, message_id=message.id, qa_object=qa_object
        )
        db.add(interview)
        db.commit()

        # Unix 秒时间戳由数据库默认值填充
        assert chat_session.created_at > 0
        assert message.created_at > 0
        assert interview.created_at > 0

        assert interview.interview_duration == 0
        assert interview.status == 0
        assert interview.qa_object == qa_object


def test_updated_at_is_refreshed_on_update(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    with Session(sqlite_engine) as db:
        user, chat_session = _add_user_and_session(db)
        message = ChatMessage(
            user_id=user.id,
            session_id=chat_session.id,
            select_model=4,
            request_id="req-0002",
            request_text="q",
            response_text="a",
        )
        db.add(message)
        db.flush()
        interview = Interview(session_id=chat_session.id, user_id=user.id, message_id=message.id, qa_object=[])
        db.add(interview)
        db.commit()

        monkeypatch.setattr(models.time, "time", lambda: 1_800_000_000)
        interview.status = 1
        db.commit()

        assert interview.updated_at == 1_800_000_000
        assert interview.updated_at > interview.created_at


def test_entry_message_is_unique(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as db:
        user, chat_session = _add_user_and_session(db)
        messages = [
            ChatMessage(
                user_id=user.id,
                session_id=chat_session.id,
                select_model=4,
                request_id=f"req-{index}",
                request_text="q",
                response_text="a",
            )
            for index in (1, 2)
        ]
        db.add_all(messages)
        db.flush()

        db.add(Interview(session_id=chat_session.id, user_id=user.id, message_id=messages[0].id, qa_object=[]))
        db.commit()

        db.add(Interview(session_id=chat_session.id, user_id=user.id, message_id=messages[0].id, qa_object=[]))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_message_segments_json_round_trip(sqlite_engine: Engine) -> None:
    """附件段只存 file/image/audio 三类，直接以 JSON 落库并可原样读回。"""
    request_segments = [{"type": "image", "resource_id": 11}, {"type": "file", "resource_id": 12}]
    response_segments = [{"type": "audio", "resource_id": 13}]

    with Session(sqlite_engine) as db:
        user, chat_session = _add_user_and_session(db)
        message = ChatMessage(
            user_id=user.id,
            session_id=chat_session.id,
            select_model=0,
            request_id="req-seg-1",
            request_text="看下这两张图",
            response_text="收到",
            request_segments=request_segments,
            response_segments=response_segments,
        )
        db.add(message)
        db.commit()

        assert message.request_segments == request_segments
        assert message.response_segments == response_segments
