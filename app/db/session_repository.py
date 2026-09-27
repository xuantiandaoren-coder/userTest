"""数据库层：会话表数据访问。"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ChatSession


class ChatSessionRepository:
    """基于 SQLAlchemy Session 的会话表仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, session_id: int, user_id: int) -> ChatSession | None:
        """按 id + user_id 查询（带上 user_id，越权访问直接查不到）。"""
        stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        return self.session.scalars(stmt).first()

    def list_by_user(self, user_id: int, *, page: int, page_size: int) -> tuple[Sequence[ChatSession], int]:
        """分页返回某个用户的会话，新建的在前。"""
        condition = ChatSession.user_id == user_id
        total = self.session.scalar(select(func.count()).select_from(ChatSession).where(condition)) or 0
        stmt = (
            select(ChatSession)
            .where(condition)
            .order_by(ChatSession.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return self.session.scalars(stmt).all(), total

    def create(self, *, user_id: int, title: str, session_model: int) -> ChatSession:
        """新建会话；flush + refresh 拿到数据库填充的 created_at。"""
        chat_session = ChatSession(user_id=user_id, title=title, session_model=session_model)
        self.session.add(chat_session)
        self.session.flush()
        self.session.refresh(chat_session)
        return chat_session

    def save(self, chat_session: ChatSession) -> ChatSession:
        """提交对已加载会话的修改。"""
        self.session.flush()
        return chat_session

    def delete(self, chat_session: ChatSession) -> None:
        """删除会话。"""
        self.session.delete(chat_session)
        self.session.flush()
