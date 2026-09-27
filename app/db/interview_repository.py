"""数据库层：面试记录表数据访问。"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import Interview


class InterviewRepository:
    """基于 SQLAlchemy Session 的面试记录仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, interview_id: int, user_id: int) -> Interview | None:
        """按 interview_id + user_id 查询，避免读到别人的面试记录。"""
        stmt = select(Interview).where(Interview.id == interview_id, Interview.user_id == user_id)
        return self.session.scalars(stmt).first()

    def find_by_message_ids(self, message_ids: Sequence[int]) -> dict[int, Interview]:
        """按入口消息 id 批量取面试记录，供消息列表拼 interview_id / status。"""
        if not message_ids:
            return {}
        stmt = select(Interview).where(Interview.message_id.in_(message_ids))
        return {interview.message_id: interview for interview in self.session.scalars(stmt).all()}

    def delete_by_session(self, session_id: int) -> None:
        """删除会话下的全部面试记录。"""
        self.session.execute(delete(Interview).where(Interview.session_id == session_id))
