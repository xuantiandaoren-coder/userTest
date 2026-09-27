"""数据库层：用户背景画像表数据访问（提示词变量注入的数据来源）。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import UserProfile


class UserProfileRepository:
    """基于 SQLAlchemy Session 的 user_profiles 仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_user(self, user_id: int) -> UserProfile | None:
        """按 user_id 查询画像（一人一行），未命中返回 None。"""
        return self.session.scalars(select(UserProfile).where(UserProfile.user_id == user_id)).first()

    def upsert(self, user_id: int, **fields: Any) -> UserProfile:
        """按 user_id 写入或更新画像，返回最新行。"""
        profile = self.get_by_user(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, **fields)
            self.session.add(profile)
        else:
            for key, value in fields.items():
                setattr(profile, key, value)
        self.session.flush()
        self.session.refresh(profile)
        return profile
