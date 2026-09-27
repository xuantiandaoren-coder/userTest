"""数据库层：用户表数据访问（所有 SQL 都收敛在这里）。"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User


class UserRepository:
    """基于 SQLAlchemy Session 的用户表仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, user_id: int) -> User | None:
        """按主键查询，未命中返回 None。"""
        return self.session.get(User, user_id)

    def find_by_username(self, user_name: str) -> User | None:
        """按用户名精确查询（走唯一索引），未命中返回 None。"""
        stmt = select(User).where(User.user_name == user_name)
        return self.session.scalars(stmt).first()

    def list_all(self) -> Sequence[User]:
        """按 id 升序返回全部用户。"""
        return self.session.scalars(select(User).order_by(User.id)).all()

    def create(self, *, user_name: str, password: str) -> User:
        """插入一行；flush 后 refresh，以便拿到数据库自动填充的 create_time。"""
        user = User(user_name=user_name, password=password)
        self.session.add(user)
        self.session.flush()
        self.session.refresh(user)
        return user

    def save(self, user: User) -> User:
        """提交对已加载对象的修改。"""
        self.session.flush()
        self.session.refresh(user)
        return user

    def delete(self, user: User) -> None:
        """删除一行。"""
        self.session.delete(user)
        self.session.flush()
