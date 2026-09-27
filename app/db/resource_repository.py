"""数据库层：资源元数据表数据访问（所有 SQL 都收敛在这里）。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Resource


class ResourceRepository:
    """基于 SQLAlchemy Session 的 resources 表仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def find_by_hash(self, file_hash: str, user_id: int) -> Resource | None:
        """按 (file_hash, user_id) 预查，命中说明该用户已上传过同样内容。"""
        stmt = select(Resource).where(Resource.file_hash == file_hash, Resource.user_id == user_id)
        return self.session.scalars(stmt).first()

    def find_by_ids(self, resource_ids: Sequence[int]) -> dict[int, Resource]:
        """按主键批量取资源，供消息附件段补齐 name / url / size。"""
        if not resource_ids:
            return {}
        stmt = select(Resource).where(Resource.id.in_(resource_ids))
        return {resource.id: resource for resource in self.session.scalars(stmt).all()}

    def create(self, **fields: object) -> Resource:
        """插入一行元数据；flush 交给唯一索引做并发兜底（冲突抛 IntegrityError）。"""
        resource = Resource(**fields)
        self.session.add(resource)
        self.session.flush()
        self.session.refresh(resource)
        return resource

    def list_expired(self, now: datetime) -> Sequence[Resource]:
        """按 id 升序返回已到期（expire_time <= now）的资源。"""
        stmt = (
            select(Resource)
            .where(Resource.expire_time.is_not(None), Resource.expire_time <= now)
            .order_by(Resource.id)
        )
        return self.session.scalars(stmt).all()

    def delete(self, resource: Resource) -> None:
        """删除一行元数据。"""
        self.session.delete(resource)
        self.session.flush()
