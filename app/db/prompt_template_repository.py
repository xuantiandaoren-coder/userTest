"""数据库层：提示词模板表数据访问（SQL 全部收敛在这里）。

「同一 (agent_name, scene) 仅一条 is_active=1」由本仓储的
`deactivate_group` + `activate` 在同一个事务里完成，服务层负责调用顺序。
公共模板的 agent_name 为 NULL，因此条件一律用 `is_(None)` 而不是 `== None`。
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from app.db.models import PromptTemplate


class PromptTemplateRepository:
    """基于 SQLAlchemy Session 的提示词模板表仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, template_id: int) -> PromptTemplate | None:
        """按主键查询，未命中返回 None。"""
        return self.session.get(PromptTemplate, template_id)

    def _group_stmt(self, agent_name: str | None, scene: str) -> Select[tuple[PromptTemplate]]:
        """(agent_name, scene) 分组条件：agent_name 为空即公共模板。"""
        condition = PromptTemplate.agent_name.is_(None) if agent_name is None else PromptTemplate.agent_name == agent_name
        return select(PromptTemplate).where(condition, PromptTemplate.scene == scene)

    def find_group(self, agent_name: str | None, scene: str) -> Sequence[PromptTemplate]:
        """返回一组模板的全部版本（版本号倒序，最新在前）。"""
        return self.session.scalars(
            self._group_stmt(agent_name, scene).order_by(PromptTemplate.version.desc())
        ).all()

    def find_active(self, agent_name: str | None, scene: str) -> PromptTemplate | None:
        """返回该组当前生效版本，未命中返回 None。"""
        stmt = self._group_stmt(agent_name, scene).where(PromptTemplate.is_active == 1)
        return self.session.scalars(stmt.order_by(PromptTemplate.version.desc())).first()

    def list_active(self, *, scene: str | None = None) -> Sequence[PromptTemplate]:
        """返回所有生效版本（启动加载到 Redis 用），可按 scene 过滤。"""
        stmt = select(PromptTemplate).where(PromptTemplate.is_active == 1)
        if scene:
            stmt = stmt.where(PromptTemplate.scene == scene)
        return self.session.scalars(stmt.order_by(PromptTemplate.scene, PromptTemplate.agent_name)).all()

    def list_scenes(self) -> Sequence[str]:
        """返回所有已出现过的场景名（去重、升序）。"""
        return self.session.scalars(select(PromptTemplate.scene).distinct().order_by(PromptTemplate.scene)).all()

    def next_version(self, agent_name: str | None, scene: str) -> int:
        """计算该组的下一版本号：无历史版本时从 1 开始。"""
        stmt = select(func.max(PromptTemplate.version)).where(
            PromptTemplate.scene == scene,
            PromptTemplate.agent_name.is_(None) if agent_name is None else PromptTemplate.agent_name == agent_name,
        )
        current = self.session.scalar(stmt) or 0
        return int(current) + 1

    def has_version(self, agent_name: str | None, scene: str, version: int) -> bool:
        """该组是否已存在指定版本号（并发写入时用于兜底重试）。"""
        stmt = self._group_stmt(agent_name, scene).where(PromptTemplate.version == version)
        return self.session.scalars(stmt).first() is not None

    def deactivate_group(self, agent_name: str | None, scene: str) -> int:
        """把该组所有生效版本置为 0，返回受影响行数。"""
        condition = PromptTemplate.agent_name.is_(None) if agent_name is None else PromptTemplate.agent_name == agent_name
        result = self.session.execute(
            update(PromptTemplate)
            .where(condition, PromptTemplate.scene == scene, PromptTemplate.is_active == 1)
            .values(is_active=0)
        )
        return int(result.rowcount or 0)

    def activate(self, template: PromptTemplate) -> PromptTemplate:
        """把指定版本置为生效（调用前应先 deactivate_group）。"""
        template.is_active = 1
        self.session.flush()
        self.session.refresh(template)
        return template

    def create(self, **values: object) -> PromptTemplate:
        """插入一行；flush + refresh 以便拿到数据库填充的 id / created_at。"""
        template = PromptTemplate(**values)
        self.session.add(template)
        self.session.flush()
        self.session.refresh(template)
        return template
