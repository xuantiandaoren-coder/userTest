"""数据库层：工作流运行表数据访问（SQL 全部收敛在这里）。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import WorkflowRun


class WorkflowRunRepository:
    """基于 SQLAlchemy Session 的 workflow_runs 仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        run_id: str,
        user_id: int,
        session_id: int,
        workflow_type: str,
        status: str,
        state: dict[str, Any] | None = None,
    ) -> WorkflowRun:
        """新建一次工作流运行；flush + refresh 拿到数据库填充的时间戳。"""
        run = WorkflowRun(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
            workflow_type=workflow_type,
            status=status,
            state_json=state,
        )
        self.session.add(run)
        self.session.flush()
        self.session.refresh(run)
        return run

    def get(self, run_id: str, user_id: int) -> WorkflowRun | None:
        """按 run_id + user_id 查询（带上 user_id，越权访问直接查不到）。"""
        stmt = select(WorkflowRun).where(WorkflowRun.run_id == run_id, WorkflowRun.user_id == user_id)
        return self.session.scalars(stmt).first()

    def update(
        self,
        run: WorkflowRun,
        *,
        status: str | None = None,
        state: dict[str, Any] | None = None,
        error_message: str | None = None,
        completed_at: int | None = None,
    ) -> WorkflowRun:
        """更新运行状态 / 状态快照；由请求级事务统一提交。"""
        if status is not None:
            run.status = status
        if state is not None:
            run.state_json = state
        if error_message is not None:
            run.error_message = error_message[:1000]
        if completed_at is not None:
            run.completed_at = completed_at
        self.session.flush()
        self.session.refresh(run)
        return run

    def mark_failed(self, run: WorkflowRun, message: str) -> WorkflowRun:
        """标记失败：写 status + error_message，并立即提交（失败信息不能随异常回滚）。"""
        run.status = "failed"
        run.error_message = (message or "")[:1000]
        self.session.flush()
        self.session.commit()
        self.session.refresh(run)
        return run
