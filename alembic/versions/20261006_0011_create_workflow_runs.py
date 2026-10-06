"""create workflow_runs table

Revision ID: 20261006_0011
Revises: 20261006_0010
Create Date: 2026-10-06

新增 workflow_runs 表：保存跨 HTTP 请求的工作流状态。学习测评拆成两次请求
（start 生成题目 / submit 提交评分），中间状态完整序列化进 state_json，
第二次请求按 run_id 恢复题目与上下文继续执行。

- run_id：UUID hex，唯一索引，对外标识一次运行
- status：pending/running/quiz_ready/evaluated/failed
- state_json：完整工作流状态（JSON）
- create_at / update_at / completed_at：Unix 秒时间戳（update_at 由 ORM onupdate 刷新）

与 app/db/models.py 的 WorkflowRun 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20261006_0011"
down_revision: str | None = "20261006_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_KWARGS = {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"}


def _unix_timestamp() -> sa.TextClause:
    """Unix 秒时间戳的数据库默认值。"""
    return sa.text("(UNIX_TIMESTAMP())")


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("run_id", sa.String(length=64), nullable=False, comment="运行 id（对外标识一次工作流）"),
        sa.Column("user_id", sa.Integer(), nullable=False, comment="所属用户 id"),
        sa.Column("session_id", sa.Integer(), nullable=False, comment="所属会话 id"),
        sa.Column(
            "workflow_type",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'learning_assessment'"),
            comment="工作流类型：learning_assessment=学习测评",
        ),
        sa.Column("status", sa.String(length=32), nullable=False, comment="状态：pending/running/quiz_ready/evaluated/failed"),
        sa.Column("state_json", sa.JSON(), nullable=True, comment="完整工作流状态（跨请求恢复题目与评分上下文）"),
        sa.Column("error_message", sa.String(length=1000), nullable=True, comment="失败原因（仅 failed 时有值）"),
        sa.Column("create_at", sa.BigInteger(), nullable=False, server_default=_unix_timestamp(), comment="创建时间（Unix 秒）"),
        sa.Column("update_at", sa.BigInteger(), nullable=False, server_default=_unix_timestamp(), comment="更新时间（Unix 秒）"),
        sa.Column("completed_at", sa.BigInteger(), nullable=True, comment="完成时间（Unix 秒，评分完成后写入）"),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_runs"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], name="fk_workflow_runs_user_id_user"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], name="fk_workflow_runs_session_id_sessions"),
        sa.UniqueConstraint("run_id", name="uq_workflow_runs_run_id"),
        comment="工作流运行表（跨请求流程状态）",
        **TABLE_KWARGS,
    )
    op.create_index("ix_workflow_runs_user_id", "workflow_runs", ["user_id"])
    op.create_index("ix_workflow_runs_status", "workflow_runs", ["status"])
    op.create_index("ix_workflow_runs_user_id_workflow_type", "workflow_runs", ["user_id", "workflow_type"])


def downgrade() -> None:
    op.drop_index("ix_workflow_runs_user_id_workflow_type", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_status", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_user_id", table_name="workflow_runs")
    op.drop_table("workflow_runs")
