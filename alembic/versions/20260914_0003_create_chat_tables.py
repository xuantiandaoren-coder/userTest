"""create chat tables

Revision ID: 20260914_0003
Revises: 20260913_0002
Create Date: 2026-09-14

新增会话、消息、面试记录三张业务表，与 app/db/models.py 中的模型保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "20260914_0003"
down_revision: str | None = "20260913_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_KWARGS = {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"}


def _medium_text() -> sa.Text:
    """MySQL 用 MEDIUMTEXT，SQLite（测试）退化为 TEXT。"""
    return sa.Text().with_variant(mysql.MEDIUMTEXT(), "mysql")


def _unix_timestamp() -> sa.TextClause:
    """Unix 秒时间戳的数据库默认值。"""
    return sa.text("(UNIX_TIMESTAMP())")


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("user_id", sa.Integer(), nullable=False, comment="所属用户 id"),
        sa.Column("session_model", sa.Integer(), nullable=False, comment="会话类型：0=学习，1=面试，2=笔记"),
        sa.Column("title", sa.String(length=255), nullable=False, comment="会话标题"),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="会话创建时间（Unix 秒）",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sessions")),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], name=op.f("fk_sessions_user_id_user")),
        comment="会话表",
        **TABLE_KWARGS,
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("user_id", sa.Integer(), nullable=False, comment="所属用户 id"),
        sa.Column("session_id", sa.Integer(), nullable=False, comment="所属会话 id"),
        sa.Column(
            "select_model",
            sa.Integer(),
            nullable=False,
            comment="选择模式：0=默认，1=知识精讲，2=刷题，3=简历优化，4=模拟面试，5=面试复盘",
        ),
        sa.Column("request_id", sa.String(length=64), nullable=False, comment="请求唯一标识"),
        sa.Column("request_text", _medium_text(), nullable=False, comment="提问文本"),
        sa.Column("response_text", _medium_text(), nullable=False, comment="回答文本"),
        sa.Column("file_extracted_text", _medium_text(), nullable=True, comment="从文件中提取的完整文本（对话上下文用）"),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="创建时间（Unix 秒）",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_messages")),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], name=op.f("fk_chat_messages_user_id_user")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], name=op.f("fk_chat_messages_session_id_sessions")),
        comment="消息表",
        **TABLE_KWARGS,
    )
    op.create_index("ix_chat_messages_session_id_created_at", "chat_messages", ["session_id", "created_at"])
    op.create_index("ix_chat_messages_request_id", "chat_messages", ["request_id"])

    op.create_table(
        "interviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("session_id", sa.Integer(), nullable=False, comment="所属会话 id"),
        sa.Column("message_id", sa.Integer(), nullable=False, comment="开启本次模拟面试的入口消息 id"),
        sa.Column("qa_object", sa.JSON(), nullable=False, comment="一问一答对象列表"),
        sa.Column("interview_duration", sa.Integer(), nullable=False, server_default=sa.text("0"), comment="累计面试时长（秒）"),
        sa.Column(
            "status",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="面试状态：0=进行中，1=已结束，2=异常终止",
        ),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="面试开始时间（Unix 秒）",
        ),
        sa.Column(
            "updated_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="面试更新时间（Unix 秒）",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_interviews")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], name=op.f("fk_interviews_session_id_sessions")),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_messages.id"],
            name=op.f("fk_interviews_message_id_chat_messages"),
        ),
        sa.UniqueConstraint("message_id", name=op.f("uq_interviews_message_id")),
        comment="面试记录表",
        **TABLE_KWARGS,
    )
    op.create_index("ix_interviews_session_id_message_id", "interviews", ["session_id", "message_id"])
    op.create_index("ix_interviews_status", "interviews", ["status"])


def downgrade() -> None:
    op.drop_index("ix_interviews_status", table_name="interviews")
    op.drop_index("ix_interviews_session_id_message_id", table_name="interviews")
    op.drop_table("interviews")

    op.drop_index("ix_chat_messages_request_id", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_id_created_at", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
