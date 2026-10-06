"""add long-term memory fields to user_profiles

Revision ID: 20261006_0010
Revises: 20261004_0009
Create Date: 2026-10-06

user_profiles 增加长期记忆字段，供 LongTermMemory 服务沉淀跨会话用户画像：

- learning_goal：学习目标（字符串，可空）
- learning_style：学习风格（字符串，可空）
- interview_focus：面试关注点（JSON 数组，可空）
- long_term_summary：长期记忆摘要（MEDIUMTEXT，可空）

与 app/db/models.py 的 UserProfile 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "20261006_0010"
down_revision: str | None = "20261004_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _medium_text() -> sa.Text:
    """MySQL 用 MEDIUMTEXT，SQLite（测试）退化为 TEXT。"""
    return sa.Text().with_variant(mysql.MEDIUMTEXT(), "mysql")


def upgrade() -> None:
    op.add_column(
        "user_profiles",
        sa.Column("learning_goal", sa.String(length=255), nullable=True, comment="学习目标（长期记忆）"),
    )
    op.add_column(
        "user_profiles",
        sa.Column("learning_style", sa.String(length=255), nullable=True, comment="学习风格（长期记忆）"),
    )
    op.add_column(
        "user_profiles",
        sa.Column("interview_focus", sa.JSON(), nullable=True, comment="面试关注点（JSON 数组，长期记忆）"),
    )
    op.add_column(
        "user_profiles",
        sa.Column("long_term_summary", _medium_text(), nullable=True, comment="长期记忆摘要"),
    )


def downgrade() -> None:
    op.drop_column("user_profiles", "long_term_summary")
    op.drop_column("user_profiles", "interview_focus")
    op.drop_column("user_profiles", "learning_style")
    op.drop_column("user_profiles", "learning_goal")
