"""create user table

Revision ID: 20260912_0001
Revises:
Create Date: 2026-09-12

初始迁移：与 app/db/models.py 中的 User 模型保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260912_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("userName", sa.String(length=20), nullable=False, comment="用户名"),
        sa.Column("password", sa.String(length=255), nullable=False, comment="密码哈希"),
        sa.Column(
            "create_time",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
            comment="创建时间",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user")),
        comment="用户信息表",
        mysql_charset="utf8mb4",
        mysql_engine="InnoDB",
    )
    op.create_index("uk_user_userName", "user", ["userName"], unique=True)


def downgrade() -> None:
    op.drop_index("uk_user_userName", table_name="user")
    op.drop_table("user")
