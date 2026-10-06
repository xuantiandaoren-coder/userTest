"""rename workflow_runs timestamps to created_at / updated_at

Revision ID: 20261006_0012
Revises: 20261006_0011
Create Date: 2026-10-06

把 workflow_runs 的 create_at / update_at 统一为仓库通用的
created_at / updated_at（与 sessions / chat_messages 等表命名一致）。

0011 已经创建并应用过旧列名，这里用重命名迁移保持历史不可变：
- 全新库：0011 建旧列 -> 0012 立即改名
- 已应用 0011 的库：直接改名，不丢数据

与 app/db/models.py 的 WorkflowRun 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20261006_0012"
down_revision: str | None = "20261006_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "workflow_runs",
        "create_at",
        new_column_name="created_at",
        existing_type=sa.BigInteger(),
        existing_nullable=False,
        existing_server_default=sa.text("(UNIX_TIMESTAMP())"),
    )
    op.alter_column(
        "workflow_runs",
        "update_at",
        new_column_name="updated_at",
        existing_type=sa.BigInteger(),
        existing_nullable=False,
        existing_server_default=sa.text("(UNIX_TIMESTAMP())"),
    )


def downgrade() -> None:
    op.alter_column(
        "workflow_runs",
        "updated_at",
        new_column_name="update_at",
        existing_type=sa.BigInteger(),
        existing_nullable=False,
        existing_server_default=sa.text("(UNIX_TIMESTAMP())"),
    )
    op.alter_column(
        "workflow_runs",
        "created_at",
        new_column_name="create_at",
        existing_type=sa.BigInteger(),
        existing_nullable=False,
        existing_server_default=sa.text("(UNIX_TIMESTAMP())"),
    )
