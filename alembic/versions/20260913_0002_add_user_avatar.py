"""add user avatar

Revision ID: 20260913_0002
Revises: 20260912_0001
Create Date: 2026-09-13

用户头像：上传图片时写入相对存储路径。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260913_0002"
down_revision: str | None = "20260912_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("user", sa.Column("avatar", sa.String(length=255), nullable=True, comment="头像文件相对路径"))


def downgrade() -> None:
    op.drop_column("user", "avatar")
