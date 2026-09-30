"""add resources.doc_category

Revision ID: 20260930_0007
Revises: 20260915_0006
Create Date: 2026-09-30

resources 表新增文档分类列：resume / study_material / general，可空，
仅文件类型（resource_type=0）有意义，图片 / 音频恒为 NULL。
与 app/db/models.py 的 Resource 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260930_0007"
down_revision: str | None = "20260915_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "resources",
        sa.Column(
            "doc_category",
            sa.String(length=32),
            nullable=True,
            comment="文档分类：resume/study_material/general；仅文件类型(resource_type=0)有意义，其余为空",
        ),
    )


def downgrade() -> None:
    op.drop_column("resources", "doc_category")
