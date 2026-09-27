"""add chat message segments and interview user_id

Revision ID: 20260915_0005
Revises: 20260915_0004
Create Date: 2026-09-15

1. chat_messages 增加 request_segments / response_segments：只存附件段（file / image / audio）
2. interviews 增加 user_id：面试记录按 interview_id + user_id 查询，历史数据按所属会话回填
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260915_0005"
down_revision: str | None = "20260915_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "request_segments",
            sa.JSON(),
            nullable=True,
            comment='请求附件段（仅 file/image/audio），元素如 {"type":"image","resource_id":13}',
        ),
    )
    op.add_column(
        "chat_messages",
        sa.Column(
            "response_segments",
            sa.JSON(),
            nullable=True,
            comment='回复附件段（仅 file/image/audio），元素如 {"type":"file","resource_id":14}',
        ),
    )

    # 先加可空列（SQLite 不支持直接 ADD COLUMN 带外键，外键放到下面的 batch 里补）
    op.add_column("interviews", sa.Column("user_id", sa.Integer(), nullable=True, comment="所属用户 id"))
    # 历史数据回填：面试记录归属 = 其会话归属的用户（MySQL / SQLite 都支持这种关联子查询写法）
    op.execute(
        "UPDATE interviews SET user_id = "
        "(SELECT sessions.user_id FROM sessions WHERE sessions.id = interviews.session_id) "
        "WHERE user_id IS NULL"
    )
    # batch_alter_table：SQLite 走「建新表 + 搬数据」，MySQL 走普通 ALTER
    with op.batch_alter_table("interviews") as batch:
        batch.alter_column("user_id", existing_type=sa.Integer(), nullable=False)
        batch.create_foreign_key("fk_interviews_user_id_user", "user", ["user_id"], ["id"])
        batch.create_index("ix_interviews_user_id", ["user_id"])


def downgrade() -> None:
    with op.batch_alter_table("interviews") as batch:
        batch.drop_index("ix_interviews_user_id")
        batch.drop_constraint("fk_interviews_user_id_user", type_="foreignkey")
        batch.drop_column("user_id")
    op.drop_column("chat_messages", "response_segments")
    op.drop_column("chat_messages", "request_segments")
