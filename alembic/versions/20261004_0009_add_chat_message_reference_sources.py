"""add chat_messages.reference_sources

Revision ID: 20261004_0009
Revises: 20261004_0008
Create Date: 2026-10-04

chat_messages 增加 reference_sources：本轮回答引用的知识片段，只存引用不存全文
（元素形如 {"chunk_id": "<knowledge_chunks.vector_id>", "score": 0.78}），
历史消息接口据此回查 knowledge_chunks + resources 重新拼出 sources。
与 app/db/models.py 的 ChatMessage 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20261004_0009"
down_revision: str | None = "20261004_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "reference_sources",
            sa.JSON(),
            nullable=True,
            comment='回答引用的知识片段（只存引用不存全文），元素如 {"chunk_id":"<vector_id>","score":0.78}',
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "reference_sources")
