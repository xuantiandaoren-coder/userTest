"""create knowledge_chunks table

Revision ID: 20261004_0008
Revises: 20260930_0007
Create Date: 2026-10-04

新增知识库分块原文表：向量存 Qdrant（只存向量 + 过滤字段），原文存本表，
两边用入库时生成的同一个 UUID 关联（Qdrant point id == knowledge_chunks.vector_id）。
resource_id 外键指向 resources.id，资源过期被清理时本表跟随删除。
与 app/db/models.py 的 KnowledgeChunk 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "20261004_0008"
down_revision: str | None = "20260930_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _big_int() -> sa.BigInteger:
    """MySQL 用 BIGINT，SQLite（测试）退化为 INTEGER 以保留主键自增。"""
    return sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def _medium_text() -> sa.Text:
    """MySQL 用 MEDIUMTEXT，SQLite 无法编译该类型，退化为 TEXT。"""
    return sa.Text().with_variant(mysql.MEDIUMTEXT(), "mysql")


def _unix_timestamp() -> sa.TextClause:
    """数据库侧默认值：当前 Unix 秒时间戳（与 sessions / chat_messages 一致）。"""
    return sa.text("(UNIX_TIMESTAMP())")


def upgrade() -> None:
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", _big_int(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("resource_id", _big_int(), nullable=False, comment="所属资源ID（resources.id）"),
        sa.Column(
            "vector_id",
            sa.String(length=36),
            nullable=False,
            comment="Qdrant point id：入库时生成的 UUID，与向量库同值关联",
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False, comment="文件内分块序号，从 0 开始"),
        sa.Column("char_count", sa.Integer(), nullable=False, comment="分块字符数"),
        sa.Column("text", _medium_text(), nullable=False, comment="分块原文"),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="入库时间（Unix 秒）",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunks")),
        sa.ForeignKeyConstraint(
            ["resource_id"],
            ["resources.id"],
            name=op.f("fk_knowledge_chunks_resource_id_resources"),
            ondelete="CASCADE",
        ),
        comment="知识库分块原文表",
        mysql_charset="utf8mb4",
        mysql_engine="InnoDB",
    )
    op.create_index("uk_knowledge_chunks_vector_id", "knowledge_chunks", ["vector_id"], unique=True)
    op.create_index("ix_knowledge_chunks_resource_id", "knowledge_chunks", ["resource_id"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_resource_id", table_name="knowledge_chunks")
    op.drop_index("uk_knowledge_chunks_vector_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
