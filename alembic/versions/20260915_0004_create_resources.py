"""create resources table

Revision ID: 20260915_0004
Revises: 20260914_0003
Create Date: 2026-09-15

新增资源元数据表：MySQL 只存元数据，原文件放 SeaweedFS，
(file_hash, user_id) 唯一索引实现用户级去重。与 app/db/models.py 的 Resource 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "20260915_0004"
down_revision: str | None = "20260914_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tiny_int() -> sa.Integer:
    """MySQL 用 TINYINT，SQLite（测试）退化为 INTEGER。"""
    return sa.Integer().with_variant(mysql.TINYINT(), "mysql")


def _big_int() -> sa.BigInteger:
    """MySQL 用 BIGINT，SQLite（测试）退化为 INTEGER 以保留主键自增。"""
    return sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "resources",
        sa.Column("id", _big_int(), autoincrement=True, nullable=False, comment="资源主键ID"),
        sa.Column("resource_type", _tiny_int(), nullable=False, comment="资源类型：0=文件，1=图片，2=音频"),
        sa.Column(
            "storage_scene",
            _tiny_int(),
            nullable=False,
            server_default=sa.text("0"),
            comment="存储场景：0=长过期(1个月)，1=短过期(2小时)，2=只提取内容不存原文件",
        ),
        sa.Column(
            "upload_purpose",
            _tiny_int(),
            nullable=False,
            server_default=sa.text("0"),
            comment="上传用途：0=普通资源，1=用户头像",
        ),
        sa.Column("file_name", sa.String(length=255), nullable=False, comment="用户上传原始文件名"),
        sa.Column("file_hash", sa.String(length=64), nullable=False, comment="文件MD5，去重核心字段"),
        sa.Column("storage_path", sa.String(length=512), nullable=False, comment="SeaweedFS 对象存储路径"),
        sa.Column("user_id", _big_int(), nullable=False, comment="上传用户ID"),
        sa.Column("expire_time", sa.DateTime(), nullable=True, comment="资源过期时间"),
        sa.Column(
            "create_time",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
            comment="创建时间",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resources")),
        comment="资源元数据表",
        mysql_charset="utf8mb4",
        mysql_engine="InnoDB",
    )
    op.create_index("uk_file_hash_user_id", "resources", ["file_hash", "user_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uk_file_hash_user_id", table_name="resources")
    op.drop_table("resources")
