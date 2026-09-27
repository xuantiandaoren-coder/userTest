"""create prompt_templates and user_profiles tables

Revision ID: 20260915_0006
Revises: 20260915_0005
Create Date: 2026-09-15

两张新表，支撑「提示词版本管理 + 变量注入」：

1. prompt_templates：提示词模板版本表
   - 私有模板 agent_name 有值 + template_type=1；公共模板 agent_name=NULL + template_type=2
   - 同一 (agent_name, scene) 内 version 整数自增，且只有一条 is_active=1（服务层在事务内保证）
   - variables 存 JSON 数组字符串；template_content 用 MEDIUMTEXT
2. user_profiles：用户背景画像表（一人一行），提示词变量注入的数据来源
   - target_skills / weak_topics 为 JSON 数组
   - user_id 唯一索引 + 外键指向 user.id

与 app/db/models.py 的 PromptTemplate / UserProfile 保持一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "20260915_0006"
down_revision: str | None = "20260915_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_KWARGS = {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"}


def _tiny_int() -> sa.Integer:
    """MySQL 用 TINYINT，SQLite（测试）退化为 INTEGER。"""
    return sa.Integer().with_variant(mysql.TINYINT(), "mysql")


def _medium_text() -> sa.Text:
    """MySQL 用 MEDIUMTEXT，SQLite（测试）退化为 TEXT。"""
    return sa.Text().with_variant(mysql.MEDIUMTEXT(), "mysql")


def _unix_timestamp() -> sa.TextClause:
    """Unix 秒时间戳的数据库默认值。"""
    return sa.text("(UNIX_TIMESTAMP())")


def upgrade() -> None:
    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("agent_name", sa.String(length=64), nullable=True, comment="智能体名；公共模板为 NULL"),
        sa.Column("scene", sa.String(length=64), nullable=False, comment="场景：chat/resume/interview/plan"),
        sa.Column(
            "template_type",
            _tiny_int(),
            nullable=False,
            server_default=sa.text("1"),
            comment="模板类型：1=私有（agent_name 有值），2=公共（agent_name 为空）",
        ),
        sa.Column("template_content", _medium_text(), nullable=False, comment="提示词模板正文，用 {变量} 占位"),
        sa.Column(
            "variables",
            sa.String(length=512),
            nullable=True,
            comment='变量名列表（JSON 数组字符串），如 ["target_job","weak_topics"]',
        ),
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
            comment="版本号：同一 (agent_name, scene) 内整数自增",
        ),
        sa.Column(
            "is_active",
            _tiny_int(),
            nullable=False,
            server_default=sa.text("0"),
            comment="是否生效：0=否，1=是；同组最多一条为 1",
        ),
        sa.Column("description", sa.String(length=255), nullable=True, comment="版本说明（改了什么）"),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="创建时间（Unix 秒）",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prompt_templates")),
        comment="提示词模板版本表",
        **TABLE_KWARGS,
    )
    op.create_index(
        "ix_prompt_templates_agent_scene",
        "prompt_templates",
        ["agent_name", "scene", "is_active"],
    )

    op.create_table(
        "user_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="自增主键"),
        sa.Column("user_id", sa.Integer(), nullable=False, comment="所属用户 id（一人一行）"),
        sa.Column("target_job", sa.String(length=128), nullable=True, comment="目标岗位"),
        sa.Column("years_experience", sa.Integer(), nullable=True, comment="工作经验（年）"),
        sa.Column("target_level", sa.String(length=32), nullable=True, comment="目标等级，如 P6 / 高级"),
        sa.Column("target_skills", sa.JSON(), nullable=True, comment="已掌握技能（JSON 数组）"),
        sa.Column("weak_topics", sa.JSON(), nullable=True, comment="薄弱点（JSON 数组）"),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            nullable=False,
            server_default=_unix_timestamp(),
            comment="创建时间（Unix 秒）",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_profiles")),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], name=op.f("fk_user_profiles_user_id_user")),
        comment="用户背景画像表（提示词变量来源）",
        **TABLE_KWARGS,
    )
    op.create_index("uk_user_profiles_user_id", "user_profiles", ["user_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uk_user_profiles_user_id", table_name="user_profiles")
    op.drop_table("user_profiles")
    op.drop_index("ix_prompt_templates_agent_scene", table_name="prompt_templates")
    op.drop_table("prompt_templates")
