"""数据库层：ORM 模型（SQLAlchemy 2.0 声明式风格）。

包含用户、会话、消息、面试记录、资源元数据、提示词模板、用户背景画像七张表。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.mysql import MEDIUMTEXT, TINYINT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.expression import FunctionElement

from app.db.base import Base

def medium_text() -> Text:
    """MySQL 用 MEDIUMTEXT；测试用的 SQLite 无法编译该类型，退化为 TEXT。"""
    return Text().with_variant(MEDIUMTEXT(), "mysql")

def tiny_int() -> Integer:
    """MySQL 用 TINYINT；测试用的 SQLite 无法编译该类型，退化为 INTEGER。"""
    return Integer().with_variant(TINYINT(), "mysql")

def big_int() -> BigInteger:
    """MySQL 用 BIGINT；SQLite 退化为 INTEGER，保证主键自增行为一致。"""
    return BigInteger().with_variant(Integer(), "sqlite")

class UnixTimestamp(FunctionElement):
    """数据库侧默认值：Unix 秒时间戳。

    按方言渲染：MySQL 用 UNIX_TIMESTAMP()，SQLite（测试）用 strftime()。
    """

    type = BigInteger()
    inherit_cache = True

@compiles(UnixTimestamp)
def _compile_unix_timestamp(element: UnixTimestamp, compiler: Any, **kw: Any) -> str:
    return "(UNIX_TIMESTAMP())"

@compiles(UnixTimestamp, "sqlite")
def _compile_unix_timestamp_sqlite(element: UnixTimestamp, compiler: Any, **kw: Any) -> str:
    return "CAST(strftime('%s', 'now') AS INTEGER)"

def unix_timestamp() -> UnixTimestamp:
    """列默认值：由数据库写入当前 Unix 秒时间戳。"""
    return UnixTimestamp()

def _now_unix() -> int:
    """Python 侧当前 Unix 秒，用于 ORM 更新时刷新 updated_at。"""
    return int(time.time())

class User(Base):
    """用户表。

    - id：自增主键
    - userName：唯一索引（数据库列名保持驼峰，Python 侧为 user_name）
    - password：密码哈希（不落库明文）
    - avatar：头像文件相对路径（上传图片时写入）
    - create_time：由数据库自动填充当前时间
    """

    __tablename__ = "user"
    __table_args__ = (
        Index("uk_user_userName", "userName", unique=True),
        {"comment": "用户信息表", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    user_name: Mapped[str] = mapped_column("userName", String(20), nullable=False, comment="用户名")
    password: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码哈希")
    avatar: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="头像文件相对路径")
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        comment="创建时间",
    )

class ChatSession(Base):
    """会话表：一次学习 / 面试 / 笔记会话。

    - user_id：所属用户，外键关联 user.id
    - session_model：0=学习，1=面试，2=笔记
    - created_at：创建时间（Unix 秒，数据库默认填充）
    """

    __tablename__ = "sessions"
    __table_args__ = {"comment": "会话表", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id"),
        nullable=False,
        index=True,
        comment="所属用户 id",
    )
    session_model: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="会话类型：0=学习，1=面试，2=笔记",
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, comment="会话标题")
    created_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=unix_timestamp(),
        comment="会话创建时间（Unix 秒）",
    )

class ChatMessage(Base):
    """消息表：一轮提问与回答，隶属于某个会话。

    - user_id / session_id：外键，分别关联 user.id、sessions.id
    - select_model：0=默认，1=知识精讲，2=刷题，3=简历优化，4=模拟面试，5=面试复盘
    - request_id：请求唯一标识，用于幂等 / 链路追踪
    - request_text / response_text：聊天正文主字段（提问 / 回答的完整文本）
    - request_segments / response_segments：只存附件段（file / image / audio），
      元素形如 ``{"type": "image", "resource_id": 13}``，无附件时为 NULL；
      返回时按 resource_id 关联 resources 补上 name / url / size
    - file_extracted_text：上传文件中提取的文本，作为对话上下文，无文件时为空
    - (session_id, created_at) 复合索引：按会话拉取消息列表
    """

    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_id_created_at", "session_id", "created_at"),
        {"comment": "消息表", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False, comment="所属用户 id")
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False, comment="所属会话 id")
    select_model: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="选择模式：0=默认，1=知识精讲，2=刷题，3=简历优化，4=模拟面试，5=面试复盘",
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True, comment="请求唯一标识")
    request_text: Mapped[str] = mapped_column(medium_text(), nullable=False, comment="提问文本")
    response_text: Mapped[str] = mapped_column(medium_text(), nullable=False, comment="回答文本")
    request_segments: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="请求附件段（仅 file/image/audio），元素如 {\"type\":\"image\",\"resource_id\":13}",
    )
    response_segments: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="回复附件段（仅 file/image/audio），元素如 {\"type\":\"file\",\"resource_id\":14}",
    )
    file_extracted_text: Mapped[str | None] = mapped_column(
        medium_text(),
        nullable=True,
        comment="从文件中提取的完整文本（对话上下文用）",
    )
    created_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=unix_timestamp(),
        comment="创建时间（Unix 秒）",
    )

class Interview(Base):
    """面试记录表：一次模拟面试的问答集合。

    - message_id：唯一，指向开启本次模拟面试的入口消息
    - user_id：所属用户，外部按 interview_id + user_id 查询，避免越权读取他人面试
    - qa_object：一问一答列表，元素形如
      ``{"id": uuid, "question": str, "answer": str, "created_at": Unix 秒}``
    - interview_duration：累计面试时长（秒）
    - status：0=进行中，1=已结束，2=异常终止
    - updated_at：ORM 更新时自动刷新为当前 Unix 秒
    """

    __tablename__ = "interviews"
    __table_args__ = (
        Index("ix_interviews_session_id_message_id", "session_id", "message_id"),
        {"comment": "面试记录表", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False, comment="所属会话 id")
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id"),
        nullable=False,
        index=True,
        comment="所属用户 id",
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_messages.id"),
        nullable=False,
        unique=True,
        comment="开启本次模拟面试的入口消息 id",
    )
    qa_object: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, comment="一问一答对象列表")
    interview_duration: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="累计面试时长（秒）",
    )
    status: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        index=True,
        comment="面试状态：0=进行中，1=已结束，2=异常终止",
    )
    created_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=unix_timestamp(),
        comment="面试开始时间（Unix 秒）",
    )
    updated_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=unix_timestamp(),
        onupdate=_now_unix,
        comment="面试更新时间（Unix 秒）",
    )

class Resource(Base):
    """资源元数据表：原文件存 SeaweedFS，本表只存元数据，两者解耦。

    - resource_type：0=文件，1=图片，2=音频
    - doc_category：文档分类（resume/study_material/general），仅 resource_type=0(文件) 有意义，其余为 NULL
    - storage_scene：0=长过期（1 个月），1=短过期（2 小时），2=只提取内容不存原文件
    - upload_purpose：0=普通资源，1=用户头像（仅图片会把对象键写进 user.avatar）
    - file_hash：文件内容 MD5；(file_hash, user_id) 唯一索引实现用户级去重
    - storage_path：SeaweedFS 对象键；expire_time 到期后由清理任务先删对象、再删本行
    """

    __tablename__ = "resources"
    __table_args__ = (
        Index("uk_file_hash_user_id", "file_hash", "user_id", unique=True),
        {"comment": "资源元数据表", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )

    id: Mapped[int] = mapped_column(big_int(), primary_key=True, autoincrement=True, comment="资源主键ID")
    resource_type: Mapped[int] = mapped_column(
        tiny_int(),
        nullable=False,
        comment="资源类型：0=文件，1=图片，2=音频",
    )
    doc_category: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="文档分类：resume/study_material/general；仅文件类型(resource_type=0)有意义，其余为空",
    )
    storage_scene: Mapped[int] = mapped_column(
        tiny_int(),
        nullable=False,
        server_default=text("0"),
        comment="存储场景：0=长过期(1个月)，1=短过期(2小时)，2=只提取内容不存原文件",
    )
    upload_purpose: Mapped[int] = mapped_column(
        tiny_int(),
        nullable=False,
        server_default=text("0"),
        comment="上传用途：0=普通资源，1=用户头像",
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, comment="用户上传原始文件名")
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, comment="文件MD5，去重核心字段")
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False, comment="SeaweedFS 对象存储路径")
    user_id: Mapped[int] = mapped_column(big_int(), nullable=False, comment="上传用户ID")
    expire_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="资源过期时间")
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        comment="创建时间",
    )


class PromptTemplate(Base):
    """提示词模板版本表：一个 (agent_name, scene) 一组，多版本共存、单版本生效。

    - agent_name：私有模板填智能体名（如 tutor）；公共模板为 NULL
    - scene：场景（workflow / resume / quiz / interview / evaluation / study / note），公共模板按 scene 被所有智能体复用
    - template_type：1=私有，2=公共（与 agent_name 是否为空互相印证）
    - variables：模板变量名列表（JSON 数组字符串，如 ["target_job","weak_topics"]）
    - version：同一 (agent_name, scene) 内整数自增，从 1 开始
    - is_active：0/1，同一 (agent_name, scene) 内最多一条为 1（回滚即切换这一列）

    说明：agent_name 为 NULL 的公共模板在 MySQL / SQLite 的唯一索引里 NULL 互不相等，
    因此「同组仅一条生效」与「版本不重复」都由服务层在事务内保证，这里只建普通索引。
    """

    __tablename__ = "prompt_templates"
    __table_args__ = (
        Index("ix_prompt_templates_agent_scene", "agent_name", "scene", "is_active"),
        {"comment": "提示词模板版本表", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    agent_name: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="智能体名；公共模板为 NULL",
    )
    scene: Mapped[str] = mapped_column(String(64), nullable=False, comment="场景：workflow/resume/quiz/interview/evaluation/study/note")
    template_type: Mapped[int] = mapped_column(
        tiny_int(),
        nullable=False,
        server_default=text("1"),
        comment="模板类型：1=私有（agent_name 有值），2=公共（agent_name 为空）",
    )
    template_content: Mapped[str] = mapped_column(medium_text(), nullable=False, comment="提示词模板正文，用 {变量} 占位")
    variables: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment='变量名列表（JSON 数组字符串），如 ["target_job","weak_topics"]',
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
        comment="版本号：同一 (agent_name, scene) 内整数自增",
    )
    is_active: Mapped[int] = mapped_column(
        tiny_int(),
        nullable=False,
        server_default=text("0"),
        comment="是否生效：0=否，1=是；同组最多一条为 1",
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="版本说明（改了什么）")
    created_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=unix_timestamp(),
        comment="创建时间（Unix 秒）",
    )


class UserProfile(Base):
    """用户背景画像表：提示词变量注入的数据来源（一人一行）。

    - target_job / years_experience / target_level：目标岗位、工作经验、目标等级
    - target_skills / weak_topics：已掌握技能、薄弱点（JSON 数组，注入前转成顿号分隔文本）

    注入前会经过「验证层 -> 转换层 -> 填充层」三层处理，详见 app/prompts/injector.py。
    """

    __tablename__ = "user_profiles"
    __table_args__ = (
        Index("uk_user_profiles_user_id", "user_id", unique=True),
        {"comment": "用户背景画像表（提示词变量来源）", "mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id"),
        nullable=False,
        comment="所属用户 id（一人一行）",
    )
    target_job: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="目标岗位")
    years_experience: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="工作经验（年）")
    target_level: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="目标等级，如 P6 / 高级")
    target_skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True, comment="已掌握技能（JSON 数组）")
    weak_topics: Mapped[list[str] | None] = mapped_column(JSON, nullable=True, comment="薄弱点（JSON 数组）")
    created_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=unix_timestamp(),
        comment="创建时间（Unix 秒）",
    )
