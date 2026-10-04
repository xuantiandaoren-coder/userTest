"""校验层：会话 / 聊天消息 / 面试记录模型。

- 正文用 request_text / response_text
- 附件用 *_segments，只允许 file / image / audio 三类段
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SegmentType = Literal["file", "image", "audio"]


class SessionCreate(BaseModel):
    """创建会话：主题 + 会话类型。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255, description="会话标题")
    session_model: int = Field(ge=0, le=2, description="会话类型：0=学习，1=面试，2=笔记")


class SessionUpdate(BaseModel):
    """编辑会话标题。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255, description="新的会话标题")


class SessionPublic(BaseModel):
    """会话对外响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    session_model: int
    created_at: int


class MessageSegment(BaseModel):
    """附件段：type 只有 file / image / audio，resource_id 指向 resources 表。

    name / url 由后端按 resource_id 关联补全，落库时只存 type + resource_id。
    """

    type: SegmentType
    resource_id: int
    name: str | None = None
    url: str | None = None


class MessageSource(BaseModel):
    """回答引用的知识片段。

    落库只存 chunk_id + score（见 chat_messages.reference_sources），
    其余字段（资源 / 分块序号 / 文件名 / 原文）在读取时回查 knowledge_chunks + resources。
    片段被删除时 text 为「该参考片段已经删除」，出处字段为 null。
    """

    chunk_id: str = Field(description="知识分块 id，等于 knowledge_chunks.vector_id（Qdrant point id）")
    resource_id: int | None = Field(default=None, description="所属资源 id（resources.id）；片段已删除时为 null")
    chunk_index: int | None = Field(default=None, description="文件内分块序号，从 0 开始")
    file_name: str | None = Field(default=None, description="来源文件名")
    score: float | None = Field(default=None, description="检索相似度")
    text: str = Field(description="分块原文；片段已删除时为「该参考片段已经删除」")


class MessagePublic(BaseModel):
    """聊天消息对外响应：正文 + 附件段 + 面试卡片信息。"""

    id: int
    session_id: int
    select_model: int
    request_id: str
    request_text: str
    response_text: str
    request_segments: list[MessageSegment] = Field(default_factory=list)
    response_segments: list[MessageSegment] = Field(default_factory=list)
    sources: list[MessageSource] = Field(default_factory=list, description="本轮回答引用的知识片段（历史按引用回查）")
    status: int | None = Field(default=None, description="关联面试的状态：0=进行中，1=已结束，2=异常终止；非面试消息为 null")
    interview_id: int | None = Field(default=None, description="该消息开启的面试记录 id，非面试消息为 null")
    created_at: int


class InterviewPublic(BaseModel):
    """面试详情：问答列表 + 状态，供面试卡片 / 复盘页使用。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    message_id: int
    qa_object: list[dict[str, Any]]
    interview_duration: int
    status: int
    created_at: int
    updated_at: int


class StreamChatRequest(BaseModel):
    """流式聊天请求（POST /sessions/{session_id}/stream-chat）。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=8000, description="本轮提问正文")
    agent_name: str | None = Field(
        default=None,
        max_length=64,
        description="智能体名（见 AGENT_CONFIG）；留空按会话类型自动选择",
    )
    scene: str | None = Field(default=None, max_length=64, description="模板场景；留空取该智能体的默认场景")
    use_search: bool = Field(default=True, description="是否启用记忆层的搜索增强")
    search_query: str | None = Field(
        default=None,
        max_length=500,
        description="检索关键词；留空用 message 抽取",
    )


class StreamChatRequestPublic(BaseModel):
    """流式聊天请求的落库摘要（SSE done 事件里回带，便于前端对齐）。"""

    request_id: str
    message_id: int | None = None
    session_id: int
    agent_name: str
    scene: str
    prompt_versions: str
    search_hits: int = 0
