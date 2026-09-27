"""业务层：会话 / 聊天消息 / 面试记录。

规则：
- 正文（request_text / response_text）是聊天主字段，附件只放 *_segments（file / image / audio）
- 消息返回带 status + interview_id，供前端渲染面试卡片
- 会话删除是级联删除：消息、面试记录，以及消息附件引用到的资源（对象存储 + 元数据）
- 所有读写都按「会话 / 面试归属当前用户」过滤，越权一律 404
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.exceptions import BusinessError
from app.core.seaweedfs import SeaweedFSClient
from app.db.chat_message_repository import ChatMessageRepository
from app.db.interview_repository import InterviewRepository
from app.db.models import ChatMessage, ChatSession, Interview, Resource, User
from app.db.resource_repository import ResourceRepository
from app.db.session_repository import ChatSessionRepository
from app.schemas.chat import MessagePublic, MessageSegment, SessionCreate, SessionUpdate

SEGMENT_TYPES = frozenset({"file", "image", "audio"})


class SessionNotFoundError(BusinessError):
    """业务异常：会话不存在或不属于当前用户。"""

    code = "SESSION_NOT_FOUND"
    http_status = 404
    message = "会话不存在"


class InterviewNotFoundError(BusinessError):
    """业务异常：面试记录不存在或不属于当前用户。"""

    code = "INTERVIEW_NOT_FOUND"
    http_status = 404
    message = "面试记录不存在"


class ChatService:
    """会话与消息业务逻辑。"""

    def __init__(
        self,
        sessions: ChatSessionRepository,
        messages: ChatMessageRepository,
        interviews: InterviewRepository,
        resources: ResourceRepository,
        storage: SeaweedFSClient,
    ) -> None:
        self.sessions = sessions
        self.messages = messages
        self.interviews = interviews
        self.resources = resources
        self.storage = storage

    def list_sessions(self, user: User, *, page: int, page_size: int) -> tuple[Sequence[ChatSession], int]:
        """分页返回当前用户的会话列表（新建的在前）。"""
        return self.sessions.list_by_user(user.id, page=page, page_size=page_size)

    def create_session(self, user: User, payload: SessionCreate) -> ChatSession:
        """创建会话主题。"""
        return self.sessions.create(user_id=user.id, title=payload.title, session_model=payload.session_model)

    def rename_session(self, user: User, session_id: int, payload: SessionUpdate) -> ChatSession:
        """编辑会话标题。"""
        chat_session = self._require_session(user, session_id)
        chat_session.title = payload.title
        return self.sessions.save(chat_session)

    async def delete_session(self, user: User, session_id: int) -> None:
        """级联删除会话：面试记录 -> 消息 -> 会话 -> 附件资源（先删对象，再删元数据）。"""
        chat_session = self._require_session(user, session_id)

        resource_ids = self.messages.resource_ids_of_session(session_id)
        keep_ids = self.messages.resource_ids_in_use(user.id, exclude_session_id=session_id)

        self.interviews.delete_by_session(session_id)
        self.messages.delete_by_session(session_id)
        self.sessions.delete(chat_session)

        for resource in self.resources.find_by_ids(sorted(resource_ids - keep_ids)).values():
            await run_in_threadpool(self.storage.delete_object, resource.storage_path)
            self.resources.delete(resource)

    def list_messages(
        self, user: User, session_id: int, *, page: int, page_size: int
    ) -> tuple[list[MessagePublic], int]:
        """分页返回会话内的消息，附件段补齐 name / url / size，并带上面试卡片信息。"""
        self._require_session(user, session_id)
        messages, total = self.messages.list_by_session(session_id, page=page, page_size=page_size)
        interviews = self.interviews.find_by_message_ids([message.id for message in messages])
        resources = self.resources.find_by_ids(_segment_resource_ids(messages))

        return [self._to_public(message, interviews.get(message.id), resources) for message in messages], total

    def get_interview(self, user: User, interview_id: int) -> Interview:
        """按 interview_id + 当前用户查询面试详情（含 qa_object）。"""
        interview = self.interviews.get(interview_id, user.id)
        if interview is None:
            raise InterviewNotFoundError(detail=f"interview_id={interview_id} user_id={user.id}")
        return interview

    def _require_session(self, user: User, session_id: int) -> ChatSession:
        """取会话并校验归属，不存在或不属于当前用户时抛 404。"""
        chat_session = self.sessions.get(session_id, user.id)
        if chat_session is None:
            raise SessionNotFoundError(detail=f"session_id={session_id} user_id={user.id}")
        return chat_session

    def _to_public(
        self,
        message: ChatMessage,
        interview: Interview | None,
        resources: dict[int, Resource],
    ) -> MessagePublic:
        """消息行 -> 对外响应：正文 + 附件段 + 面试状态。"""
        return MessagePublic(
            id=message.id,
            session_id=message.session_id,
            select_model=message.select_model,
            request_id=message.request_id,
            request_text=message.request_text,
            response_text=message.response_text,
            request_segments=_enrich(message.request_segments, resources, self.storage),
            response_segments=_enrich(message.response_segments, resources, self.storage),
            status=interview.status if interview else None,
            interview_id=interview.id if interview else None,
            created_at=message.created_at,
        )


def _segment_resource_ids(messages: Sequence[ChatMessage]) -> list[int]:
    """收集这一页消息附件引用的 resource_id，便于一次查库补齐。"""
    ids: set[int] = set()
    for message in messages:
        for segments in (message.request_segments, message.response_segments):
            for segment in segments or []:
                if isinstance(segment, dict) and isinstance(segment.get("resource_id"), int):
                    ids.add(segment["resource_id"])
    return sorted(ids)


def _enrich(
    segments: Sequence[Any] | None,
    resources: dict[int, Resource],
    storage: SeaweedFSClient,
) -> list[MessageSegment]:
    """把落库的附件段（type + resource_id）补成前端可直接渲染的段。"""
    enriched: list[MessageSegment] = []
    for segment in segments or []:
        if not isinstance(segment, dict) or segment.get("type") not in SEGMENT_TYPES:
            continue  # 跳过异常数据，避免一条脏段让整个接口报错
        resource_id = segment.get("resource_id")
        if not isinstance(resource_id, int):  # 同样跳过缺 resource_id 的脏段
            continue
        resource = resources.get(resource_id)
        enriched.append(
            MessageSegment(
                type=segment["type"],
                resource_id=resource_id,
                name=resource.file_name if resource else None,
                url=storage.url_for(resource.storage_path) if resource else None,
            )
        )
    return enriched
