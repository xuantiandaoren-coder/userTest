"""数据库层：聊天消息表数据访问。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from app.db.models import ChatMessage


class ChatMessageRepository:
    """基于 SQLAlchemy Session 的消息表仓储。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_by_session(self, session_id: int, *, page: int, page_size: int) -> tuple[Sequence[ChatMessage], int]:
        """分页返回会话内的消息，按时间正序（同一秒内按 id 保证顺序稳定）。"""
        condition = ChatMessage.session_id == session_id
        total = self.session.scalar(select(func.count()).select_from(ChatMessage).where(condition)) or 0
        stmt = (
            select(ChatMessage)
            .where(condition)
            .order_by(ChatMessage.created_at, ChatMessage.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return self.session.scalars(stmt).all(), total

    def delete_by_session(self, session_id: int) -> None:
        """删除会话下的全部消息。"""
        self.session.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))

    def create(
        self,
        *,
        user_id: int,
        session_id: int,
        select_model: int,
        request_id: str,
        request_text: str,
        response_text: str,
        request_segments: list[dict[str, Any]] | None = None,
        response_segments: list[dict[str, Any]] | None = None,
        file_extracted_text: str | None = None,
    ) -> ChatMessage:
        """写入一轮问答（流式聊天在流结束后调用），flush + refresh 拿到自增 id。"""
        message = ChatMessage(
            user_id=user_id,
            session_id=session_id,
            select_model=select_model,
            request_id=request_id,
            request_text=request_text,
            response_text=response_text,
            request_segments=request_segments,
            response_segments=response_segments,
            file_extracted_text=file_extracted_text,
        )
        self.session.add(message)
        self.session.flush()
        self.session.refresh(message)
        return message

    def recent_by_session(self, session_id: int, limit: int) -> Sequence[ChatMessage]:
        """返回会话内**最后** limit 条消息，按时间正序（记忆层构建历史用）。

        先按倒序取 N 条再翻正序：SQL 侧只用一次排序，避免全表排序后取尾部的写法。
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        return list(reversed(self.session.scalars(stmt).all()))

    def search_text(
        self,
        user_id: int,
        keywords: Sequence[str],
        *,
        exclude_session_id: int | None = None,
        limit: int = 60,
    ) -> Sequence[ChatMessage]:
        """按关键词做跨会话检索（记忆层的「搜索增强」数据来源）。

        命中范围：提问正文 / 回答正文 / 文件提取文本；大小写不敏感。
        这里只做候选召回，真正的排序打分在记忆层完成（见 app/memory/memory.py）。
        """
        if not keywords:
            return []
        columns = (ChatMessage.request_text, ChatMessage.response_text, ChatMessage.file_extracted_text)
        conditions = [
            func.lower(column).like(f"%{keyword.lower()}%") for keyword in keywords for column in columns
        ]
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id, or_(*conditions))
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        if exclude_session_id is not None:
            # 当前会话的历史已经进对话上下文（MessagesPlaceholder），检索只补充其它会话的资料
            stmt = stmt.where(ChatMessage.session_id != exclude_session_id)
        return self.session.scalars(stmt).all()

    def resource_ids_of_session(self, session_id: int) -> set[int]:
        """会话内所有消息附件引用的 resource_id（级联删除时收集要清理的文件）。"""
        stmt = select(ChatMessage.request_segments, ChatMessage.response_segments).where(
            ChatMessage.session_id == session_id
        )
        return self._collect_resource_ids(stmt)

    def resource_ids_in_use(self, user_id: int, *, exclude_session_id: int) -> set[int]:
        """该用户其他会话仍在引用的 resource_id（级联删除时避免删掉别处还在用的文件）。"""
        stmt = select(ChatMessage.request_segments, ChatMessage.response_segments).where(
            ChatMessage.user_id == user_id,
            ChatMessage.session_id != exclude_session_id,
        )
        return self._collect_resource_ids(stmt)

    def _collect_resource_ids(self, stmt: Select[tuple[Any, Any]]) -> set[int]:
        """把查询结果里的附件段解析成 resource_id 集合。"""
        used: set[int] = set()
        for request_segments, response_segments in self.session.execute(stmt):
            for segments in (request_segments, response_segments):
                for segment in segments or []:
                    if isinstance(segment, dict) and isinstance(segment.get("resource_id"), int):
                        used.add(segment["resource_id"])
        return used
