"""记忆层统一入口：把「短期记忆 + 搜索增强 + 长期记忆」组装成一次对话的记忆上下文。

分工：

===========  ==========================================================
短期记忆      ``InstantMemory``（app/memory/instant.py）：本会话最近 N 轮，按字符预算裁剪
搜索增强      ``SearchBackend``（app/memory/memory.py）：跨会话关键词召回相关片段
长期记忆      ``LongTermMemory``（app/memory/long_term_memory.py）：跨会话用户画像，注入为 SystemMessage
工作记忆      预留：当前任务的中间状态（如本轮的检索结果、待确认项）
===========  ==========================================================

上层（``StreamChatService``）只依赖 ``load()`` 一个方法，后续接入工作记忆时
在 ``load()`` 内部扩展即可，调用方不用改。组合好的结果仍是 ``MemoryContext``，
提示词层只认这个结构。

长期画像的注入方式与搜索增强不同：搜索增强是拼进 system 的文本块，长期画像则包装成
``SystemMessage`` 插到历史消息列表最前面，由提示词层在组装最终 system 时统一合并
（见 app/prompts/prompt_layer.py 的 ``merge_system_messages``），
这样画像既能进 system，又不会污染历史消息的角色序列。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.messages import SystemMessage
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.chat_message_repository import ChatMessageRepository
from app.db.user_profile_repository import UserProfileRepository
from app.memory.instant import InstantMemory
from app.memory.long_term_memory import LongTermMemory
from app.memory.memory import (
    DatabaseSearchBackend,
    MemoryConfig,
    MemoryContext,
    SearchBackend,
    render_search_context,
)

logger = logging.getLogger("app.memory.service")

__all__ = ["MemoryService"]


class MemoryService:
    """记忆层入口：短期记忆 + 搜索增强 + 长期记忆（工作记忆的接入点）。"""

    def __init__(
        self,
        messages: ChatMessageRepository,
        *,
        config: MemoryConfig | None = None,
        instant: InstantMemory | None = None,
        search_backend: SearchBackend | None = None,
        profiles: UserProfileRepository | None = None,
        session_factory: Callable[[], Session] | None = None,
        model_factory: Callable[[], object] | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self.messages = messages
        self.config = config or MemoryConfig(max_turns=settings.llm_history_turns)
        self.instant = instant or InstantMemory(messages, config=self.config)
        self.search_backend = search_backend or DatabaseSearchBackend(messages)
        # 长期记忆：不传实现时按默认依赖构造（读用请求会话、写用独立会话工厂）
        self.long_term = long_term or LongTermMemory(
            profiles=profiles if profiles is not None else UserProfileRepository(messages.session),
            session_factory=session_factory,
            model_factory=model_factory,  # type: ignore[arg-type]
        )

    def load(
        self,
        *,
        user_id: int,
        session_id: int,
        query: str,
        use_search: bool | None = None,
        enable_long_term: bool = True,
    ) -> MemoryContext:
        """构建本次对话的记忆上下文（短期记忆 + 可选搜索增强 + 可选长期画像）。

        ``use_search=None`` 时按 ``MemoryConfig.search_enabled`` 决定；
        搜索增强失败只记 warning 并降级（历史照常注入），不让整轮对话失败。

        ``enable_long_term=True`` 时读用户长期画像；有内容就包装成 ``SystemMessage``
        插到历史消息最前面（角色为 system，提示词层会把它合并进最终 system）。
        """
        instant = self.instant.load(session_id)
        context = MemoryContext(turns=instant.turns, messages=instant.messages)

        if enable_long_term:
            self._attach_long_term_memory(context, user_id)

        want_search = self.config.search_enabled if use_search is None else use_search
        if not want_search:
            return context

        try:
            hits = self.search_backend.search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                limit=self.config.search_candidate_limit,
            )
        except Exception as exc:  # noqa: BLE001 - 检索是增强项：失败不能让聊天整体不可用
            logger.warning("搜索增强失败，已跳过：%r", exc)
            context.warnings.append("搜索增强失败，本次未注入检索资料")
            return context

        context.search_hits = hits[: self.config.search_top_k]
        context.search_context = render_search_context(context.search_hits)
        return context

    def _attach_long_term_memory(self, context: MemoryContext, user_id: int) -> None:
        """读长期画像并插入历史消息最前面；失败只记日志，不影响本轮对话。"""
        try:
            memory_text = self.long_term.load_memory_context(user_id)
        except Exception as exc:  # noqa: BLE001 - 长期记忆是增强项，失败要能降级
            logger.warning("长期记忆读取失败，已跳过 user_id=%s：%r", user_id, exc)
            context.warnings.append("长期记忆读取失败，本次未注入用户画像")
            return
        if not memory_text:
            return
        context.long_term_text = memory_text
        context.messages.insert(0, SystemMessage(content=memory_text))
