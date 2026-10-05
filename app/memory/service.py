"""记忆层统一入口：把「短期记忆 + 搜索增强」组装成一次对话的记忆上下文。

分工：

===========  ==========================================================
短期记忆      ``InstantMemory``（app/memory/instant.py）：本会话最近 N 轮，按字符预算裁剪
搜索增强      ``SearchBackend``（app/memory/memory.py）：跨会话关键词召回相关片段
工作记忆      预留：当前任务的中间状态（如本轮的检索结果、待确认项）
长期记忆      预留：跨会话沉淀的知识与用户画像摘要
===========  ==========================================================

上层（``StreamChatService``）只依赖 ``load()`` 一个方法，后续接入工作记忆 / 长期记忆时
在 ``load()`` 内部扩展即可，调用方不用改。组合好的结果仍是 ``MemoryContext``，
提示词层只认这个结构。
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.db.chat_message_repository import ChatMessageRepository
from app.memory.instant import InstantMemory
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
    """记忆层入口：短期记忆 + 搜索增强（工作记忆 / 长期记忆的接入点）。"""

    def __init__(
        self,
        messages: ChatMessageRepository,
        *,
        config: MemoryConfig | None = None,
        instant: InstantMemory | None = None,
        search_backend: SearchBackend | None = None,
    ) -> None:
        self.messages = messages
        self.config = config or MemoryConfig(max_turns=settings.llm_history_turns)
        self.instant = instant or InstantMemory(messages, config=self.config)
        self.search_backend = search_backend or DatabaseSearchBackend(messages)

    def load(
        self,
        *,
        user_id: int,
        session_id: int,
        query: str,
        use_search: bool | None = None,
    ) -> MemoryContext:
        """构建本次对话的记忆上下文（短期记忆 + 可选搜索增强）。

        ``use_search=None`` 时按 ``MemoryConfig.search_enabled`` 决定；
        搜索增强失败只记 warning 并降级（历史照常注入），不让整轮对话失败。
        """
        instant = self.instant.load(session_id)
        context = MemoryContext(turns=instant.turns, messages=instant.messages)

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
