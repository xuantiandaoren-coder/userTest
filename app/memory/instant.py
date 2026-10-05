"""短期记忆读取：从 chat_messages 读最近 N 轮 -> 字符预算裁剪 -> LangChain 消息序列。

只做「怎么查历史」这一件事，和主业务流程（stream_chat_service）解耦：

- 不含检索增强（那是 app/memory/memory.py 的 SearchBackend）
- 不拼提示词（提示词层 app/prompts/prompt_layer.py）
- 不落库、不选模型、不发网络请求

对外入口 ``InstantMemory.load(session_id)``，内部顺序固定三步：

1. ``_load_recent_rows``：按 created_at 倒序取最近 N 轮（仓储内部再翻回正序）
2. ``_fit_char_budget``：超出字符预算就从**最旧**的一轮开始丢
3. ``build_chat_history_from_rows``：提问 -> HumanMessage，回答 -> AIMessage

配置统一走 ``app.core.config.settings``（``LLM_HISTORY_TURNS`` 是默认轮数上限）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import settings
from app.db.chat_message_repository import ChatMessageRepository
from app.db.models import ChatMessage
from app.memory.memory import MemoryConfig

logger = logging.getLogger("app.memory.instant")

__all__ = ["InstantMemory", "InstantMemoryContext", "build_chat_history_from_rows"]


@dataclass(frozen=True)
class InstantMemoryContext:
    """短期记忆读取结果：原始行 + 消息序列 + 轮次 + 实际字符数。"""

    rows: list[ChatMessage]
    messages: list[BaseMessage]
    turns: list[tuple[str, str]]
    chars: int = 0
    truncated: bool = False   # 是否因为字符预算丢掉了更早的轮次

    @property
    def summary(self) -> str:
        """一行摘要，用于日志 / SSE 元事件。"""
        return f"history_turns={len(self.turns)} history_chars={self.chars} truncated={self.truncated}"


class InstantMemory:
    """短期记忆（会话内最近若干轮问答）。"""

    def __init__(self, messages: ChatMessageRepository, *, config: MemoryConfig | None = None) -> None:
        self.messages = messages
        self.config = config or MemoryConfig(max_turns=settings.llm_history_turns)

    def load(self, session_id: int) -> InstantMemoryContext:
        """读一条会话的短期记忆（读不到就是空上下文，不报错）。"""
        rows = self._load_recent_rows(session_id)
        rows, truncated = self._fit_char_budget(rows)
        context = InstantMemoryContext(
            rows=rows,
            messages=build_chat_history_from_rows(rows),
            turns=rows_to_turns(rows),
            chars=sum(row_chars(row) for row in rows),
            truncated=truncated,
        )
        if truncated:
            logger.info(
                "短期记忆按字符预算裁剪 session_id=%s 保留轮次=%s 字符=%s 上限=%s",
                session_id,
                len(context.turns),
                context.chars,
                self.config.max_chars,
            )
        return context

    def _load_recent_rows(self, session_id: int) -> list[ChatMessage]:
        """按 created_at 倒序取最近 N 轮（仓储翻回正序返回，便于按时间拼接消息）。

        一条消息就是一轮（提问 + 回答同表），取 ``max_turns * 2`` 留出余量；
        再按"最近 N 轮"截取尾部；``max_turns <= 0`` 表示不注入历史。
        """
        max_turns = max(self.config.max_turns, 0)
        if max_turns == 0:
            return []
        rows = list(self.messages.recent_by_session(session_id, max_turns * 2))
        return rows[-max_turns:]  # 只留最近 N 轮

    def _fit_char_budget(self, rows: Sequence[ChatMessage]) -> tuple[list[ChatMessage], bool]:
        """字符预算裁剪：总字符数超过 ``max_chars`` 时从最旧的轮次开始丢。"""
        budget = self.config.max_chars
        if budget <= 0:
            return [], bool(rows)

        kept = list(rows)
        truncated = False
        while kept and sum(row_chars(row) for row in kept) > budget:
            kept.pop(0)
            truncated = True
        return kept, truncated


# ---------------------------------------------------------------------------
# 纯函数：消息行 -> 轮次 / LangChain 消息
# ---------------------------------------------------------------------------
def build_chat_history_from_rows(rows: Sequence[ChatMessage]) -> list[BaseMessage]:
    """消息行 -> LangChain 消息序列：提问 HumanMessage、回答 AIMessage，空内容跳过。

    历史以角色消息进上下文（MessagesPlaceholder("history")），不塞进 system。
    """
    history: list[BaseMessage] = []
    for row in rows:
        question = (row.request_text or "").strip()
        answer = (row.response_text or "").strip()
        if question:
            history.append(HumanMessage(content=question))
        if answer:
            history.append(AIMessage(content=answer))
    return history


def rows_to_turns(rows: Sequence[ChatMessage]) -> list[tuple[str, str]]:
    """消息行 -> (提问, 回答) 列表（两边都空的行丢弃）。"""
    turns: list[tuple[str, str]] = []
    for row in rows:
        question = (row.request_text or "").strip()
        answer = (row.response_text or "").strip()
        if question or answer:
            turns.append((question, answer))
    return turns


def row_chars(row: ChatMessage) -> int:
    """一轮问答的字符数（提问 + 回答），字符预算按这个口径累计。"""
    return len(row.request_text or "") + len(row.response_text or "")
