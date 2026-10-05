"""短期记忆读取：InstantMemory.load() 的查询顺序、字符预算裁剪与消息格式化。

只测记忆读取本身（不涉及检索增强 / 提示词 / 模型），数据用内存 SQLite。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from app.db.chat_message_repository import ChatMessageRepository
from app.db.models import ChatMessage
from app.memory.instant import (
    InstantMemory,
    build_chat_history_from_rows,
    row_chars,
    rows_to_turns,
)
from app.memory.memory import MemoryConfig

SESSION_ID = 1
USER_ID = 7


def _add(
    db: Session,
    *,
    request_text: str,
    response_text: str = "回答",
    session_id: int = SESSION_ID,
    request_id: str = "req",
) -> ChatMessage:
    message = ChatMessage(
        user_id=USER_ID,
        session_id=session_id,
        select_model=0,
        request_id=request_id,
        request_text=request_text,
        response_text=response_text,
    )
    db.add(message)
    db.flush()
    return message


def _instant(db: Session, **config: object) -> InstantMemory:
    return InstantMemory(ChatMessageRepository(db), config=MemoryConfig(**config))  # type: ignore[arg-type]


def test_load_reads_recent_turns_in_time_order(db_session: Session) -> None:
    """只保留最近 N 轮，且按时间正序（提问 -> 回答）。"""
    for index in range(1, 6):
        _add(db_session, request_text=f"问题{index}", response_text=f"回答{index}", request_id=f"r{index}")

    context = _instant(db_session, max_turns=2, max_chars=2000).load(SESSION_ID)

    assert context.turns == [("问题4", "回答4"), ("问题5", "回答5")]
    assert [type(item) for item in context.messages] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert [item.content for item in context.messages] == ["问题4", "回答4", "问题5", "回答5"]
    assert context.truncated is False


def test_load_returns_ascending_order_even_though_query_is_desc(db_session: Session) -> None:
    """底层按 created_at 倒序取，返回给模型必须是正序（否则历史被读反）。"""
    first = _add(db_session, request_text="最早的问题", request_id="r1")
    second = _add(db_session, request_text="最新的问题", request_id="r2")
    # created_at 是数据库侧默认的 Unix 秒，同一秒内靠 id 保证顺序
    assert second.id > first.id

    context = _instant(db_session, max_turns=5).load(SESSION_ID)

    assert [question for question, _ in context.turns] == ["最早的问题", "最新的问题"]


def test_fit_char_budget_drops_oldest_turns(db_session: Session) -> None:
    """超出字符预算时从最旧的一轮开始丢，并标记 truncated。"""
    _add(db_session, request_text="旧" * 15, response_text="旧答", request_id="old")
    _add(db_session, request_text="新问题", response_text="新回答", request_id="new")

    context = _instant(db_session, max_turns=5, max_chars=20).load(SESSION_ID)

    assert context.turns == [("新问题", "新回答")]
    assert context.truncated is True
    assert context.chars == row_chars(context.rows[0])
    assert "truncated=True" in context.summary


def test_fit_char_budget_keeps_everything_within_budget(db_session: Session) -> None:
    """预算够时一轮都不丢。"""
    _add(db_session, request_text="问题一", response_text="回答一", request_id="r1")
    _add(db_session, request_text="问题二", response_text="回答二", request_id="r2")

    context = _instant(db_session, max_turns=5, max_chars=10_000).load(SESSION_ID)

    assert len(context.turns) == 2
    assert context.truncated is False
    assert context.chars == 6 + 6  # (问题一 + 回答一) * 2 轮


def test_load_zero_turns_or_zero_budget_returns_empty(db_session: Session) -> None:
    """max_turns=0 或 max_chars=0 表示不注入历史（不是报错）。"""
    _add(db_session, request_text="问题", response_text="回答")

    no_turns = _instant(db_session, max_turns=0).load(SESSION_ID)
    no_budget = _instant(db_session, max_turns=5, max_chars=0).load(SESSION_ID)

    assert no_turns.messages == [] and no_turns.turns == []
    assert no_budget.messages == [] and no_budget.truncated is True


def test_load_empty_session_returns_empty_context(db_session: Session) -> None:
    """会话还没有消息时返回空上下文。"""
    context = _instant(db_session).load(SESSION_ID)

    assert context.rows == [] and context.messages == [] and context.turns == []
    assert context.chars == 0 and context.truncated is False


def test_load_ignores_other_sessions(db_session: Session) -> None:
    """只读本会话的历史，别的会话不混进来。"""
    _add(db_session, request_text="本会话问题", request_id="mine")
    _add(db_session, request_text="别的会话问题", session_id=SESSION_ID + 1, request_id="other")

    context = _instant(db_session).load(SESSION_ID)

    assert [question for question, _ in context.turns] == ["本会话问题"]


def test_build_chat_history_from_rows_maps_roles_and_skips_empty() -> None:
    """消息行 -> Human / AI 消息，空内容跳过。"""
    rows = [
        ChatMessage(user_id=1, session_id=1, select_model=0, request_id="a", request_text="问题一", response_text="回答一"),
        ChatMessage(user_id=1, session_id=1, select_model=0, request_id="b", request_text="问题二", response_text=""),
        ChatMessage(user_id=1, session_id=1, select_model=0, request_id="c", request_text="", response_text="无提问的回答"),
    ]

    messages = build_chat_history_from_rows(rows)

    assert [type(item) for item in messages] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert [item.content for item in messages] == ["问题一", "回答一", "问题二", "无提问的回答"]
    assert build_chat_history_from_rows([]) == []


def test_rows_to_turns_drops_fully_empty_rows() -> None:
    """提问与回答都空的行不进轮次（脏数据兜底）。"""
    rows = [
        ChatMessage(user_id=1, session_id=1, select_model=0, request_id="a", request_text="问", response_text="答"),
        ChatMessage(user_id=1, session_id=1, select_model=0, request_id="b", request_text="  ", response_text=""),
    ]

    assert rows_to_turns(rows) == [("问", "答")]


def test_memory_service_uses_injected_instant_memory(db_session: Session) -> None:
    """MemoryService 走注入的短期记忆实现（方便替换 / 单测）。"""
    from app.memory.service import MemoryService

    seen: list[int] = []
    real = _instant(db_session, max_turns=2)

    class _Spy:
        def load(self, session_id: int):
            seen.append(session_id)
            return real.load(session_id)

    service = MemoryService(
        ChatMessageRepository(db_session),
        instant=_Spy(),  # type: ignore[arg-type]
        config=MemoryConfig(max_turns=2, search_enabled=False),
    )
    _add(db_session, request_text="问题", response_text="回答")

    context = service.load(user_id=USER_ID, session_id=SESSION_ID, query="问题")

    assert seen == [SESSION_ID]
    assert context.turns == [("问题", "回答")]


def test_instant_memory_defaults_to_configured_history_turns() -> None:
    """不传 config 时用 LLM_HISTORY_TURNS 作为轮数上限。"""
    from app.core.config import settings

    memory = InstantMemory(ChatMessageRepository.__new__(ChatMessageRepository))

    assert memory.config.max_turns == settings.llm_history_turns


def test_instant_memory_is_exported_from_package() -> None:
    """记忆层包导出 InstantMemory 与 MemoryService（统一入口）。"""
    import app.memory as memory_package

    assert {"InstantMemory", "InstantMemoryContext", "MemoryService"} <= set(memory_package.__all__)
