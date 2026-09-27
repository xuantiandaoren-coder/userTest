"""记忆层测试：历史轮次构建 + 搜索增强（关键词抽取、打分、片段、失败降级）。

记忆层只做两件事：把本会话最近 N 轮问答变成 LangChain 消息序列，以及从历史消息里
检索出跨会话的相关片段拼成参考文本。模型与提示词都不在这一层，所以这里不需要网络。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from app.db.chat_message_repository import ChatMessageRepository
from app.db.models import ChatMessage
from app.memory.memory import (
    DatabaseSearchBackend,
    MemoryBuilder,
    MemoryConfig,
    build_searchable_text,
    extract_keywords,
    history_from_pairs,
    make_snippet,
    rank_hits,
)

USER_ID = 7
CURRENT_SESSION = 1


def _add_message(
    db: Session,
    *,
    session_id: int,
    request_text: str,
    response_text: str = "回答",
    user_id: int = USER_ID,
    file_extracted_text: str | None = None,
    request_id: str = "req",
) -> ChatMessage:
    """直接落一条消息（聊天正文没有写入接口，测试里用 ORM 造数据）。"""
    message = ChatMessage(
        user_id=user_id,
        session_id=session_id,
        select_model=0,
        request_id=request_id,
        request_text=request_text,
        response_text=response_text,
        file_extracted_text=file_extracted_text,
    )
    db.add(message)
    db.flush()
    return message


@pytest.fixture()
def builder(db_session: Session) -> MemoryBuilder:
    """挂在测试会话上的记忆层构建器（历史轮数上限 3 轮，便于断言截断）。"""
    return MemoryBuilder(
        ChatMessageRepository(db_session),
        config=MemoryConfig(max_turns=3, max_chars=2000, search_top_k=2),
    )


# ---------------------------------------------------------------------------
# 纯函数：关键词、片段、打分
# ---------------------------------------------------------------------------
def test_extract_keywords_mixes_english_words_and_chinese_grams() -> None:
    keywords = extract_keywords("MySQL 索引优化怎么做")

    assert "MySQL" in keywords
    assert "索引优化怎么做" in keywords  # 连续中文片段整体保留
    assert "索引" in keywords            # 同时给出 2-gram，兼顾「索引」这类短词召回
    assert "优化" in keywords
    assert len(keywords) == len(set(keywords))  # 去重
    assert extract_keywords("") == []
    assert extract_keywords("a") == []  # 单字符不参与检索


def test_make_snippet_centers_on_keyword_and_marks_truncation() -> None:
    text = "前缀" * 60 + "并发编程" + "后缀" * 60

    snippet = make_snippet(text, "并发编程", width=30)

    assert "并发编程" in snippet
    assert snippet.startswith("…") and snippet.endswith("…")
    assert len(snippet) <= 32


def test_make_snippet_keeps_short_text_untouched() -> None:
    assert make_snippet("很短的一段话", "短") == "很短的一段话"


def test_rank_hits_scores_by_keyword_length_and_prefers_newer_messages(db_session: Session) -> None:
    longer = _add_message(db_session, session_id=2, request_text="聊聊并发编程", request_id="m1")
    shorter = _add_message(db_session, session_id=3, request_text="聊聊并发", request_id="m2")
    miss = _add_message(db_session, session_id=4, request_text="完全无关的话题", request_id="m3")

    hits = rank_hits([shorter, longer, miss], ["并发编程", "并发"])

    by_id = {hit.message_id: hit for hit in hits}
    assert miss.id not in by_id                                  # 没命中关键词的候选被丢弃
    assert by_id[longer.id].score > by_id[shorter.id].score      # 长关键词权重更高
    assert [hit.message_id for hit in hits] == [longer.id, shorter.id]
    assert by_id[longer.id].source == "history"
    assert rank_hits([longer, shorter], ["并发", "并发"])[0].score == 4  # 命中几个词就累加几分


def test_rank_hits_marks_file_hits(db_session: Session) -> None:
    message = _add_message(
        db_session,
        session_id=2,
        request_text="看看这份笔记",
        file_extracted_text="MySQL 索引优化要点",
        request_id="f1",
    )

    hits = rank_hits([message], ["MySQL"])

    assert hits and hits[0].source == "file"
    assert "索引优化" in hits[0].snippet
    assert hits[0].as_dict()["source"] == "file"


def test_build_searchable_text_merges_body_and_file_text(db_session: Session) -> None:
    message = _add_message(
        db_session,
        session_id=2,
        request_text="问题",
        response_text="回答",
        file_extracted_text="附件正文",
        request_id="t1",
    )

    assert build_searchable_text(message) == "问题 回答 附件正文"


def test_history_from_pairs_builds_role_messages() -> None:
    messages = history_from_pairs([("问题一", "回答一"), ("问题二", "")])

    assert [type(item) for item in messages] == [HumanMessage, AIMessage, HumanMessage]
    assert messages[0].content == "问题一"
    assert messages[-1].content == "问题二"
    assert history_from_pairs([]) == []


# ---------------------------------------------------------------------------
# 历史记忆
# ---------------------------------------------------------------------------
def test_build_keeps_recent_turns_in_order(builder: MemoryBuilder, db_session: Session) -> None:
    for index in range(1, 6):
        _add_message(db_session, session_id=CURRENT_SESSION, request_text=f"问题{index}", response_text=f"回答{index}")

    context = builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="随便问问", use_search=False)

    assert context.turns == [("问题3", "回答3"), ("问题4", "回答4"), ("问题5", "回答5")]  # 只保留最近 3 轮
    assert [item.content for item in context.messages] == ["问题3", "回答3", "问题4", "回答4", "问题5", "回答5"]
    assert context.summary == "history_turns=3 search_hits=0"


def test_build_drops_oldest_turns_when_history_is_too_long(db_session: Session) -> None:
    builder = MemoryBuilder(
        ChatMessageRepository(db_session),
        config=MemoryConfig(max_turns=5, max_chars=20, search_enabled=False),
    )
    _add_message(db_session, session_id=CURRENT_SESSION, request_text="旧" * 15, response_text="旧答")
    _add_message(db_session, session_id=CURRENT_SESSION, request_text="新问题", response_text="新回答")

    context = builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="q")

    assert context.turns == [("新问题", "新回答")]  # 超出字符上限的最旧轮次被丢掉


# ---------------------------------------------------------------------------
# 搜索增强
# ---------------------------------------------------------------------------
def test_build_injects_search_hits_from_other_sessions(builder: MemoryBuilder, db_session: Session) -> None:
    _add_message(
        db_session,
        session_id=2,
        request_text="MySQL 索引优化",
        response_text="用覆盖索引",
        request_id="other-1",
    )
    _add_message(db_session, session_id=CURRENT_SESSION, request_text="MySQL 索引优化", request_id="current-1")
    _add_message(db_session, session_id=3, request_text="MySQL 索引优化", user_id=USER_ID + 1, request_id="other-user")

    context = builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="MySQL 索引优化")

    assert len(context.search_hits) == 1                                  # 当前会话 / 别人的消息都不算
    assert context.search_hits[0].session_id == 2
    assert "【检索" not in context.search_context                          # 标题由提示词层排版
    assert "MySQL 索引优化" in context.search_context
    assert context.memory_text == context.search_context
    assert context.summary == "history_turns=1 search_hits=1"  # 当前会话那轮进历史，检索只补别的会话


def test_build_respects_search_top_k(builder: MemoryBuilder, db_session: Session) -> None:
    for index in range(5):
        _add_message(db_session, session_id=10 + index, request_text=f"索引优化 {index}", request_id=f"h{index}")

    context = builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="索引优化")

    assert len(context.search_hits) == 2  # search_top_k=2


def test_build_skips_search_when_disabled_or_query_is_empty(builder: MemoryBuilder, db_session: Session) -> None:
    _add_message(db_session, session_id=2, request_text="索引优化", request_id="h1")

    assert builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="索引优化", use_search=False).search_hits == []
    assert builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="") .search_hits == []


def test_search_failure_only_adds_warning(db_session: Session) -> None:
    class BrokenBackend:
        """检索后端不可用（外部搜索超时等）。"""

        def search(self, **_: object) -> list[object]:
            raise RuntimeError("search backend down")

    builder = MemoryBuilder(
        ChatMessageRepository(db_session),
        search_backend=BrokenBackend(),  # type: ignore[arg-type]
        config=MemoryConfig(max_turns=3),
    )
    _add_message(db_session, session_id=CURRENT_SESSION, request_text="问题", response_text="回答")

    context = builder.build(user_id=USER_ID, session_id=CURRENT_SESSION, query="问题")

    assert context.turns == [("问题", "回答")]      # 检索失败不影响历史
    assert context.search_context == ""
    assert context.warnings == ["搜索增强失败，本次未注入检索资料"]


def test_database_search_backend_end_to_end(db_session: Session) -> None:
    backend = DatabaseSearchBackend(ChatMessageRepository(db_session))
    _add_message(db_session, session_id=5, request_text="讲讲 GIL 是什么", request_id="s1")

    hits = backend.search(query="GIL 是什么", user_id=USER_ID, session_id=CURRENT_SESSION, limit=10)

    assert [hit.session_id for hit in hits] == [5]
