"""长期记忆测试：画像读取、写入意图（语义 + 关键词兜底）、异步抽取回写与接入。

向量后端用假实现（不加载 fastembed / 不联网），模型用假的 BaseChatModel，
数据库走内存 SQLite，验证 LongTermMemory 与 MemoryService / 流式服务的接线。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_chat_model_dep, get_persist_session_factory
from app.core.config import settings
from app.db.chat_message_repository import ChatMessageRepository
from app.db.models import ChatMessage, User, UserProfile
from app.db.user_profile_repository import UserProfileRepository
from app.main import app
from app.memory.long_term_memory import (
    NON_PROFILE_EXAMPLES,
    PROFILE_INTENT_EXAMPLES,
    LongTermMemory,
    format_profile_memory,
)
from app.memory.memory import MemoryConfig
from app.memory.service import MemoryService
from app.rag import dialogue
from tests.conftest import FakeRedis
from tests.test_stream_chat import FakeStreamingChatModel, _create_session, _create_template, _login, parse_sse

SESSIONS = "/sessions"


# ---------------------------------------------------------------------------
# 假实现
# ---------------------------------------------------------------------------
class FakeEmbeddingBackend:
    """可控向量后端：画像例句 -> [1,0]，非画像例句 -> [0,1]，查询向量由测试指定。"""

    def __init__(self) -> None:
        self.query_text = ""
        self.query_vector: list[float] = [1.0, 0.0]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text == self.query_text:
                vectors.append(self.query_vector)
            elif text in PROFILE_INTENT_EXAMPLES:
                vectors.append([1.0, 0.0])
            elif text in NON_PROFILE_EXAMPLES:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


class BrokenEmbeddingBackend:
    """向量后端不可用（fastembed 未安装 / 模型缺失）。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend down")


class FakeExtractionModel(BaseChatModel):
    """按预设文本返回 JSON 的假模型。"""

    response: str = "{}"

    @property
    def _llm_type(self) -> str:
        return "fake-extraction-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])


# ---------------------------------------------------------------------------
# 画像读取
# ---------------------------------------------------------------------------
def test_format_profile_memory_only_includes_non_empty_fields() -> None:
    profile = UserProfile(
        user_id=1,
        target_job="后端工程师",
        learning_goal="半年内转行后端",
        interview_focus=["系统设计", "并发"],
        long_term_summary="三年全栈经验",
    )

    text = format_profile_memory(profile)

    assert text.startswith("【用户长期画像】")
    assert "目标岗位：后端工程师" in text
    assert "学习目标：半年内转行后端" in text
    assert "面试关注点：系统设计、并发" in text
    assert "画像摘要：三年全栈经验" in text
    assert "目标等级" not in text  # 空字段不输出


def test_load_memory_context_reads_profile(db_session: Session) -> None:
    db_session.add(UserProfile(user_id=9, learning_style="看视频 + 动手做项目"))
    db_session.flush()
    memory = LongTermMemory(profiles=UserProfileRepository(db_session), embeddings=FakeEmbeddingBackend(), enabled=True)

    assert "学习风格：看视频 + 动手做项目" in memory.load_memory_context(9)
    assert memory.load_memory_context(999) == ""  # 没有画像 -> 空文本


def test_load_memory_context_failure_degrades_silently() -> None:
    class BrokenRepo:
        def get_by_user(self, user_id: int) -> UserProfile | None:
            raise RuntimeError("db down")

    memory = LongTermMemory(profiles=BrokenRepo(), embeddings=FakeEmbeddingBackend(), enabled=True)  # type: ignore[arg-type]

    assert memory.load_memory_context(1) == ""


# ---------------------------------------------------------------------------
# 写入意图
# ---------------------------------------------------------------------------
def test_should_write_uses_contrastive_semantic_scores() -> None:
    backend = FakeEmbeddingBackend()
    memory = LongTermMemory(
        embeddings=backend,
        similarity_threshold=0.6,
        similarity_margin=0.08,
        enabled=True,
    )
    backend.query_text = "我来说说自己的情况"

    backend.query_vector = [1.0, 0.0]  # 更像画像句
    assert memory.should_write(backend.query_text, 1) is True

    backend.query_vector = [0.0, 1.0]  # 更像非画像句
    assert memory.should_write(backend.query_text, 1) is False


def test_should_write_falls_back_to_keywords_when_embeddings_unavailable() -> None:
    memory = LongTermMemory(embeddings=BrokenEmbeddingBackend(), enabled=True)

    assert memory.should_write("我的学习目标是转行做后端", 1) is True
    assert memory.should_write("我不太会并发编程", 1) is True
    assert memory.should_write("今天天气怎么样", 1) is False
    assert memory.should_write("这是啥", 1) is False


def test_should_write_disabled_returns_false() -> None:
    memory = LongTermMemory(embeddings=FakeEmbeddingBackend(), enabled=False)

    assert memory.should_write("我的学习目标是转行", 1) is False


# ---------------------------------------------------------------------------
# 异步抽取 + 回写
# ---------------------------------------------------------------------------
def _seed_user(engine: Engine, user_id: int = 1) -> None:
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    session.add(User(id=user_id, user_name=f"u{user_id}", password="x"))
    session.commit()
    session.close()


def test_submit_profile_update_async_extracts_and_persists(sqlite_engine: Engine) -> None:
    _seed_user(sqlite_engine)
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    response = json.dumps(
        {
            "learning_goal": "转行后端",
            "learning_style": "项目驱动",
            "interview_focus": ["系统设计", "并发"],
            "long_term_summary": "三年全栈，目标后端",
        },
        ensure_ascii=False,
    )
    memory = LongTermMemory(
        profiles=UserProfileRepository(factory()),
        session_factory=factory,
        model_factory=lambda: FakeExtractionModel(response=response),
        embeddings=FakeEmbeddingBackend(),
        enabled=True,
    )

    thread = memory.submit_profile_update_async(1, "我的学习目标是转行", ["用户：目标后端", "助手：好的"])
    assert thread is not None
    thread.join(timeout=10)

    session = factory()
    profile = UserProfileRepository(session).get_by_user(1)
    session.close()
    assert profile is not None
    assert profile.learning_goal == "转行后端"
    assert profile.learning_style == "项目驱动"
    assert profile.interview_focus == ["系统设计", "并发"]
    assert profile.long_term_summary == "三年全栈，目标后端"


def test_long_term_write_is_logged(
    sqlite_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_user(sqlite_engine)
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    memory = LongTermMemory(
        profiles=UserProfileRepository(factory()),
        session_factory=factory,
        model_factory=lambda: FakeExtractionModel(response='{"learning_goal": "转行后端"}'),
        embeddings=FakeEmbeddingBackend(),
        enabled=True,
    )

    with caplog.at_level(logging.INFO, logger="app.memory.long_term"):
        thread = memory.submit_profile_update_async(1, "我的学习目标是转行", "上下文")
        assert thread is not None
        thread.join(timeout=10)

    messages = [record.getMessage() for record in caplog.records if record.name == "app.memory.long_term"]
    assert any("长期记忆写入成功" in message and "changed=['learning_goal']" in message for message in messages)


def test_submit_profile_update_async_skips_without_session_or_model() -> None:
    memory = LongTermMemory(embeddings=FakeEmbeddingBackend(), enabled=True)

    assert memory.submit_profile_update_async(1, "文本", "上下文") is None


def test_submit_profile_update_async_ignores_malformed_model_output(sqlite_engine: Engine) -> None:
    _seed_user(sqlite_engine)
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    memory = LongTermMemory(
        profiles=UserProfileRepository(factory()),
        session_factory=factory,
        model_factory=lambda: FakeExtractionModel(response="我不能理解这个请求"),
        embeddings=FakeEmbeddingBackend(),
        enabled=True,
    )

    thread = memory.submit_profile_update_async(1, "文本", "上下文")
    assert thread is not None
    thread.join(timeout=10)

    session = factory()
    profile = UserProfileRepository(session).get_by_user(1)
    session.close()
    assert profile is None  # 解析失败 -> 不写脏数据


# ---------------------------------------------------------------------------
# MemoryService 接入
# ---------------------------------------------------------------------------
def test_memory_service_prepends_long_term_system_message(db_session: Session) -> None:
    db_session.add(UserProfile(user_id=7, learning_goal="转行后端"))
    db_session.add(
        ChatMessage(user_id=7, session_id=1, select_model=0, request_id="r1", request_text="问题", response_text="回答")
    )
    db_session.flush()
    service = MemoryService(
        ChatMessageRepository(db_session),
        config=MemoryConfig(max_turns=3),
        profiles=UserProfileRepository(db_session),
        long_term=LongTermMemory(profiles=UserProfileRepository(db_session), embeddings=FakeEmbeddingBackend(), enabled=True),
    )

    context = service.load(user_id=7, session_id=1, query="随便", use_search=False)

    assert context.long_term_text.startswith("【用户长期画像】")
    assert isinstance(context.messages[0], SystemMessage)
    assert context.messages[0].content == context.long_term_text
    assert [item.content for item in context.messages[1:]] == ["问题", "回答"]


def test_memory_service_can_disable_long_term(db_session: Session) -> None:
    db_session.add(UserProfile(user_id=7, learning_goal="转行后端"))
    db_session.flush()
    service = MemoryService(
        ChatMessageRepository(db_session),
        config=MemoryConfig(max_turns=3),
        profiles=UserProfileRepository(db_session),
        long_term=LongTermMemory(profiles=UserProfileRepository(db_session), embeddings=FakeEmbeddingBackend(), enabled=True),
    )

    context = service.load(user_id=7, session_id=1, query="随便", use_search=False, enable_long_term=False)

    assert context.long_term_text == ""
    assert all(not isinstance(item, SystemMessage) for item in context.messages)


# ---------------------------------------------------------------------------
# 流式服务接线：命中意图跳过 RAG + 后台更新；未命中照常 RAG
# ---------------------------------------------------------------------------
@pytest.fixture()
def llm_override(fake_model: FakeStreamingChatModel, sqlite_engine: Engine) -> Iterator[FakeStreamingChatModel]:
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    app.dependency_overrides[get_chat_model_dep] = lambda: fake_model
    app.dependency_overrides[get_persist_session_factory] = lambda: factory
    try:
        yield fake_model
    finally:
        app.dependency_overrides.pop(get_chat_model_dep, None)
        app.dependency_overrides.pop(get_persist_session_factory, None)


@pytest.fixture()
def fake_model() -> FakeStreamingChatModel:
    return FakeStreamingChatModel(tokens=["你好", "，", "世界"])


@pytest.mark.anyio
async def test_profile_intent_skips_rag_and_submits_async_update(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "long_term_memory_enabled", True)
    monkeypatch.setattr(LongTermMemory, "should_write", lambda self, text, user_id: True)
    submitted: list[tuple[int, str, list[BaseMessage]]] = []
    monkeypatch.setattr(
        LongTermMemory,
        "submit_profile_update_async",
        lambda self, user_id, request_text, context: submitted.append((user_id, request_text, list(context))),
    )
    rag_calls: list[int] = []
    monkeypatch.setattr(dialogue, "search_similar_chunks", lambda *a, **kw: rag_calls.append(1) or [])

    headers, user_id = await _login(httpx_client, "ltm_intent")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, agent_name=None, content="公共规则")

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "我的学习目标是转行做后端"},
        headers=headers,
    )

    assert response.status_code == 200
    assert rag_calls == []  # 命中画像意图 -> 跳过 RAG
    assert len(submitted) == 1
    submitted_user, request_text, context = submitted[0]
    assert submitted_user == user_id
    assert request_text == "我的学习目标是转行做后端"
    assert context[-1].content == "你好，世界"  # 本轮 AI 回复在上下文末尾

    done = dict(parse_sse(response.text))["done"]
    assert done["sources"] == []


@pytest.mark.anyio
async def test_non_profile_message_still_runs_rag(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "long_term_memory_enabled", True)
    monkeypatch.setattr(LongTermMemory, "should_write", lambda self, text, user_id: False)
    submitted: list[int] = []
    monkeypatch.setattr(
        LongTermMemory,
        "submit_profile_update_async",
        lambda self, *a, **kw: submitted.append(1),
    )
    rag_calls: list[int] = []
    monkeypatch.setattr(dialogue, "search_similar_chunks", lambda *a, **kw: rag_calls.append(1) or [])

    headers, _ = await _login(httpx_client, "ltm_no_intent")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, agent_name=None, content="公共规则")

    with caplog.at_level(logging.INFO, logger="app.stream_chat"):
        response = await httpx_client.post(
            f"{SESSIONS}/{session['id']}/stream-chat",
            json={"message": "HashMap 的扩容因子是多少"},
            headers=headers,
        )

    assert response.status_code == 200
    assert rag_calls == [1]  # 未命中画像意图 -> 正常走 RAG
    assert submitted == []   # 不触发画像更新
    messages = [record.getMessage() for record in caplog.records if record.name == "app.stream_chat"]
    assert any("短期记忆写入成功" in message and "reason=done" in message for message in messages)
