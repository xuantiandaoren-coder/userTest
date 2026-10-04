"""RAG 对话链路端到端：意图判断 -> 资料注入 -> SSE sources -> 落库引用 -> 历史回显。

检索用假实现（只替换 ``app.rag.dialogue.search_similar_chunks``），模型与向量库都不连；
MySQL 侧走内存 SQLite（分块原文表与消息表都在里面）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_chat_model_dep, get_persist_session_factory
from app.db.models import ChatMessage, KnowledgeChunk, Resource, User
from app.main import app
from app.rag import dialogue
from app.rag.retriever import ChunkHit
from tests.conftest import FakeRedis
from tests.test_stream_chat import FakeStreamingChatModel, event_names, parse_sse

SESSIONS = "/sessions"
PROMPT = "/prompt"
PASSWORD = "HZq7mK2p"


@pytest.fixture()
def fake_model() -> FakeStreamingChatModel:
    return FakeStreamingChatModel(tokens=["数组", "加链表"])


@pytest.fixture()
def llm_override(fake_model: FakeStreamingChatModel, sqlite_engine: Engine) -> Iterator[FakeStreamingChatModel]:
    """模型与落库会话工厂都换成测试实现（落库走独立会话）。"""
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    app.dependency_overrides[get_chat_model_dep] = lambda: fake_model
    app.dependency_overrides[get_persist_session_factory] = lambda: factory
    try:
        yield fake_model
    finally:
        app.dependency_overrides.pop(get_chat_model_dep, None)
        app.dependency_overrides.pop(get_persist_session_factory, None)


async def _login(client: httpx.AsyncClient, username: str) -> tuple[dict[str, str], int]:
    created = await client.post("/auth/register", json={"username": username, "password": PASSWORD})
    tokens = (await client.post("/auth/login", json={"username": username, "password": PASSWORD})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}, created.json()["id"]


async def _create_session(client: httpx.AsyncClient, headers: dict[str, str]) -> dict:
    response = await client.post(SESSIONS, json={"title": "RAG 对话", "session_model": 0}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


async def _create_template(client: httpx.AsyncClient, headers: dict[str, str]) -> None:
    response = await client.post(
        f"{PROMPT}/templates",
        json={"scene": "study", "agent_name": "difficulty_learner", "template_content": "你是学习助手，按流程引导。"},
        headers=headers,
    )
    assert response.status_code == 201, response.text


def _seed_chunk(
    db: Session,
    *,
    user_id: int,
    vector_id: str = "chunk-1",
    text: str = "HashMap 默认容量 16，扩容因子 0.75。",
    file_name: str = "java.pdf",
) -> Resource:
    """造一份已入库的资料：resources + knowledge_chunks。"""
    if db.get(User, user_id) is None:
        db.add(User(id=user_id, user_name=f"rag_dialogue_{user_id}", password="x" * 60))
        db.flush()
    digest = hashlib.md5(f"{user_id}:{file_name}".encode(), usedforsecurity=False).hexdigest()
    resource = Resource(
        resource_type=0,
        doc_category="study_material",
        storage_scene=0,
        upload_purpose=0,
        file_name=file_name,
        file_hash=digest,
        storage_path=f"{user_id}/{digest}.pdf",
        user_id=user_id,
    )
    db.add(resource)
    db.flush()
    db.add(KnowledgeChunk(resource_id=resource.id, vector_id=vector_id, chunk_index=3, char_count=len(text), text=text))
    db.commit()
    return resource


def _fake_hit(resource_id: int) -> ChunkHit:
    return ChunkHit(
        vector_id="chunk-1",
        resource_id=resource_id,
        text="HashMap 默认容量 16，扩容因子 0.75。",
        score=0.8123,
        file_name="java.pdf",
        doc_category="study_material",
        chunk_index=3,
    )


def _messages(model: FakeStreamingChatModel) -> list[Any]:
    return model.captured[-1]


def _human_text(model: FakeStreamingChatModel) -> str:
    return next(item.content for item in _messages(model) if type(item).__name__ == "HumanMessage")


def _system_text(model: FakeStreamingChatModel) -> str:
    return next(item.content for item in _messages(model) if type(item).__name__ == "SystemMessage")


# ---------------------------------------------------------------------------
# 流式对话：注入 + sources + 落库引用
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_stream_chat_injects_context_and_returns_sources(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    db_session: Session,
    httpx_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """命中资料时：参考资料拼到问题前、system 补 RAG 规则、done 回 sources、落库只存引用。"""
    headers, user_id = await _login(httpx_client, "rag_stream_hit")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers)
    resource = _seed_chunk(db_session, user_id=user_id)
    monkeypatch.setattr(dialogue, "search_similar_chunks", lambda *a, **kw: [_fake_hit(resource.id)])

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "HashMap 的扩容因子是多少"},
        headers=headers,
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert event_names(events) == ["meta", "delta", "delta", "done"]

    # 模型拿到的是「参考资料 + 用户问题」，system 里补了 RAG 回答规则
    human = _human_text(llm_override)
    assert human.startswith(dialogue.RAG_CONTEXT_HEADING)
    assert "[1] 来源：java.pdf（第 4 块，相似度 0.81）" in human
    assert "HashMap 默认容量 16" in human
    assert human.endswith("【用户问题】\nHashMap 的扩容因子是多少")
    assert dialogue.RAG_RULES in _system_text(llm_override)

    # done 事件回完整 sources
    done = dict(events)["done"]
    assert done["sources"] == [
        {
            "chunk_id": "chunk-1",
            "resource_id": resource.id,
            "chunk_index": 3,
            "file_name": "java.pdf",
            "score": 0.8123,
            "text": "HashMap 默认容量 16，扩容因子 0.75。",
        }
    ]

    # 落库：正文是原始提问（不含注入的参考资料），引用只存 chunk_id + score
    message = db_session.scalars(select(ChatMessage)).one()
    assert message.request_text == "HashMap 的扩容因子是多少"
    assert dialogue.RAG_CONTEXT_HEADING not in (message.request_text or "")
    assert message.reference_sources == [{"chunk_id": "chunk-1", "score": 0.8123}]


@pytest.mark.anyio
async def test_stream_chat_skips_retrieval_for_small_talk(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    db_session: Session,
    httpx_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认语 / 推进语不检索：不调向量库、不补 RAG 规则、sources 为空、引用不落库。"""
    headers, user_id = await _login(httpx_client, "rag_stream_skip")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers)
    _seed_chunk(db_session, user_id=user_id)

    def _boom(*args: object, **kwargs: object) -> list[ChunkHit]:
        raise AssertionError("不该发起检索")

    monkeypatch.setattr(dialogue, "search_similar_chunks", _boom)

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "好的，继续"},
        headers=headers,
    )

    done = dict(parse_sse(response.text))["done"]
    assert done["sources"] == []
    assert _human_text(llm_override) == "好的，继续"
    assert dialogue.RAG_RULES not in _system_text(llm_override)
    message = db_session.scalars(select(ChatMessage)).one()
    assert message.reference_sources is None


@pytest.mark.anyio
async def test_stream_chat_keeps_working_when_search_fails(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检索异常不影响对话：照常回答，只是没有 sources。"""
    headers, _ = await _login(httpx_client, "rag_stream_fail")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers)

    def _boom(*args: object, **kwargs: object) -> list[ChunkHit]:
        raise RuntimeError("qdrant 挂了")

    monkeypatch.setattr(dialogue, "search_similar_chunks", _boom)

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "HashMap 的扩容因子是多少"},
        headers=headers,
    )

    events = parse_sse(response.text)
    assert "error" not in event_names(events)
    assert dict(events)["done"]["sources"] == []


# ---------------------------------------------------------------------------
# 历史消息：按引用回查拼 sources
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_history_messages_rebuild_sources_from_references(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    db_session: Session,
    httpx_client: httpx.AsyncClient,
) -> None:
    """刷新历史后 sources 仍在：靠 reference_sources 回查 knowledge_chunks + resources。"""
    headers, user_id = await _login(httpx_client, "rag_history_ok")
    session = await _create_session(httpx_client, headers)
    resource = _seed_chunk(db_session, user_id=user_id)
    db_session.add(
        ChatMessage(
            user_id=user_id,
            session_id=session["id"],
            select_model=0,
            request_id="req-history",
            request_text="HashMap 的扩容因子是多少",
            response_text="0.75",
            reference_sources=[{"chunk_id": "chunk-1", "score": 0.8123}],
        )
    )
    db_session.commit()

    response = await httpx_client.get(f"{SESSIONS}/{session['id']}/messages", headers=headers)

    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["sources"] == [
        {
            "chunk_id": "chunk-1",
            "resource_id": resource.id,
            "chunk_index": 3,
            "file_name": "java.pdf",
            "score": 0.8123,
            "text": "HashMap 默认容量 16，扩容因子 0.75。",
        }
    ]


@pytest.mark.anyio
async def test_history_messages_mark_deleted_chunk(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    db_session: Session,
    httpx_client: httpx.AsyncClient,
) -> None:
    """引用的知识块已被删除时，历史里返回「该参考片段已经删除」。"""
    headers, user_id = await _login(httpx_client, "rag_history_deleted")
    session = await _create_session(httpx_client, headers)
    _seed_chunk(db_session, user_id=user_id)
    db_session.add(
        ChatMessage(
            user_id=user_id,
            session_id=session["id"],
            select_model=0,
            request_id="req-deleted",
            request_text="HashMap 的扩容因子是多少",
            response_text="0.75",
            reference_sources=[{"chunk_id": "chunk-1", "score": 0.8}, {"chunk_id": "never-existed", "score": 0.5}],
        )
    )
    db_session.commit()
    # 模拟资源过期清理 / 重新上传覆盖：分块行没了
    db_session.execute(KnowledgeChunk.__table__.delete())
    db_session.commit()

    response = await httpx_client.get(f"{SESSIONS}/{session['id']}/messages", headers=headers)

    assert response.status_code == 200, response.text
    sources = response.json()["items"][0]["sources"]
    assert [source["text"] for source in sources] == [dialogue.DELETED_CHUNK_TEXT, dialogue.DELETED_CHUNK_TEXT]
    assert [source["resource_id"] for source in sources] == [None, None]
    assert [source["score"] for source in sources] == [0.8, 0.5]  # 分数仍来自落库引用


@pytest.mark.anyio
async def test_history_messages_without_references_have_empty_sources(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    db_session: Session,
    httpx_client: httpx.AsyncClient,
) -> None:
    """没有引用的消息 sources 为空数组（前端无需判空）。"""
    headers, user_id = await _login(httpx_client, "rag_history_none")
    session = await _create_session(httpx_client, headers)
    db_session.add(
        ChatMessage(
            user_id=user_id,
            session_id=session["id"],
            select_model=0,
            request_id="req-plain",
            request_text="你好",
            response_text="你好，有什么可以帮你",
        )
    )
    db_session.commit()

    response = await httpx_client.get(f"{SESSIONS}/{session['id']}/messages", headers=headers)

    assert response.json()["items"][0]["sources"] == []
