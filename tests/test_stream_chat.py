"""流式聊天接口测试（SSE）：`POST /sessions/{session_id}/stream-chat`。

这条用例把三层串起来跑：模型层（假的 BaseChatModel，不连网络）、提示词层（模板渲染 +
链路组装）、记忆层（历史轮次 + 搜索增强），最后断言 SSE 帧序列与落库结果。

依赖替换：
- `get_chat_model_dep` -> FakeStreamingChatModel（记录模型实际收到的消息，便于断言提示词）
- `get_persist_session_factory` -> 测试库的会话工厂（流结束后的落库走独立会话）
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
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
from app.db.models import ChatMessage, ChatSession, UserProfile
from app.main import app
from tests.conftest import FakeRedis

SESSIONS = "/sessions"
PROMPT = "/prompt"
PASSWORD = "HZq7mK2p"


class FakeStreamingChatModel(BaseChatModel):
    """按 token 逐块输出的假模型：不联网，并把收到的消息记录下来供断言。"""

    tokens: list[str] = Field(default_factory=list)
    error: str | None = None
    captured: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-streaming-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured.append(list(messages))
        if self.error:
            raise RuntimeError(self.error)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer()))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.captured.append(list(messages))
        if self.error:
            raise RuntimeError(self.error)
        for token in self.tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager is not None:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def answer(self) -> str:
        """完整回答 = 所有 token 拼接。"""
        return "".join(self.tokens)


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """把 SSE 响应体拆成 [(event, data), ...]。"""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        event = ""
        data: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event:
            events.append((event, data))
    return events


def event_names(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in events]


async def _login(client: httpx.AsyncClient, username: str) -> tuple[dict[str, str], int]:
    created = await client.post("/auth/register", json={"username": username, "password": PASSWORD})
    tokens = (await client.post("/auth/login", json={"username": username, "password": PASSWORD})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}, created.json()["id"]


async def _create_session(client: httpx.AsyncClient, headers: dict[str, str], model: int = 0) -> dict:
    response = await client.post(SESSIONS, json={"title": "流式聊天", "session_model": model}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


async def _create_template(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    *,
    content: str,
    agent_name: str | None = "difficulty_learner",
    scene: str = "study",
) -> dict:
    payload: dict[str, Any] = {"scene": scene, "template_content": content}
    if agent_name is not None:
        payload["agent_name"] = agent_name
    response = await client.post(f"{PROMPT}/templates", json=payload, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def fake_model() -> FakeStreamingChatModel:
    return FakeStreamingChatModel(tokens=["你好", "，", "世界"])


@pytest.fixture()
def llm_override(fake_model: FakeStreamingChatModel, sqlite_engine: Engine) -> Iterator[FakeStreamingChatModel]:
    """把模型与落库会话工厂换成测试实现。"""
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    app.dependency_overrides[get_chat_model_dep] = lambda: fake_model
    app.dependency_overrides[get_persist_session_factory] = lambda: factory
    try:
        yield fake_model
    finally:
        app.dependency_overrides.pop(get_chat_model_dep, None)
        app.dependency_overrides.pop(get_persist_session_factory, None)


def _system_prompt(model: FakeStreamingChatModel) -> str:
    """取出模型最后一次调用里的 system 文本。"""
    messages = model.captured[-1]
    system = [item for item in messages if isinstance(item, SystemMessage)]
    assert system, "模型没有收到 system 消息"
    return system[0].content  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 入参校验：这些错误发生在 SSE 开始之前，仍是标准 JSON
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_stream_chat_requires_authentication(httpx_client: httpx.AsyncClient) -> None:
    response = await httpx_client.post(f"{SESSIONS}/1/stream-chat", json={"message": "在吗"})

    assert response.status_code == 401


@pytest.mark.anyio
async def test_stream_chat_rejects_unknown_session(
    db_override: None, redis_override: FakeRedis, llm_override: FakeStreamingChatModel, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "stream_404")

    response = await httpx_client.post(f"{SESSIONS}/999/stream-chat", json={"message": "在吗"}, headers=headers)

    assert response.status_code == 404
    assert response.json()["code"] == "SESSION_NOT_FOUND"


@pytest.mark.anyio
async def test_stream_chat_rejects_other_users_session(
    db_override: None, redis_override: FakeRedis, llm_override: FakeStreamingChatModel, httpx_client: httpx.AsyncClient
) -> None:
    alice, _ = await _login(httpx_client, "stream_alice")
    bob, _ = await _login(httpx_client, "stream_bob")
    session = await _create_session(httpx_client, alice)

    response = await httpx_client.post(f"{SESSIONS}/{session['id']}/stream-chat", json={"message": "越权"}, headers=bob)

    assert response.status_code == 404  # 归属隔离：别人的会话等同于不存在


@pytest.mark.anyio
async def test_stream_chat_rejects_unknown_agent(
    db_override: None, redis_override: FakeRedis, llm_override: FakeStreamingChatModel, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "stream_agent")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, content="模板")

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "在吗", "agent_name": "ghost"},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["code"] == "AGENT_NOT_FOUND"


@pytest.mark.anyio
async def test_stream_chat_reports_missing_template(
    db_override: None, redis_override: FakeRedis, llm_override: FakeStreamingChatModel, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "stream_no_template")
    session = await _create_session(httpx_client, headers)

    response = await httpx_client.post(f"{SESSIONS}/{session['id']}/stream-chat", json={"message": "在吗"}, headers=headers)

    assert response.status_code == 404
    assert response.json()["code"] == "PROMPT_TEMPLATE_NOT_FOUND"


# ---------------------------------------------------------------------------
# 正常链路：meta -> delta* -> done，并把这一轮写入 chat_messages
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_stream_chat_emits_sse_and_persists_the_round(
    db_override: None,
    db_session: Session,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, user_id = await _login(httpx_client, "stream_happy")
    session = await _create_session(httpx_client, headers)
    await _create_template(
        httpx_client,
        headers,
        agent_name=None,
        content="公共规则：场景 {scene}，用户 {user_name}，今天 {current_date}",
    )
    await _create_template(httpx_client, headers, content="私有指令：目标岗位 {target_job}，薄弱点 {weak_topics}")
    # 用户画像（变量注入的数据来源）
    db_session.add(
        UserProfile(
            user_id=user_id,
            target_job="后端工程师",
            years_experience=3,
            target_level="P6",
            target_skills=["Python", "MySQL"],
            weak_topics=["并发", "索引"],
        )
    )
    # 另一个会话里的历史消息 -> 记忆层的搜索增强
    other = await _create_session(httpx_client, headers)
    db_session.add(
        ChatMessage(
            user_id=user_id,
            session_id=other["id"],
            select_model=0,
            request_id="prior",
            request_text="MySQL 索引优化",
            response_text="回表是主键索引再查一次",
        )
    )
    # 当前会话里已有一轮问答 -> 记忆层的历史
    db_session.add(
        ChatMessage(
            user_id=user_id,
            session_id=session["id"],
            select_model=0,
            request_id="history",
            request_text="先聊聊索引",
            response_text="索引是排序的数据结构",
        )
    )
    db_session.flush()

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "MySQL 索引优化怎么学"},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"  # 反代不许缓冲，否则流会被攒成一坨

    events = parse_sse(response.text)
    names = event_names(events)
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert set(names) == {"meta", "delta", "done"}

    meta = events[0][1]
    assert meta["agent_name"] == "difficulty_learner"  # 会话类型 0 -> 默认智能体
    assert meta["scene"] == "study"
    assert meta["prompt_versions"] == "common=v1,private=v1"
    assert meta["history_turns"] == 2             # 历史问答各 1 条
    assert [hit["session_id"] for hit in meta["search_hits"]] == [other["id"]]
    assert meta["warnings"] == []

    streamed = "".join(data["content"] for name, data in events if name == "delta")
    assert streamed == "你好，世界"

    # 提示词层：公共 + 私有模板拼接，并把画像变量填进占位符
    system = _system_prompt(llm_override)
    assert "公共规则：场景 study" in system
    assert "用户 stream_happy" in system
    assert "私有指令：目标岗位 后端工程师" in system
    assert "薄弱点 并发、索引" in system
    assert "3 年" not in system           # 模板没引用 {years_experience}，变量不会自己混进来
    assert "未填写" not in system         # 画像齐全，没有落到默认值

    # 记忆层：检索增强与历史进上下文的方式不同（检索进 system，历史进消息序列）
    assert "回表是主键索引再查一次" in system
    assert "【回答要求】" in system
    messages = llm_override.captured[-1]
    assert [item.content for item in messages[1:-1]] == ["先聊聊索引", "索引是排序的数据结构"]
    assert messages[-1].content == "MySQL 索引优化怎么学"

    # 落库：一轮问答（request_id 与 meta 对齐）
    done = events[-1][1]
    assert done["message_id"]
    assert done["answer_length"] == len("你好，世界")
    assert done["request_id"] == meta["request_id"]

    persisted = db_session.scalars(select(ChatMessage).where(ChatMessage.session_id == session["id"])).all()
    row = [item for item in persisted if item.request_id == meta["request_id"]]
    assert len(row) == 1
    assert row[0].request_text == "MySQL 索引优化怎么学"
    assert row[0].response_text == "你好，世界"
    assert row[0].user_id == user_id
    assert row[0].select_model == 1  # difficulty_learner 在 AGENT_CONFIG 里的 select_model

    # 消息接口也能读到刚写入的这一轮
    listed = (await httpx_client.get(f"{SESSIONS}/{session['id']}/messages", headers=headers)).json()
    assert listed["total"] == 2
    assert [item["response_text"] for item in listed["items"]] == ["索引是排序的数据结构", "你好，世界"]  # 按时间正序


@pytest.mark.anyio
async def test_stream_chat_surfaces_unfilled_placeholder_warnings(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    """模板占位符没被赋值时不让请求失败，而是在 meta 事件里给前端一条告警。"""
    headers, _ = await _login(httpx_client, "stream_warning")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, content="模板 {typo_variable}")

    response = await httpx_client.post(f"{SESSIONS}/{session['id']}/stream-chat", json={"message": "在吗"}, headers=headers)

    meta = parse_sse(response.text)[0][1]
    assert meta["warnings"] == ["模板占位符未赋值：typo_variable"]
    assert "模板 {typo_variable}" in _system_prompt(llm_override)  # 原样保留，便于模板作者发现拼写错误


@pytest.mark.anyio
async def test_stream_chat_can_switch_agent_and_scene(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, _ = await _login(httpx_client, "stream_agent_switch")
    session = await _create_session(httpx_client, headers)
    await _create_template(
        httpx_client, headers, agent_name="resume_parser", scene="resume", content="简历模板 {target_job}"
    )

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "帮我改简历", "agent_name": "resume_parser"},
        headers=headers,
    )

    meta = parse_sse(response.text)[0][1]
    assert (meta["agent_name"], meta["scene"]) == ("resume_parser", "resume")
    assert "简历模板 未填写" in _system_prompt(llm_override)  # 没填画像 -> 默认值兜底


@pytest.mark.anyio
async def test_stream_chat_folds_legacy_agent_alias(
    db_override: None,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    """老名字（resume_optimizer）折叠成现行名字，并复用现行名字的模板分组。"""
    headers, _ = await _login(httpx_client, "stream_alias")
    session = await _create_session(httpx_client, headers)
    await _create_template(
        httpx_client, headers, agent_name="resume_parser", scene="resume", content="简历模板 {target_job}"
    )

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "帮我改简历", "agent_name": "resume_optimizer"},
        headers=headers,
    )

    assert response.status_code == 200
    meta = parse_sse(response.text)[0][1]
    assert (meta["agent_name"], meta["scene"]) == ("resume_parser", "resume")
    assert "简历模板 未填写" in _system_prompt(llm_override)


@pytest.mark.anyio
async def test_stream_chat_can_disable_search_enhancement(
    db_override: None,
    db_session: Session,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, user_id = await _login(httpx_client, "stream_no_search")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, content="模板 {target_job}")
    other = await _create_session(httpx_client, headers)
    db_session.add(
        ChatMessage(
            user_id=user_id,
            session_id=other["id"],
            select_model=0,
            request_id="prior",
            request_text="MySQL 索引优化",
            response_text="回表是主键索引再查一次",
        )
    )
    db_session.flush()

    response = await httpx_client.post(
        f"{SESSIONS}/{session['id']}/stream-chat",
        json={"message": "MySQL 索引优化", "use_search": False},
        headers=headers,
    )

    meta = parse_sse(response.text)[0][1]
    assert meta["search_hits"] == []
    assert "回表是主键索引再查一次" not in _system_prompt(llm_override)


# ---------------------------------------------------------------------------
# 模型故障：响应头已发出，只能通过 error 事件告知前端
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_stream_chat_reports_model_failure_as_error_event(
    db_override: None,
    db_session: Session,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, _ = await _login(httpx_client, "stream_llm_error")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, content="模板")
    llm_override.error = "upstream 503"

    response = await httpx_client.post(f"{SESSIONS}/{session['id']}/stream-chat", json={"message": "在吗"}, headers=headers)

    assert response.status_code == 200  # 已经开始下发，HTTP 状态码改不了了
    events = parse_sse(response.text)
    assert event_names(events) == ["meta", "error"]
    assert events[-1][1] == {"code": "LLM_ERROR", "message": "模型调用失败，请稍后重试"}
    # 失败的一轮不落库（没有回答可存）
    assert db_session.scalars(select(ChatMessage).where(ChatMessage.session_id == session["id"])).all() == []


@pytest.mark.anyio
async def test_stream_chat_does_not_persist_empty_answer(
    db_override: None,
    db_session: Session,
    redis_override: FakeRedis,
    llm_override: FakeStreamingChatModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, _ = await _login(httpx_client, "stream_empty")
    session = await _create_session(httpx_client, headers)
    await _create_template(httpx_client, headers, content="模板")
    llm_override.tokens = [""]  # 模型有响应，但内容为空

    response = await httpx_client.post(f"{SESSIONS}/{session['id']}/stream-chat", json={"message": "在吗"}, headers=headers)

    events = parse_sse(response.text)
    assert event_names(events) == ["meta", "done"]
    assert events[-1][1]["message_id"] is None
    assert db_session.scalars(select(ChatMessage).where(ChatMessage.session_id == session["id"])).all() == []
