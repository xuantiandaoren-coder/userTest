"""学习测评工作流测试：两段 LangGraph 流程 + 跨请求状态恢复 + 评分闭环。

模型用假的 BaseChatModel（按调用顺序返回出题 / 评分 JSON），不联网；
数据库走内存 SQLite，验证 start -> submit -> get 的完整闭环与状态持久化。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_workflow_model_factory
from app.core.json_parse import (
    JsonParseError,
    extract_json_array,
    extract_json_object,
    loads_json,
)
from app.db.models import WorkflowRun
from app.main import app
from app.workflows.learning_assessment import (
    normalize_evaluation,
    normalize_quiz,
    public_quiz,
)

LEARNING = "/learning-workflows"
PASSWORD = "HZq7mK2p"

QUIZ_JSON = """```json
{"questions": [
  {"question_id": "q1", "type": "single_choice", "stem": "HashMap 默认容量是多少？",
   "options": [{"key": "A", "text": "8"}, {"key": "B", "text": "16"}, {"key": "C", "text": "32"}, {"key": "D", "text": "64"}],
   "answer": "B"},
  {"question_id": "q2", "type": "short_answer", "stem": "简述 volatile 的作用", "answer": "保证可见性与有序性"}
]}
```"""

EVALUATION_JSON = """{"score": 80,
 "evaluation": [
   {"question_id": "q1", "correct": true, "comment": "正确，默认容量 16。"},
   {"question_id": "q2", "correct": false, "comment": "漏了禁止指令重排序。"}
 ],
 "weaknesses": ["JMM 可见性", "指令重排", "JMM 可见性"],
 "review": "建议复习 JMM 与 volatile 的内存语义。"}"""


class FakeWorkflowModel(BaseChatModel):
    """按调用顺序回放出题 / 评分 JSON 的假模型。"""

    responses: list[str] = Field(default_factory=list)
    captured: list[list[BaseMessage]] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-workflow-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured.append(list(messages))
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        text = self.responses[index] if self.responses else "{}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


async def _login(client: httpx.AsyncClient, username: str) -> tuple[dict[str, str], int]:
    created = await client.post("/auth/register", json={"username": username, "password": PASSWORD})
    tokens = (await client.post("/auth/login", json={"username": username, "password": PASSWORD})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}, created.json()["id"]


async def _create_session(client: httpx.AsyncClient, headers: dict[str, str]) -> dict:
    response = await client.post("/sessions", json={"title": "测评", "session_model": 0}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def workflow_model() -> FakeWorkflowModel:
    return FakeWorkflowModel(responses=[QUIZ_JSON, EVALUATION_JSON])


@pytest.fixture()
def model_override(workflow_model: FakeWorkflowModel) -> Iterator[FakeWorkflowModel]:
    app.dependency_overrides[get_workflow_model_factory] = lambda: (lambda provider=None: workflow_model)
    try:
        yield workflow_model
    finally:
        app.dependency_overrides.pop(get_workflow_model_factory, None)


async def _start(client: httpx.AsyncClient, headers: dict[str, str], session_id: int, text: str = "Java 基础") -> dict:
    response = await client.post(
        f"{LEARNING}/start",
        json={"session_id": session_id, "request_text": text, "question_count": 2},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 完整闭环
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_start_returns_standard_quiz_without_answers(
    db_override: None,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, user_id = await _login(httpx_client, "lw_start")
    session = await _create_session(httpx_client, headers)

    body = await _start(httpx_client, headers, session["id"])

    assert body["status"] == "quiz_ready"
    assert body["run_id"]
    assert len(body["quiz"]) == 2
    first = body["quiz"][0]
    assert first["question_id"] == "q1"
    assert first["type"] == "single_choice"
    assert first["stem"]
    assert [option["key"] for option in first["options"]] == ["A", "B", "C", "D"]
    assert all("answer" not in question for question in body["quiz"])  # 不泄露答案


@pytest.mark.anyio
async def test_full_loop_persists_state_and_recovers_quiz(
    db_override: None,
    db_session: Session,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, user_id = await _login(httpx_client, "lw_loop")
    session = await _create_session(httpx_client, headers)
    started = await _start(httpx_client, headers, session["id"])

    response = await httpx_client.post(
        f"{LEARNING}/{started['run_id']}/submit",
        json={
            "answers": [
                {"question_id": "q1", "value": "B"},
                {"question_id": "q2", "value": "保证可见性"},
            ]
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "evaluated"
    assert body["score"] == 80.0
    assert [item["question_id"] for item in body["evaluation"]] == ["q1", "q2"]
    assert body["evaluation"][0]["correct"] is True
    assert body["weaknesses"] == ["JMM 可见性", "指令重排"]  # 去重
    assert "JMM" in body["review"]

    # 评分图确实拿到了 start 阶段持久化的题目（跨请求恢复）
    submit_messages = model_override.captured[-1]
    assert "HashMap 默认容量" in submit_messages[-1].content
    assert "q1" in submit_messages[-1].content

    # 落库：状态、分数、完成时间都写入
    run = db_session.scalars(select(WorkflowRun).where(WorkflowRun.run_id == started["run_id"])).one()
    assert run.status == "evaluated"
    assert run.state_json["score"] == 80.0
    assert run.completed_at is not None


@pytest.mark.anyio
async def test_get_returns_status_and_state(
    db_override: None,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, _ = await _login(httpx_client, "lw_get")
    session = await _create_session(httpx_client, headers)
    started = await _start(httpx_client, headers, session["id"])

    response = await httpx_client.get(f"{LEARNING}/{started['run_id']}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == started["run_id"]
    assert body["status"] == "quiz_ready"
    assert len(body["state"]["quiz"]) == 2
    assert all("answer" not in question for question in body["state"]["quiz"])  # GET 也不泄露答案


# ---------------------------------------------------------------------------
# 边界：鉴权 / 归属 / 状态
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_endpoints_require_authentication(httpx_client: httpx.AsyncClient) -> None:
    assert (await httpx_client.post(f"{LEARNING}/start", json={"session_id": 1, "request_text": "x"})).status_code == 401
    assert (
        await httpx_client.post(
            f"{LEARNING}/abc/submit",
            json={"answers": [{"question_id": "q1", "value": "x"}]},
        )
    ).status_code == 401
    assert (await httpx_client.get(f"{LEARNING}/abc")).status_code == 401


@pytest.mark.anyio
async def test_start_rejects_other_users_session(
    db_override: None,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    alice, _ = await _login(httpx_client, "lw_alice")
    bob, _ = await _login(httpx_client, "lw_bob")
    session = await _create_session(httpx_client, alice)

    response = await httpx_client.post(
        f"{LEARNING}/start",
        json={"session_id": session["id"], "request_text": "越权"},
        headers=bob,
    )

    assert response.status_code == 404
    assert response.json()["code"] == "SESSION_NOT_FOUND"


@pytest.mark.anyio
async def test_other_user_cannot_read_run(
    db_override: None,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    alice, _ = await _login(httpx_client, "lw_owner")
    bob, _ = await _login(httpx_client, "lw_other")
    session = await _create_session(httpx_client, alice)
    started = await _start(httpx_client, headers=alice, session_id=session["id"])

    response = await httpx_client.get(f"{LEARNING}/{started['run_id']}", headers=bob)

    assert response.status_code == 404
    assert response.json()["code"] == "WORKFLOW_RUN_NOT_FOUND"


@pytest.mark.anyio
async def test_submit_twice_is_conflict(
    db_override: None,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, _ = await _login(httpx_client, "lw_twice")
    session = await _create_session(httpx_client, headers)
    started = await _start(httpx_client, headers, session["id"])
    payload = {"answers": [{"question_id": "q1", "value": "B"}, {"question_id": "q2", "value": "x"}]}

    first = await httpx_client.post(f"{LEARNING}/{started['run_id']}/submit", json=payload, headers=headers)
    second = await httpx_client.post(f"{LEARNING}/{started['run_id']}/submit", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["code"] == "WORKFLOW_STATE_INVALID"


@pytest.mark.anyio
async def test_submit_rejects_unknown_question_id(
    db_override: None,
    model_override: FakeWorkflowModel,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers, _ = await _login(httpx_client, "lw_unknown_q")
    session = await _create_session(httpx_client, headers)
    started = await _start(httpx_client, headers, session["id"])

    response = await httpx_client.post(
        f"{LEARNING}/{started['run_id']}/submit",
        json={"answers": [{"question_id": "q999", "value": "x"}]},
        headers=headers,
    )

    assert response.status_code == 409
    assert response.json()["code"] == "WORKFLOW_STATE_INVALID"


@pytest.mark.anyio
async def test_model_failure_marks_run_failed(
    db_override: None,
    db_session: Session,
    httpx_client: httpx.AsyncClient,
) -> None:
    alice, _ = await _login(httpx_client, "lw_fail")
    session = await _create_session(httpx_client, alice)
    broken_response = "这不是 JSON"
    previous = app.dependency_overrides.get(get_workflow_model_factory)

    class _Broken(FakeWorkflowModel):
        pass

    app.dependency_overrides[get_workflow_model_factory] = lambda: (
        lambda provider=None: _Broken(responses=[broken_response])
    )
    try:
        response = await httpx_client.post(
            f"{LEARNING}/start",
            json={"session_id": session["id"], "request_text": "出题"},
            headers=alice,
        )
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_workflow_model_factory, None)
        else:
            app.dependency_overrides[get_workflow_model_factory] = previous

    assert response.status_code == 500
    assert response.json()["code"] == "WORKFLOW_GENERATION_FAILED"
    run = db_session.scalars(select(WorkflowRun).where(WorkflowRun.user_id != 0)).first()
    assert run is not None and run.status == "failed"
    assert run.error_message


# ---------------------------------------------------------------------------
# 纯函数：JSON 解析 / 归一化
# ---------------------------------------------------------------------------
def test_extract_json_handles_code_fence_and_noise() -> None:
    assert extract_json_object('说明\n```json\n{"a": 1}\n```\n结束') == {"a": 1}
    assert extract_json_array("题目：[1, 2, 3]") == [1, 2, 3]
    assert extract_json_object("没有 JSON") is None
    with pytest.raises(JsonParseError):
        loads_json("没有 JSON")
    with pytest.raises(JsonParseError):
        loads_json('{"a": 1}', expect=list)


def test_normalize_quiz_drops_invalid_and_keeps_answer_server_side() -> None:
    quiz = normalize_quiz(
        [
            {"question_id": "q1", "type": "单选题", "stem": "题干", "options": ["甲", "乙"], "answer": "甲"},
            {"stem": ""},  # 无题干 -> 丢弃
            {"type": "single_choice", "stem": "第二题"},  # question_id 兜底
        ]
    )

    assert [item["question_id"] for item in quiz] == ["q1", "q2"]
    assert quiz[0]["type"] == "short_answer"  # 非法 type 退化为简答，且不带 options
    assert quiz[0]["answer"] == "甲"
    assert all("answer" not in item for item in public_quiz(quiz))  # 对外剥离答案


def test_normalize_evaluation_pads_missing_questions() -> None:
    quiz = [{"question_id": "q1"}, {"question_id": "q2"}]
    score, evaluation, weaknesses, review = normalize_evaluation(
        {"score": "88.5", "evaluation": [{"question_id": "q1", "correct": "yes", "comment": "好"}]},
        quiz,
    )

    assert score == 88.5
    assert [item["question_id"] for item in evaluation] == ["q1", "q2"]
    assert evaluation[0]["correct"] is True
    assert evaluation[1]["correct"] is None  # 漏评题目补空项
    assert weaknesses == []
    assert review == ""
