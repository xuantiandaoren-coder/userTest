"""会话 / 聊天消息 / 面试接口测试：分页、归属隔离、附件段、级联删除。

消息没有写入接口（由 AI 侧写入），因此用例直接用 ORM 造数据，再用 HTTP 接口读。
"""

from __future__ import annotations

import base64

import httpx
import pytest
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, Interview, Resource
from tests.conftest import FakeSeaweedFSClient

SESSIONS = "/sessions"
INTERVIEWS = "/interviews"

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


async def _login(client: httpx.AsyncClient, username: str) -> tuple[dict[str, str], int]:
    """注册并登录，返回 (Authorization 头, 用户 id)。"""
    password = "HZq7mK2p"
    created = await client.post("/auth/register", json={"username": username, "password": password})
    tokens = (await client.post("/auth/login", json={"username": username, "password": password})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}, created.json()["id"]


async def _create_session(client: httpx.AsyncClient, headers: dict[str, str], title: str, model: int = 0) -> dict:
    response = await client.post(SESSIONS, json={"title": title, "session_model": model}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def _add_message(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    request_text: str,
    response_text: str = "回答",
    request_segments: list[dict] | None = None,
    response_segments: list[dict] | None = None,
    request_id: str = "req-1",
) -> ChatMessage:
    message = ChatMessage(
        user_id=user_id,
        session_id=session_id,
        select_model=0,
        request_id=request_id,
        request_text=request_text,
        response_text=response_text,
        request_segments=request_segments,
        response_segments=response_segments,
    )
    db.add(message)
    db.flush()
    return message


@pytest.mark.anyio
async def test_sessions_require_authentication(httpx_client: httpx.AsyncClient) -> None:
    assert (await httpx_client.get(SESSIONS)).status_code == 401
    assert (await httpx_client.post(SESSIONS, json={"title": "x", "session_model": 0})).status_code == 401


@pytest.mark.anyio
async def test_create_and_list_sessions_with_pagination(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "chat_pager")
    for index in range(1, 4):
        await _create_session(httpx_client, headers, f"会话{index}", model=index % 3)

    first = (await httpx_client.get(f"{SESSIONS}?page=1&page_size=2", headers=headers)).json()
    second = (await httpx_client.get(f"{SESSIONS}?page=2&page_size=2", headers=headers)).json()

    assert first["total"] == 3 and first["page"] == 1 and first["page_size"] == 2
    assert [item["title"] for item in first["items"]] == ["会话3", "会话2"]  # 新建的在前
    assert [item["title"] for item in second["items"]] == ["会话1"]
    assert set(first["items"][0]) == {"id", "title", "session_model", "created_at"}


@pytest.mark.anyio
async def test_session_pagination_params_are_validated(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "chat_params")

    assert (await httpx_client.get(f"{SESSIONS}?page=0", headers=headers)).status_code == 422
    assert (await httpx_client.get(f"{SESSIONS}?page_size=0", headers=headers)).status_code == 422
    assert (await httpx_client.get(f"{SESSIONS}?page_size=101", headers=headers)).status_code == 422
    assert (
        await httpx_client.post(SESSIONS, json={"title": "", "session_model": 0}, headers=headers)
    ).status_code == 422
    assert (
        await httpx_client.post(SESSIONS, json={"title": "x", "session_model": 9}, headers=headers)
    ).status_code == 422


@pytest.mark.anyio
async def test_update_session_title(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "chat_renamer")
    session = await _create_session(httpx_client, headers, "旧标题")

    response = await httpx_client.put(f"{SESSIONS}/{session['id']}", json={"title": "新标题"}, headers=headers)

    assert response.status_code == 200
    assert response.json()["title"] == "新标题"
    listed = (await httpx_client.get(SESSIONS, headers=headers)).json()
    assert [item["title"] for item in listed["items"]] == ["新标题"]


@pytest.mark.anyio
async def test_sessions_are_isolated_between_users(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    alice, _ = await _login(httpx_client, "chat_alice")
    bob, _ = await _login(httpx_client, "chat_bob")
    alice_session = await _create_session(httpx_client, alice, "alice 的会话")

    assert (await httpx_client.get(SESSIONS, headers=bob)).json()["total"] == 0
    renamed = await httpx_client.put(f"{SESSIONS}/{alice_session['id']}", json={"title": "改"}, headers=bob)
    assert renamed.status_code == 404
    assert (await httpx_client.delete(f"{SESSIONS}/{alice_session['id']}", headers=bob)).status_code == 404
    assert (await httpx_client.get(f"{SESSIONS}/{alice_session['id']}/messages", headers=bob)).status_code == 404


@pytest.mark.anyio
async def test_messages_pagination_shape_and_interview_card_fields(
    db_override: None, db_session: Session, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, user_id = await _login(httpx_client, "chat_msgs")
    session = await _create_session(httpx_client, headers, "模拟面试", model=1)

    messages = [
        _add_message(
            db_session,
            user_id=user_id,
            session_id=session["id"],
            request_text=f"问题{index}",
            request_id=f"req-{index}",
        )
        for index in range(1, 4)
    ]
    interview = Interview(
        session_id=session["id"],
        user_id=user_id,
        message_id=messages[0].id,
        qa_object=[{"id": "q1", "question": "讲讲索引", "answer": "…", "created_at": 1717171717}],
        status=1,
    )
    db_session.add(interview)
    db_session.commit()

    page = (await httpx_client.get(f"{SESSIONS}/{session['id']}/messages?page=1&page_size=2", headers=headers)).json()

    assert page["total"] == 3 and len(page["items"]) == 2
    assert [item["request_text"] for item in page["items"]] == ["问题1", "问题2"]  # 时间正序
    first = page["items"][0]
    assert set(first) == {
        "id",
        "session_id",
        "select_model",
        "request_id",
        "request_text",
        "response_text",
        "request_segments",
        "response_segments",
        "status",
        "interview_id",
        "created_at",
    }
    assert first["interview_id"] == interview.id  # 面试卡片：入口消息带 interview_id + status
    assert first["status"] == 1
    assert first["created_at"] > 0
    assert page["items"][1]["interview_id"] is None and page["items"][1]["status"] is None


@pytest.mark.anyio
async def test_message_segments_are_enriched_from_resources(
    db_override: None, db_session: Session, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, user_id = await _login(httpx_client, "chat_segments")
    session = await _create_session(httpx_client, headers, "带附件的会话")

    uploaded = (
        await httpx_client.post(
            "/files/upload", files={"file": ("pic.png", PNG_BYTES, "image/png")}, headers=headers
        )
    ).json()
    _add_message(
        db_session,
        user_id=user_id,
        session_id=session["id"],
        request_text="这张图是什么",
        response_segments=[{"type": "image", "resource_id": uploaded["resource_id"]}],
    )
    db_session.commit()

    item = (await httpx_client.get(f"{SESSIONS}/{session['id']}/messages", headers=headers)).json()["items"][0]

    assert item["request_segments"] == []
    segment = item["response_segments"][0]
    assert segment["type"] == "image"
    assert segment["resource_id"] == uploaded["resource_id"]
    assert segment["name"] == "pic.png"
    assert segment["url"] == f"https://fake-seaweedfs.test/{uploaded['path']}"


@pytest.mark.anyio
async def test_message_segments_skip_broken_entries(
    db_override: None, db_session: Session, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    """脏段（缺字段 / 类型非法 / 资源已删）不能让整个消息列表接口挂掉。"""
    headers, user_id = await _login(httpx_client, "chat_dirty")
    session = await _create_session(httpx_client, headers, "脏数据")

    _add_message(
        db_session,
        user_id=user_id,
        session_id=session["id"],
        request_text="x",
        request_segments=[
            {"type": "video", "resource_id": 1},
            {"resource_id": 2},
            {"type": "file", "resource_id": 999},
        ],
    )
    db_session.commit()

    item = (await httpx_client.get(f"{SESSIONS}/{session['id']}/messages", headers=headers)).json()["items"][0]

    assert item["request_segments"] == [{"type": "file", "resource_id": 999, "name": None, "url": None}]


@pytest.mark.anyio
async def test_get_interview_detail_by_id_and_user(
    db_override: None, db_session: Session, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    alice, alice_id = await _login(httpx_client, "iv_alice")
    bob, _ = await _login(httpx_client, "iv_bob")
    session = await _create_session(httpx_client, alice, "模拟面试", model=1)
    message = _add_message(db_session, user_id=alice_id, session_id=session["id"], request_text="开始模拟面试")
    qa_object = [{"id": "q1", "question": "讲讲索引", "answer": "…", "created_at": 1717171717}]
    interview = Interview(
        session_id=session["id"], user_id=alice_id, message_id=message.id, qa_object=qa_object, status=0
    )
    db_session.add(interview)
    db_session.commit()

    response = await httpx_client.get(f"{INTERVIEWS}/{interview.id}", headers=alice)

    assert response.status_code == 200
    body = response.json()
    assert body["qa_object"] == qa_object
    assert body["status"] == 0 and body["interview_duration"] == 0
    assert body["message_id"] == message.id
    assert (await httpx_client.get(f"{INTERVIEWS}/{interview.id}", headers=bob)).status_code == 404  # 别人的面试
    assert (await httpx_client.get(f"{INTERVIEWS}/{interview.id + 1000}", headers=alice)).status_code == 404


@pytest.mark.anyio
async def test_delete_session_cascades_messages_interviews_and_resources(
    db_override: None, db_session: Session, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, user_id = await _login(httpx_client, "chat_cascade")
    session = await _create_session(httpx_client, headers, "待删除")
    other = await _create_session(httpx_client, headers, "保留")

    uploaded = (
        await httpx_client.post(
            "/files/upload", files={"file": ("doc.png", PNG_BYTES, "image/png")}, headers=headers
        )
    ).json()
    message = _add_message(
        db_session,
        user_id=user_id,
        session_id=session["id"],
        request_text="附件消息",
        request_segments=[{"type": "image", "resource_id": uploaded["resource_id"]}],
    )
    interview = Interview(session_id=session["id"], user_id=user_id, message_id=message.id, qa_object=[])
    db_session.add(interview)
    db_session.commit()
    assert uploaded["path"] in storage_override.objects

    response = await httpx_client.delete(f"{SESSIONS}/{session['id']}", headers=headers)

    assert response.status_code == 204
    assert (await httpx_client.get(SESSIONS, headers=headers)).json()["items"][0]["id"] == other["id"]
    assert db_session.query(ChatMessage).filter_by(session_id=session["id"]).count() == 0
    assert db_session.query(Interview).filter_by(id=interview.id).count() == 0
    assert db_session.query(Resource).filter_by(id=uploaded["resource_id"]).count() == 0  # 元数据删除
    assert uploaded["path"] not in storage_override.objects  # 对象存储文件也删掉
    assert (await httpx_client.get(f"{INTERVIEWS}/{interview.id}", headers=headers)).status_code == 404


@pytest.mark.anyio
async def test_delete_session_keeps_resources_still_used_elsewhere(
    db_override: None, db_session: Session, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    """同一份文件被别的会话引用时，不能跟着被删掉。"""
    headers, user_id = await _login(httpx_client, "chat_shared")
    first = await _create_session(httpx_client, headers, "会话A")
    second = await _create_session(httpx_client, headers, "会话B")

    uploaded = (
        await httpx_client.post(
            "/files/upload", files={"file": ("shared.png", PNG_BYTES, "image/png")}, headers=headers
        )
    ).json()
    segment = [{"type": "image", "resource_id": uploaded["resource_id"]}]
    _add_message(db_session, user_id=user_id, session_id=first["id"], request_text="A", request_segments=segment)
    _add_message(db_session, user_id=user_id, session_id=second["id"], request_text="B", request_segments=segment)
    db_session.commit()

    await httpx_client.delete(f"{SESSIONS}/{first['id']}", headers=headers)

    assert db_session.query(Resource).filter_by(id=uploaded["resource_id"]).count() == 1
    assert uploaded["path"] in storage_override.objects

    await httpx_client.delete(f"{SESSIONS}/{second['id']}", headers=headers)

    assert db_session.query(Resource).filter_by(id=uploaded["resource_id"]).count() == 0
    assert uploaded["path"] not in storage_override.objects


@pytest.mark.anyio
async def test_session_lookup_returns_404_for_missing_id(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "chat_missing")

    response = await httpx_client.get(f"{SESSIONS}/999999/messages", headers=headers)

    assert response.status_code == 404
    assert response.json()["code"] == "SESSION_NOT_FOUND"


@pytest.mark.anyio
async def test_sessions_are_reachable_under_api_v1_prefix(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers, _ = await _login(httpx_client, "chat_prefix")

    response = await httpx_client.post(
        "/api/v1/sessions", json={"title": "带前缀", "session_model": 2}, headers=headers
    )

    assert response.status_code == 201
    assert (await httpx_client.get("/api/v1/sessions", headers=headers)).json()["total"] == 1
