"""用户接口测试：增删改查、密码哈希与参数校验。"""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import User
from app.services.user_service import pwd_context

USER_PATH = "/api/v1/users"


def _create(client: TestClient, username: str = "alice", password: str = "secret123"):
    return client.post(USER_PATH, json={"username": username, "password": password})


def test_create_user_hashes_password(client: TestClient, db_session: Session) -> None:
    response = _create(client)
    assert response.status_code == 201

    data = response.json()
    assert {"id", "username"} <= set(data)  # 响应不泄露任何密码字段
    assert set(data) <= {"id", "username", "avatar", "avatar_url"}
    assert data["username"] == "alice"

    row = db_session.get(User, data["id"])
    assert row is not None
    assert row.user_name == "alice"
    assert row.password != "secret123"  # 数据库层不保存明文
    assert pwd_context.verify("secret123", row.password)  # 存的是可校验的哈希
    assert row.create_time is not None  # create_time 由数据库自动填充


def test_duplicate_username_returns_conflict(client: TestClient) -> None:
    assert _create(client).status_code == 201
    duplicate = _create(client)
    assert duplicate.status_code == 409


def test_validation_rejects_bad_payload(client: TestClient) -> None:
    short_password = client.post(USER_PATH, json={"username": "alice", "password": "123"})
    assert short_password.status_code == 422

    short_username = client.post(USER_PATH, json={"username": "a", "password": "secret123"})
    assert short_username.status_code == 422


def test_list_and_get_user(client: TestClient) -> None:
    created = _create(client, username="bob").json()

    listed = client.get(USER_PATH)
    assert listed.status_code == 200
    assert [user["id"] for user in listed.json()] == [created["id"]]

    detail = client.get(f"{USER_PATH}/{created['id']}")
    assert detail.status_code == 200
    assert detail.json() == created

    missing = client.get(f"{USER_PATH}/999")
    assert missing.status_code == 404


def test_update_username_and_password(client: TestClient, db_session: Session) -> None:
    created = _create(client, username="carol").json()
    user_id = created["id"]

    response = client.patch(
        f"{USER_PATH}/{user_id}",
        json={"username": "carol2", "password": "new-secret-456"},
    )
    assert response.status_code == 200
    assert response.json() == {"id": user_id, "username": "carol2", "avatar": None, "avatar_url": None}

    row = db_session.get(User, user_id)
    assert row is not None
    assert row.user_name == "carol2"
    assert pwd_context.verify("new-secret-456", row.password)
    assert not pwd_context.verify("secret123", row.password)


def test_update_duplicate_username_returns_conflict(client: TestClient) -> None:
    first = _create(client, username="dave").json()
    _create(client, username="erin")

    response = client.patch(f"{USER_PATH}/{first['id']}", json={"username": "erin"})
    assert response.status_code == 409


def test_delete_user(client: TestClient) -> None:
    created = _create(client).json()

    deleted = client.delete(f"{USER_PATH}/{created['id']}")
    assert deleted.status_code == 204

    assert client.get(f"{USER_PATH}/{created['id']}").status_code == 404
    assert client.delete(f"{USER_PATH}/{created['id']}").status_code == 404
