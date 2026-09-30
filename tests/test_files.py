"""文件上传测试：需认证、类型识别、SeaweedFS 存原文件、resources 存元数据、MD5 去重。"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta

import httpx
import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.storage import DocCategory, InvalidDocCategoryError, normalize_doc_category
from app.db.models import Resource, User
from tests.conftest import FakeSeaweedFSClient

UPLOAD = "/files/upload"
UPLOAD_ALIAS = "/upload/file"

# 一张真实的 1x1 PNG（含正确魔数）
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
TXT_BYTES = b"hello upload\n"
MP3_BYTES = b"ID3\x03\x00\x00\x00\x00\x00\x00audio-payload"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("Resume", DocCategory.RESUME),
        (" study_material ", DocCategory.STUDY_MATERIAL),
        ("GENERAL", DocCategory.GENERAL),
    ],
)
def test_normalize_doc_category(raw: str | None, expected: DocCategory | None) -> None:
    """空串 / 空白视为可空，合法值大小写与首尾空白都能归一化。"""
    assert normalize_doc_category(raw) == expected


def test_normalize_doc_category_rejects_unknown_value() -> None:
    with pytest.raises(InvalidDocCategoryError):
        normalize_doc_category("cv")


async def _login(client: httpx.AsyncClient, username: str) -> dict[str, str]:
    """注册并登录，返回 Authorization 头。"""
    password = "HZq7mK2p"
    assert (await client.post("/auth/register", json={"username": username, "password": password})).status_code == 201
    tokens = (await client.post("/auth/login", json={"username": username, "password": password})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _resource(session: Session, user_id: int) -> Resource:
    return session.query(Resource).filter_by(user_id=user_id).one()


@pytest.mark.anyio
async def test_upload_requires_authentication(httpx_client: httpx.AsyncClient) -> None:
    response = await httpx_client.post(UPLOAD, files={"file": ("a.png", PNG_BYTES, "image/png")})

    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_INVALID"


@pytest.mark.anyio
async def test_image_with_avatar_purpose_updates_avatar(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "uploader")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("avatar.png", PNG_BYTES, "image/png")},
        data={"upload_purpose": 1},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "image"
    assert body["resource_type"] == 1
    assert body["avatar_updated"] is True
    assert body["size"] == len(PNG_BYTES)
    assert body["deduplicated"] is False
    assert body["storage_scene"] == 0

    # 原文件在对象存储：键为 <user_id>/<md5>.png
    user_id = _user_id(db_session, "uploader")
    assert body["path"] == f"{user_id}/{body['md5']}.png"
    assert storage_override.objects[body["path"]] == PNG_BYTES
    assert body["url"] == f"https://fake-seaweedfs.test/{body['path']}"  # 可直接访问的地址

    # 元数据在 MySQL：只有一条，字段与请求一致
    row = _resource(db_session, user_id)
    assert row.file_hash == body["md5"]
    assert row.file_name == "avatar.png"
    assert row.storage_path == body["path"]
    assert row.resource_type == 1
    assert row.upload_purpose == 1

    # 头像写的是对象键
    assert _user(db_session, "uploader").avatar == body["path"]

    # 用户接口同时给出 avatar 与可直接 <img src> 的 avatar_url
    me = (await httpx_client.get("/auth/me", headers=headers)).json()
    assert me["avatar"] == body["path"]
    assert me["avatar_url"] == body["url"]


@pytest.mark.anyio
async def test_image_without_avatar_purpose_keeps_avatar_empty(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "generalimg")

    response = await httpx_client.post(UPLOAD, files={"file": ("pic.png", PNG_BYTES, "image/png")}, headers=headers)

    assert response.status_code == 201
    assert response.json()["avatar_updated"] is False
    assert _user(db_session, "generalimg").avatar is None  # 普通资源不动头像


@pytest.mark.anyio
async def test_document_is_only_stored(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "docuploader")

    response = await httpx_client.post(UPLOAD, files={"file": ("note.txt", TXT_BYTES, "text/plain")}, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "document"
    assert body["resource_type"] == 0
    assert body["avatar_updated"] is False
    assert storage_override.objects[body["path"]] == TXT_BYTES
    assert _user(db_session, "docuploader").avatar is None  # 文档不动 avatar


@pytest.mark.anyio
async def test_audio_is_detected_as_audio_resource(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "audiouploader")

    response = await httpx_client.post(UPLOAD, files={"file": ("clip.mp3", MP3_BYTES, "audio/mpeg")}, headers=headers)

    assert response.status_code == 201
    assert response.json()["kind"] == "audio"
    assert response.json()["resource_type"] == 2
    assert _resource(db_session, _user_id(db_session, "audiouploader")).resource_type == 2


@pytest.mark.anyio
async def test_document_doc_category_is_persisted(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """文件类型可带文档分类：请求值原样落库，并在响应里回显。"""
    headers = await _login(httpx_client, "resumeuploader")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("resume.txt", TXT_BYTES, "text/plain")},
        data={"doc_category": "resume"},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "document"
    assert body["doc_category"] == "resume"
    row = _resource(db_session, _user_id(db_session, "resumeuploader"))
    assert row.doc_category == "resume"


@pytest.mark.anyio
async def test_document_doc_category_is_optional(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """文档分类可空：不传、传空串都落 NULL，不报错。"""
    headers = await _login(httpx_client, "nocategory")

    omitted = await httpx_client.post(
        UPLOAD, files={"file": ("a.txt", b"no category", "text/plain")}, headers=headers
    )
    empty = await httpx_client.post(
        UPLOAD,
        files={"file": ("b.txt", b"blank category", "text/plain")},
        data={"doc_category": ""},
        headers=headers,
    )

    assert omitted.status_code == 201
    assert omitted.json()["doc_category"] is None
    assert empty.status_code == 201
    assert empty.json()["doc_category"] is None
    rows = db_session.query(Resource).filter_by(user_id=_user_id(db_session, "nocategory")).all()
    assert [row.doc_category for row in rows] == [None, None]


@pytest.mark.anyio
async def test_doc_category_is_ignored_for_non_document(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """仅文件类型有意义：图片 / 音频传了文档分类也不落库。"""
    headers = await _login(httpx_client, "imgcategory")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("pic.png", PNG_BYTES, "image/png")},
        data={"doc_category": "resume"},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["doc_category"] is None
    assert _resource(db_session, _user_id(db_session, "imgcategory")).doc_category is None


@pytest.mark.anyio
async def test_invalid_doc_category_is_rejected(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "badcategory")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("a.txt", TXT_BYTES, "text/plain")},
        data={"doc_category": "not-a-category"},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_DOC_CATEGORY"
    assert db_session.query(Resource).count() == 0  # 脏值不落库
    assert storage_override.objects == {}


@pytest.mark.anyio
async def test_dedup_updates_doc_category(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """重复上传同一内容时，显式带来的分类会覆盖旧值；不传则保留。"""
    headers = await _login(httpx_client, "recategorize")

    first = await httpx_client.post(
        UPLOAD,
        files={"file": ("notes.txt", b"same content", "text/plain")},
        data={"doc_category": "general"},
        headers=headers,
    )
    second = await httpx_client.post(
        UPLOAD,
        files={"file": ("notes.txt", b"same content", "text/plain")},
        data={"doc_category": "study_material"},
        headers=headers,
    )
    third = await httpx_client.post(
        UPLOAD, files={"file": ("notes.txt", b"same content", "text/plain")}, headers=headers
    )

    assert first.json()["doc_category"] == "general"
    assert second.json()["deduplicated"] is True
    assert second.json()["doc_category"] == "study_material"
    assert third.json()["doc_category"] == "study_material"  # 不传时保留已有分类
    assert db_session.query(Resource).filter_by(user_id=_user_id(db_session, "recategorize")).count() == 1
    assert _resource(db_session, _user_id(db_session, "recategorize")).doc_category == "study_material"


@pytest.mark.anyio
async def test_same_content_is_deduplicated(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "deduper")

    first = await httpx_client.post(UPLOAD, files={"file": ("one.png", PNG_BYTES, "image/png")}, headers=headers)
    second = await httpx_client.post(UPLOAD, files={"file": ("two.png", PNG_BYTES, "image/png")}, headers=headers)

    assert first.json()["md5"] == second.json()["md5"]
    assert first.json()["path"] == second.json()["path"]
    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True  # 预查命中，未重复上传
    assert len(storage_override.objects) == 1
    assert db_session.query(Resource).filter_by(user_id=_user_id(db_session, "deduper")).count() == 1


@pytest.mark.anyio
async def test_files_are_isolated_per_user(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    alice = await _login(httpx_client, "alice_uploader")
    bob = await _login(httpx_client, "bob_uploader")

    a = await httpx_client.post(UPLOAD, files={"file": ("a.txt", TXT_BYTES, "text/plain")}, headers=alice)
    b = await httpx_client.post(UPLOAD, files={"file": ("b.txt", TXT_BYTES, "text/plain")}, headers=bob)

    # 同内容不同用户：各自一条元数据、各自一个对象键
    assert a.json()["path"] != b.json()["path"]
    assert a.json()["path"].startswith(f"{_user_id(db_session, 'alice_uploader')}/")
    assert b.json()["path"].startswith(f"{_user_id(db_session, 'bob_uploader')}/")
    assert len(storage_override.objects) == 2


@pytest.mark.anyio
async def test_extract_only_scene_keeps_no_file_and_no_metadata(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """storage_scene=2：只提取内容，不上传原文件、不写元数据、不动头像。"""
    headers = await _login(httpx_client, "extractor")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("note.txt", TXT_BYTES, "text/plain")},
        data={"storage_scene": 2, "upload_purpose": 1},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["extracted_text"] == TXT_BYTES.decode()
    assert body["path"] == ""
    assert body["url"] == ""  # 没存对象，自然没有访问地址
    assert storage_override.objects == {}  # 没存原文件
    assert db_session.query(Resource).count() == 0  # 没写元数据
    assert _user(db_session, "extractor").avatar is None


@pytest.mark.anyio
async def test_expire_time_follows_storage_scene(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "ttluser")

    await httpx_client.post(UPLOAD, files={"file": ("long.txt", b"long-lived", "text/plain")}, headers=headers)
    await httpx_client.post(
        UPLOAD,
        files={"file": ("short.txt", b"short-lived", "text/plain")},
        data={"storage_scene": 1},
        headers=headers,
    )

    rows = {row.file_name: row for row in db_session.query(Resource).all()}
    long_ttl = rows["long.txt"].expire_time - datetime.now()
    short_ttl = rows["short.txt"].expire_time - datetime.now()

    assert timedelta(seconds=settings.resource_ttl_long_seconds) - timedelta(minutes=1) < long_ttl
    assert long_ttl <= timedelta(seconds=settings.resource_ttl_long_seconds)
    assert short_ttl <= timedelta(seconds=settings.resource_ttl_short_seconds)
    assert rows["short.txt"].storage_scene == 1


@pytest.mark.anyio
async def test_upload_alias_path_is_available(
    db_override: None,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """兼容 /upload/file 与 /files/upload 两个路径。"""
    headers = await _login(httpx_client, "aliasuser")

    response = await httpx_client.post(
        UPLOAD_ALIAS, files={"file": ("note.txt", TXT_BYTES, "text/plain")}, headers=headers
    )

    assert response.status_code == 201
    assert response.json()["kind"] == "document"


@pytest.mark.anyio
async def test_unsupported_type_is_rejected(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "badtype")

    response = await httpx_client.post(
        UPLOAD, files={"file": ("evil.exe", b"MZ\x90\x00binary", "application/x-msdownload")}, headers=headers
    )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_FILE_TYPE"
    assert storage_override.objects == {}  # 未上传任何对象
    assert db_session.query(Resource).count() == 0


@pytest.mark.anyio
async def test_fake_image_is_rejected(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """声明 image/png 但内容不是图片：不能写进 avatar。"""
    headers = await _login(httpx_client, "fakeimg")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("fake.png", b"definitely not an image", "image/png")},
        data={"upload_purpose": 1},
        headers=headers,
    )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_FILE_TYPE"
    assert _user(db_session, "fakeimg").avatar is None
    assert storage_override.objects == {}


@pytest.mark.anyio
async def test_empty_file_is_rejected(db_override: None, httpx_client: httpx.AsyncClient) -> None:
    headers = await _login(httpx_client, "emptyfile")

    response = await httpx_client.post(UPLOAD, files={"file": ("empty.txt", b"", "text/plain")}, headers=headers)

    assert response.status_code == 400
    assert response.json()["code"] == "EMPTY_FILE"


@pytest.mark.anyio
async def test_oversized_file_is_rejected(
    db_override: None,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = await _login(httpx_client, "toobig")
    monkeypatch.setattr(settings, "max_upload_bytes", 16)

    response = await httpx_client.post(UPLOAD, files={"file": ("big.txt", b"x" * 64, "text/plain")}, headers=headers)

    assert response.status_code == 413
    assert response.json()["code"] == "FILE_TOO_LARGE"
    assert storage_override.objects == {}  # 超限的文件不会传到对象存储


@pytest.mark.anyio
async def test_path_traversal_filename_is_sanitized(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    headers = await _login(httpx_client, "traversal")

    response = await httpx_client.post(
        UPLOAD, files={"file": ("../../evil.txt", TXT_BYTES, "text/plain")}, headers=headers
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "evil.txt"  # 只保留 basename
    assert body["path"].split("/")[0] == str(_user_id(db_session, "traversal"))  # 键首段是用户 ID，没跳出目录


@pytest.mark.anyio
async def test_invalid_scene_or_purpose_is_rejected(
    db_override: None, storage_override: FakeSeaweedFSClient, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "badparams")

    scene = await httpx_client.post(
        UPLOAD, files={"file": ("a.txt", TXT_BYTES, "text/plain")}, data={"storage_scene": 9}, headers=headers
    )
    purpose = await httpx_client.post(
        UPLOAD, files={"file": ("b.txt", TXT_BYTES, "text/plain")}, data={"upload_purpose": 9}, headers=headers
    )

    assert scene.status_code == 422
    assert purpose.status_code == 422


def _user(session: Session, username: str) -> User:
    return session.query(User).filter_by(user_name=username).one()


def _user_id(session: Session, username: str) -> int:
    return _user(session, username).id
