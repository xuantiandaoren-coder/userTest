"""上传触发 RAG 入库：调用时机（仅文件类型 + 新落库）、携带的文档分类、失败不影响上传。"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator

import httpx
import pytest
from sqlalchemy.orm import Session

from app.api.deps import get_rag_ingest_service
from app.db.models import Resource, User
from app.main import app
from app.services.rag_service import RagIngestResult, RagIngestService
from tests.conftest import FakeSeaweedFSClient

UPLOAD = "/files/upload"

# 一张真实的 1x1 PNG（含正确魔数）
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
TXT_BYTES = b"knowledge point 1\nknowledge point 2\n"


class FakeRagIngestService:
    """假 RAG 服务：只记录调用，不连 DashScope / Qdrant。"""

    def __init__(self, result: RagIngestResult | None = None) -> None:
        self.result = result or RagIngestResult(ingested=True, doc_category="resume", chunk_count=3)
        self.calls: list[dict[str, object]] = []

    async def ingest(
        self,
        *,
        data: bytes,
        file_name: str,
        user_id: int,
        doc_category: str | None = None,
    ) -> RagIngestResult:
        self.calls.append({"data": data, "file_name": file_name, "user_id": user_id, "doc_category": doc_category})
        return self.result


@pytest.fixture()
def install_rag() -> Iterator[Callable[[FakeRagIngestService], FakeRagIngestService]]:
    """把上传链路里的 RAG 服务换成假实现，用例结束自动还原。"""

    def _install(service: FakeRagIngestService) -> FakeRagIngestService:
        app.dependency_overrides[get_rag_ingest_service] = lambda: service
        return service

    try:
        yield _install
    finally:
        app.dependency_overrides.pop(get_rag_ingest_service, None)


async def _login(client: httpx.AsyncClient, username: str) -> dict[str, str]:
    """注册并登录，返回 Authorization 头。"""
    password = "HZq7mK2p"
    assert (await client.post("/auth/register", json={"username": username, "password": password})).status_code == 201
    tokens = (await client.post("/auth/login", json={"username": username, "password": password})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _user_id(session: Session, username: str) -> int:
    return session.query(User).filter_by(user_name=username).one().id


@pytest.mark.anyio
async def test_document_upload_ingests_into_rag_with_doc_category(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    install_rag: Callable[[FakeRagIngestService], FakeRagIngestService],
) -> None:
    """文件类型上传：把前端传的文档分类原样带给 RAG，并在响应里回显入库结果。"""
    rag = install_rag(FakeRagIngestService())
    headers = await _login(httpx_client, "ragdoc")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("resume.txt", TXT_BYTES, "text/plain")},
        data={"doc_category": "resume"},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["doc_category"] == "resume"
    assert body["rag_ingested"] is True
    assert body["rag_chunk_count"] == 3
    assert body["rag_error"] is None

    assert len(rag.calls) == 1
    call = rag.calls[0]
    assert call["file_name"] == "resume.txt"
    assert call["doc_category"] == "resume"  # 前端传的文件类型一路透传到 RAG
    assert call["data"] == TXT_BYTES  # 直接复用上传时读到的字节，不重复读文件
    assert call["user_id"] == _user_id(db_session, "ragdoc")


@pytest.mark.anyio
async def test_document_upload_without_doc_category_passes_empty(
    db_override: None,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    install_rag: Callable[[FakeRagIngestService], FakeRagIngestService],
) -> None:
    """不传文档分类时按空处理，由 RAG 侧自己按关键字分类。"""
    rag = install_rag(FakeRagIngestService())
    headers = await _login(httpx_client, "ragnocategory")

    await httpx_client.post(UPLOAD, files={"file": ("notes.txt", TXT_BYTES, "text/plain")}, headers=headers)

    assert [call["doc_category"] for call in rag.calls] == [None]


@pytest.mark.anyio
async def test_non_document_upload_does_not_touch_rag(
    db_override: None,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    install_rag: Callable[[FakeRagIngestService], FakeRagIngestService],
) -> None:
    """仅文件类型入库：图片 / 音频不触发 RAG 调用。"""
    rag = install_rag(FakeRagIngestService())
    headers = await _login(httpx_client, "ragimage")

    response = await httpx_client.post(UPLOAD, files={"file": ("pic.png", PNG_BYTES, "image/png")}, headers=headers)

    assert response.status_code == 201
    assert response.json()["rag_ingested"] is False
    assert response.json()["rag_error"] is None  # 没尝试入库，不算错误
    assert rag.calls == []


@pytest.mark.anyio
async def test_extract_only_scene_does_not_ingest(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    install_rag: Callable[[FakeRagIngestService], FakeRagIngestService],
) -> None:
    """storage_scene=2 不落库，也不入库（没有元数据就不必进知识库）。"""
    rag = install_rag(FakeRagIngestService())
    headers = await _login(httpx_client, "ragextract")

    response = await httpx_client.post(
        UPLOAD,
        files={"file": ("note.txt", TXT_BYTES, "text/plain")},
        data={"storage_scene": 2},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["rag_ingested"] is False
    assert db_session.query(Resource).count() == 0
    assert rag.calls == []


@pytest.mark.anyio
async def test_deduplicated_upload_does_not_reingest(
    db_override: None,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    install_rag: Callable[[FakeRagIngestService], FakeRagIngestService],
) -> None:
    """同一内容重复上传：第一次入库，第二次命中去重不再入库，避免向量库里堆重复点。"""
    rag = install_rag(FakeRagIngestService())
    headers = await _login(httpx_client, "ragdedup")

    first = await httpx_client.post(UPLOAD, files={"file": ("a.txt", TXT_BYTES, "text/plain")}, headers=headers)
    second = await httpx_client.post(UPLOAD, files={"file": ("b.txt", TXT_BYTES, "text/plain")}, headers=headers)

    assert first.json()["rag_ingested"] is True
    assert second.json()["deduplicated"] is True
    assert second.json()["rag_ingested"] is False
    assert second.json()["rag_error"] is None
    assert len(rag.calls) == 1


@pytest.mark.anyio
async def test_rag_failure_does_not_break_upload(
    db_override: None,
    db_session: Session,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
    install_rag: Callable[[FakeRagIngestService], FakeRagIngestService],
) -> None:
    """RAG 未配置 / 调用失败：上传照样 201、元数据照样写，只在响应里给出错误码。"""
    install_rag(FakeRagIngestService(RagIngestResult(ingested=False, error="RAG_NOT_CONFIGURED")))
    headers = await _login(httpx_client, "ragfail")

    response = await httpx_client.post(UPLOAD, files={"file": ("note.txt", TXT_BYTES, "text/plain")}, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["rag_ingested"] is False
    assert body["rag_error"] == "RAG_NOT_CONFIGURED"
    assert storage_override.objects[body["path"]] == TXT_BYTES  # 原文件已存对象存储
    assert db_session.query(Resource).count() == 1  # 元数据已落库


@pytest.mark.anyio
async def test_rag_disabled_by_config_skips_ingest(
    db_override: None,
    storage_override: FakeSeaweedFSClient,
    httpx_client: httpx.AsyncClient,
) -> None:
    """RAG_INGEST_ENABLED=false（测试默认）：走真实服务但直接跳过，不去连向量库。"""
    headers = await _login(httpx_client, "ragdisabled")

    response = await httpx_client.post(UPLOAD, files={"file": ("note.txt", TXT_BYTES, "text/plain")}, headers=headers)

    assert response.status_code == 201
    assert response.json()["rag_ingested"] is False
    assert response.json()["rag_error"] == "RAG_DISABLED"


@pytest.mark.anyio
async def test_rag_service_disabled_returns_disabled_result() -> None:
    """开关关闭时连 ingest_file 都不会 import / 调用。"""
    result = await RagIngestService(enabled=False).ingest(
        data=TXT_BYTES, file_name="note.txt", user_id=1, doc_category="resume"
    )

    assert result == RagIngestResult(ingested=False, error="RAG_DISABLED")


@pytest.mark.anyio
async def test_rag_service_maps_ingest_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实服务的返回值映射：RAG 给出的最终分类与分块数透出给接口。"""
    app_rag = pytest.importorskip("app.rag")
    seen: dict[str, object] = {}

    def fake_ingest_file(
        source: bytes,
        file_name: str | None = None,
        *,
        user_id: int,
        doc_category: str | None = None,
    ) -> dict[str, object]:
        seen.update({"source": source, "file_name": file_name, "user_id": user_id, "doc_category": doc_category})
        return {
            "file_name": file_name,
            "user_id": user_id,
            "doc_category": "study_material",
            "chunk_count": 5,
            "point_ids": [],
        }

    monkeypatch.setattr(app_rag, "ingest_file", fake_ingest_file)

    result = await RagIngestService(enabled=True).ingest(
        data=TXT_BYTES, file_name="course.txt", user_id=7, doc_category="study_material"
    )

    assert seen == {
        "source": TXT_BYTES,
        "file_name": "course.txt",
        "user_id": 7,
        "doc_category": "study_material",
    }
    assert result == RagIngestResult(ingested=True, doc_category="study_material", chunk_count=5)


@pytest.mark.anyio
async def test_rag_service_swallows_rag_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG 抛业务异常时不向上冒泡，转成错误码返回给接口。"""
    app_rag = pytest.importorskip("app.rag")
    from app.rag.core import RagNotConfiguredError

    def fake_ingest_file(
        source: bytes,
        file_name: str | None = None,
        *,
        user_id: int,
        doc_category: str | None = None,
    ) -> dict[str, object]:
        raise RagNotConfiguredError(detail="missing dashscope_api_key")

    monkeypatch.setattr(app_rag, "ingest_file", fake_ingest_file)

    result = await RagIngestService(enabled=True).ingest(data=TXT_BYTES, file_name="note.txt", user_id=1)

    assert result == RagIngestResult(ingested=False, error="RAG_NOT_CONFIGURED")


@pytest.mark.anyio
async def test_rag_service_swallows_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """未预期异常同样只记日志，不让上传接口 500。"""
    app_rag = pytest.importorskip("app.rag")

    def fake_ingest_file(
        source: bytes,
        file_name: str | None = None,
        *,
        user_id: int,
        doc_category: str | None = None,
    ) -> dict[str, object]:
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr(app_rag, "ingest_file", fake_ingest_file)

    result = await RagIngestService(enabled=True).ingest(data=TXT_BYTES, file_name="note.txt", user_id=1)

    assert result == RagIngestResult(ingested=False, error="RAG_INGEST_FAILED")
