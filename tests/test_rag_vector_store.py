"""向量库写入：同一（user_id, file_name）覆盖语义（先删旧分块，再写新分块）。

用例用假 Qdrant 客户端替换真实连接，只验证调用顺序、过滤条件与异常转换。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunk, Resource, User
from app.rag import core as rag_core


class _FakeQdrantClient:
    """假 Qdrant：记录调用顺序，可指定 count 返回值或让删除抛错。"""

    def __init__(
        self,
        *,
        existing: int = 0,
        collections: tuple[str, ...] = (rag_core.COLLECTION_NAME,),
        fail_on: str | None = None,
    ) -> None:
        self.existing = existing
        self.collections = collections
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.deleted_selectors: list[object] = []
        self.upserted: list[object] = []
        self.created: list[str] = []

    # --- 被 core.py 用到的接口 ---
    def get_collections(self) -> object:
        self.calls.append("get_collections")
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.calls.append("create_collection")
        self.created.append(collection_name)

    def count(self, *, collection_name: str, count_filter: object) -> object:
        self.calls.append("count")
        if self.fail_on == "count":
            raise RuntimeError("qdrant count 挂了")
        return SimpleNamespace(count=self.existing)

    def delete(self, *, collection_name: str, points_selector: object) -> None:
        self.calls.append("delete")
        if self.fail_on == "delete":
            raise RuntimeError("qdrant delete 挂了")
        self.deleted_selectors.append(points_selector)

    def upsert(self, *, collection_name: str, points: list[object]) -> None:
        self.calls.append("upsert")
        if self.fail_on == "upsert":
            raise RuntimeError("qdrant upsert 挂了")
        self.upserted.append(points)


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeQdrantClient:
    """把 core 里的 QdrantClient 换成假实现（默认：库里已有 3 条同文件旧分块）。"""
    client = _FakeQdrantClient(existing=3)
    monkeypatch.setattr(rag_core, "QdrantClient", lambda **kwargs: client)
    return client


def _write(*, user_id: int = 7, file_name: str = "resume.pdf") -> list[str]:
    return rag_core._write_qdrant(
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        user_id=user_id,
        doc_category="resume",
        file_name=file_name,
    )


def test_delete_happens_before_upsert(fake_client: _FakeQdrantClient) -> None:
    """覆盖语义：先删除同名文件的旧分块，再写入新分块。"""
    _write()

    assert fake_client.calls.index("count") < fake_client.calls.index("delete")
    assert fake_client.calls.index("delete") < fake_client.calls.index("upsert")


def test_delete_filter_targets_user_and_file(fake_client: _FakeQdrantClient) -> None:
    """过滤条件必须是（user_id, file_name）两个字段，不能误删同用户其它文件。"""
    _write(user_id=42, file_name="note.docx")

    selector = fake_client.deleted_selectors[0]
    conditions = {condition.key: condition.match.value for condition in selector.must}
    assert conditions == {"user_id": 42, "file_name": "note.docx"}


def test_no_delete_when_nothing_to_overwrite(fake_client: _FakeQdrantClient) -> None:
    """首次上传（库里没有同名分块）时不发删除请求，省一次写操作。"""
    fake_client.existing = 0

    _write()

    assert "delete" not in fake_client.calls
    assert "upsert" in fake_client.calls


def test_created_payload_carries_filters_but_not_text(fake_client: _FakeQdrantClient) -> None:
    """payload 只放检索过滤字段：带（user_id, file_name）用于覆盖删除，且不再存原文。"""
    point_ids = _write(user_id=9, file_name="study.pdf")

    points = fake_client.upserted[0]
    assert len(points) == len(point_ids) == 2
    payloads = [point.payload for point in points]
    assert all(payload["user_id"] == 9 and payload["file_name"] == "study.pdf" for payload in payloads)
    assert all(payload["doc_category"] == "resume" for payload in payloads)
    assert [payload["chunk_index"] for payload in payloads] == [0, 1]
    assert all("text" not in payload for payload in payloads)  # 原文改存 MySQL knowledge_chunks
    assert [point.id for point in points] == point_ids  # 返回值就是写进向量库的 point id


def test_delete_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """删除旧分块失败时抛 VECTOR_STORE_FAILED，且不继续写新分块（避免新旧并存）。"""
    client = _FakeQdrantClient(existing=2, fail_on="delete")
    monkeypatch.setattr(rag_core, "QdrantClient", lambda **kwargs: client)

    with pytest.raises(rag_core.VectorStoreError) as excinfo:
        _write()

    assert excinfo.value.code == "VECTOR_STORE_FAILED"
    assert "upsert" not in client.calls


def test_upsert_failure_is_wrapped(fake_client: _FakeQdrantClient) -> None:
    """写入失败同样转成系统异常（旧分块已被删，重传一次即可恢复）。"""
    fake_client.fail_on = "upsert"

    with pytest.raises(rag_core.VectorStoreError) as excinfo:
        _write()

    assert excinfo.value.code == "VECTOR_STORE_FAILED"


def test_ingest_file_writes_chunks_to_mysql_with_matching_vector_ids(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """跑到 ingest_file 这一层：先删旧点再写新点，并把原文落到 MySQL，两边 UUID 同值。"""
    client = _FakeQdrantClient(existing=5)
    monkeypatch.setattr(rag_core, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(rag_core, "embed_texts", lambda texts: [[0.0, 1.0] for _ in texts])

    user = User(user_name="vectorstore", password="x" * 60)
    db_session.add(user)
    db_session.flush()
    resource = Resource(
        resource_type=0,
        storage_scene=0,
        upload_purpose=0,
        file_name="resume.txt",
        file_hash="f" * 32,
        storage_path=f"{user.id}/{'f' * 32}.txt",
        user_id=user.id,
    )
    db_session.add(resource)
    db_session.flush()

    summary = rag_core.ingest_file(
        "# 个人简历\n姓名：张三".encode(),
        "resume.txt",
        user_id=user.id,
        db=db_session,
        resource_id=resource.id,
    )

    assert summary["chunk_count"] == 1
    assert client.calls.index("delete") < client.calls.index("upsert")
    assert summary["stored_chunks"] == 1

    rows = db_session.scalars(select(KnowledgeChunk)).all()
    assert [row.text for row in rows] == ["# 个人简历\n姓名：张三"]
    assert [row.vector_id for row in rows] == summary["point_ids"]  # Qdrant point id == MySQL vector_id
    assert rows[0].resource_id == resource.id


def test_ingest_file_skips_mysql_when_no_session(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """没传 db / resource_id 时不落原文，但要留警告（避免线上静默丢内容）。"""
    client = _FakeQdrantClient(existing=0)
    monkeypatch.setattr(rag_core, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(rag_core, "embed_texts", lambda texts: [[0.0, 1.0] for _ in texts])

    with caplog.at_level("WARNING"):
        summary = rag_core.ingest_file("只有一行的知识点内容。".encode(), "note.txt", user_id=1)

    assert summary["stored_chunks"] == 0
    assert "分块原文未落库" in caplog.text
