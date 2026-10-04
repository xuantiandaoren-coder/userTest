"""RAG 检索：问题向量化 -> Qdrant 近邻（带 user_id 过滤）-> 回 MySQL 取原文。

用例用假 Qdrant 客户端 + 内存 SQLite，不连真实向量库与模型。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunk, Resource, User
from app.rag import chunk_store
from app.rag import retriever
from app.rag.retriever import RetrievalError, search_similar_chunks


class _FakeQdrantClient:
    """假 Qdrant：返回预设命中点，并记录查询参数。"""

    def __init__(
        self,
        points: list[object] | None = None,
        *,
        collections: tuple[str, ...] = (retriever.COLLECTION_NAME,),
        fail: bool = False,
    ) -> None:
        self.points = points or []
        self.collections = collections
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def get_collections(self) -> object:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    def query_points(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("qdrant 挂了")
        return SimpleNamespace(points=self.points)


def _point(vector_id: str, score: float, **payload: object) -> object:
    return SimpleNamespace(id=vector_id, score=score, payload=payload or None)


def _seed(db: Session, *, user_id: int, vector_ids: list[str], texts: list[str], file_name: str = "resume.pdf") -> None:
    """造一个用户 + 资源 + 分块行，模拟入库后的 MySQL 状态。"""
    user = User(id=user_id, user_name=f"u{user_id}", password="x" * 60)
    db.add(user)
    db.flush()
    resource = Resource(
        resource_type=0,
        storage_scene=0,
        upload_purpose=0,
        file_name=file_name,
        file_hash=f"{user_id:032d}",
        storage_path=f"{user_id}/f.txt",
        user_id=user_id,
    )
    db.add(resource)
    db.flush()
    db.add_all(
        [
            KnowledgeChunk(
                resource_id=resource.id,
                vector_id=vector_id,
                chunk_index=index,
                char_count=len(text),
                text=text,
            )
            for index, (vector_id, text) in enumerate(zip(vector_ids, texts, strict=True))
        ]
    )
    db.flush()


def _install(monkeypatch: pytest.MonkeyPatch, client: _FakeQdrantClient, vector: list[float] | None = None) -> None:
    monkeypatch.setattr(retriever, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [vector or [0.1, 0.2] for _ in texts])


def test_search_returns_hits_ordered_by_score(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """命中按相似度降序返回，原文来自 MySQL。"""
    _seed(db_session, user_id=1, vector_ids=["v1", "v2"], texts=["第一块原文", "第二块原文"])
    client = _FakeQdrantClient([_point("v2", 0.91, file_name="resume.pdf", doc_category="resume", chunk_index=1),
                                _point("v1", 0.77, file_name="resume.pdf", doc_category="resume", chunk_index=0)])
    _install(monkeypatch, client)

    hits = search_similar_chunks("二叉树怎么遍历", user_id=1, db=db_session)

    assert [hit.vector_id for hit in hits] == ["v2", "v1"]
    assert [hit.text for hit in hits] == ["第二块原文", "第一块原文"]
    assert [hit.score for hit in hits] == [0.91, 0.77]
    assert hits[0].file_name == "resume.pdf"
    assert hits[0].doc_category == "resume"
    assert hits[0].chunk_index == 1


def test_search_filters_by_user_id(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """检索必须带 user_id 过滤：多用户共用一个 collection，不过滤会串数据。"""
    _seed(db_session, user_id=7, vector_ids=["v1"], texts=["我的资料"])
    client = _FakeQdrantClient([_point("v1", 0.9)])
    _install(monkeypatch, client)

    search_similar_chunks("问题", user_id=7, db=db_session)

    query_filter = client.calls[0]["query_filter"]
    conditions = {condition.key: condition.match.value for condition in query_filter.must}
    assert conditions == {"user_id": 7}


def test_search_passes_top_k_and_question_vector(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """top_k 透传到 Qdrant，问题文本走同一个向量化函数。"""
    _seed(db_session, user_id=1, vector_ids=["v1"], texts=["甲"])
    client = _FakeQdrantClient([_point("v1", 0.5)])
    _install(monkeypatch, client, vector=[9.0, 8.0])

    search_similar_chunks("知识点", user_id=1, db=db_session, top_k=3)

    call = client.calls[0]
    assert call["limit"] == 3
    assert call["query"] == [9.0, 8.0]
    assert call["collection_name"] == retriever.COLLECTION_NAME
    assert call["with_vectors"] is False
    assert call["score_threshold"] == retriever.MIN_SCORE  # 阈值下推到服务端


def test_search_default_top_k_is_three(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """top_k 外部可控，不传时默认 3。"""
    _seed(db_session, user_id=1, vector_ids=["v1"], texts=["甲"])
    client = _FakeQdrantClient([_point("v1", 0.5)])
    _install(monkeypatch, client)

    search_similar_chunks("问题", user_id=1, db=db_session)

    assert retriever.DEFAULT_TOP_K == 3
    assert client.calls[0]["limit"] == 3


def test_search_drops_hits_below_score_threshold(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """相似度低于 0.4 的命中直接丢弃，避免把不相关片段塞进上下文。"""
    _seed(db_session, user_id=1, vector_ids=["high", "low"], texts=["相关的原文", "不相关的原文"])
    client = _FakeQdrantClient([_point("high", 0.61), _point("low", 0.39)])
    _install(monkeypatch, client)

    hits = search_similar_chunks("问题", user_id=1, db=db_session, top_k=5)

    assert [hit.vector_id for hit in hits] == ["high"]
    assert retriever.MIN_SCORE == 0.4


def test_search_keeps_hit_exactly_on_threshold(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """阈值是"低于才丢"：正好 0.4 的命中保留。"""
    _seed(db_session, user_id=1, vector_ids=["edge"], texts=["边界原文"])
    _install(monkeypatch, _FakeQdrantClient([_point("edge", 0.4)]))

    assert [hit.vector_id for hit in search_similar_chunks("问题", user_id=1, db=db_session)] == ["edge"]


def test_search_truncates_long_chunk_text(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """单条原文超过 500 字截断（只影响返回值，库里原文不动）。"""
    long_text = "知识点。" * 300          # 1200 字
    _seed(db_session, user_id=1, vector_ids=["long"], texts=[long_text])
    _install(monkeypatch, _FakeQdrantClient([_point("long", 0.9)]))

    hit = search_similar_chunks("问题", user_id=1, db=db_session)[0]

    assert len(hit.text) == retriever.MAX_TEXT_CHARS == 500
    assert hit.text == long_text[:500]
    # 库里原文没被截断
    assert chunk_store.fetch_chunks_by_vector_ids(db_session, ["long"])["long"].text == long_text


def test_search_keeps_short_chunk_text_intact(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """不超过 500 字的原文原样返回。"""
    _seed(db_session, user_id=1, vector_ids=["short"], texts=["短原文"])
    _install(monkeypatch, _FakeQdrantClient([_point("short", 0.9)]))

    assert search_similar_chunks("问题", user_id=1, db=db_session)[0].text == "短原文"


def test_search_skips_dangling_vector(db_session: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """向量库有、MySQL 没有的悬空点跳过并告警，不影响其余命中。"""
    _seed(db_session, user_id=1, vector_ids=["v1"], texts=["还在的原文"])
    client = _FakeQdrantClient([_point("ghost", 0.99), _point("v1", 0.8)])
    _install(monkeypatch, client)

    with caplog.at_level("WARNING"):
        hits = search_similar_chunks("问题", user_id=1, db=db_session)

    assert [hit.vector_id for hit in hits] == ["v1"]
    assert "悬空点" in caplog.text


def test_search_does_not_leak_other_users_chunks(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """即使向量库返回了别人的 point id，也取不到原文（MySQL 侧只有本人数据）。"""
    _seed(db_session, user_id=1, vector_ids=["mine"], texts=["我的资料"])
    _seed(db_session, user_id=2, vector_ids=["others"], texts=["别人的资料"])
    client = _FakeQdrantClient([_point("others", 0.9), _point("mine", 0.8)])
    _install(monkeypatch, client)

    hits = search_similar_chunks("问题", user_id=1, db=db_session)

    assert [hit.vector_id for hit in hits] == ["mine"]
    assert "别人的资料" not in [hit.text for hit in hits]


@pytest.mark.parametrize("query", ["", "   "])
def test_search_returns_empty_for_blank_query(db_session: Session, monkeypatch: pytest.MonkeyPatch, query: str) -> None:
    """空问题直接返回空列表：不调模型、不查向量库。"""
    client = _FakeQdrantClient()
    _install(monkeypatch, client)
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: pytest.fail("空问题不该调向量化"))

    assert search_similar_chunks(query, user_id=1, db=db_session) == []
    assert client.calls == []


@pytest.mark.parametrize("top_k", [0, -3])
def test_search_returns_empty_for_non_positive_top_k(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, top_k: int
) -> None:
    """top_k <= 0 直接返回空列表，不浪费一次向量化与检索。"""
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: pytest.fail("不该调向量化"))

    assert search_similar_chunks("问题", user_id=1, db=db_session, top_k=top_k) == []


def test_search_returns_empty_when_collection_missing(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """知识库还没建过 collection 时返回空列表，不抛异常。"""
    client = _FakeQdrantClient(collections=())
    _install(monkeypatch, client)

    with caplog.at_level("INFO"):
        assert search_similar_chunks("问题", user_id=1, db=db_session) == []

    assert client.calls == []
    assert "还没有 collection" in caplog.text


def test_search_returns_empty_when_no_neighbour(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """向量库没有命中时返回空列表。"""
    _install(monkeypatch, _FakeQdrantClient([]))

    assert search_similar_chunks("问题", user_id=1, db=db_session) == []


def test_search_wraps_qdrant_failure(monkeypatch: pytest.MonkeyPatch, db_session: Session) -> None:
    """向量库异常转成 RETRIEVAL_FAILED，细节留在 detail 里。"""
    _install(monkeypatch, _FakeQdrantClient(fail=True))

    with pytest.raises(RetrievalError) as excinfo:
        search_similar_chunks("问题", user_id=1, db=db_session)

    assert excinfo.value.code == "RETRIEVAL_FAILED"
    assert "qdrant 挂了" in (excinfo.value.detail or "")


def test_chunk_hit_as_dict_is_serializable(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """ChunkHit.as_dict 给出接口可序列化的结构。"""
    _seed(db_session, user_id=1, vector_ids=["v1"], texts=["原文"], file_name="note.txt")
    _install(monkeypatch, _FakeQdrantClient([_point("v1", 0.66, file_name="note.txt", doc_category="general", chunk_index=2)]))

    hit = search_similar_chunks("问题", user_id=1, db=db_session)[0]

    assert hit.as_dict() == {
        "vector_id": "v1",
        "resource_id": hit.resource_id,   # 出处：knowledge_chunks.resource_id
        "text": "原文",
        "score": 0.66,
        "file_name": "note.txt",
        "doc_category": "general",
        "chunk_index": 2,
    }


def test_fetch_chunks_by_vector_ids_returns_map(db_session: Session) -> None:
    """取原文函数按 vector_id 返回字典，供检索侧按分值顺序自己排。"""
    _seed(db_session, user_id=1, vector_ids=["a", "b"], texts=["甲", "乙"])

    rows = chunk_store.fetch_chunks_by_vector_ids(db_session, ["b", "missing", "a"])

    assert sorted(rows) == ["a", "b"]
    assert rows["b"].text == "乙"
    assert chunk_store.fetch_chunks_by_vector_ids(db_session, []) == {}


def test_fetch_chunks_by_vector_ids_can_scope_to_user(db_session: Session) -> None:
    """传 user_id 时按 resources 校验归属：别人的 vector_id 即使传进来也取不到原文。"""
    _seed(db_session, user_id=1, vector_ids=["mine"], texts=["我的资料"])
    _seed(db_session, user_id=2, vector_ids=["others"], texts=["别人的资料"])

    assert sorted(chunk_store.fetch_chunks_by_vector_ids(db_session, ["mine", "others"], user_id=1)) == ["mine"]
    assert sorted(chunk_store.fetch_chunks_by_vector_ids(db_session, ["mine", "others"])) == ["mine", "others"]


def test_retriever_module_only_exports_search() -> None:
    """retriever 对外只暴露 search_similar_chunks。"""
    assert retriever.__all__ == ["search_similar_chunks"]
