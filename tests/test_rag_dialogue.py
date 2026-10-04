"""RAG 对话链路：意图判断、参考资料注入、来源组装与历史回显。

用例不连模型 / 向量库：检索用假实现替换，回显走内存 SQLite。
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunk, Resource, User
from app.rag import dialogue
from app.rag.retriever import ChunkHit


def _hit(
    vector_id: str = "chunk-1",
    *,
    score: float = 0.82,
    text: str = "HashMap 默认容量 16，扩容因子 0.75。",
    resource_id: int | None = 11,
    file_name: str | None = "java.pdf",
    chunk_index: int | None = 0,
) -> ChunkHit:
    return ChunkHit(
        vector_id=vector_id,
        resource_id=resource_id,
        text=text,
        score=score,
        file_name=file_name,
        doc_category="study_material",
        chunk_index=chunk_index,
    )


class _QdrantStub:
    """假 Qdrant：只记录被调用过几次。"""


# ---------------------------------------------------------------------------
# 1. 意图判断
# ---------------------------------------------------------------------------
def test_should_retrieve_skips_empty_and_blank() -> None:
    """空输入 / 纯空白不检索。"""
    assert dialogue.should_retrieve("") is False
    assert dialogue.should_retrieve("   \n\t ") is False
    assert dialogue.should_retrieve(None) is False  # type: ignore[arg-type]


def test_should_retrieve_skips_too_short_input() -> None:
    """极短输入不检索（去掉空白标点后短于阈值）。"""
    assert dialogue.should_retrieve("嗯") is False
    assert dialogue.should_retrieve("？") is False
    assert dialogue.MIN_QUERY_CHARS == 4


def test_should_retrieve_skips_confirm_polite_and_continue_phrases() -> None:
    """确认语 / 礼貌语 / 推进语都不检索，包括它们的短组合。"""
    for text in ("好的", "嗯嗯", "收到", "没错", "ok", "你好", "谢谢", "辛苦了", "继续", "下一题", "接着"):
        assert dialogue.should_retrieve(text) is False, text
    assert dialogue.should_retrieve("好的，继续！") is False
    assert dialogue.should_retrieve("谢谢，继续") is False
    assert dialogue.should_retrieve("你好，在吗") is False


def test_should_retrieve_keeps_real_questions() -> None:
    """有信息量的问题要检索（含礼貌语开头的长句）。"""
    assert dialogue.should_retrieve("什么是红黑树") is True
    assert dialogue.should_retrieve("HashMap 的扩容因子是多少？") is True
    assert dialogue.should_retrieve("谢谢，那红黑树的时间复杂度呢") is True  # 长句里的礼貌语不影响


def test_skip_phrase_matching_is_deterministic() -> None:
    """短语表来自集合，匹配必须与迭代顺序无关（固定点剥离）。"""
    for _ in range(20):
        assert dialogue.should_retrieve("好的继续") is False
        assert dialogue.should_retrieve("麻烦你了，接着") is False


# ---------------------------------------------------------------------------
# 2. 检索与注入
# ---------------------------------------------------------------------------
def test_retrieve_knowledge_skips_search_for_skip_phrases(db_session: Session, monkeypatch) -> None:
    """意图判断不过时连向量检索都不发起。"""

    def _boom(*args: object, **kwargs: object) -> list[ChunkHit]:
        raise AssertionError("不该发起检索")

    monkeypatch.setattr(dialogue, "search_similar_chunks", _boom)

    context = dialogue.retrieve_knowledge("好的", user_id=1, db=db_session)

    assert context.retrieved is False
    assert context.retrieved_text == ""
    assert context.sources == []


def test_retrieve_knowledge_passes_user_and_top_k(db_session: Session, monkeypatch) -> None:
    """检索按 user_id 隔离，top_k 默认 3。"""
    seen: dict[str, object] = {}

    def _fake_search(query: str, *, user_id: int, db: Session, top_k: int) -> list[ChunkHit]:
        seen.update({"query": query, "user_id": user_id, "top_k": top_k, "db": db})
        return [_hit()]

    monkeypatch.setattr(dialogue, "search_similar_chunks", _fake_search)

    context = dialogue.retrieve_knowledge("  什么是红黑树  ", user_id=42, db=db_session)

    assert seen == {"query": "什么是红黑树", "user_id": 42, "top_k": dialogue.RAG_TOP_K, "db": db_session}
    assert context.retrieved is True
    assert dialogue.RAG_TOP_K == 3
    assert len(context.sources) == 1


def test_retrieve_knowledge_degrades_on_failure(db_session: Session, monkeypatch) -> None:
    """检索失败不让对话挂掉：返回空上下文。"""

    def _boom(*args: object, **kwargs: object) -> list[ChunkHit]:
        raise RuntimeError("qdrant 挂了")

    monkeypatch.setattr(dialogue, "search_similar_chunks", _boom)

    context = dialogue.retrieve_knowledge("什么是红黑树", user_id=1, db=db_session)

    assert context.retrieved is False
    assert context.has_context is False


def test_format_retrieved_chunks_lists_origin_and_text() -> None:
    """参考资料文本：编号 + 出处 + 原文。"""
    text = dialogue.format_retrieved_chunks([_hit(), _hit("chunk-2", score=0.5, text="第二块", chunk_index=1)])

    assert text.startswith(dialogue.RAG_CONTEXT_HEADING)
    assert "[1] 来源：java.pdf（第 1 块，相似度 0.82）" in text
    assert "[2] 来源：java.pdf（第 2 块，相似度 0.50）" in text
    assert "第二块" in text
    assert dialogue.format_retrieved_chunks([]) == ""


def test_compose_question_puts_context_first() -> None:
    """参考资料拼在本轮问题前，空资料时原样返回问题。"""
    composed = dialogue.compose_question("【参考资料】\n[1] 内容", "什么是红黑树")

    assert composed.endswith("【用户问题】\n什么是红黑树")
    assert composed.startswith("【参考资料】")
    assert dialogue.compose_question("", "什么是红黑树") == "什么是红黑树"


def test_build_rag_system_prompt_only_appends_rules() -> None:
    """只在原 system 后补 RAG 规则，不改写模板正文。"""
    merged = dialogue.build_rag_system_prompt("你是学习助手，按流程引导。")

    assert merged.startswith("你是学习助手，按流程引导。")
    assert dialogue.RAG_RULES in merged
    assert dialogue.build_rag_system_prompt("") == dialogue.RAG_RULES


# ---------------------------------------------------------------------------
# 3. 来源与落库引用
# ---------------------------------------------------------------------------
def test_build_sources_has_full_shape() -> None:
    """sources 字段齐全：chunk_id / resource_id / chunk_index / file_name / score / text。"""
    sources = dialogue.build_sources([_hit()])

    assert sources == [
        {
            "chunk_id": "chunk-1",
            "resource_id": 11,
            "chunk_index": 0,
            "file_name": "java.pdf",
            "score": 0.82,
            "text": "HashMap 默认容量 16，扩容因子 0.75。",
        }
    ]


def test_build_references_stores_only_id_and_score() -> None:
    """落库引用只有 chunk_id + score：不在消息表里重复保存知识块全文。"""
    references = dialogue.build_references([_hit(), _hit("chunk-2", score=0.6666)])

    assert references == [{"chunk_id": "chunk-1", "score": 0.82}, {"chunk_id": "chunk-2", "score": 0.6666}]
    assert all(set(reference) == {"chunk_id", "score"} for reference in references)


def _seed_chunk(db: Session, *, user_id: int, vector_id: str, text: str = "原文内容", file_name: str = "java.pdf") -> None:
    """造一个用户 + 资源 + 一条分块；同一 (user_id, file_name) 复用已有资源行。"""
    if db.get(User, user_id) is None:
        db.add(User(id=user_id, user_name=f"u{user_id}", password="x" * 60))
        db.flush()
    resource = db.scalars(
        select(Resource).where(Resource.user_id == user_id, Resource.file_name == file_name)
    ).first()
    if resource is None:
        digest = hashlib.md5(f"{user_id}:{file_name}".encode(), usedforsecurity=False).hexdigest()
        resource = Resource(
            resource_type=0,
            storage_scene=0,
            upload_purpose=0,
            file_name=file_name,
            file_hash=digest,
            storage_path=f"{user_id}/{digest}.txt",
            user_id=user_id,
        )
        db.add(resource)
        db.flush()
    db.add(KnowledgeChunk(resource_id=resource.id, vector_id=vector_id, chunk_index=0, char_count=len(text), text=text))
    db.flush()


def test_resolve_reference_sources_rebuilds_from_db(db_session: Session) -> None:
    """历史回显：按 chunk_id 回查原文与出处。"""
    _seed_chunk(db_session, user_id=5, vector_id="chunk-1", text="回查到的原文", file_name="note.txt")

    sources = dialogue.resolve_reference_sources(
        db_session, [{"chunk_id": "chunk-1", "score": 0.71}], user_id=5
    )

    assert sources == [
        {
            "chunk_id": "chunk-1",
            "resource_id": sources[0]["resource_id"],
            "chunk_index": 0,
            "file_name": "note.txt",
            "score": 0.71,
            "text": "回查到的原文",
        }
    ]
    assert sources[0]["resource_id"] is not None


def test_resolve_reference_sources_marks_deleted_chunk(db_session: Session) -> None:
    """片段已被删除时返回占位文本，出处字段留空。"""
    sources = dialogue.resolve_reference_sources(db_session, [{"chunk_id": "gone", "score": 0.6}], user_id=5)

    assert sources == [
        {
            "chunk_id": "gone",
            "resource_id": None,
            "chunk_index": None,
            "file_name": None,
            "score": 0.6,
            "text": dialogue.DELETED_CHUNK_TEXT,
        }
    ]
    assert dialogue.DELETED_CHUNK_TEXT == "该参考片段已经删除"


def test_resolve_reference_sources_skips_dirty_rows(db_session: Session) -> None:
    """脏引用（缺 chunk_id / 非字典）直接跳过，不让历史接口报错。"""
    _seed_chunk(db_session, user_id=5, vector_id="ok", text="正常原文")

    sources = dialogue.resolve_reference_sources(
        db_session,
        [{"score": 0.5}, "not-a-dict", {"chunk_id": ""}, {"chunk_id": "ok", "score": 0.9}],  # type: ignore[list-item]
        user_id=5,
    )

    assert [source["chunk_id"] for source in sources] == ["ok"]
    assert dialogue.resolve_reference_sources(db_session, None) == []


def test_resolve_reference_sources_respects_user_scope(db_session: Session) -> None:
    """别人的 chunk_id 即使出现在引用里也回查不到（按 user_id 校验归属）。"""
    _seed_chunk(db_session, user_id=5, vector_id="mine", text="我的原文")
    _seed_chunk(db_session, user_id=6, vector_id="others", text="别人的原文")

    sources = dialogue.resolve_reference_sources(
        db_session,
        [{"chunk_id": "mine", "score": 0.8}, {"chunk_id": "others", "score": 0.7}],
        user_id=5,
    )

    assert [source["text"] for source in sources] == ["我的原文", dialogue.DELETED_CHUNK_TEXT]


def test_load_source_index_batches_lookup(db_session: Session) -> None:
    """列表接口按页复用一次查询：index 里命中的才带出处。"""
    _seed_chunk(db_session, user_id=5, vector_id="a", text="甲")
    _seed_chunk(db_session, user_id=5, vector_id="b", text="乙")

    index = dialogue.load_source_index(db_session, ["a", "b", "missing"], user_id=5)

    assert sorted(index) == ["a", "b"]
    assert index["a"]["text"] == "甲"
