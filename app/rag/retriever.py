"""知识库检索：用户提问 -> 问题向量化 -> Qdrant 近邻 -> 回 MySQL 取原文。

对外只暴露 ``search_similar_chunks``，内部三步：

1. 用与入库**同一个模型 / 维度**把问题向量化（``core.embed_texts``）
2. 在 Qdrant 里按余弦相似度取最近的 ``top_k`` 个点，**必须带 user_id 过滤**：
   所有用户共用 ``knowledge_chunks`` 这一个 collection，不过滤会把别人的资料召回
3. 拿命中的 point id（== ``knowledge_chunks.vector_id``）回 MySQL 批量取原文
   （``chunk_store.fetch_chunks_by_vector_ids``），按相似度从高到低组装结果

入库时 Qdrant 只存向量与过滤字段（user_id / doc_category / file_name / chunk_index），
原文在 MySQL，所以第 3 步不可省：只拿向量库结果是没有文本可回答的。

结果整形（都在本模块内，不改动库里数据）：
- **分数阈值**：低于 ``MIN_SCORE``（0.4）的命中直接丢弃，避免把不相关片段塞进上下文
- **单条截断**：单条原文超过 ``MAX_TEXT_CHARS``（500 字）时截断，控制拼进提示词的体积
- **top_k**：默认 ``DEFAULT_TOP_K``（3），由调用方按需传入
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import SystemError
from app.rag.chunk_store import fetch_chunks_by_vector_ids
from app.rag.core import COLLECTION_NAME, embed_texts

logger = logging.getLogger("app.rag.retriever")

__all__ = ["search_similar_chunks"]

# 默认召回条数：够拼一段上下文，又不至于把整篇文档塞给模型
DEFAULT_TOP_K = 3

# 相似度阈值：低于该值的命中说明和问题不相关（0.4 以下在余弦空间基本是噪声）
MIN_SCORE = 0.4

# 单条原文截断长度（字符）：一条 500 字，3 条也就 1500 字，提示词体积可控
MAX_TEXT_CHARS = 500


class RetrievalError(SystemError):
    """系统异常：向量检索失败（Qdrant 不可用等）。"""

    code = "RETRIEVAL_FAILED"
    message = "知识库检索失败，请稍后重试"


@dataclass(frozen=True)
class ChunkHit:
    """一条命中的知识片段：原文 + 出处 + 相似度。"""

    vector_id: str          # Qdrant point id，等于 knowledge_chunks.vector_id
    resource_id: int | None  # 所属资源 id（knowledge_chunks.resource_id，回查文件名等出处信息）
    text: str               # 分块原文（来自 MySQL）
    score: float            # 余弦相似度，越大越相关
    file_name: str | None   # 出自哪个上传文件
    doc_category: str | None  # 文档分类（resume / study_material / general）
    chunk_index: int | None   # 文件内的分块序号

    def as_dict(self) -> dict[str, object]:
        """转成可 JSON 序列化的字典（接口直接返回时用）。"""
        return {
            "vector_id": self.vector_id,
            "resource_id": self.resource_id,
            "text": self.text,
            "score": self.score,
            "file_name": self.file_name,
            "doc_category": self.doc_category,
            "chunk_index": self.chunk_index,
        }


def search_similar_chunks(
    query: str,
    *,
    user_id: int,
    db: Session,
    top_k: int = DEFAULT_TOP_K,
) -> list[ChunkHit]:
    """检索当前用户知识库里与 ``query`` 最相关的 ``top_k`` 个片段。

    入参：
    - ``query``：用户提问；空串直接返回空列表（不调模型、不查库）
    - ``user_id``：**必填**，检索只看这个用户上传的资料
    - ``db``：取分块原文用的会话（只读查询，不提交事务）
    - ``top_k``：召回条数（默认 ``DEFAULT_TOP_K`` = 3），<= 0 返回空列表

    返回前会丢掉相似度低于 ``MIN_SCORE`` 的命中，并把超过 ``MAX_TEXT_CHARS`` 的
    单条原文截断（只影响返回值，库里原文不动）。

    用法::

        hits = search_similar_chunks("二叉树的前序遍历怎么写", user_id=7, db=session, top_k=3)
        context = "\\n\\n".join(hit.text for hit in hits)
    """
    question = query.strip()
    limit = min(max(top_k, 0), 100)
    if not question or limit == 0:
        return []

    vector = embed_texts([question])[0]
    points = _query_points(vector, user_id=user_id, limit=limit)
    if not points:
        return []

    # 命中顺序就是相似度顺序，回表取原文后仍按这个顺序组装
    # 回表带 user_id：Qdrant 过滤之外再做一层归属校验
    rows = fetch_chunks_by_vector_ids(db, [str(point.id) for point in points], user_id=user_id)
    hits: list[ChunkHit] = []
    for point in points:
        score = float(point.score)
        if score < MIN_SCORE:  # Qdrant 侧已按阈值过滤，这里再兜一次，不依赖服务端版本
            logger.debug("命中低于阈值已丢弃 vector_id=%s score=%s", point.id, score)
            continue
        row = rows.get(str(point.id))
        if row is None:  # 悬空点：向量还在、原文已随资源清理删除
            logger.warning("向量库存在无原文的悬空点 vector_id=%s，已跳过", point.id)
            continue
        payload = point.payload or {}
        hits.append(
            ChunkHit(
                vector_id=row.vector_id,
                resource_id=row.resource_id,
                text=_truncate(row.text),
                score=score,
                file_name=payload.get("file_name"),
                doc_category=payload.get("doc_category"),
                chunk_index=payload.get("chunk_index"),
            )
        )
    return hits


def _query_points(vector: list[float], *, user_id: int, limit: int) -> list[object]:
    """在 Qdrant 里做一次近邻查询，返回按分值降序的命中点（拿不到就返回空列表）。"""
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    try:
        collections = {collection.name for collection in client.get_collections().collections}
        if COLLECTION_NAME not in collections:
            logger.info("向量库还没有 collection=%s，检索直接返回空", COLLECTION_NAME)
            return []

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=_user_filter(user_id),
            limit=limit,
            score_threshold=MIN_SCORE,  # 服务端先滤掉低分点，少传无用的 payload
            with_payload=True,   # 出处信息（file_name / doc_category / chunk_index）在这里
            with_vectors=False,  # 检索不需要回传向量，省带宽
        )
    except Exception as exc:  # noqa: BLE001 - 统一转系统异常，细节写日志
        raise RetrievalError(detail=f"Qdrant 检索失败：{exc}") from exc
    return list(response.points)


def _user_filter(user_id: int) -> Filter:
    """检索过滤条件：多用户共用一个 collection，必须按 user_id 隔离。"""
    return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """单条原文截断（超长才截，不补省略号，保持原文可引用）。"""
    return text if len(text) <= limit else text[:limit]
