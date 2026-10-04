"""知识库分块原文存储：向量在 Qdrant，原文在 MySQL 的 knowledge_chunks 表。

对外暴露两个函数：

``save_chunks``（写）：

1. **覆盖语义**：写入前先删掉同一（user_id, file_name）的旧分块。旧分块通过
   resources 表定位（file_name 存在 resources 上），与 Qdrant 侧按同一条件删点保持一致
2. **UUID 关联**：``vector_ids`` 与 Qdrant point id 是同一批值，落库进 ``vector_id`` 列，
   检索时先用向量召回 point id，再按 vector_id 回本表取原文
3. **只 flush 不 commit**：事务边界交给调用方（``RagIngestService``），
   写库包在 savepoint 里，失败只回滚分块写入，不牵连调用方同一事务里的其它改动

``fetch_chunks_by_vector_ids``（读）：按 vector_id 批量取回分块原文，供检索侧组装回答。

配置统一走 ``app.core.config.settings``，本模块不直接读环境变量。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunk, Resource

logger = logging.getLogger("app.rag.chunk_store")

__all__ = ["save_chunks", "fetch_chunks_by_vector_ids", "fetch_chunks_with_resources"]


def save_chunks(
    db: Session,
    *,
    resource_id: int,
    user_id: int,
    file_name: str,
    texts: list[str],
    vector_ids: list[str],
) -> int:
    """把一个文件的分块原文写进 knowledge_chunks，返回写入条数。

    入参：
    - ``db``：调用方的会话（与写 resources 同一个事务）
    - ``resource_id``：本次上传的文件在 resources 里的主键
    - ``user_id`` / ``file_name``：覆盖语义的定位键，用于删除旧分块
    - ``texts``：分块原文，顺序与 ``vector_ids`` 一一对应
    - ``vector_ids``：Qdrant point id（入库时生成的 UUID）

    用法::

        save_chunks(db, resource_id=12, user_id=1, file_name="resume.pdf",
                    texts=chunks, vector_ids=point_ids)
    """
    if not texts:
        return 0
    if len(texts) != len(vector_ids):
        raise ValueError(f"分块数与向量数不一致：texts={len(texts)} vector_ids={len(vector_ids)}")

    # 旧分块通过 resources 定位：同一个用户重复上传同名文件属于覆盖
    old_resource_ids = select(Resource.id).where(
        Resource.user_id == user_id,
        Resource.file_name == file_name,
    )
    with db.begin_nested():  # 失败只回滚分块写入，不影响调用方事务里的 resources 等改动
        db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.resource_id.in_(old_resource_ids)))
        db.add_all(
            [
                KnowledgeChunk(
                    resource_id=resource_id,
                    vector_id=vector_id,
                    chunk_index=chunk_index,
                    char_count=len(text),
                    text=text,
                )
                for chunk_index, (text, vector_id) in enumerate(zip(texts, vector_ids, strict=True))
            ]
        )
        db.flush()

    return len(texts)


def fetch_chunks_by_vector_ids(
    db: Session,
    vector_ids: Sequence[str],
    *,
    user_id: int | None = None,
) -> dict[str, KnowledgeChunk]:
    """按 vector_id 批量取分块原文，返回 ``{vector_id: 分块行}``。

    检索链路的第二步：Qdrant 只回 point id（== vector_id），原文在这里取。
    返回字典而不是列表，因为**顺序由检索侧的分值决定**，调用方按命中顺序自己排；
    字典里缺失的 vector_id 说明向量库有悬空点（原文已随资源清理删除）。

    ``user_id`` 传了就走 resources 关联校验归属（多用户共用一套表和向量库，
    检索侧建议传，作为 Qdrant 过滤之外的第二道隔离，避免返回别人的资料）。

    用法::

        rows = fetch_chunks_by_vector_ids(db, [hit.vector_id for hit in hits], user_id=user_id)
        text = rows[hit.vector_id].text
    """
    if not vector_ids:
        return {}
    stmt = select(KnowledgeChunk).where(KnowledgeChunk.vector_id.in_(list(vector_ids)))
    if user_id is not None:
        stmt = stmt.join(Resource, Resource.id == KnowledgeChunk.resource_id).where(Resource.user_id == user_id)
    return {row.vector_id: row for row in db.scalars(stmt).all()}


def fetch_chunks_with_resources(
    db: Session,
    vector_ids: Sequence[str],
    *,
    user_id: int | None = None,
) -> dict[str, tuple[KnowledgeChunk, Resource]]:
    """按 vector_id 取「分块原文 + 所属资源」，返回 ``{vector_id: (分块行, 资源行)}``。

    历史消息回显用：``reference_sources`` 里只有 chunk_id 与 score，
    出处（resource_id / file_name / chunk_index）与原文都要回这两张表补。
    ``user_id`` 传了同样会校验归属，避免拿别人的资源行拼来源。
    """
    if not vector_ids:
        return {}
    stmt = (
        select(KnowledgeChunk, Resource)
        .join(Resource, Resource.id == KnowledgeChunk.resource_id)
        .where(KnowledgeChunk.vector_id.in_(list(vector_ids)))
    )
    if user_id is not None:
        stmt = stmt.where(Resource.user_id == user_id)
    return {chunk.vector_id: (chunk, resource) for chunk, resource in db.execute(stmt).all()}
