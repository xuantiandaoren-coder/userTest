"""业务层：上传文件后的 RAG 向量化入库（best-effort，不阻断上传）。

链路：上传落库成功 -> ``ingest_file``（app/rag/core.py）解析 -> 清洗 -> 分类 -> 分块
-> DashScope 向量化 -> 写 Qdrant，payload 带 user_id / doc_category / file_name / chunk_index。

约定：
- 只有**文件类型**（document）且本次真的写了 ``resources`` 元数据的上传才入库；
  图片 / 音频、``storage_scene=2``（只提取内容不落库）、去重命中的重复上传都不入库
- 向量化属于增强能力，任何失败（没配 Key、类型不支持、Qdrant 不可用等）都只记日志，
  由 ``RagIngestResult.error`` 回给调用方，上传接口本身仍然成功
- 开关：``settings.rag_ingest_enabled``（``RAG_INGEST_ENABLED``，默认开）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagIngestResult:
    """一次 RAG 入库的结果：``ingested=False`` 时 ``error`` 给出跳过 / 失败原因。"""

    ingested: bool
    doc_category: str | None = None
    chunk_count: int = 0
    error: str | None = None


class RagIngestService:
    """把文件字节内容向量化写入向量库；不抛异常，失败通过返回值表达。"""

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = settings.rag_ingest_enabled if enabled is None else enabled

    async def ingest(
        self,
        *,
        data: bytes,
        file_name: str,
        user_id: int,
        doc_category: str | None = None,
    ) -> RagIngestResult:
        """向量化入库；``doc_category``（resume/study_material/general）优先于关键字自动分类。"""
        if not self.enabled:
            return RagIngestResult(ingested=False, error="RAG_DISABLED")

        try:
            # 延迟导入：rag 依赖较重（文档解析 / 向量化 / 向量库），缺失时只影响入库，不影响上传
            from app.rag import ingest_file

            summary = await run_in_threadpool(
                ingest_file,
                data,
                file_name,
                user_id=user_id,
                doc_category=doc_category,
            )
        except ImportError as exc:  # rag 依赖未安装：跳过入库
            logger.warning("RAG 依赖不可用，跳过入库 file=%s error=%r", file_name, exc)
            return RagIngestResult(ingested=False, error="RAG_UNAVAILABLE")
        except AppError as exc:  # 未配置 Key / 类型不支持 / 向量库写入失败等
            logger.warning("RAG 入库跳过 file=%s code=%s detail=%s", file_name, exc.code, exc.detail)
            return RagIngestResult(ingested=False, error=exc.code)
        except Exception as exc:  # noqa: BLE001 - 任何异常都不能影响上传结果
            logger.exception("RAG 入库异常 file=%s error=%r", file_name, exc)
            return RagIngestResult(ingested=False, error="RAG_INGEST_FAILED")

        category = summary.get("doc_category")
        return RagIngestResult(
            ingested=True,
            doc_category=str(category) if category else None,
            chunk_count=int(summary.get("chunk_count") or 0),
        )
