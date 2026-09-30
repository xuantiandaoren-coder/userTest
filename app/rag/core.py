"""RAG 文件向量化入库：解析 -> 清洗 -> 分类 -> 分块 -> 向量化 -> 写 Qdrant。

对外只暴露 ``ingest_file``，其余函数均为模块内私有实现（单文件聚合，不过度抽象）。

配置统一走 ``app.core.config.settings``：
- ``dashscope_api_key``：DashScope 文本向量化 Key
- ``qdrant_host`` / ``qdrant_port``：Qdrant 服务地址
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import uuid
from collections.abc import Callable
from pathlib import Path

import PyPDF2
from dashscope import TextEmbedding
from docx import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.core.exceptions import SystemError
from app.core.storage import UnsupportedFileTypeError

logger = logging.getLogger("app.rag")

__all__ = ["ingest_file"]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
COLLECTION_NAME = "knowledge_chunks"
EMBEDDING_MODEL = "text_embedding-v4"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 10

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CATEGORY_RESUME = "resume"
CATEGORY_STUDY = "study_material"
CATEGORY_GENERAL = "general"
VALID_CATEGORIES = frozenset({CATEGORY_RESUME, CATEGORY_STUDY, CATEGORY_GENERAL})

# 分块回退切分点（在窗口内优先按这些字符断开，避免切开句子）
SENTENCE_TERMINATORS = "。！？\n"

# 分类关键字：命中即计分（大小写不敏感）
_RESUME_KEYWORDS = (
    "简历", "个人简历", "求职", "应聘", "教育背景", "工作经历", "项目经验", "项目经历",
    "实习经历", "技能特长", "专业技能", "自我评价", "联系方式", "毕业院校", "学历",
    "岗位职责", "获奖情况", "任职", "求职意向",
)
_STUDY_KEYWORDS = (
    "知识点", "学习目标", "课程", "教材", "章节", "习题", "例题", "考点", "复习",
    "大纲", "笔记", "理论", "实验", "参考资料", "教学内容", "课后", "概念", "定义",
    "作业", "解析",
)

# 清洗用正则
_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"
_ZERO_WIDTH_TABLE = {ord(char): None for char in _ZERO_WIDTH_CHARS}
_HORIZONTAL_WS = re.compile(r"[ \t\u3000\xa0]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 独立页码行：如 “12”“- 12 -”“12 / 100”“第 12 页”
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:\d+|[-—–.·]+\s*\d+\s*[-—–.·]*|\d+\s*/\s*\d+|第\s*\d+\s*页)\s*$"
)


class RagNotConfiguredError(SystemError):
    """系统异常：RAG 依赖未配置（缺 DashScope Key 等）。"""

    code = "RAG_NOT_CONFIGURED"
    message = "向量化服务未配置，请联系管理员"


class EmbeddingError(SystemError):
    """系统异常：DashScope 向量化调用失败。"""

    code = "EMBEDDING_FAILED"
    message = "文本向量化失败，请稍后重试"


class VectorStoreError(SystemError):
    """系统异常：Qdrant 写入失败。"""

    code = "VECTOR_STORE_FAILED"
    message = "向量库写入失败，请稍后重试"


# ---------------------------------------------------------------------------
# 解析：按扩展名字典分派
# ---------------------------------------------------------------------------
def _parse_pdf(data: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(data: bytes) -> str:
    document = Document(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _parse_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


_PARSERS: dict[str, Callable[[bytes], str]] = {
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "txt": _parse_txt,
}


def _parse(file_name: str, data: bytes) -> str:
    """按扩展名分派解析器；不支持的类型抛 UnsupportedFileTypeError。"""
    extension = Path(file_name).suffix.lstrip(".").lower()
    parser = _PARSERS.get(extension)
    if parser is None:
        raise UnsupportedFileTypeError(
            detail=f"RAG 不支持的文件类型：file={file_name!r} extension={extension or 'unknown'}",
        )
    return parser(data)


# ---------------------------------------------------------------------------
# 清洗：去零宽字符、合并空白、压缩多换行、删独立页码行
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = text.translate(_ZERO_WIDTH_TABLE)

    lines: list[str] = []
    for raw in text.split("\n"):
        line = _HORIZONTAL_WS.sub(" ", raw).strip()
        if _PAGE_NUMBER_LINE.match(line):
            continue  # 丢弃独立成行的页码
        lines.append(line)

    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)  # 连续空行压成一行空行
    return text.strip()


# ---------------------------------------------------------------------------
# 分类：关键字计分，显示值（显式传入的分类）优先
# ---------------------------------------------------------------------------
def _classify(text: str, doc_category: str | None) -> str:
    if doc_category:
        normalized = doc_category.strip().lower()
        if normalized in VALID_CATEGORIES:
            return normalized

    lowered = text.lower()
    scores = {
        CATEGORY_RESUME: sum(lowered.count(keyword) for keyword in _RESUME_KEYWORDS),
        CATEGORY_STUDY: sum(lowered.count(keyword) for keyword in _STUDY_KEYWORDS),
    }
    best = max(scores, key=lambda category: scores[category])
    return best if scores[best] > 0 else CATEGORY_GENERAL


# ---------------------------------------------------------------------------
# 分块：滑动窗口 size=500 overlap=50，回退句子终止切分，MD5 去重
# ---------------------------------------------------------------------------
def _last_terminator(text: str, start: int, end: int) -> int:
    """在 [start, end) 内找最后一个句子终止符，返回断点（终止符之后）；找不到返回 end。"""
    for position in range(end - 1, start, -1):
        if text[position] in SENTENCE_TERMINATORS:
            return position + 1
    return end


def _chunk(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    seen: set[str] = set()  # MD5 去重
    length = len(text)
    start = 0

    while start < length:
        end = min(start + CHUNK_SIZE, length)
        if end < length:
            # 窗口尾部回退到最近的句子终止符，避免把句子切断
            end = _last_terminator(text, start, end)

        piece = text[start:end].strip()
        if piece:
            digest = hashlib.md5(piece.encode("utf-8"), usedforsecurity=False).hexdigest()
            if digest not in seen:
                seen.add(digest)
                chunks.append(piece)

        if end >= length:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)  # 保证前进，不会死循环

    return chunks


# ---------------------------------------------------------------------------
# 向量化：DashScope text_embedding-v4，维度 1024，每批 10 条，按 text_index 对齐
# ---------------------------------------------------------------------------
def _embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    api_key = settings.dashscope_api_key.get_secret_value().strip()
    if not api_key:
        raise RagNotConfiguredError(
            detail="缺少 dashscope_api_key，请在 .env / 环境变量中配置 DASHSCOPE_API_KEY",
        )

    vectors: list[list[float]] = []
    for offset in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[offset : offset + EMBEDDING_BATCH_SIZE]
        response = TextEmbedding.call(
            model=EMBEDDING_MODEL,
            input=batch,
            dimension=EMBEDDING_DIM,
            api_key=api_key,
        )
        if getattr(response, "status_code", None) != 200:
            raise EmbeddingError(
                detail=f"DashScope 返回异常：code={getattr(response, 'code', None)} message={getattr(response, 'message', None)}",
            )

        embeddings = response.output["embeddings"]
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                detail=f"向量条数与输入不一致：expected={len(batch)} got={len(embeddings)}",
            )
        # 按 text_index 对齐，保证与 batch 顺序一一对应
        for item in sorted(embeddings, key=lambda entry: entry["text_index"]):
            vectors.append(item["embedding"])

    return vectors


# ---------------------------------------------------------------------------
# 写 Qdrant：collection 不存在自动建，余弦距离
# ---------------------------------------------------------------------------
def _ensure_collection(client: QdrantClient) -> None:
    existing = {collection.name for collection in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("创建 Qdrant collection=%s dim=%s", COLLECTION_NAME, EMBEDDING_DIM)


def _write_qdrant(
    *,
    texts: list[str],
    vectors: list[list[float]],
    user_id: int,
    doc_category: str,
    file_name: str,
) -> list[str]:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_collection(client)

    point_ids: list[str] = []
    points: list[PointStruct] = []
    for chunk_index, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "user_id": user_id,
                    "doc_category": doc_category,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                },
            )
        )

    try:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as exc:  # noqa: BLE001 - 统一转系统异常，细节写日志
        raise VectorStoreError(detail=f"Qdrant upsert 失败：{exc}") from exc

    return point_ids


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def _resolve_source(source: str | Path | bytes, file_name: str | None) -> tuple[str, bytes]:
    if isinstance(source, (bytes, bytearray)):
        if not file_name:
            raise ValueError("以字节内容调用时必须提供 file_name")
        return file_name, bytes(source)
    path = Path(source)
    return file_name or path.name, path.read_bytes()


def ingest_file(
    source: str | Path | bytes,
    file_name: str | None = None,
    *,
    user_id: int,
    doc_category: str | None = None,
) -> dict[str, object]:
    """把一个文件向量化并写入 Qdrant，返回入库摘要。

    入参：
    - ``source``：文件路径（str / Path），或文件字节内容（bytes）
    - ``file_name``：展示名 / 类型识别用；传路径时可省略（默认取路径文件名），传 bytes 时必填
    - ``user_id``：归属用户
    - ``doc_category``：显式分类（resume / study_material / general），传入则优先于关键字计分

    用法::

        ingest_file("/data/a.pdf", user_id=1)
        ingest_file(b"...", "note.txt", user_id=1)

    返回：``{"file_name", "user_id", "doc_category", "chunk_count", "point_ids"}``
    """
    resolved_name, data = _resolve_source(source, file_name)

    text = _clean(_parse(resolved_name, data))
    category = _classify(text, doc_category)
    chunks = _chunk(text)

    point_ids: list[str] = []
    if chunks:
        vectors = _embed(chunks)
        point_ids = _write_qdrant(
            texts=chunks,
            vectors=vectors,
            user_id=user_id,
            doc_category=category,
            file_name=resolved_name,
        )

    logger.info(
        "RAG 入库完成 file=%s category=%s chunks=%s points=%s",
        resolved_name,
        category,
        len(chunks),
        len(point_ids),
    )
    return {
        "file_name": resolved_name,
        "user_id": user_id,
        "doc_category": category,
        "chunk_count": len(chunks),
        "point_ids": point_ids,
    }
