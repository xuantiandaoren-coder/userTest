"""RAG 文件向量化入库：解析 -> 清洗 -> 分类 -> 分块 -> 向量化 -> 写 Qdrant。

对外只暴露 ``ingest_file``，其余函数均为模块内私有实现（单文件聚合，不过度抽象）。
PDF 解析用 pypdfium2 抽文本层，抽不出内容（平均每页字符数过低，典型是扫描件）时
转 ``app.rag.ocr.ocr_pdf`` 走 OCR 兜底；OCR 实现单独放在 app/rag/ocr.py。

配置统一走 ``app.core.config.settings``：
- ``dashscope_api_key``：DashScope 文本向量化 Key
- ``embedding_model`` / ``embedding_dim`` / ``embedding_batch_size``：向量化模型与批大小
- ``ocr_enabled`` / ``pdf_scanned_min_chars_per_page``：扫描件判定与 OCR 兜底开关
- ``qdrant_host`` / ``qdrant_port``：Qdrant 服务地址

分块用 langchain 的 ``RecursiveCharacterTextSplitter`` 递归切片：优先在 Markdown 标题
（DOCX 的 Heading 1/2/3 会注入 ``#`` / ``##`` / ``###``）断开，其次按段落、换行、
中文句读、空格，最后退到字符级；切片结果仍按 MD5 去重。

分块数据分开存：Qdrant 只存**向量 + 检索过滤字段**（user_id / doc_category / file_name /
chunk_index），原文写 MySQL 的 knowledge_chunks（app/rag/chunk_store.py），
两边用入库时生成的同一个 UUID 关联（Qdrant point id == vector_id）。
写库是**覆盖**语义：先按（user_id, file_name）删掉旧分块再写新分块，
同一个用户重复上传同名文件（内容变了）不会出现新旧点并存、检索到重复片段的情况。
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import uuid
from collections.abc import Callable
from pathlib import Path

import pypdfium2 as pdfium
from dashscope import TextEmbedding
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import SystemError
from app.core.storage import UnsupportedFileTypeError
from app.rag.chunk_store import save_chunks
from app.rag.ocr import ocr_pdf

logger = logging.getLogger("app.rag")

__all__ = ["ingest_file", "embed_texts"]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
COLLECTION_NAME = "knowledge_chunks"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CATEGORY_RESUME = "resume"
CATEGORY_STUDY = "study_material"
CATEGORY_GENERAL = "general"
VALID_CATEGORIES = frozenset({CATEGORY_RESUME, CATEGORY_STUDY, CATEGORY_GENERAL})

# Word 标题样式 -> Markdown 标题标记（中文版 Word 的样式名是 "标题 1"，一并映射）
DOCX_HEADING_PREFIXES = {
    "Heading 1": "#",
    "Heading 2": "##",
    "Heading 3": "###",
    "标题 1": "#",
    "标题 2": "##",
    "标题 3": "###",
}

# 递归切片的分隔符优先级（从粗到细）：标题 -> 段落 -> 换行 -> 中文句读 -> 空格 -> 字符兜底
CHUNK_SEPARATORS = ["\n###", "\n##", "\n#", "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

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
    """系统异常：向量化调用失败。"""

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
    """PDF -> 文本；抽不出文本层（扫描件）时走 OCR 兜底。

    判定口径：pypdfium2 抽出全文后算“平均每页字符数”，低于
    ``settings.pdf_scanned_min_chars_per_page``（默认 20）即视为扫描件——
    这类 PDF 的文字只存在于图片里，文本层要么为空、要么只有零散水印字符。
    """
    text, page_count = _extract_pdf_text(data)
    if not page_count or not settings.ocr_enabled:
        return text
    if len(text) / page_count >= settings.pdf_scanned_min_chars_per_page:
        return text

    logger.info(
        "PDF 判定为扫描件 pages=%s text_chars=%s，转 OCR 兜底",
        page_count,
        len(text),
    )
    return ocr_pdf(data)


def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    """pypdfium2 逐页抽文本层，返回（全文, 页数）。"""
    pages: list[str] = []
    with pdfium.PdfDocument(data) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    pages.append(textpage.get_text_range())
                finally:
                    textpage.close()
            finally:
                page.close()
    return "\n".join(pages), len(pages)


def _parse_docx(data: bytes) -> str:
    """DOCX -> 文本；Heading 1/2/3 段落注入 # / ## / ### 标题标记。

    注入的标记会被切片按 ``\n###`` / ``\n##`` / ``\n#`` 优先识别，标题连同其正文
    更容易落在同一块里；非标题段落保持原样，空段落保留（段落边界对切片有用）。
    """
    document = Document(io.BytesIO(data))
    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            lines.append("")
            continue
        style_name = (getattr(paragraph.style, "name", "") or "").strip()
        prefix = DOCX_HEADING_PREFIXES.get(style_name, "")
        lines.append(f"{prefix} {text}" if prefix else text)
    return "\n".join(lines)


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
# 分块：递归切片 size=500 overlap=50（标题 -> 段落 -> 句读 -> 字符兜底），MD5 去重
# ---------------------------------------------------------------------------
def _chunk(text: str) -> list[str]:
    """递归智能切片：按分隔符优先级从粗到细找断点，尽量不切开标题与句子。"""
    text = text.strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        # 分隔符跟在下段开头：标点不丢，DOCX 注入的 "### 标题" 也不会和标题分家
        keep_separator=True,
    )

    chunks: list[str] = []
    seen: set[str] = set()  # MD5 去重
    for piece in splitter.split_text(text):
        piece = piece.strip()
        if not piece:
            continue
        digest = hashlib.md5(piece.encode("utf-8"), usedforsecurity=False).hexdigest()
        if digest not in seen:
            seen.add(digest)
            chunks.append(piece)

    return chunks


# ---------------------------------------------------------------------------
# 向量化：DashScope qwen3.7-text-embedding，维度 / 批大小走配置，按 text_index 对齐
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """把文本向量化（返回顺序与入参一致）；入库与检索共用这一条链路。

    检索侧只需要 ``embed_texts([query])[0]``，因此这里不做单条特判，
    维度、批量大小、异常类型（RAG_NOT_CONFIGURED / EMBEDDING_FAILED）两边保持一致。
    """
    if not texts:
        return []

    api_key = settings.dashscope_api_key.get_secret_value().strip()
    if not api_key:
        raise RagNotConfiguredError(
            detail="缺少 dashscope_api_key，请在 .env / 环境变量中配置 DASHSCOPE_API_KEY",
        )

    batch_size = max(1, settings.embedding_batch_size)
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        response = TextEmbedding.call(
            model=settings.embedding_model,
            input=batch,
            dimension=settings.embedding_dim,
            api_key=api_key,
        )
        if getattr(response, "status_code", None) != 200:
            raise EmbeddingError(
                detail=f"向量化服务返回异常：code={getattr(response, 'code', None)} message={getattr(response, 'message', None)}",
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
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        logger.info("创建 Qdrant collection=%s dim=%s", COLLECTION_NAME, settings.embedding_dim)


def _delete_existing_points(client: QdrantClient, *, user_id: int, file_name: str) -> int:
    """删除同一（user_id, file_name）的旧分块，返回删除条数。

    覆盖语义：同一个用户重复上传同名文件时，旧内容的分块必须先清掉，否则新旧点并存，
    检索会把过期片段和新片段一起召回（同一份资料的重复答案）。
    """
    selector = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="file_name", match=MatchValue(value=file_name)),
        ]
    )
    try:
        existing = client.count(collection_name=COLLECTION_NAME, count_filter=selector).count
        if existing:
            client.delete(collection_name=COLLECTION_NAME, points_selector=selector)
    except Exception as exc:  # noqa: BLE001 - 统一转系统异常，细节写日志
        raise VectorStoreError(detail=f"Qdrant 删除同名文件旧分块失败：{exc}") from exc
    return existing


def _write_qdrant(
    *,
    vectors: list[list[float]],
    user_id: int,
    doc_category: str,
    file_name: str,
) -> list[str]:
    """写向量库：先按（user_id, file_name）删旧点，再写本次的新点；返回新点的 id。

    payload 只放检索过滤字段，**不存原文**（原文在 MySQL 的 knowledge_chunks，
    两边用这里生成的 point id == vector_id 关联）。
    """
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_collection(client)

    removed = _delete_existing_points(client, user_id=user_id, file_name=file_name)
    if removed:
        logger.info(
            "覆盖写入：先删除旧分块 file=%s user_id=%s removed=%s",
            file_name,
            user_id,
            removed,
        )

    point_ids: list[str] = []
    points: list[PointStruct] = []
    for chunk_index, vector in enumerate(vectors):
        point_id = str(uuid.uuid4())  # 同值写进 Qdrant point id 与 MySQL knowledge_chunks.vector_id
        point_ids.append(point_id)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
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
    db: Session | None = None,
    resource_id: int | None = None,
) -> dict[str, object]:
    """把一个文件向量化入库，返回入库摘要。

    入参：
    - ``source``：文件路径（str / Path），或文件字节内容（bytes）
    - ``file_name``：展示名 / 类型识别用；传路径时可省略（默认取路径文件名），传 bytes 时必填
    - ``user_id``：归属用户
    - ``doc_category``：显式分类（resume / study_material / general），传入则优先于关键字计分
    - ``db`` / ``resource_id``：写分块原文用的会话与资源主键（由上传链路传入）；
      不传则原文不落库（向量库已不存原文，检索会取不到内容），仅用于离线调试

    用法::

        ingest_file("/data/a.pdf", user_id=1, db=session, resource_id=12)
        ingest_file(b"...", "note.txt", user_id=1)

    顺序：写 Qdrant（先删同文件旧点）-> 写 MySQL knowledge_chunks（先删同文件旧分块）。
    返回：``{"file_name", "user_id", "doc_category", "chunk_count", "point_ids", "stored_chunks"}``
    """
    resolved_name, data = _resolve_source(source, file_name)

    text = _clean(_parse(resolved_name, data))
    category = _classify(text, doc_category)
    chunks = _chunk(text)

    point_ids: list[str] = []
    stored_chunks = 0
    if chunks:
        vectors = embed_texts(chunks)
        point_ids = _write_qdrant(
            vectors=vectors,
            user_id=user_id,
            doc_category=category,
            file_name=resolved_name,
        )
        stored_chunks = _save_chunk_texts(
            db=db,
            resource_id=resource_id,
            user_id=user_id,
            file_name=resolved_name,
            texts=chunks,
            vector_ids=point_ids,
        )

    logger.info(
        "RAG 入库完成 file=%s category=%s chunks=%s points=%s 原文入库=%s",
        resolved_name,
        category,
        len(chunks),
        len(point_ids),
        stored_chunks,
    )
    return {
        "file_name": resolved_name,
        "user_id": user_id,
        "doc_category": category,
        "chunk_count": len(chunks),
        "point_ids": point_ids,
        "stored_chunks": stored_chunks,
    }


def _save_chunk_texts(
    *,
    db: Session | None,
    resource_id: int | None,
    user_id: int,
    file_name: str,
    texts: list[str],
    vector_ids: list[str],
) -> int:
    """把分块原文写进 MySQL knowledge_chunks；缺 db / resource_id 时只记日志。"""
    if db is None or resource_id is None:
        logger.warning(
            "未传 db/resource_id，分块原文未落库（向量库只存向量，检索将取不到内容）file=%s chunks=%s",
            file_name,
            len(texts),
        )
        return 0
    return save_chunks(
        db,
        resource_id=resource_id,
        user_id=user_id,
        file_name=file_name,
        texts=texts,
        vector_ids=vector_ids,
    )
