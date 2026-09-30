"""文件存储：类型自动识别（本地 / 对象存储共用）+ 本地落盘（旧逻辑）。

- 类型识别：扩展名 / 客户端 Content-Type 初判，图片再用魔数校验内容（不轻信客户端声明）
- 上传读取：read_upload 一次性读全内容、校验大小 / 非空、算 MD5，供对象存储链路使用
- 命名去重：`<内容 MD5><扩展名>`，同一内容重复上传直接复用已有文件
- 落盘位置：`STORAGE_ROOT/<user_id>/`，先写临时文件再原子替换，避免半截文件

落盘逻辑（store_upload）只服务于旧的本地存储链路，运行链路见 app/services/upload_service_seaweedfs.py。
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import BusinessError

CHUNK_SIZE = 1024 * 1024
_MAGIC_SAMPLE_SIZE = 512


class FileKind(str, Enum):
    """自动识别出的文件大类。"""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"


class DocCategory(str, Enum):
    """文档分类：仅文件类型（resource_type=0）有意义，图片 / 音频恒为空。"""

    RESUME = "resume"
    STUDY_MATERIAL = "study_material"
    GENERAL = "general"


# 文档分类取值清单，用于校验与提示
DOC_CATEGORIES: tuple[str, ...] = tuple(category.value for category in DocCategory)


# resource_type：0=文件，1=图片，2=音频（与 resources 表注释一致）
RESOURCE_TYPE_BY_KIND: dict[FileKind, int] = {
    FileKind.DOCUMENT: 0,
    FileKind.IMAGE: 1,
    FileKind.AUDIO: 2,
}


class UnsupportedFileTypeError(BusinessError):
    """业务异常：文件类型不在支持范围内。"""

    code = "UNSUPPORTED_FILE_TYPE"
    http_status = 415
    message = "不支持的文件类型"


class InvalidDocCategoryError(BusinessError):
    """业务异常：文档分类不在 resume/study_material/general 之内。"""

    code = "INVALID_DOC_CATEGORY"
    http_status = 422
    message = "文档分类不合法"


class FileTooLargeError(BusinessError):
    """业务异常：文件超过大小上限。"""

    code = "FILE_TOO_LARGE"
    http_status = 413
    message = "文件过大"


class EmptyFileError(BusinessError):
    """业务异常：文件内容为空。"""

    code = "EMPTY_FILE"
    http_status = 400
    message = "文件内容为空"


@dataclass(frozen=True)
class StoredFile:
    """落盘结果。"""

    kind: FileKind
    md5: str
    size: int
    original_name: str
    relative_path: str  # 相对 STORAGE_ROOT，如 1/ab12...ef.png
    path: Path
    deduplicated: bool  # 命中已有文件（内容相同）


@dataclass(frozen=True)
class UploadContent:
    """一次上传的完整内容：原文件名、字节内容、MD5、大小与识别出的类型。"""

    original_name: str
    data: bytes
    md5: str
    size: int
    kind: FileKind
    content_type: str | None


# 图片内容魔数：声明是图片时必须匹配其中之一
_IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"RIFF",
)

_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp"})
_AUDIO_EXTENSIONS = frozenset({"mp3", "wav", "m4a", "aac", "flac", "ogg", "oga", "opus", "wma"})
_AUDIO_MEDIA_TYPES = frozenset(
    {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac", "audio/flac", "audio/ogg"}
)
_DOCUMENT_EXTENSIONS = frozenset(
    {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "md", "csv", "rtf", "odt", "ods", "odp"}
)
_DOCUMENT_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/markdown",
        "text/csv",
    }
)
_SAFE_EXTENSION = re.compile(r"^[A-Za-z0-9]{1,10}$")


def classify_file(file_name: str, content_type: str | None, head: bytes) -> FileKind:
    """识别文件类型；不支持时抛 UnsupportedFileTypeError。"""
    extension = safe_extension(file_name)
    media_type = (content_type or "").split(";")[0].strip().lower()

    if media_type.startswith("image/") or extension in _IMAGE_EXTENSIONS:
        if any(head.startswith(magic) for magic in _IMAGE_MAGIC):
            return FileKind.IMAGE
        # 只声明是图片、内容却对不上（改名或伪造 Content-Type）时拒绝，避免把非图片写成头像
        raise UnsupportedFileTypeError(detail=f"declared image but unsupported content: file={file_name!r}")

    if extension in _DOCUMENT_EXTENSIONS or media_type in _DOCUMENT_MEDIA_TYPES:
        return FileKind.DOCUMENT

    if extension in _AUDIO_EXTENSIONS or media_type in _AUDIO_MEDIA_TYPES:
        return FileKind.AUDIO

    raise UnsupportedFileTypeError(detail=f"file={file_name!r} content_type={media_type or 'unknown'}")


async def read_upload(upload: UploadFile) -> UploadContent:
    """读完整份上传内容（受 max_upload_bytes 限制），返回内容 + MD5 + 类型。

    对象存储需要把整个对象一次性 put 出去，因此这里整体读入内存；
    大小上限默认 10MB，超限在读完前即抛 413，不会无限吃内存。
    """
    original_name = Path(upload.filename or "").name  # 只取文件名，防目录穿越
    hasher = hashlib.md5(usedforsecurity=False)  # 仅用于去重，不用于安全校验
    buffer = bytearray()
    head = b""

    while chunk := await upload.read(CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) > settings.max_upload_bytes:
            raise FileTooLargeError(detail=f"max={settings.max_upload_bytes} bytes")
        head = head or chunk[:_MAGIC_SAMPLE_SIZE]
        hasher.update(chunk)

    if not buffer:
        raise EmptyFileError(detail=f"file={original_name!r}")

    return UploadContent(
        original_name=original_name,
        data=bytes(buffer),
        md5=hasher.hexdigest(),
        size=len(buffer),
        kind=classify_file(original_name, upload.content_type, head),
        content_type=upload.content_type,
    )


async def store_upload(upload: UploadFile, *, user_id: int) -> StoredFile:
    """把上传内容流式写入 `STORAGE_ROOT/<user_id>/`，返回落盘结果。"""
    original_name = Path(upload.filename or "").name  # 只取文件名，防目录穿越
    target_dir = settings.storage_root / str(user_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.md5(usedforsecurity=False)  # 仅用于去重命名，不用于安全校验
    size = 0
    head = b""
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(dir=target_dir, prefix=".upload-", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await upload.read(CHUNK_SIZE):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise FileTooLargeError(detail=f"max={settings.max_upload_bytes} bytes")
                head = head or chunk[:_MAGIC_SAMPLE_SIZE]
                hasher.update(chunk)
                tmp.write(chunk)

        if size == 0:
            raise EmptyFileError(detail=f"file={original_name!r}")

        kind = classify_file(original_name, upload.content_type, head)
        md5 = hasher.hexdigest()
        extension = safe_extension(original_name)
        final_path = target_dir / f"{md5}.{extension}" if extension else target_dir / md5

        deduplicated = final_path.exists()
        if deduplicated:  # 内容相同：复用已有文件，丢弃临时文件
            tmp_path.unlink(missing_ok=True)
        else:
            os.replace(tmp_path, final_path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    return StoredFile(
        kind=kind,
        md5=md5,
        size=size,
        original_name=original_name,
        relative_path=f"{user_id}/{final_path.name}",
        path=final_path,
        deduplicated=deduplicated,
    )


def safe_extension(file_name: str) -> str:
    """取安全扩展名（小写、白名单字符），拿不到则返回空串。"""
    suffix = Path(file_name).suffix.lstrip(".").lower()
    return suffix if _SAFE_EXTENSION.match(suffix) else ""


def normalize_doc_category(value: str | None) -> DocCategory | None:
    """归一化文档分类：None / 空串 / 纯空白一律视为未分类（None）。

    取值只允许 resume / study_material / general（大小写不敏感），
    其他值抛 InvalidDocCategoryError（422），避免脏值落库。
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    try:
        return DocCategory(normalized)
    except ValueError:
        raise InvalidDocCategoryError(detail=f"doc_category={value!r} allowed={list(DOC_CATEGORIES)}") from None
