"""业务层（运行链路）：上传到 SeaweedFS 对象存储。

架构：MySQL resources 存元数据，SeaweedFS 存原文件，两者解耦，接口路径不变。

- 去重：先按 (file_hash, user_id) 预查；并发写入由唯一索引兜底（IntegrityError 后回查）
- storage_scene：0=长过期(1 个月)、1=短过期(2 小时) 存对象；2=只提取内容，不存原文件、不写元数据
- upload_purpose：仅 1(avatar) 且类型为图片时，把对象键写进 user.avatar
- doc_category：文档分类 resume / study_material / general，仅文件类型有意义，其余类型强制为空
- RAG：文件类型落库成功后调用 RagIngestService 向量化入库（失败只记日志，不影响上传）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.extract import extract_text
from app.core.seaweedfs import SeaweedFSClient
from app.core.storage import (
    RESOURCE_TYPE_BY_KIND,
    DocCategory,
    FileKind,
    UploadContent,
    normalize_doc_category,
    safe_extension,
    read_upload,
)
from app.db.models import Resource, User
from app.db.resource_repository import ResourceRepository
from app.db.user_repository import UserRepository
from app.schemas.file import UploadResult
from app.services.rag_service import RagIngestResult, RagIngestService

logger = logging.getLogger(__name__)

# 存储场景
SCENE_LONG_EXPIRY = 0   # 长过期：1 个月
SCENE_SHORT_EXPIRY = 1  # 短过期：2 小时
SCENE_EXTRACT_ONLY = 2  # 只提取内容，不存原文件

# 上传用途
PURPOSE_GENERAL = 0
PURPOSE_AVATAR = 1


class SeaweedFSUploadService:
    """上传业务逻辑：读内容 -> 去重 -> 存对象 + 写元数据 -> 按用途更新头像。"""

    def __init__(
        self,
        resources: ResourceRepository,
        users: UserRepository,
        storage: SeaweedFSClient,
        rag: RagIngestService | None = None,
    ) -> None:
        self.resources = resources
        self.users = users
        self.storage = storage
        self.rag = rag

    async def upload(
        self,
        user: User,
        upload: UploadFile,
        *,
        storage_scene: int = SCENE_LONG_EXPIRY,
        upload_purpose: int = PURPOSE_GENERAL,
        doc_category: str | None = None,
    ) -> UploadResult:
        """处理一次上传：按存储场景落对象 / 只提取内容，按上传用途决定是否更新头像。

        doc_category 仅对文件类型生效：图片 / 音频即使传了也按空处理。
        """
        content = await read_upload(upload)
        category = _document_category(content.kind, doc_category)

        if storage_scene == SCENE_EXTRACT_ONLY:
            # 只提取内容：不写对象存储，也不落元数据，内容直接回给调用方
            return self._extract_only_result(content, storage_scene, upload_purpose)

        resource, deduplicated = await self._persist(user, content, storage_scene, upload_purpose, category)
        path = resource.storage_path

        avatar_updated = upload_purpose == PURPOSE_AVATAR and content.kind is FileKind.IMAGE
        if avatar_updated:
            user.avatar = path
            self.users.save(user)

        # 文件类型落库成功后再向量化入库：失败不影响上传结果（见 app/services/rag_service.py）
        rag = await self._ingest_to_rag(user, content, resource, deduplicated)

        return UploadResult(
            kind=content.kind.value,
            resource_id=resource.id,
            resource_type=RESOURCE_TYPE_BY_KIND[content.kind],
            md5=content.md5,
            size=content.size,
            filename=content.original_name,
            path=path,
            url=self.storage.url_for(path),
            avatar_updated=avatar_updated,
            deduplicated=deduplicated,
            storage_scene=storage_scene,
            upload_purpose=upload_purpose,
            doc_category=resource.doc_category,
            rag_ingested=rag is not None and rag.ingested,
            rag_chunk_count=rag.chunk_count if rag else 0,
            rag_error=rag.error if rag else None,
        )

    async def _persist(
        self,
        user: User,
        content: UploadContent,
        storage_scene: int,
        upload_purpose: int,
        doc_category: DocCategory | None,
    ) -> tuple[Resource, bool]:
        """存对象 + 写元数据；返回 (元数据行, 是否命中已有资源)。"""
        existing = self.resources.find_by_hash(content.md5, user.id)
        if existing is not None:
            # 去重命中：本次显式带了分类就同步更新，没带（空）则保留原值
            if doc_category is not None and existing.doc_category != doc_category.value:
                existing.doc_category = doc_category.value
                self.resources.session.flush()
            return existing, True

        key = object_key(user.id, content.md5, content.original_name)
        await run_in_threadpool(self.storage.put_object, key, content.data, content.content_type)
        try:
            resource = self.resources.create(
                resource_type=RESOURCE_TYPE_BY_KIND[content.kind],
                doc_category=doc_category.value if doc_category else None,
                storage_scene=storage_scene,
                upload_purpose=upload_purpose,
                file_name=content.original_name,
                file_hash=content.md5,
                storage_path=key,
                user_id=user.id,
                expire_time=_expire_time(storage_scene),
            )
        except IntegrityError:
            # 并发上传同内容：唯一索引兜底，回查先写入的那条，并清掉本次多写的对象
            self.resources.session.rollback()
            existing = self.resources.find_by_hash(content.md5, user.id)
            if existing is None:  # 理论上不可达：索引冲突却查不到记录
                raise
            if existing.storage_path != key:
                await run_in_threadpool(self.storage.delete_object, key)
            return existing, True
        return resource, False

    async def _ingest_to_rag(
        self,
        user: User,
        content: UploadContent,
        resource: Resource,
        deduplicated: bool,
    ) -> RagIngestResult | None:
        """把刚落库的文件送进 RAG（解析 -> 分块 -> 向量化 -> 写 Qdrant）。

        只对文件类型、且本次真的新增了元数据的上传入库：
        - 图片 / 音频：没有文档语义，不入库
        - 去重命中：内容早就入过库，重复入库只会在向量库里堆重复点
        - 未接 RAG 服务（测试 / 未注入依赖）：直接跳过
        """
        if self.rag is None or content.kind is not FileKind.DOCUMENT:
            return None
        if deduplicated:
            logger.info("命中已有资源，跳过 RAG 入库 file=%s resource_id=%s", resource.file_name, resource.id)
            return None
        return await self.rag.ingest(
            data=content.data,
            file_name=resource.file_name,
            user_id=user.id,
            doc_category=resource.doc_category,
        )

    @staticmethod
    def _extract_only_result(content: UploadContent, storage_scene: int, upload_purpose: int) -> UploadResult:
        """storage_scene=2 的结果：只带提取到的内容，不落对象、不落元数据。"""
        return UploadResult(
            kind=content.kind.value,
            resource_type=RESOURCE_TYPE_BY_KIND[content.kind],
            md5=content.md5,
            size=content.size,
            filename=content.original_name,
            path="",
            avatar_updated=False,
            deduplicated=False,
            storage_scene=storage_scene,
            upload_purpose=upload_purpose,
            extracted_text=extract_text(content.original_name, content.data),
        )


def object_key(user_id: int, file_hash: str, file_name: str) -> str:
    """对象键：`<user_id>/<MD5><扩展名>`，同一用户同一内容恒定，重复上传即覆盖。"""
    extension = safe_extension(file_name)
    return f"{user_id}/{file_hash}.{extension}" if extension else f"{user_id}/{file_hash}"


def _document_category(kind: FileKind, doc_category: str | None) -> DocCategory | None:
    """只有文件类型才有文档分类：图片 / 音频即使传了也丢弃（返回 None）。"""
    if kind is not FileKind.DOCUMENT:
        return None
    return normalize_doc_category(doc_category)


def _expire_time(storage_scene: int) -> datetime:
    """按存储场景计算过期时间：0=1 个月，1=2 小时。"""
    seconds = settings.resource_ttl_long_seconds
    if storage_scene == SCENE_SHORT_EXPIRY:
        seconds = settings.resource_ttl_short_seconds
    return datetime.now() + timedelta(seconds=seconds)
