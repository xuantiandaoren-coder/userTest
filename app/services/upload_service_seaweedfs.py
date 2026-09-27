"""业务层（运行链路）：上传到 SeaweedFS 对象存储。

架构：MySQL resources 存元数据，SeaweedFS 存原文件，两者解耦，接口路径不变。

- 去重：先按 (file_hash, user_id) 预查；并发写入由唯一索引兜底（IntegrityError 后回查）
- storage_scene：0=长过期(1 个月)、1=短过期(2 小时) 存对象；2=只提取内容，不存原文件、不写元数据
- upload_purpose：仅 1(avatar) 且类型为图片时，把对象键写进 user.avatar
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.extract import extract_text
from app.core.seaweedfs import SeaweedFSClient
from app.core.storage import RESOURCE_TYPE_BY_KIND, FileKind, UploadContent, safe_extension, read_upload
from app.db.models import Resource, User
from app.db.resource_repository import ResourceRepository
from app.db.user_repository import UserRepository
from app.schemas.file import UploadResult

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
    ) -> None:
        self.resources = resources
        self.users = users
        self.storage = storage

    async def upload(
        self,
        user: User,
        upload: UploadFile,
        *,
        storage_scene: int = SCENE_LONG_EXPIRY,
        upload_purpose: int = PURPOSE_GENERAL,
    ) -> UploadResult:
        """处理一次上传：按存储场景落对象 / 只提取内容，按上传用途决定是否更新头像。"""
        content = await read_upload(upload)

        if storage_scene == SCENE_EXTRACT_ONLY:
            # 只提取内容：不写对象存储，也不落元数据，内容直接回给调用方
            return self._extract_only_result(content, storage_scene, upload_purpose)

        resource, deduplicated = await self._persist(user, content, storage_scene, upload_purpose)
        path = resource.storage_path

        avatar_updated = upload_purpose == PURPOSE_AVATAR and content.kind is FileKind.IMAGE
        if avatar_updated:
            user.avatar = path
            self.users.save(user)

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
        )

    async def _persist(
        self,
        user: User,
        content: UploadContent,
        storage_scene: int,
        upload_purpose: int,
    ) -> tuple[Resource, bool]:
        """存对象 + 写元数据；返回 (元数据行, 是否命中已有资源)。"""
        existing = self.resources.find_by_hash(content.md5, user.id)
        if existing is not None:
            return existing, True

        key = object_key(user.id, content.md5, content.original_name)
        await run_in_threadpool(self.storage.put_object, key, content.data, content.content_type)
        try:
            resource = self.resources.create(
                resource_type=RESOURCE_TYPE_BY_KIND[content.kind],
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


def _expire_time(storage_scene: int) -> datetime:
    """按存储场景计算过期时间：0=1 个月，1=2 小时。"""
    seconds = settings.resource_ttl_long_seconds
    if storage_scene == SCENE_SHORT_EXPIRY:
        seconds = settings.resource_ttl_short_seconds
    return datetime.now() + timedelta(seconds=seconds)
