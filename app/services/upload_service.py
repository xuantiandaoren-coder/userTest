"""业务层（教学保留，运行链路不经过这里）：本地磁盘上传。

规则：图片 -> 落盘并更新用户 avatar；文档 -> 只落盘。
存储细节（类型识别、目录、去重命名）在 app/core/storage.py。

运行链路请见 app/services/upload_service_seaweedfs.py：
MySQL 存元数据、SeaweedFS 存原文件，两者解耦。本文件只用于对照讲解旧实现。
"""

from fastapi import UploadFile

from app.core.storage import RESOURCE_TYPE_BY_KIND, FileKind, store_upload
from app.db.models import User
from app.db.user_repository import UserRepository
from app.schemas.file import UploadResult


class FileService:
    """文件业务逻辑：本地落盘 + 按类型决定是否更新用户头像。"""

    def __init__(self, repository: UserRepository) -> None:
        self.repository = repository

    async def upload(self, user: User, upload: UploadFile) -> UploadResult:
        """保存文件到本地；识别为图片时把相对路径写入用户 avatar 字段。"""
        stored = await store_upload(upload, user_id=user.id)

        avatar_updated = stored.kind is FileKind.IMAGE
        if avatar_updated:
            user.avatar = stored.relative_path
            self.repository.save(user)

        return UploadResult(
            kind=stored.kind.value,
            resource_type=RESOURCE_TYPE_BY_KIND[stored.kind],
            md5=stored.md5,
            size=stored.size,
            filename=stored.original_name,
            path=stored.relative_path,
            avatar_updated=avatar_updated,
            deduplicated=stored.deduplicated,
        )
