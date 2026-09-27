"""路由层：需认证的通用文件上传接口。

类型识别与存储规则在 app/services/upload_service_seaweedfs.py；
本层只负责鉴权、参数声明与响应。
"""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.api.deps import CurrentUserDep, UploadServiceDep
from app.schemas.file import UploadResult

router = APIRouter(tags=["files"])


@router.post(
    "/files/upload",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
    summary="上传文件（需登录；对象存 SeaweedFS，元数据存 MySQL）",
)
@router.post(
    "/upload/file",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def upload_file(
    file: Annotated[UploadFile, File(description="待上传文件")],
    current_user: CurrentUserDep,
    service: UploadServiceDep,
    storage_scene: Annotated[int, Form(ge=0, le=2, description="0=长过期(1个月)，1=短过期(2小时)，2=只提取内容")] = 0,
    upload_purpose: Annotated[int, Form(ge=0, le=1, description="0=普通资源，1=用户头像")] = 0,
) -> UploadResult:
    """登录用户上传文件：按内容识别类型，原文件存 SeaweedFS，元数据写 resources 表。"""
    return await service.upload(
        current_user,
        file,
        storage_scene=storage_scene,
        upload_purpose=upload_purpose,
    )
