"""校验层：文件上传响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field


class UploadResult(BaseModel):
    """上传结果：元数据（MySQL）+ 对象位置（SeaweedFS）。"""

    kind: Literal["image", "document", "audio"] = Field(description="自动识别出的类型")
    resource_id: int | None = Field(default=None, description="resources 表主键，聊天附件段用它关联；只提取内容时为 null")
    resource_type: int = Field(description="资源类型：0=文件，1=图片，2=音频")
    md5: str = Field(description="文件内容 MD5，去重核心字段")
    size: int = Field(description="文件大小（字节）")
    filename: str = Field(description="原始文件名")
    path: str = Field(description="SeaweedFS 对象键，如 1/ab12...ef.png；只提取内容时为空串")
    url: str = Field(default="", description="对象的可直接访问地址；只提取内容时为空串")
    avatar_updated: bool = Field(description="upload_purpose=1 且为图片时会写入用户 avatar")
    deduplicated: bool = Field(description="命中该用户已有资源（内容相同），未重复上传")
    storage_scene: int = Field(default=0, description="存储场景：0=长过期，1=短过期，2=只提取内容")
    upload_purpose: int = Field(default=0, description="上传用途：0=普通资源，1=用户头像")
    doc_category: str | None = Field(
        default=None,
        description="文档分类：resume/study_material/general；仅文件类型有意义，其余为 null",
    )
    rag_ingested: bool = Field(
        default=False,
        description="该文件是否已向量化入知识库（仅文件类型会尝试；失败不影响上传成功）",
    )
    rag_chunk_count: int = Field(default=0, description="入知识库时切分出的分块数；未入库为 0")
    rag_error: str | None = Field(
        default=None,
        description="未入库原因 / 错误码，如 RAG_DISABLED / RAG_NOT_CONFIGURED / UNSUPPORTED_FILE_TYPE；入库成功或未尝试为 null",
    )
    extracted_text: str | None = Field(default=None, description="storage_scene=2 时提取到的文件内容")
