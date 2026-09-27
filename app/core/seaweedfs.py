"""SeaweedFS 对象存储客户端（S3 网关）。

元数据与文件解耦：MySQL 只存元数据，原文件以对象（object）形式放在 SeaweedFS。
本模块只负责「把对象放进去 / 删掉」，业务规则在 app/services/upload_service_seaweedfs.py。

客户端惰性创建：导入本模块、构造 SeaweedFSClient 都不会建立连接，
真正用到时（首个 put / delete）才创建 boto3 客户端，测试可整体替换该依赖。
"""

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings


class SeaweedFSClient:
    """SeaweedFS S3 客户端：对象的 put / delete。"""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str,
        public_base_url: str = "",
        presign_expire_seconds: int = 3600,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._public_base_url = public_base_url.rstrip("/")
        self._presign_expire_seconds = presign_expire_seconds

    @property
    def _signed(self) -> bool:
        """是否配置了 S3 凭据（配了就要签名访问）。"""
        return bool(self._access_key and self._secret_key)

    @cached_property
    def _client(self) -> Any:
        """boto3 S3 客户端：path-style 寻址，适配 SeaweedFS 网关。

        配了 Access/Secret Key 就用 s3v4 签名；都没配则走匿名请求（UNSIGNED），
        对应 SeaweedFS 未开启鉴权的部署方式。
        """
        signature_version = "s3v4" if self._signed else UNSIGNED
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self._access_key or None,
            aws_secret_access_key=self._secret_key or None,
            region_name=self._region,
            config=Config(signature_version=signature_version, s3={"addressing_style": "path"}),
        )

    def put_object(self, key: str, body: bytes, content_type: str | None = None) -> str:
        """上传对象，返回对象键。同键重复上传即覆盖（内容相同，天然幂等）。"""
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type or "application/octet-stream",
        )
        return key

    def delete_object(self, key: str) -> None:
        """删除对象；对象不存在时 S3 语义同样返回成功，故无需先判断存在性。"""
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def object_exists(self, key: str) -> bool:
        """对象是否存在（head 失败一律视为不存在，供校验 / 历史数据迁移使用）。"""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError:
            return False
        return True

    def url_for(self, key: str) -> str:
        """对象的可访问地址，直接给前端用。

        - 配了 `SEAWEEDFS_PUBLIC_BASE_URL`（CDN / 反向代理）：返回 `<base>/<key>`，由 CDN 承担访问与鉴权
        - 否则开启了鉴权（配了 Access/Secret Key）：返回 s3v4 预签名 URL（有有效期，过期后重新获取）
        - 否则（匿名网关）：返回 `<endpoint>/<bucket>/<key>`
        """
        if self._public_base_url:
            return f"{self._public_base_url}/{key}"
        if self._signed:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self._presign_expire_seconds,
            )
        return f"{self.endpoint}/{self.bucket}/{key}"


@lru_cache(maxsize=1)
def get_seaweedfs_client() -> SeaweedFSClient:
    """进程级单例：连接参数全部来自环境变量 / .env。"""
    return SeaweedFSClient(
        endpoint=settings.seaweedfs_endpoint,
        access_key=settings.seaweedfs_access_key.get_secret_value(),
        secret_key=settings.seaweedfs_secret_key.get_secret_value(),
        bucket=settings.seaweedfs_bucket,
        region=settings.seaweedfs_region,
        public_base_url=settings.seaweedfs_public_base_url,
        presign_expire_seconds=settings.seaweedfs_presign_expire_seconds,
    )
