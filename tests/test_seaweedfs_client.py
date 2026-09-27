"""SeaweedFS 客户端测试：对象访问地址（url_for）在三种部署方式下的拼接规则。"""

from __future__ import annotations

from app.core.seaweedfs import SeaweedFSClient

KEY = "11/937bf0c0eb2370deab143ae4fe8aec95.jpg"


def _client(**overrides: object) -> SeaweedFSClient:
    params: dict[str, object] = {
        "endpoint": "http://seaweedfs.internal:8333/",
        "access_key": "",
        "secret_key": "",
        "bucket": "test",
        "region": "us-east-1",
    }
    params.update(overrides)
    return SeaweedFSClient(**params)  # type: ignore[arg-type]


def test_anonymous_client_returns_gateway_url() -> None:
    """未配凭据（匿名网关）：<endpoint>/<bucket>/<key>，不会去签名。"""
    assert _client().url_for(KEY) == f"http://seaweedfs.internal:8333/test/{KEY}"


def test_public_base_url_wins() -> None:
    """配了 CDN / 反向代理：<base>/<key>，由 CDN 承担访问与鉴权。"""
    client = _client(access_key="ak", secret_key="sk", public_base_url="https://cdn.example.com/")

    assert client.url_for(KEY) == f"https://cdn.example.com/{KEY}"


def test_signed_client_returns_presigned_url() -> None:
    """配了 Access/Secret Key：返回带签名的预签名 URL（本地签名，不联网）。"""
    client = _client(access_key="ak", secret_key="sk", presign_expire_seconds=60)

    url = client.url_for(KEY)

    assert url.startswith("http://seaweedfs.internal:8333/test/" + KEY)
    assert "X-Amz-Signature=" in url
    assert "X-Amz-Expires=60" in url
