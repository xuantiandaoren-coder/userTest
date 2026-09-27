"""带自动刷新的调用方客户端（服务端运行时不依赖它，供 SDK / 脚本 / 测试使用）。

访问令牌过期时：用 refresh token 换新的 access token，然后自动重放原请求，
调用方完全无感知。浏览器端等价实现是 axios / fetch 的 401 拦截器。
"""

from __future__ import annotations

import asyncio

import httpx

TOKEN_EXPIRED_CODE = "TOKEN_EXPIRED"


class RefreshTokenRejectedError(RuntimeError):
    """刷新令牌不可用，调用方需要重新登录。"""


class RefreshableClient:
    """包装 httpx.AsyncClient，透明处理访问令牌过期。

    用法::

        async with httpx.AsyncClient(base_url=...) as client:
            api = RefreshableClient(client)
            await api.login("alice", "secret123")
            await api.get("/api/v1/users")   # access 过期时自动刷新并重试
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        login_path: str = "/auth/login",
        refresh_path: str = "/auth/refresh",
    ) -> None:
        self._client = client
        self._login_path = login_path
        self._refresh_path = refresh_path
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        # 并发请求同时过期时，只让一个请求真正去刷新
        self._refresh_lock = asyncio.Lock()

    @property
    def access_token(self) -> str | None:
        return self._access_token

    def set_tokens(self, *, access_token: str | None = None, refresh_token: str | None = None) -> None:
        """直接注入令牌（例如从本地缓存恢复登录态）。"""
        if access_token is not None:
            self._access_token = access_token
        if refresh_token is not None:
            self._refresh_token = refresh_token

    async def login(self, username: str, password: str) -> httpx.Response:
        """登录并记住令牌对。"""
        response = await self._client.post(self._login_path, json={"username": username, "password": password})
        if response.status_code == httpx.codes.OK:
            self._apply_tokens(response.json())
        return response

    async def refresh(self) -> httpx.Response:
        """用刷新令牌换新的访问令牌；失败则清除登录态并抛错。"""
        async with self._refresh_lock:
            if self._refresh_token is None:
                raise RefreshTokenRejectedError("缺少刷新令牌，请先登录")

            response = await self._client.post(self._refresh_path, json={"refresh_token": self._refresh_token})
            if response.status_code != httpx.codes.OK:
                self._access_token = self._refresh_token = None
                raise RefreshTokenRejectedError(f"刷新令牌不可用（HTTP {response.status_code}），请重新登录")

            self._apply_tokens(response.json())
            return response

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """发请求；若因访问令牌过期被拒，自动刷新后重放原请求（只重试一次）。"""
        response = await self._send(method, url, **kwargs)
        if response.status_code == httpx.codes.UNAUTHORIZED and _is_token_expired(response):
            await self.refresh()
            response = await self._send(method, url, **kwargs)
        return response

    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", url, **kwargs)

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        headers = dict(kwargs.pop("headers", None) or {})
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return await self._client.request(method, url, headers=headers, **kwargs)

    def _apply_tokens(self, payload: dict) -> None:
        self._access_token = payload["access_token"]
        self._refresh_token = payload["refresh_token"]


def _is_token_expired(response: httpx.Response) -> bool:
    """按统一错误结构判断 401 的原因是否为“访问令牌过期”。"""
    try:
        return response.json().get("code") == TOKEN_EXPIRED_CODE
    except ValueError:
        return False
