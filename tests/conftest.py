"""pytest 公共夹具：用内存 SQLite 替换 MySQL，保证用例独立且无需外部依赖。

环境变量优先级高于 .env，因此在导入 app（会加载配置并初始化日志）之前
先固定测试环境，避免受本机 .env 影响，也避免日志写进仓库目录。
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from tempfile import gettempdir

os.environ["APP_ENV"] = "dev"
os.environ["LOG_DIR"] = str(Path(gettempdir()) / "user-api-test-logs")
os.environ["LOG_ACCESS"] = "true"
os.environ["STORAGE_ROOT"] = str(Path(gettempdir()) / "user-api-test-storage")
# 启动预热会连真实数据库 / Redis，测试里关掉（用例需要时直接调用 warm_cache()）
os.environ["PROMPT_CACHE_WARMUP"] = "false"
# 上传后自动向量化入库默认关闭，避免用例连 DashScope / Qdrant；
# 需要验证上传触发 RAG 的用例用 dependency_overrides 注入假实现
os.environ["RAG_INGEST_ENABLED"] = "false"
# JWT 密钥只走环境变量（生产即如此），测试用独立的固定值，长度满足最小要求
os.environ["JWT_SECRET_KEY"] = "pytest-only-jwt-secret-key-0123456789abcdef"

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine, create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.rate_limit import login_limiter, register_limiter  # noqa: E402
from app.api.deps import get_redis, get_storage_client  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402


class FakeSeaweedFSClient:
    """内存对象存储：模拟 SeaweedFS 的 put_object / delete_object，测试不连真实网关。"""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_delete: set[str] = set()  # 需要模拟删除失败的对象键

    def put_object(self, key: str, body: bytes, content_type: str | None = None) -> str:
        self.objects[key] = body
        return key

    def delete_object(self, key: str) -> None:
        if key in self.fail_delete:
            raise RuntimeError(f"delete failed: {key}")
        self.objects.pop(key, None)

    def url_for(self, key: str) -> str:
        return f"https://fake-seaweedfs.test/{key}"


@pytest.fixture()
def anyio_backend() -> str:
    """异步用例只跑 asyncio 后端（环境未安装 trio）。"""
    return "asyncio"


@pytest.fixture()
def sqlite_engine() -> Iterator[Engine]:
    """每个用例一套全新的内存表结构（StaticPool 让多线程共享同一连接）。"""
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def db_session(sqlite_engine: Engine) -> Iterator[Session]:
    """与线上一致的会话：请求结束提交，异常回滚。"""
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def db_override(db_session: Session) -> Iterator[None]:
    """把 get_db 依赖换成测试会话，接口仍走完整的服务层与仓储层。"""

    def _override_get_db() -> Iterator[Session]:
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def client(db_override: None) -> Iterator[TestClient]:
    """同步接口测试客户端（starlette TestClient）。"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_rate_limiters() -> None:
    """限流器是进程级状态，每个用例前清零，避免用例相互影响。"""
    register_limiter.reset()
    login_limiter.reset()


@pytest.fixture()
async def httpx_client() -> AsyncIterator[httpx.AsyncClient]:
    """进程内 httpx 异步客户端：直连 ASGI 应用，不经过网络。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture()
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个用例独立的存储目录，避免上传文件在用例之间互相影响。"""
    root = tmp_path / "storage"
    monkeypatch.setattr(settings, "storage_root", root)
    return root


@pytest.fixture()
def object_store() -> FakeSeaweedFSClient:
    """每个用例独立的内存对象存储。"""
    return FakeSeaweedFSClient()


@pytest.fixture()
def storage_override(object_store: FakeSeaweedFSClient) -> Iterator[FakeSeaweedFSClient]:
    """把 SeaweedFS 客户端依赖换成内存实现。"""
    app.dependency_overrides[get_storage_client] = lambda: object_store
    try:
        yield object_store
    finally:
        app.dependency_overrides.pop(get_storage_client, None)


class FakeRedis:
    """内存版 Redis：只实现提示词缓存用到的 Hash 命令子集。

    `broken=True` 用来模拟 Redis 不可用（连接失败 / 超时），验证业务能降级回数据库。
    """

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expires: list[tuple[str, int]] = []
        self.broken = False

    def _guard(self) -> None:
        """模拟「Redis 挂了」：所有命令直接抛错。"""
        if self.broken:
            raise RuntimeError("fake redis is down")

    def ping(self) -> bool:
        self._guard()
        return True

    def hget(self, name: str, key: str) -> str | None:
        self._guard()
        return self.hashes.get(name, {}).get(key)

    def hgetall(self, name: str) -> dict[str, str]:
        self._guard()
        return dict(self.hashes.get(name, {}))

    def hset(
        self,
        name: str,
        key: str | None = None,
        value: str | None = None,
        mapping: dict[str, str] | None = None,
    ) -> int:
        self._guard()
        bucket = self.hashes.setdefault(name, {})
        added = 0
        if key is not None:
            if key not in bucket:
                added += 1
            bucket[key] = value if value is not None else ""
        for field, field_value in (mapping or {}).items():
            if field not in bucket:
                added += 1
            bucket[field] = field_value
        return added

    def hdel(self, name: str, *keys: str) -> int:
        self._guard()
        bucket = self.hashes.get(name, {})
        removed = 0
        for key in keys:
            if key in bucket:
                del bucket[key]
                removed += 1
        return removed

    def delete(self, *names: str) -> int:
        self._guard()
        return sum(1 for name in names if self.hashes.pop(name, None) is not None)

    def expire(self, name: str, time: int) -> bool:
        self._guard()
        self.expires.append((name, time))
        return True


@pytest.fixture()
def fake_redis() -> FakeRedis:
    """每个用例独立的内存 Redis。"""
    return FakeRedis()


@pytest.fixture()
def redis_override(fake_redis: FakeRedis) -> Iterator[FakeRedis]:
    """把提示词缓存依赖换成内存实现（不连真实 Redis）。"""
    app.dependency_overrides[get_redis] = lambda: fake_redis
    try:
        yield fake_redis
    finally:
        app.dependency_overrides.pop(get_redis, None)
