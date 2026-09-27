"""提示词模板版本管理测试：版本自增、公共/私有分组、Redis 缓存、拼接、一键回滚。

不连真实 Redis：用内存 FakeRedis 模拟 Hash；另外覆盖「Redis 不可用则全程回退数据库」。
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.db.prompt_template_repository import PromptTemplateRepository
from app.prompts.prompt_template_manager import (
    COMMON_AGENT_KEY,
    SOURCE_DB,
    SOURCE_REDIS,
    PromptTemplateInvalidError,
    PromptTemplateManager,
    PromptTemplateNotFoundError,
    cache_field,
)
from tests.conftest import FakeRedis

COMMON_SCENE = "chat"


@pytest.fixture()
def manager(db_session: Session, fake_redis: FakeRedis) -> PromptTemplateManager:
    """带内存 Redis 的模板管理器（仓储挂在测试会话上）。"""
    return PromptTemplateManager(PromptTemplateRepository(db_session), fake_redis)


@pytest.fixture()
def db_only_manager(db_session: Session) -> PromptTemplateManager:
    """未启用 Redis 的管理器：所有读取都应直达数据库。"""
    return PromptTemplateManager(PromptTemplateRepository(db_session), None)


# ---------------------------------------------------------------------------
# 版本号：同组自增、跨组独立
# ---------------------------------------------------------------------------
def test_next_version_starts_at_one_and_increases_per_group(manager: PromptTemplateManager) -> None:
    assert manager.next_version("tutor", COMMON_SCENE) == 1

    manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="私有 v1")
    manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="私有 v2")
    manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="公共 v1")

    assert manager.next_version("tutor", COMMON_SCENE) == 3
    assert manager.next_version(None, COMMON_SCENE) == 2          # 公共模板独立计数
    assert manager.next_version("quiz_coach", COMMON_SCENE) == 1  # 换智能体重新从 1 开始
    assert manager.next_version("tutor", "interview") == 1        # 换场景重新从 1 开始


def test_template_type_is_inferred_from_agent_name(manager: PromptTemplateManager) -> None:
    private = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="私有")
    common = manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="公共")

    assert (private.template_type, private.agent_name) == (1, "tutor")
    assert (common.template_type, common.agent_name) == (2, None)


def test_public_and_private_must_match_their_agent_name(manager: PromptTemplateManager) -> None:
    """公共模板不能挂智能体，私有模板必须挂智能体。"""
    with pytest.raises(PromptTemplateInvalidError):
        manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="x", template_type=2)
    with pytest.raises(PromptTemplateInvalidError):
        manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="x", template_type=1)


def test_variables_are_extracted_from_placeholders(manager: PromptTemplateManager) -> None:
    view = manager.create_version(
        agent_name="tutor",
        scene=COMMON_SCENE,
        template_content="目标岗位 {target_job}，薄弱点 {weak_topics}，再写一次 {target_job}",
    )

    assert sorted(view.variables) == ["target_job", "weak_topics"]  # 去重 + 自动提取
    assert view.variables == manager.get_template("tutor", COMMON_SCENE).variables  # 落库后仍一致

    explicit = manager.create_version(
        agent_name="tutor",
        scene=COMMON_SCENE,
        template_content="正文 {target_job}",
        variables=["target_job", "weak_topics"],
    )
    assert explicit.variables == ["target_job", "weak_topics"]  # 显式声明优先

    assert manager.next_version("tutor", COMMON_SCENE) == 3


def test_create_version_rejects_empty_content(manager: PromptTemplateManager) -> None:
    with pytest.raises(PromptTemplateInvalidError):
        manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="   ")


# ---------------------------------------------------------------------------
# 生效版本：同组仅一条 is_active=1
# ---------------------------------------------------------------------------
def test_only_latest_version_stays_active_in_a_group(manager: PromptTemplateManager, db_session: Session) -> None:
    first = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1")
    second = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v2")
    other_agent = manager.create_version(agent_name="quiz_coach", scene=COMMON_SCENE, template_content="别的智能体")

    versions = manager.list_versions("tutor", COMMON_SCENE)
    assert [(view.version, view.is_active) for view in versions] == [(2, True), (1, False)]
    assert manager.get_template("tutor", COMMON_SCENE).version == 2
    assert manager.get_template("quiz_coach", COMMON_SCENE).version == other_agent.version
    # 换组不影响：另一个智能体的模板仍然生效
    assert [view.is_active for view in manager.list_versions("quiz_coach", COMMON_SCENE)] == [True]
    assert first.version == 1 and second.version == 2


def test_create_version_can_skip_activation(manager: PromptTemplateManager) -> None:
    active = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1")
    draft = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v2", activate=False)

    assert draft.is_active is False
    assert manager.get_template("tutor", COMMON_SCENE).version == active.version == 1


# ---------------------------------------------------------------------------
# Redis 缓存
# ---------------------------------------------------------------------------
def test_warm_cache_writes_active_templates_into_one_hash(manager: PromptTemplateManager, fake_redis: FakeRedis) -> None:
    manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="私有模板")
    manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="公共模板")
    manager.create_version(agent_name="study_planner", scene="plan", template_content="规划", activate=False)

    written = manager.warm_cache()

    assert written == 2
    bucket = fake_redis.hashes[manager.cache_key]
    assert set(bucket) == {cache_field("tutor", COMMON_SCENE), cache_field(None, COMMON_SCENE)}
    assert cache_field(None, COMMON_SCENE) == f"{COMMON_AGENT_KEY}:{COMMON_SCENE}"
    assert f"{COMMON_AGENT_KEY}:{COMMON_SCENE}" in bucket
    # 预热会带上 TTL，避免缓存与数据库长期不一致
    assert (manager.cache_key, manager.cache_ttl) in fake_redis.expires


def test_warm_cache_without_redis_is_a_noop(db_only_manager: PromptTemplateManager) -> None:
    db_only_manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1")

    assert db_only_manager.warm_cache() == 0  # 未启用 Redis：只记日志，不抛错


def test_get_template_prefers_cache_over_database(manager: PromptTemplateManager, db_session: Session) -> None:
    view = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="缓存中的版本")

    # 绕过服务层直接改库，模拟「缓存尚未失效」时读到旧内容
    row = manager.templates.get(view.template_id)
    row.template_content = "数据库里的新内容"
    db_session.flush()

    cached = manager.get_template("tutor", COMMON_SCENE)
    assert cached.content == "缓存中的版本"
    assert cached.source == SOURCE_REDIS
    assert cached.cache_field == cache_field("tutor", COMMON_SCENE)


def test_get_template_falls_back_to_database_and_backfills_cache(
    manager: PromptTemplateManager, db_session: Session, fake_redis: FakeRedis
) -> None:
    repository = PromptTemplateRepository(db_session)
    repository.create(
        agent_name="tutor",
        scene=COMMON_SCENE,
        template_type=1,
        template_content="只有数据库里有",
        variables='["target_job"]',
        version=1,
        is_active=1,
        description=None,
    )
    db_session.flush()

    view = manager.get_template("tutor", COMMON_SCENE)

    assert view is not None and view.source == SOURCE_DB
    # 回写缓存：下次直接从 Redis 命中
    assert fake_redis.hget(manager.cache_key, cache_field("tutor", COMMON_SCENE))


def test_get_template_returns_none_when_no_active_version(manager: PromptTemplateManager) -> None:
    manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1", activate=False)

    assert manager.get_template("tutor", COMMON_SCENE) is None


def test_broken_redis_degrades_to_database(manager: PromptTemplateManager, fake_redis: FakeRedis) -> None:
    """Redis 挂掉只影响缓存命中率，读写都要能继续走数据库。"""
    view = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1")
    fake_redis.broken = True

    assert manager.read_cache("tutor", COMMON_SCENE) is None
    assert manager.warm_cache() == 0
    assert manager.get_template("tutor", COMMON_SCENE).content == "v1"   # 回退数据库
    manager.invalidate("tutor", COMMON_SCENE)                            # 不抛异常
    assert manager.rollback(view.template_id).is_active is True


def test_corrupted_cache_payload_is_ignored(manager: PromptTemplateManager, fake_redis: FakeRedis) -> None:
    fake_redis.hset(manager.cache_key, cache_field("tutor", COMMON_SCENE), "{不是 JSON")

    assert manager.read_cache("tutor", COMMON_SCENE) is None


def test_invalidate_removes_single_field_or_whole_hash(manager: PromptTemplateManager, fake_redis: FakeRedis) -> None:
    manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1")
    manager.create_version(agent_name="quiz_coach", scene=COMMON_SCENE, template_content="v1")

    manager.invalidate("tutor", COMMON_SCENE)
    assert fake_redis.hget(manager.cache_key, cache_field("quiz_coach", COMMON_SCENE))
    assert fake_redis.hget(manager.cache_key, cache_field("tutor", COMMON_SCENE)) is None

    manager.invalidate(None)
    assert manager.cache_key not in fake_redis.hashes


# ---------------------------------------------------------------------------
# 公共 + 私有拼接
# ---------------------------------------------------------------------------
def test_compose_joins_common_then_private_and_merges_variables(manager: PromptTemplateManager) -> None:
    manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="公共：{scene} 通用规则")
    manager.create_version(
        agent_name="tutor",
        scene=COMMON_SCENE,
        template_content="私有：目标岗位 {target_job}",
    )

    composed = manager.compose("tutor", COMMON_SCENE)

    assert composed.content == "公共：{scene} 通用规则\n\n私有：目标岗位 {target_job}"
    assert composed.variables == ["scene", "target_job"]
    assert composed.common.template_type == 2 and composed.private.template_type == 1
    assert composed.version_label == "common=v1,private=v1"


def test_compose_keeps_common_only_template(manager: PromptTemplateManager) -> None:
    manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="只有公共模板")

    composed = manager.compose("tutor", COMMON_SCENE)

    assert composed.content == "只有公共模板"
    assert composed.private is None
    assert composed.version_label == "common=v1,private=v0"


def test_compose_raises_when_group_has_no_template(manager: PromptTemplateManager) -> None:
    with pytest.raises(PromptTemplateNotFoundError) as excinfo:
        manager.compose("tutor", "resume")

    assert excinfo.value.http_status == 404
    assert excinfo.value.code == "PROMPT_TEMPLATE_NOT_FOUND"


# ---------------------------------------------------------------------------
# 一键回滚
# ---------------------------------------------------------------------------
def test_rollback_switches_active_version_and_refreshes_cache(
    manager: PromptTemplateManager, fake_redis: FakeRedis
) -> None:
    v1 = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v1 内容")
    manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="v2 内容")

    rolled_back = manager.rollback(v1.template_id)

    assert (rolled_back.version, rolled_back.is_active) == (1, True)
    versions = manager.list_versions("tutor", COMMON_SCENE)
    assert [(view.version, view.is_active) for view in versions] == [(2, False), (1, True)]
    # 缓存里的生效版本同步切换（否则读路径还会拿 v2）
    assert manager.read_cache("tutor", COMMON_SCENE).version == 1
    assert manager.read_cache("tutor", COMMON_SCENE).content == "v1 内容"
    assert fake_redis.hget(manager.cache_key, cache_field("tutor", COMMON_SCENE))


def test_rollback_only_touches_its_own_group(manager: PromptTemplateManager) -> None:
    common_v1 = manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="公共 v1")
    manager.create_version(agent_name=None, scene=COMMON_SCENE, template_content="公共 v2")
    private = manager.create_version(agent_name="tutor", scene=COMMON_SCENE, template_content="私有 v1")

    manager.rollback(common_v1.template_id)

    assert manager.get_template(None, COMMON_SCENE).version == 1
    assert manager.get_template("tutor", COMMON_SCENE).version == private.version  # 私有模板不受影响


def test_rollback_unknown_template_raises_not_found(manager: PromptTemplateManager) -> None:
    with pytest.raises(PromptTemplateNotFoundError):
        manager.rollback(9999)
