"""提示词模板管理接口测试：查看生效模板、新建版本、一键回滚、智能体配置。

走完整链路（路由 -> 服务 -> 仓储 -> 内存 Redis），只把数据库换成 SQLite、Redis 换成内存实现。
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import AGENT_CONFIG
from tests.conftest import FakeRedis

PROMPT = "/prompt"
PASSWORD = "HZq7mK2p"


async def _login(client: httpx.AsyncClient, username: str) -> dict[str, str]:
    """注册并登录，返回 Authorization 头。"""
    await client.post("/auth/register", json={"username": username, "password": PASSWORD})
    tokens = (await client.post("/auth/login", json={"username": username, "password": PASSWORD})).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


async def _create_template(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    *,
    scene: str = "chat",
    content: str = "模板正文",
    agent_name: str | None = "tutor",
    **extra: object,
) -> dict:
    payload: dict[str, object] = {"scene": scene, "template_content": content}
    if agent_name is not None:
        payload["agent_name"] = agent_name
    payload.update(extra)
    response = await client.post(f"{PROMPT}/templates", json=payload, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_prompt_endpoints_require_authentication(httpx_client: httpx.AsyncClient) -> None:
    """模板属于服务端配置，匿名一律 401。"""
    assert (await httpx_client.get(f"{PROMPT}/templates/tutor/chat")).status_code == 401
    assert (await httpx_client.get(f"{PROMPT}/config/agents")).status_code == 401
    assert (await httpx_client.post(f"{PROMPT}/templates", json={"scene": "chat", "template_content": "x"})).status_code == 401
    assert (await httpx_client.post(f"{PROMPT}/rollback", json={"template_id": 1})).status_code == 401


# ---------------------------------------------------------------------------
# 新建版本
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_create_template_extracts_variables_and_activates_it(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_create")

    created = await _create_template(
        httpx_client,
        headers,
        content="目标岗位 {target_job}，薄弱点 {weak_topics}，再写一次 {target_job}",
        description="首个版本",
    )

    assert created["version"] == 1
    assert created["is_active"] is True
    assert created["template_type"] == 1
    assert created["template_type_label"] == "私有"
    assert created["agent_name"] == "tutor"
    assert created["description"] == "首个版本"
    assert created["variables"] == ["target_job", "weak_topics"]  # 自动提取 + 去重
    assert created["created_at"]


@pytest.mark.anyio
async def test_common_template_has_null_agent_name(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_common")

    created = await _create_template(httpx_client, headers, agent_name=None, content="公共规则 {scene}")

    assert created["agent_name"] is None
    assert created["template_type"] == 2
    assert created["template_type_label"] == "公共"


@pytest.mark.anyio
async def test_version_numbers_are_independent_per_group(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_versions")

    first = await _create_template(httpx_client, headers, content="tutor v1")
    second = await _create_template(httpx_client, headers, content="tutor v2")
    other_agent = await _create_template(httpx_client, headers, agent_name="quiz_coach", content="coach v1")
    other_scene = await _create_template(httpx_client, headers, agent_name="tutor", scene="interview", content="interview v1")

    assert (first["version"], second["version"]) == (1, 2)
    assert other_agent["version"] == 1   # 换智能体重新计数
    assert other_scene["version"] == 1   # 换场景重新计数

    detail = (await httpx_client.get(f"{PROMPT}/templates/tutor/chat", headers=headers)).json()
    assert [item["version"] for item in detail["versions"]] == [2, 1]  # 新 -> 旧
    assert [item["is_active"] for item in detail["versions"]] == [True, False]


@pytest.mark.anyio
async def test_create_template_validates_input(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_invalid")

    public_with_agent = await httpx_client.post(
        f"{PROMPT}/templates",
        json={"scene": "chat", "template_content": "x", "agent_name": "tutor", "template_type": 2},
        headers=headers,
    )
    private_without_agent = await httpx_client.post(
        f"{PROMPT}/templates",
        json={"scene": "chat", "template_content": "x", "template_type": 1},
        headers=headers,
    )
    empty_content = await httpx_client.post(
        f"{PROMPT}/templates", json={"scene": "chat", "template_content": ""}, headers=headers
    )

    assert public_with_agent.status_code == 422
    assert public_with_agent.json()["code"] == "PROMPT_TEMPLATE_INVALID"
    assert private_without_agent.status_code == 422
    assert empty_content.status_code == 422


# ---------------------------------------------------------------------------
# 查看生效模板（公共 + 私有 + 历史版本）
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_get_template_detail_composes_common_and_private(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_detail")
    await _create_template(httpx_client, headers, agent_name=None, content="公共规则 {scene} {current_date}")
    private = await _create_template(httpx_client, headers, content="私有指令 {target_job}")

    response = await httpx_client.get(f"{PROMPT}/templates/tutor/chat", headers=headers)

    assert response.status_code == 200
    detail = response.json()
    assert detail["agent_name"] == "tutor"
    assert detail["common"]["template_type"] == 2
    assert detail["private"]["id"] == private["id"]
    # 自动提取的变量名按字母序落库，合并时保持「公共在前、私有在后」
    assert detail["composed_variables"] == ["current_date", "scene", "target_job"]
    assert detail["cache_key"] == "prompt:templates:active"
    assert detail["common_field"] == "__common__:chat"
    assert detail["private_field"] == "tutor:chat"
    assert [item["version"] for item in detail["versions"]] == [1]


@pytest.mark.anyio
async def test_get_template_accepts_common_alias_and_reports_404(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_alias")
    await _create_template(httpx_client, headers, agent_name=None, content="公共规则")

    common = (await httpx_client.get(f"{PROMPT}/templates/__common__/chat", headers=headers)).json()
    missing = await httpx_client.get(f"{PROMPT}/templates/tutor/resume", headers=headers)

    assert common["agent_name"] == "__common__"   # URL 里的别名还原成公共模板
    assert common["private"] is None
    assert common["private_field"] == ""
    assert missing.status_code == 404
    assert missing.json()["code"] == "PROMPT_TEMPLATE_NOT_FOUND"


@pytest.mark.anyio
async def test_active_template_is_served_from_cache_and_refreshed_by_rollback(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_cache")
    v1 = await _create_template(httpx_client, headers, content="v1 内容")
    v2 = await _create_template(httpx_client, headers, content="v2 内容")

    # 新建生效版本后缓存里已经是 v2
    assert redis_override.hget("prompt:templates:active", "tutor:chat")
    first = (await httpx_client.get(f"{PROMPT}/templates/tutor/chat", headers=headers)).json()
    assert first["private"]["id"] == v2["id"]

    rolled = await httpx_client.post(f"{PROMPT}/rollback", json={"template_id": v1["id"]}, headers=headers)

    assert rolled.status_code == 200
    assert rolled.json() == {
        "agent_name": "tutor",
        "scene": "chat",
        "template_type": 1,
        "active_version": 1,
        "deactivated_versions": [2],
    }
    # 缓存同步刷新：读路径立即拿到回滚后的版本
    second = (await httpx_client.get(f"{PROMPT}/templates/tutor/chat", headers=headers)).json()
    assert second["private"]["id"] == v1["id"]
    assert second["private"]["is_active"] is True
    assert "v1 内容" in second["private"]["template_content"]
    assert redis_override.hget("prompt:templates:active", "tutor:chat")


@pytest.mark.anyio
async def test_rollback_unknown_template_returns_404(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_rollback_404")

    response = await httpx_client.post(f"{PROMPT}/rollback", json={"template_id": 999}, headers=headers)

    assert response.status_code == 404
    assert response.json()["code"] == "PROMPT_TEMPLATE_NOT_FOUND"


# ---------------------------------------------------------------------------
# 智能体 / 模型 / 缓存配置
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_agents_config_lists_all_agents_and_cache_settings(
    db_override: None, redis_override: FakeRedis, httpx_client: httpx.AsyncClient
) -> None:
    headers = await _login(httpx_client, "prompt_config")

    before = (await httpx_client.get(f"{PROMPT}/config/agents", headers=headers)).json()

    assert before["default_agent"] == "flow_controller"
    assert before["default_scene"] == "workflow"
    assert before["cache_key"] == "prompt:templates:active"  # 缓存 Key 与要求一致
    assert before["redis_enabled"] is True
    assert [item["agent_name"] for item in before["agents"]] == list(AGENT_CONFIG)
    assert len(before["agents"]) == 7
    assert {item["agent_name"]: item["label"] for item in before["agents"]} == {
        "flow_controller": "流程总控Agent",
        "resume_parser": "资料&简历解析Agent",
        "quiz_generate_workflow": "出题题库Agent",
        "interview_host": "面试主考官Agent",
        "interview_evaluator": "面试评测Agent",
        "difficulty_learner": "难点学习Agent",
        "note_archiver": "笔记归档Agent",
    }
    assert {item["agent_name"]: item["scene"] for item in before["agents"]} == {
        "flow_controller": "workflow",
        "resume_parser": "resume",
        "quiz_generate_workflow": "quiz",
        "interview_host": "interview",
        "interview_evaluator": "evaluation",
        "difficulty_learner": "study",
        "note_archiver": "note",
    }
    assert all(item["template_ready"] is False for item in before["agents"])
    assert {item["provider"] for item in before["providers"]} == {
        "deepseek", "openai", "dashscope", "moonshot", "zhipu", "ollama",
    }
    assert before["scenes"] == []

    await _create_template(
        httpx_client, headers, agent_name="difficulty_learner", scene="study", content="难点学习私有模板"
    )

    after = (await httpx_client.get(f"{PROMPT}/config/agents", headers=headers)).json()
    readiness = {item["agent_name"]: item["template_ready"] for item in after["agents"]}
    assert readiness["difficulty_learner"] is True        # scene=study，模板齐全
    assert readiness["quiz_generate_workflow"] is False   # 自己的 scene=quiz 还没有模板
    assert after["scenes"] == ["study"]
