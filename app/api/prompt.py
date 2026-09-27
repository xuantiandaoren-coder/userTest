"""路由层：提示词模板版本管理接口。

四个端点：

- `GET  /prompt/templates/{agent_name}/{scene}` 查看某智能体 + 场景的生效模板（公共 + 私有 + 历史版本）
- `POST /prompt/templates`                      新建版本（自动计算下一版本号，可同时切换生效）
- `POST /prompt/rollback`                       一键回滚到指定版本
- `GET  /prompt/config/agents`                  智能体映射 / provider / 缓存配置

路径里的 `__common__` 是公共模板的固定别名（agent_name=NULL 没法写进 URL）。
全部需要登录：模板属于服务端配置，不对匿名用户开放。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, status

from app.api.deps import CurrentUserDep, PromptManagerDep, RedisDep
from app.core.config import DEFAULT_AGENT, DEFAULT_SCENE, AGENT_CONFIG, settings
from app.llm.llm import available_providers
from app.prompts.prompt_template_manager import (
    COMMON_AGENT_KEY,
    TEMPLATE_TYPE_LABELS,
    PromptTemplateNotFoundError,
    TemplateView,
    cache_field,
)
from app.schemas.prompt import (
    AgentConfigPublic,
    PromptConfigResponse,
    PromptRollbackRequest,
    PromptRollbackResult,
    PromptTemplateCreate,
    PromptTemplateDetail,
    PromptTemplatePublic,
    ProviderPublic,
)

router = APIRouter(prefix="/prompt", tags=["prompt"])

AgentNamePath = Annotated[str, Path(description="智能体名；公共模板用 __common__")]
ScenePath = Annotated[str, Path(description="场景：workflow / resume / quiz / interview / evaluation / study / note")]
# 公共模板没有 agent_name，URL 里用 __common__ 代指
COMMON_ALIASES = frozenset({COMMON_AGENT_KEY, "__common__", "common"})


@router.get("/templates/{agent_name}/{scene}", response_model=PromptTemplateDetail, summary="查看生效模板")
def get_prompt_template(
    current_user: CurrentUserDep,
    manager: PromptManagerDep,
    agent_name: AgentNamePath,
    scene: ScenePath,
) -> PromptTemplateDetail:
    """返回该 (agent, scene) 的公共模板、私有模板与全部历史版本。"""
    agent = _agent_from_path(agent_name)
    common = manager.get_common_template(scene)
    private = manager.get_private_template(agent, scene) if agent else None
    versions = manager.list_versions(agent, scene)
    if not versions and common is None and private is None:
        raise PromptTemplateNotFoundError(detail=f"agent={agent or COMMON_AGENT_KEY} scene={scene} 无任何版本")

    return PromptTemplateDetail(
        agent_name=agent or COMMON_AGENT_KEY,
        scene=scene,
        cache_key=manager.cache_key,
        common_field=cache_field(None, scene),
        private_field=cache_field(agent, scene) if agent else "",
        common=_to_public(common) if common else None,
        private=_to_public(private) if private else None,
        composed_variables=_merge_variables(common, private),
        versions=[_to_public(view) for view in versions],
    )


@router.post(
    "/templates",
    response_model=PromptTemplatePublic,
    status_code=status.HTTP_201_CREATED,
    summary="新建模板版本",
)
def create_prompt_template(
    payload: PromptTemplateCreate,
    current_user: CurrentUserDep,
    manager: PromptManagerDep,
) -> PromptTemplatePublic:
    """新建版本：版本号自动计算（同组自增），activate=true 时立即生效并下线旧版本。"""
    view = manager.create_version(
        agent_name=payload.agent_name,
        scene=payload.scene,
        template_content=payload.template_content,
        variables=payload.variables,
        description=payload.description,
        activate=payload.activate,
        template_type=payload.template_type,
    )
    return _to_public(view)


@router.post("/rollback", response_model=PromptRollbackResult, summary="一键回滚到指定版本")
def rollback_prompt_template(
    payload: PromptRollbackRequest,
    current_user: CurrentUserDep,
    manager: PromptManagerDep,
) -> PromptRollbackResult:
    """把指定版本切为生效版本：同组其它版本自动置 is_active=0，并刷新 Redis 缓存。"""
    target = manager.templates.get(payload.template_id)
    if target is None:
        raise PromptTemplateNotFoundError(detail=f"template_id={payload.template_id}")

    # 先记下回滚前生效的其它版本号，回滚结果里回带，便于前端提示「v2 -> v5，下线 v5」之类
    deactivated = [
        view.version
        for view in manager.list_versions(target.agent_name, target.scene)
        if view.is_active and view.version != target.version
    ]
    view = manager.rollback(payload.template_id)
    return PromptRollbackResult(
        agent_name=view.agent_name,
        scene=view.scene,
        template_type=view.template_type,
        active_version=view.version,
        deactivated_versions=sorted(deactivated, reverse=True),
    )


@router.get("/config/agents", response_model=PromptConfigResponse, summary="智能体与模型配置")
def get_agent_config(
    current_user: CurrentUserDep,
    manager: PromptManagerDep,
    redis: RedisDep,
) -> PromptConfigResponse:
    """返回 7 个智能体映射、模型 provider 清单与提示词缓存配置。"""
    active_keys = {
        cache_field(template.agent_name, template.scene) for template in manager.templates.list_active()
    }
    agents: list[AgentConfigPublic] = []
    for name, setting in AGENT_CONFIG.items():
        ready = (
            cache_field(name, setting.scene) in active_keys
            or cache_field(None, setting.scene) in active_keys
        )
        agents.append(
            AgentConfigPublic(
                agent_name=name,
                label=setting.label,
                scene=setting.scene,
                description=setting.description,
                select_model=setting.select_model,
                provider=setting.provider,
                template_ready=ready,
            )
        )

    return PromptConfigResponse(
        default_agent=DEFAULT_AGENT,
        default_scene=DEFAULT_SCENE,
        current_provider=settings.llm_provider,
        current_model=settings.llm_model or "",
        cache_key=manager.cache_key,
        redis_enabled=settings.redis_enabled and redis is not None,
        agents=agents,
        providers=[ProviderPublic(**item) for item in available_providers()],
        scenes=list(manager.templates.list_scenes()),
    )


def _agent_from_path(agent_name: str) -> str | None:
    """把 URL 里的 __common__ 还原成 None（公共模板 agent_name 为 NULL）。"""
    cleaned = (agent_name or "").strip()
    return None if cleaned.lower() in COMMON_ALIASES else cleaned


def _to_public(view: TemplateView) -> PromptTemplatePublic:
    """只读视图 -> 接口响应模型。"""
    return PromptTemplatePublic(
        id=view.template_id or 0,
        agent_name=view.agent_name,
        scene=view.scene,
        template_type=view.template_type,
        template_type_label=TEMPLATE_TYPE_LABELS.get(view.template_type, "未知"),
        version=view.version,
        is_active=view.is_active,
        template_content=view.content,
        variables=view.variables,
        description=view.description,
        created_at=view.created_at,
    )


def _merge_variables(common: TemplateView | None, private: TemplateView | None) -> list[str]:
    """公共 + 私有模板的变量名合并（去重、保持顺序）。"""
    merged: list[str] = []
    for view in (common, private):
        for name in view.variables if view else []:
            if name not in merged:
                merged.append(name)
    return merged
