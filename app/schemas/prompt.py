"""校验层：提示词模板版本管理相关模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplateCreate(BaseModel):
    """新建一个模板版本。

    - agent_name 留空 = 公共模板（template_type=2）
    - template_type 不传则按 agent_name 推断（有值=1 私有，为空=2 公共）
    - variables 不传则从 template_content 里自动提取 `{变量}` 占位符
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    scene: str = Field(min_length=1, max_length=64, description="场景：workflow / resume / quiz / interview / evaluation / study / note")
    template_content: str = Field(min_length=1, description="模板正文，用 {变量} 占位")
    agent_name: str | None = Field(default=None, max_length=64, description="智能体名；公共模板留空")
    template_type: int | None = Field(default=None, ge=1, le=2, description="1=私有，2=公共；留空按 agent_name 推断")
    variables: list[str] | None = Field(default=None, description="变量名列表；留空自动提取")
    description: str | None = Field(default=None, max_length=255, description="版本说明")
    activate: bool = Field(default=True, description="是否立即生效（同组其它版本自动下线）")


class PromptTemplatePublic(BaseModel):
    """模板版本对外响应。"""

    id: int
    agent_name: str | None
    scene: str
    template_type: int
    template_type_label: str
    version: int
    is_active: bool
    template_content: str
    variables: list[str]
    description: str | None = None
    created_at: int | None = None


class PromptTemplateDetail(BaseModel):
    """某 (agent, scene) 的生效模板详情：公共 + 私有 + 全部历史版本。"""

    agent_name: str
    scene: str
    cache_key: str = Field(description="Redis Hash 的 key")
    common_field: str = Field(description="公共模板在 Hash 里的 field，形如 __common__:{scene}")
    private_field: str = Field(description="私有模板在 Hash 里的 field，形如 {agent}:{scene}")
    common: PromptTemplatePublic | None = None
    private: PromptTemplatePublic | None = None
    composed_variables: list[str] = Field(default_factory=list, description="公共 + 私有合并后的变量名")
    versions: list[PromptTemplatePublic] = Field(default_factory=list, description="该组历史版本（新 -> 旧）")


class PromptRollbackRequest(BaseModel):
    """一键回滚：把指定版本切为生效版本。"""

    template_id: int = Field(gt=0, description="要回滚到的模板版本 id")


class PromptRollbackResult(BaseModel):
    """回滚结果：切到哪个版本、下掉了哪些版本。"""

    agent_name: str | None
    scene: str
    template_type: int
    active_version: int
    deactivated_versions: list[int] = Field(default_factory=list, description="被下线的版本号")


class AgentConfigPublic(BaseModel):
    """单个智能体的配置（不含提示词正文）。"""

    agent_name: str
    label: str
    scene: str
    description: str
    select_model: int | None = None
    provider: str | None = Field(default=None, description="该智能体指定的模型 provider；null 表示跟随全局")
    template_ready: bool = Field(default=False, description="该 (agent, scene) 是否已有生效模板")


class ProviderPublic(BaseModel):
    """模型 provider 配置状态。"""

    provider: str
    label: str
    default_model: str
    base_url: str | None = None
    api_key_env: str = ""
    configured: bool = False


class PromptConfigResponse(BaseModel):
    """GET /prompt/config/agents 响应：智能体映射 + 模型 provider + 缓存配置。"""

    default_agent: str
    default_scene: str
    current_provider: str
    current_model: str
    cache_key: str
    redis_enabled: bool
    agents: list[AgentConfigPublic]
    providers: list[ProviderPublic]
    scenes: list[str] = Field(default_factory=list, description="数据库里已出现过的场景")


class PromptTemplateInput(BaseModel):
    """提示词变量注入调试用的通用载荷（保留给扩展端点）。"""

    variables: dict[str, Any] = Field(default_factory=dict)
