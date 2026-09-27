"""提示词模板管理：版本计算、Redis 缓存、公共 / 私有拼接、一键回滚。

存储模型（prompt_templates 表）：
- 私有模板：agent_name 有值，template_type=1
- 公共模板：agent_name=NULL，template_type=2
- 同一 (agent_name, scene) 下 version 独立自增，且只有一条 is_active=1

缓存模型（Redis Hash，key = `settings.prompt_cache_key`，默认 `prompt:templates:active`）：
- field：私有 `{agent}:{scene}`，公共 `__common__:{scene}`
- value：一行 JSON（version / content / variables / description / created_at）
- 读路径是 cache-aside：先读 Redis，未命中查库并回写；Redis 不可用则全程走库
- 回滚 / 新建版本后立即刷新对应 field，保证「生效版本」与缓存一致
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.core.config import DEFAULT_AGENT, settings
from app.core.exceptions import BusinessError
from app.core.redis_client import RedisLike
from app.db.models import PromptTemplate
from app.db.prompt_template_repository import PromptTemplateRepository
from app.prompts.injector import PLACEHOLDER_PATTERN

logger = logging.getLogger("app.prompts")

TEMPLATE_TYPE_PRIVATE = 1
TEMPLATE_TYPE_COMMON = 2
TEMPLATE_TYPE_LABELS = {TEMPLATE_TYPE_PRIVATE: "私有", TEMPLATE_TYPE_COMMON: "公共"}

# 公共模板在 Redis field / 日志里的固定前缀（agent_name 为 NULL，需要一个稳定的键）
COMMON_AGENT_KEY = "__common__"

MAX_TEMPLATE_CHARS = 200_000  # MEDIUMTEXT 上限 16MB，这里只做「明显异常」的保护
SOURCE_REDIS = "redis"
SOURCE_DB = "db"
SOURCE_EMPTY = "empty"


class PromptTemplateNotFoundError(BusinessError):
    """业务异常：模板不存在 / 指定分组没有可用模板。"""

    code = "PROMPT_TEMPLATE_NOT_FOUND"
    http_status = 404
    message = "提示词模板不存在"


class PromptTemplateInvalidError(BusinessError):
    """业务异常：模板内容或类型不合法。"""

    code = "PROMPT_TEMPLATE_INVALID"
    http_status = 422
    message = "提示词模板不合法"


@dataclass(frozen=True)
class TemplateView:
    """模板的只读视图：脱离 ORM 会话也能安全传递（流式接口跨线程 / 跨会话要用）。"""

    agent_name: str | None
    scene: str
    template_type: int
    version: int
    content: str
    variables: list[str] = field(default_factory=list)
    template_id: int | None = None
    description: str | None = None
    created_at: int | None = None
    is_active: bool = True
    source: str = SOURCE_DB

    @property
    def is_common(self) -> bool:
        """是否公共模板。"""
        return self.template_type == TEMPLATE_TYPE_COMMON

    @property
    def cache_field(self) -> str:
        """本模板在 Redis Hash 里的 field：`{agent}:{scene}` / `__common__:{scene}`。"""
        return cache_field(self.agent_name, self.scene)

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（接口响应 / 缓存写入共用）。"""
        return {
            "template_id": self.template_id,
            "agent_name": self.agent_name,
            "scene": self.scene,
            "template_type": self.template_type,
            "version": self.version,
            "template_content": self.content,
            "variables": list(self.variables),
            "description": self.description,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


@dataclass(frozen=True)
class ComposedPrompt:
    """公共模板 + 私有模板拼接结果（提示词层的输入）。"""

    agent_name: str
    scene: str
    content: str
    variables: list[str]
    common: TemplateView | None = None
    private: TemplateView | None = None

    @property
    def version_label(self) -> str:
        """形如 `common=v2,private=v5` 的版本标签，便于日志 / 响应排查。"""
        common_version = self.common.version if self.common else 0
        private_version = self.private.version if self.private else 0
        return f"common=v{common_version},private=v{private_version}"


def cache_field(agent_name: str | None, scene: str) -> str:
    """私有模板 `{agent}:{scene}`；公共模板 `__common__:{scene}`。"""
    return f"{agent_name or COMMON_AGENT_KEY}:{scene}"


def parse_variables(raw: str | None) -> list[str]:
    """解析 variables 列（JSON 数组字符串）；兼容逗号分隔的老数据。"""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("["):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("variables 列不是合法 JSON，按分隔符解析：%r", raw)
        else:
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
    return [item.strip() for item in text.replace("，", ",").split(",") if item.strip()]


def dump_variables(names: Sequence[str] | None) -> str:
    """变量名列表 -> JSON 数组字符串（列长度 512，超长直接报错而不是被截断）。"""
    payload = json.dumps(list(names or []), ensure_ascii=False)
    if len(payload) > 512:
        raise PromptTemplateInvalidError(detail=f"变量列表过长（{len(payload)} > 512 字符）")
    return payload


class PromptTemplateManager:
    """提示词模板服务：版本管理 + Redis 缓存 + 公共 / 私有拼接 + 一键回滚。"""

    def __init__(
        self,
        templates: PromptTemplateRepository,
        redis: RedisLike | None = None,
        *,
        cache_key: str | None = None,
        cache_ttl: int | None = None,
    ) -> None:
        self.templates = templates
        self.redis = redis
        self.cache_key = cache_key or settings.prompt_cache_key
        self.cache_ttl = settings.prompt_cache_ttl if cache_ttl is None else cache_ttl

    # ------------------------------------------------------------------
    # 版本
    # ------------------------------------------------------------------
    def next_version(self, agent_name: str | None, scene: str) -> int:
        """计算下一版本号：同一 (agent_name, scene) 内自增，无历史时从 1 开始。"""
        return self.templates.next_version(_normalize_agent(agent_name), scene)

    def create_version(
        self,
        *,
        agent_name: str | None,
        scene: str,
        template_content: str,
        variables: Sequence[str] | None = None,
        description: str | None = None,
        activate: bool = True,
        template_type: int | None = None,
    ) -> TemplateView:
        """新建一个版本；activate=True 时同时把该组切到新版本（旧版本自动置 0）。

        variables 不传则从模板内容里自动提取占位符，避免「模板写了 {target_job} 但没登记变量」。
        """
        agent = _normalize_agent(agent_name)
        scene = _normalize_scene(scene)
        content = (template_content or "").strip()
        if not content:
            raise PromptTemplateInvalidError(detail="template_content 不能为空")
        if len(content) > MAX_TEMPLATE_CHARS:
            raise PromptTemplateInvalidError(detail=f"template_content 超过 {MAX_TEMPLATE_CHARS} 字符")

        resolved_type = _resolve_template_type(agent, template_type)
        names = list(variables) if variables else sorted(set(PLACEHOLDER_PATTERN.findall(content)))
        version = self.next_version(agent, scene)

        template = self.templates.create(
            agent_name=agent,
            scene=scene,
            template_type=resolved_type,
            template_content=content,
            variables=dump_variables(names),
            version=version,
            is_active=0,
            description=(description or "").strip() or None,
        )
        if activate:
            self._switch_active(template)
        view = _to_view(template, source=SOURCE_DB)
        if activate:
            self._write_cache(view)  # 缓存里只放「生效版本」；非生效版本由查询走库
        logger.info(
            "提示词模板新增 template_id=%s agent=%s scene=%s version=%s activate=%s",
            template.id,
            agent or COMMON_AGENT_KEY,
            scene,
            version,
            activate,
        )
        return view

    def rollback(self, template_id: int) -> TemplateView:
        """一键回滚：把指定历史版本切为生效版本（同组其它版本 is_active 置 0）。"""
        template = self.templates.get(template_id)
        if template is None:
            raise PromptTemplateNotFoundError(detail=f"template_id={template_id}")
        if not template.is_active:
            self._switch_active(template)
        view = _to_view(template, source=SOURCE_DB)
        self._write_cache(view)
        logger.info(
            "提示词模板回滚 template_id=%s agent=%s scene=%s version=%s",
            template.id,
            template.agent_name or COMMON_AGENT_KEY,
            template.scene,
            template.version,
        )
        return view

    def _switch_active(self, template: PromptTemplate) -> None:
        """同组内先全部置 0、再把目标版本置 1（同一事务内完成，保证「仅一条生效」）。"""
        self.templates.deactivate_group(template.agent_name, template.scene)
        self.templates.activate(template)

    def list_versions(self, agent_name: str | None, scene: str) -> list[TemplateView]:
        """列出一组模板的全部版本（版本号倒序）。"""
        return [
            _to_view(item, source=SOURCE_DB)
            for item in self.templates.find_group(_normalize_agent(agent_name), _normalize_scene(scene))
        ]

    # ------------------------------------------------------------------
    # 缓存（Redis）
    # ------------------------------------------------------------------
    def warm_cache(self) -> int:
        """启动加载：把所有生效模板写入 Redis Hash，返回写入条数。

        Redis 不可用时只记日志返回 0，不影响服务启动。
        """
        active = self.templates.list_active()
        if not self.redis:
            logger.info("Redis 未启用，跳过提示词模板预热（共 %s 条生效模板）", len(active))
            return 0
        try:
            self.redis.delete(self.cache_key)
            if active:
                # 一次 hset 写入所有 field，避免逐个往返
                views = [_to_view(item, source=SOURCE_DB) for item in active]
                mapping = {view.cache_field: _dump_view(view) for view in views}
                self.redis.hset(self.cache_key, mapping=mapping)  # type: ignore[call-arg]
                if self.cache_ttl:
                    self.redis.expire(self.cache_key, self.cache_ttl)
        except Exception as exc:
            logger.warning("提示词模板预热失败（不影响启动，读路径回退数据库）：%r", exc)
            return 0
        logger.info("提示词模板预热完成：%s 条写入 %s", len(active), self.cache_key)
        return len(active)

    def read_cache(self, agent_name: str | None, scene: str) -> TemplateView | None:
        """从 Redis 读取某个字段（私有 `{agent}:{scene}` / 公共 `__common__:{scene}`）。"""
        if not self.redis:
            return None
        try:
            raw = self.redis.hget(self.cache_key, cache_field(_normalize_agent(agent_name), scene))
        except Exception as exc:
            logger.warning("读取提示词缓存失败，回退数据库：%r", exc)
            return None
        if not raw:
            return None
        try:
            return _load_view(raw)
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("提示词缓存内容损坏，已忽略并回退数据库：field=%s", cache_field(agent_name, scene))
            return None

    def invalidate(self, agent_name: str | None, scene: str | None = None) -> None:
        """删除缓存：指定 field，或（scene 为空时）整个 Hash。"""
        if not self.redis:
            return
        try:
            if scene is None:
                self.redis.delete(self.cache_key)
            else:
                self.redis.hdel(self.cache_key, cache_field(_normalize_agent(agent_name), scene))
        except Exception as exc:
            logger.warning("清理提示词缓存失败：%r", exc)

    def _write_cache(self, view: TemplateView) -> None:
        """回写单个字段（Redis 不可用则跳过）。"""
        if not self.redis:
            return
        try:
            self.redis.hset(self.cache_key, view.cache_field, _dump_view(view))
            if self.cache_ttl:
                self.redis.expire(self.cache_key, self.cache_ttl)
        except Exception as exc:
            logger.warning("写入提示词缓存失败（不影响本次请求）：%r", exc)

    # ------------------------------------------------------------------
    # 读取与拼接
    # ------------------------------------------------------------------
    def get_template(self, agent_name: str | None, scene: str) -> TemplateView | None:
        """取生效模板：Redis 优先，未命中查库并回写缓存。"""
        agent = _normalize_agent(agent_name)
        cached = self.read_cache(agent, scene)
        if cached is not None:
            return cached
        template = self.templates.find_active(agent, scene)
        if template is None:
            return None
        view = _to_view(template, source=SOURCE_DB)
        self._write_cache(view)
        return view

    def get_common_template(self, scene: str) -> TemplateView | None:
        """取公共模板（agent_name=NULL, template_type=2）。"""
        return self.get_template(None, scene)

    def get_private_template(self, agent_name: str, scene: str) -> TemplateView | None:
        """取私有模板（agent_name 有值, template_type=1）。"""
        return self.get_template(agent_name, scene)

    def compose(self, agent_name: str, scene: str) -> ComposedPrompt:
        """拼接完整模板：公共模板（通用规则）+ 私有模板（Agent 指令）。

        两者都没有时抛 404；只有其中一个时按有的拼（便于灰度上线新场景）。
        """
        agent = _normalize_agent(agent_name) or DEFAULT_AGENT
        common = self.get_common_template(scene)
        private = self.get_private_template(agent, scene)
        if common is None and private is None:
            raise PromptTemplateNotFoundError(
                detail=f"agent={agent} scene={scene} 未配置任何生效模板（公共/私有均缺失）",
            )

        parts = [item.content for item in (common, private) if item is not None]
        variables: list[str] = []
        for item in (common, private):
            for name in item.variables if item else []:
                if name not in variables:
                    variables.append(name)
        return ComposedPrompt(
            agent_name=agent,
            scene=scene,
            content="\n\n".join(parts),
            variables=variables,
            common=common,
            private=private,
        )


def _normalize_agent(agent_name: str | None) -> str | None:
    """空串 / 空白一律视为「公共模板」（agent_name=NULL）。"""
    if agent_name is None:
        return None
    cleaned = agent_name.strip()
    return cleaned or None


def _normalize_scene(scene: str) -> str:
    """场景名必填，去掉首尾空白。"""
    cleaned = (scene or "").strip()
    if not cleaned:
        raise PromptTemplateInvalidError(detail="scene 不能为空")
    if len(cleaned) > 64:
        raise PromptTemplateInvalidError(detail="scene 长度不能超过 64")
    return cleaned


def _resolve_template_type(agent_name: str | None, template_type: int | None) -> int:
    """类型推断 + 一致性校验：公共模板必须无 agent_name，私有模板必须有 agent_name。"""
    if template_type is None:
        resolved = TEMPLATE_TYPE_COMMON if agent_name is None else TEMPLATE_TYPE_PRIVATE
    else:
        if template_type not in TEMPLATE_TYPE_LABELS:
            raise PromptTemplateInvalidError(detail="template_type 只能是 1（私有）或 2（公共）")
        resolved = template_type
    if resolved == TEMPLATE_TYPE_COMMON and agent_name is not None:
        raise PromptTemplateInvalidError(detail="公共模板（template_type=2）的 agent_name 必须为空")
    if resolved == TEMPLATE_TYPE_PRIVATE and agent_name is None:
        raise PromptTemplateInvalidError(detail="私有模板（template_type=1）必须指定 agent_name")
    return resolved


def _to_view(template: PromptTemplate, *, source: str) -> TemplateView:
    """ORM 行 -> 只读视图。"""
    return TemplateView(
        template_id=template.id,
        agent_name=template.agent_name,
        scene=template.scene,
        template_type=template.template_type,
        version=template.version,
        content=template.template_content,
        variables=parse_variables(template.variables),
        description=template.description,
        created_at=template.created_at,
        is_active=bool(template.is_active),
        source=source,
    )


def _dump_view(view: TemplateView) -> str:
    """只读视图 -> Redis 里的一行 JSON（与 _load_view 对称）。

    统一走 TemplateView 而不是 ORM 行：新增 / 回滚 / 预热三条写缓存路径用的是同一个视图对象，
    避免「预热能写、回滚写不进去」这类只有上线后才暴露的不一致。
    """
    return json.dumps(view.as_dict(), ensure_ascii=False)


def _load_view(raw: str) -> TemplateView:
    """Redis JSON -> 只读视图。"""
    payload = json.loads(raw)
    return TemplateView(
        template_id=payload.get("template_id"),
        agent_name=payload.get("agent_name"),
        scene=payload["scene"],
        template_type=int(payload.get("template_type", TEMPLATE_TYPE_PRIVATE)),
        version=int(payload["version"]),
        content=payload["template_content"],
        variables=list(payload.get("variables") or []),
        description=payload.get("description"),
        created_at=payload.get("created_at"),
        is_active=bool(payload.get("is_active", True)),
        source=SOURCE_REDIS,
    )
