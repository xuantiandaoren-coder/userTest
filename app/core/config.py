"""配置层：环境变量分层管理，敏感信息只从这里进入应用。

分层优先级（高 -> 低）：

1. 进程环境变量（如 `export MYSQL_URL=...`）
2. `.env.<APP_ENV>`（`.env.prod` / `.env.dev`，环境专属配置）
3. `.env`（各环境共享的基础配置）
4. 代码内默认值（只放非敏感、可公开的项）

约定：数据库连接串这类敏感信息只允许写在环境变量或 `.env*` 文件里，
业务代码统一通过 `settings` 读取，不写死；字段用 `SecretStr` 包裹，
保证日志、异常、`repr()` 都不会打印出明文密码。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 敏感配置占位符：真实值由环境变量 / .env 文件提供
MYSQL_URL_PLACEHOLDER = "{{mysql_url}}"
JWT_SECRET_PLACEHOLDER = "{{jwt_secret_key}}"

# JWT 密钥最小长度（HS256 建议 >= 256bit）
JWT_SECRET_MIN_LENGTH = 32


class Env(str, Enum):
    """运行环境：只区分开发与生产，避免出现“半生产”的灰色状态。"""

    DEV = "dev"
    PROD = "prod"


def current_env() -> Env:
    """读取 APP_ENV（缺省 dev），非法值直接启动失败而不是静默按开发跑。"""
    raw = os.getenv("APP_ENV", Env.DEV.value).strip().lower()
    try:
        return Env(raw)
    except ValueError as exc:
        raise RuntimeError(f"APP_ENV 非法：{raw!r}，可选值：{[item.value for item in Env]}") from exc


def dotenv_files(env: Env | None = None, root: Path | None = None) -> tuple[Path, ...]:
    """返回按优先级从低到高排列的 .env 文件（同名字段后者覆盖前者）。"""
    env = env or current_env()
    root = root or PROJECT_ROOT
    return (root / ".env", root / f".env.{env.value}")


class Settings(BaseSettings):
    """应用配置。字段名小写，对应的大写环境变量同样生效（如 MYSQL_URL）。"""

    model_config = SettingsConfigDict(
        extra="ignore",        # .env 里出现的无关变量不报错
        case_sensitive=False,
        # 校验失败时不回显原始输入，避免密钥 / 连接串出现在启动报错与日志里
        hide_input_in_errors=True,
    )

    env: Env = Field(
        default_factory=current_env,
        validation_alias="APP_ENV",
        description="运行环境：dev / prod",
    )

    # 敏感配置：SecretStr 保证 repr / 日志 / 异常里只有 **********
    mysql_url: SecretStr = Field(default=SecretStr(MYSQL_URL_PLACEHOLDER), description="MySQL 连接串")

    db_echo: bool = False          # True 时打印 SQL，便于本地调试
    db_pool_size: int = 5          # 连接池常驻连接数
    db_max_overflow: int = 10      # 连接池峰值溢出连接数
    db_pool_recycle: int = 3600    # 连接回收秒数，避免被 MySQL 主动断开
    db_connect_timeout: int = 5    # 建连超时秒数，数据库不可达时快速失败

    # JWT：密钥只从环境变量 / .env 读取，代码里不写死
    jwt_secret_key: SecretStr = Field(default=SecretStr(JWT_SECRET_PLACEHOLDER), description="JWT 签名密钥")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_seconds: int = 30 * 60        # 访问令牌 30 分钟
    jwt_refresh_token_expire_seconds: int = 7 * 24 * 3600  # 刷新令牌 7 天

    # 文件上传：存储根目录（按用户 ID 分子目录）与单文件大小上限
    # storage_root 只服务于旧的本地落盘逻辑（app/services/upload_service.py，教学用，不参与运行）
    storage_root: Path = PROJECT_ROOT / "storage"
    max_upload_bytes: int = 10 * 1024 * 1024

    # 对象存储 SeaweedFS（S3 网关）：MySQL 只存元数据，原文件放这里
    seaweedfs_endpoint: str = "http://127.0.0.1:8333"       # S3 网关地址
    seaweedfs_access_key: SecretStr = Field(default=SecretStr(""), description="SeaweedFS Access Key")
    seaweedfs_secret_key: SecretStr = Field(default=SecretStr(""), description="SeaweedFS Secret Key")
    seaweedfs_bucket: str = "test"                          # 默认 bucket
    seaweedfs_region: str = "us-east-1"                     # S3 签名区域（SeaweedFS 不校验，仅占位）
    # 对外访问地址：CDN / 反向代理前缀，如 https://cdn.example.com；留空则用网关地址 <endpoint>/<bucket>
    seaweedfs_public_base_url: str = ""
    # 配了 Access/Secret Key（开启鉴权）时，用预签名 URL 给前端访问，有效期由这里控制
    seaweedfs_presign_expire_seconds: int = 3600

    # 资源过期时间：storage_scene=0 长过期（1 个月），=1 短过期（2 小时）
    resource_ttl_long_seconds: int = 30 * 24 * 3600
    resource_ttl_short_seconds: int = 2 * 3600

    # 过期清理：每天 03:00 扫描 resources.expire_time（先删对象，再删元数据）
    resource_cleanup_hour: int = 3
    resource_cleanup_minute: int = 0

    # 限流：固定窗口 + IP 维度
    register_rate_limit: int = 5
    register_rate_window: int = 60
    login_rate_limit: int = 10
    login_rate_window: int = 60

    # ------------------------------------------------------------------
    # LLM（模型层 app/llm/llm.py 只读这里，业务代码不直接读环境变量）
    # 默认 DeepSeek：OpenAI 兼容协议，base_url=https://api.deepseek.com
    # ------------------------------------------------------------------
    llm_provider: str = "deepseek"                      # 见 app/llm/llm.py:PROVIDERS
    llm_model: str = ""                                 # 留空则用 provider 的默认模型（DeepSeek 为 deepseek-chat）
    llm_api_key: SecretStr = Field(default=SecretStr(""), description="LLM API Key（敏感）")
    llm_base_url: str = ""                              # 留空则用 provider 默认地址
    llm_temperature: float = 0.7
    llm_timeout: int = 60                               # 单次请求超时（秒）
    llm_max_tokens: int | None = None                   # 留空则由服务端默认
    llm_history_turns: int = 10                         # 记忆层注入的历史轮数上限

    # ------------------------------------------------------------------
    # Redis：提示词模板缓存（无 Redis 时自动回退数据库，功能不受影响）
    # ------------------------------------------------------------------
    redis_enabled: bool = True
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: SecretStr = Field(default=SecretStr(""), description="Redis 密码（敏感）")
    redis_url: SecretStr = Field(default=SecretStr(""), description="Redis 连接串，配置后优先于 host/port/db")
    redis_socket_timeout: float = 2.0                   # 连接 / 读写超时，Redis 不可用时快速失败
    prompt_cache_key: str = "prompt:templates:active"   # 生效模板缓存 Key（Hash：field=agent:scene）
    prompt_cache_ttl: int = 3600                        # 缓存过期秒数，0 表示不过期
    # 启动时预热模板缓存：后台线程执行，不阻塞启动；测试环境置 false，避免连真实数据库
    prompt_cache_warmup: bool = True

    # ------------------------------------------------------------------
    # RAG：DashScope 文本向量化 + Qdrant 向量库（见 app/rag/core.py）
    # ------------------------------------------------------------------
    dashscope_api_key: SecretStr = Field(default=SecretStr(""), description="DashScope API Key（敏感）")
    qdrant_host: str = "127.0.0.1"   # Qdrant 服务地址
    qdrant_port: int = 6333          # Qdrant HTTP 端口
    # 上传文件后自动向量化入库；失败只记日志、不影响上传结果（向量化是增强能力）
    rag_ingest_enabled: bool = True

    # 生产默认关闭接口文档，避免暴露内部接口清单
    docs_enabled: bool = Field(default_factory=lambda: current_env() is Env.DEV)

    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_file: str = "app.log"
    log_max_bytes: int = 10 * 1024 * 1024   # 单个日志文件上限 10MB
    log_backup_count: int = 7               # 保留 7 个历史文件
    log_access: bool = True                 # 是否记录访问日志

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"LOG_LEVEL 非法：{value!r}，可选值如 DEBUG/INFO/WARNING/ERROR")
        return level

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """按“环境变量 > .env.<环境> > .env”的顺序装配配置来源。"""
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(settings_cls, env_file=dotenv_files(), env_file_encoding="utf-8"),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _validate_secrets(self) -> Settings:
        """密钥校验：连接串在生产必须显式配置，JWT 密钥任何环境都不允许用占位符或弱密钥。"""
        if self.env is Env.PROD and self.mysql_url.get_secret_value().strip() in {"", MYSQL_URL_PLACEHOLDER}:
            raise ValueError("生产环境（APP_ENV=prod）必须显式配置 MYSQL_URL")

        secret = self.jwt_secret_key.get_secret_value().strip()
        if secret in {"", JWT_SECRET_PLACEHOLDER}:
            raise ValueError(
                "JWT_SECRET_KEY 未配置：请通过环境变量或 .env 提供，"
                '可用 python -c "import secrets; print(secrets.token_urlsafe(48))" 生成'
            )
        if len(secret) < JWT_SECRET_MIN_LENGTH:
            raise ValueError(f"JWT_SECRET_KEY 长度至少 {JWT_SECRET_MIN_LENGTH} 个字符")
        return self

    @property
    def sqlalchemy_url(self) -> str:
        """返回可用的连接串明文，未配置时给出明确的排查提示。"""
        value = self.mysql_url.get_secret_value().strip()
        if not value or value == MYSQL_URL_PLACEHOLDER:
            raise RuntimeError(
                "MySQL 连接串未配置：请复制 .env.example 为 .env 并填写 MYSQL_URL，"
                "或直接设置环境变量 MYSQL_URL"
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """缓存配置对象，避免重复读取 .env。"""
    return Settings()


settings = get_settings()


# ============================================================================
# 智能体配置：7 个智能体，与提示词模板 (agent_name, scene) 一一对应
# ----------------------------------------------------------------------------
# - scene：该智能体默认使用的模板场景，公共模板按 scene 复用
# - select_model：对应 chat_messages.select_model（0=流程总控 1=难点学习 2=出题题库
#   3=简历解析 4=面试主考官 5=面试评测），用于按会话消息反推智能体
# - LEGACY_AGENT_ALIASES：历史智能体名的兼容别名，见 get_agent()
# - 公共模板（template_type=2，agent_name=NULL）与私有模板（template_type=1）
#   拼接后生效，键一律用 `{agent}:{scene}`
# ============================================================================


@dataclass(frozen=True)
class AgentSetting:
    """单个智能体的静态配置（不含提示词正文，正文在 prompt_templates 表里）。"""

    label: str                  # 中文展示名
    scene: str                  # 默认场景
    description: str            # 一句话说明
    select_model: int | None    # 关联会话消息的 select_model，无则为 None
    provider: str | None = None  # 指定模型 provider；None 表示跟随全局 LLM_PROVIDER

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（配置接口直接返回）。"""
        return {
            "agent_name": None,  # 由调用方填充
            "label": self.label,
            "scene": self.scene,
            "description": self.description,
            "select_model": self.select_model,
            "provider": self.provider,
        }


DEFAULT_AGENT = "flow_controller"
DEFAULT_SCENE = "workflow"

# 7 个智能体：与业务侧 AGENT_CONFIG 一一对应，(agent_name, scene) 即模板分组键
AGENT_CONFIG: dict[str, AgentSetting] = {
    "flow_controller": AgentSetting(
        label="流程总控Agent",
        scene="workflow",
        description="编排整条学习 / 求职流程，决定下一步交给哪个智能体",
        select_model=0,
    ),
    "resume_parser": AgentSetting(
        label="资料&简历解析Agent",
        scene="resume",
        description="解析简历与上传资料，抽取出可入库的结构化信息",
        select_model=3,
    ),
    "quiz_generate_workflow": AgentSetting(
        label="出题题库Agent",
        scene="quiz",
        description="围绕目标岗位与薄弱点出题、判题并维护题库",
        select_model=2,
    ),
    "interview_host": AgentSetting(
        label="面试主考官Agent",
        scene="interview",
        description="按目标岗位逐轮追问，主持模拟面试",
        select_model=4,
    ),
    "interview_evaluator": AgentSetting(
        label="面试评测Agent",
        scene="evaluation",
        description="评测面试问答，输出评分、亮点与改进点",
        select_model=5,
    ),
    "difficulty_learner": AgentSetting(
        label="难点学习Agent",
        scene="study",
        description="针对薄弱点讲透概念并安排陪练，直到掌握",
        select_model=1,
    ),
    "note_archiver": AgentSetting(
        label="笔记归档Agent",
        scene="note",
        description="提炼会话要点，整理成可检索、可复用的笔记",
        select_model=None,
    ),
}

# 历史智能体名 -> 现行智能体名。老客户端 / 老调用方仍按老名字请求时，
# 统一折叠成现行名字再查配置，避免直接 422；模板分组也随之落在新场景上。
LEGACY_AGENT_ALIASES: dict[str, str] = {
    "tutor": "difficulty_learner",
    "knowledge_explain": "difficulty_learner",
    "quiz_coach": "quiz_generate_workflow",
    "resume_optimizer": "resume_parser",
    "mock_interviewer": "interview_host",
    "interview_reviewer": "interview_evaluator",
    "study_planner": "flow_controller",
}

# 会话类型 -> 默认智能体：0=学习，1=面试，2=笔记
SESSION_MODEL_AGENT: dict[int, str] = {
    0: "difficulty_learner",
    1: "interview_host",
    2: "note_archiver",
}


def resolve_agent_name(agent_name: str | None) -> str | None:
    """把历史别名折叠成现行智能体名；未命中的名字原样返回。"""
    if not agent_name:
        return None
    cleaned = agent_name.strip()
    if not cleaned:
        return None
    return LEGACY_AGENT_ALIASES.get(cleaned, cleaned)


def get_agent(agent_name: str | None) -> AgentSetting | None:
    """按名称取智能体配置（自动兼容历史别名），未配置返回 None（路由层据此报 422）。"""
    name = resolve_agent_name(agent_name)
    if name is None:
        return None
    return AGENT_CONFIG.get(name)


def agent_for_select_model(select_model: int | None) -> str | None:
    """按消息的 select_model 反查智能体名（找不到返回 None）。"""
    if select_model is None:
        return None
    for name, setting in AGENT_CONFIG.items():
        if setting.select_model == select_model:
            return name
    return None


def agent_for_session_model(session_model: int) -> str:
    """按会话类型给出默认智能体（未知类型回退默认智能体）。"""
    return SESSION_MODEL_AGENT.get(session_model, DEFAULT_AGENT)
