"""模型层（第一层）：**只**负责 provider -> 模型实例。

职责边界（刻意收得很窄）：
- 输入：provider 名（缺省取 `settings.llm_provider`，默认 deepseek）、模型名、温度等参数
- 输出：`BaseChatModel` 实例（LangChain 标准接口，流式 `.astream()` 由上层调用）
- 不做：不拼提示词（提示词层 app/prompts/prompt_layer.py）、不读历史（记忆层 app/memory/memory.py）

DeepSeek 走 OpenAI 兼容协议（base_url=https://api.deepseek.com，模型 deepseek-chat），
因此统一用 `ChatOpenAI` 构造，其它兼容厂商只是换 base_url / 默认模型 / Key 环境变量。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings, settings
from app.core.exceptions import SystemError

logger = logging.getLogger("app.llm")


class LLMNotConfiguredError(SystemError):
    """系统异常：模型未配置（缺 API Key / provider 名写错），属服务端配置问题。"""

    code = "LLM_NOT_CONFIGURED"
    message = "模型服务未配置，请联系管理员"


@dataclass(frozen=True)
class ProviderSetting:
    """单个 provider 的静态信息。"""

    name: str
    label: str
    default_model: str
    base_url: str | None          # None = 用 SDK 默认（OpenAI 官方）
    api_key_env: str              # 缺省 API Key 时回退读取的环境变量名

    def as_dict(self) -> dict[str, object]:
        """转成可 JSON 序列化的字典（/prompt/config/agents 里附带的 provider 清单）。"""
        return {
            "provider": self.name,
            "label": self.label,
            "default_model": self.default_model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
        }


# 全部为 OpenAI 兼容协议，换 provider 只需换 base_url / 默认模型
PROVIDERS: dict[str, ProviderSetting] = {
    "deepseek": ProviderSetting(
        name="deepseek",
        label="DeepSeek",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "openai": ProviderSetting(
        name="openai",
        label="OpenAI",
        default_model="gpt-4o-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
    ),
    "dashscope": ProviderSetting(
        name="dashscope",
        label="通义千问（DashScope）",
        default_model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "moonshot": ProviderSetting(
        name="moonshot",
        label="Kimi（Moonshot）",
        default_model="moonshot-v1-8k",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
    ),
    "zhipu": ProviderSetting(
        name="zhipu",
        label="智谱 GLM",
        default_model="glm-4-plus",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPUAI_API_KEY",
    ),
    "ollama": ProviderSetting(
        name="ollama",
        label="本地 Ollama",
        default_model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",  # 本地部署不需要 Key
    ),
}


def resolve_provider(name: str | None = None, config: Settings | None = None) -> ProviderSetting:
    """provider 名 -> ProviderSetting；未配置的 provider 直接报错（不静默降级）。"""
    config = config or settings
    provider = (name or config.llm_provider or "deepseek").strip().lower()
    setting = PROVIDERS.get(provider)
    if setting is None:
        raise LLMNotConfiguredError(
            detail=f"provider={provider!r} 未定义，可选：{sorted(PROVIDERS)}",
        )
    return setting


def resolve_api_key(provider: ProviderSetting, config: Settings | None = None) -> str:
    """API Key 取值顺序：配置项 > provider 专属环境变量（如 DEEPSEEK_API_KEY）。"""
    config = config or settings
    key = config.llm_api_key.get_secret_value().strip()
    if key:
        return key
    if provider.api_key_env:
        return os.getenv(provider.api_key_env, "").strip()
    return ""


def is_configured(provider_name: str | None = None, config: Settings | None = None) -> bool:
    """该 provider 是否已具备调用条件（本地 Ollama 无 Key 也算可用）。"""
    config = config or settings
    try:
        provider = resolve_provider(provider_name, config)
    except LLMNotConfiguredError:
        return False
    return bool(resolve_api_key(provider, config)) or provider.name == "ollama"


def build_chat_model(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
    config: Settings | None = None,
) -> BaseChatModel:
    """构造聊天模型实例（未发起任何网络请求，构造是纯本地行为）。

    - provider 缺省取 `LLM_PROVIDER`（deepseek）
    - model 缺省取 `LLM_MODEL`，仍为空则用 provider 默认模型（DeepSeek => deepseek-chat）
    - api key 缺省取 `LLM_API_KEY`，再回退 provider 专属环境变量
    """
    config = config or settings
    setting = resolve_provider(provider, config)
    api_key = resolve_api_key(setting, config)
    if not api_key and setting.name != "ollama":
        raise LLMNotConfiguredError(
            detail=(
                f"provider={setting.name} 缺少 API Key，"
                f"请配置 LLM_API_KEY 或环境变量 {setting.api_key_env}"
            ),
        )

    kwargs: dict[str, object] = {
        "model": (model or config.llm_model or setting.default_model).strip(),
        "temperature": config.llm_temperature if temperature is None else temperature,
        "timeout": config.llm_timeout,
        "streaming": streaming,
        "max_retries": 2,
    }
    if api_key:
        kwargs["api_key"] = api_key
    base_url = (config.llm_base_url or setting.base_url or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    if config.llm_max_tokens:
        kwargs["max_tokens"] = config.llm_max_tokens

    logger.info("构建聊天模型 provider=%s model=%s streaming=%s", setting.name, kwargs["model"], streaming)
    return ChatOpenAI(**kwargs)  # type: ignore[arg-type]


@lru_cache(maxsize=8)
def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    streaming: bool = True,
) -> BaseChatModel:
    """进程级缓存：同 (provider, model, streaming) 复用同一个客户端（连接池复用）。"""
    return build_chat_model(provider=provider, model=model, streaming=streaming)


def reset_chat_model_cache() -> None:
    """清空模型缓存（测试替换配置 / 关闭应用时调用）。"""
    get_chat_model.cache_clear()


def available_providers() -> list[dict[str, object]]:
    """provider 清单 + 是否已配置，供配置接口展示。"""
    return [
        {**setting.as_dict(), "configured": is_configured(name)}
        for name, setting in PROVIDERS.items()
    ]
