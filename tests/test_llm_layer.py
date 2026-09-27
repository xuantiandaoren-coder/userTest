"""模型层测试：provider -> 模型实例。

模型层的职责边界很窄（选 provider、取 Key、构造客户端），构造本身是纯本地行为，
不发起网络请求，所以这里不需要真实 API Key。
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.llm.llm import (
    PROVIDERS,
    LLMNotConfiguredError,
    available_providers,
    build_chat_model,
    get_chat_model,
    is_configured,
    reset_chat_model_cache,
    resolve_api_key,
    resolve_provider,
)


@pytest.fixture()
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉所有 Key 来源：配置项与 provider 专属环境变量。"""
    monkeypatch.setattr(settings, "llm_api_key", SecretStr(""))
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "llm_model", "")
    for provider in PROVIDERS.values():
        if provider.api_key_env:
            monkeypatch.delenv(provider.api_key_env, raising=False)


@pytest.fixture()
def api_key(monkeypatch: pytest.MonkeyPatch, no_api_key: None) -> None:
    monkeypatch.setattr(settings, "llm_api_key", SecretStr("sk-test-key"))


# ---------------------------------------------------------------------------
# provider 解析
# ---------------------------------------------------------------------------
def test_resolve_provider_defaults_to_deepseek(no_api_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "")

    provider = resolve_provider()

    assert provider.name == "deepseek"
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.default_model == "deepseek-chat"


def test_resolve_provider_normalizes_name_and_rejects_unknown(no_api_key: None) -> None:
    assert resolve_provider(" Moonshot ").name == "moonshot"

    with pytest.raises(LLMNotConfiguredError) as excinfo:
        resolve_provider("not-a-provider")

    assert excinfo.value.code == "LLM_NOT_CONFIGURED"
    assert excinfo.value.http_status == 500  # 服务端配置问题，不是用户输入问题


def test_every_provider_declares_a_default_model() -> None:
    for name, provider in PROVIDERS.items():
        assert provider.name == name
        assert provider.default_model
        assert provider.label


# ---------------------------------------------------------------------------
# API Key 与可用性
# ---------------------------------------------------------------------------
def test_api_key_falls_back_to_provider_env(no_api_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = resolve_provider("deepseek")
    assert resolve_api_key(provider) == ""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    assert resolve_api_key(provider) == "sk-from-env"

    monkeypatch.setattr(settings, "llm_api_key", SecretStr("sk-from-config"))
    assert resolve_api_key(provider) == "sk-from-config"  # 配置项优先于环境变量


def test_is_configured_reflects_key_availability(no_api_key: None, api_key: None) -> None:
    assert is_configured("deepseek") is True
    assert is_configured("not-a-provider") is False
    assert is_configured("ollama") is True  # 本地部署不需要 Key


def test_build_chat_model_requires_key(no_api_key: None) -> None:
    with pytest.raises(LLMNotConfiguredError) as excinfo:
        build_chat_model(provider="deepseek")

    assert "DEEPSEEK_API_KEY" in (excinfo.value.detail or "")


# ---------------------------------------------------------------------------
# 构造模型实例
# ---------------------------------------------------------------------------
def test_build_chat_model_uses_provider_defaults(api_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_temperature", 0.3)

    model = build_chat_model(provider="deepseek")

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "deepseek-chat"          # 未指定模型 -> provider 默认模型
    assert model.openai_api_base == "https://api.deepseek.com"
    assert model.temperature == 0.3
    assert model.streaming is True                      # 流式接口依赖 astream
    assert model.openai_api_key.get_secret_value() == "sk-test-key"


def test_build_chat_model_honours_overrides(api_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "gpt-4o")

    model = build_chat_model(provider="openai", model="gpt-4o-mini", temperature=0.0, streaming=False)

    assert model.model_name == "gpt-4o-mini"  # 显式模型 > 配置模型 > provider 默认
    assert model.temperature == 0.0
    assert model.streaming is False


def test_get_chat_model_caches_and_resets(api_key: None) -> None:
    reset_chat_model_cache()
    try:
        first = get_chat_model("ollama")
        assert get_chat_model("ollama") is first          # 同参数复用同一个客户端
        # 缓存键是「显式入参」：显式指定模型名会另建一个实例（即使解析出的模型相同）
        assert get_chat_model("ollama", "qwen2.5:7b") is not first

        reset_chat_model_cache()
        assert get_chat_model("ollama") is not first      # 清缓存后重建
    finally:
        reset_chat_model_cache()


def test_available_providers_reports_configuration(api_key: None) -> None:
    providers = {item["provider"]: item for item in available_providers()}

    assert set(providers) == set(PROVIDERS)
    assert providers["deepseek"]["configured"] is True
    assert providers["openai"]["configured"] is True
    assert providers["deepseek"]["api_key_env"] == "DEEPSEEK_API_KEY"
