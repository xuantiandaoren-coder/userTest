"""配置层测试：dev/prod 分层、优先级、敏感信息保护与生产校验。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.core import config as config_module
from app.core.config import MYSQL_URL_PLACEHOLDER, Env, Settings, current_env, dotenv_files

SHARED_URL = "mysql+pymysql://shared:shared%40pwd@127.0.0.1:3306/shared_db?charset=utf8mb4"
PROD_URL = "mysql+pymysql://prod:prod%40pwd@10.0.0.1:3306/prod_db?charset=utf8mb4"


@pytest.fixture()
def layered_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """造一套 .env / .env.dev / .env.prod，并让配置层指向它，模拟真实的分层加载。"""
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("MYSQL_URL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DB_ECHO", raising=False)

    (tmp_path / ".env").write_text(
        f"MYSQL_URL={SHARED_URL}\nLOG_LEVEL=INFO\nDB_ECHO=false\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.dev").write_text("LOG_LEVEL=DEBUG\nDB_ECHO=true\n", encoding="utf-8")
    (tmp_path / ".env.prod").write_text(f"MYSQL_URL={PROD_URL}\n", encoding="utf-8")
    return tmp_path


def test_dev_merges_base_and_dev_files(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")

    settings = Settings()

    assert settings.env is Env.DEV
    assert settings.mysql_url.get_secret_value() == SHARED_URL  # 基础 .env 生效
    assert settings.log_level == "DEBUG"  # .env.dev 覆盖 .env
    assert settings.db_echo is True
    assert settings.docs_enabled is True


def test_prod_overrides_base_and_closes_docs(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")

    settings = Settings()

    assert settings.env is Env.PROD
    assert settings.mysql_url.get_secret_value() == PROD_URL  # .env.prod 覆盖 .env
    assert settings.docs_enabled is False


def test_process_env_wins_over_dotenv(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MYSQL_URL", "mysql+pymysql://cli:pw@127.0.0.1:3306/cli_db")

    settings = Settings()

    assert settings.mysql_url.get_secret_value().endswith("/cli_db")


def test_dotenv_files_are_layered(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")

    assert [path.name for path in dotenv_files()] == [".env", ".env.prod"]


def test_invalid_app_env_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")

    with pytest.raises(RuntimeError, match="APP_ENV 非法"):
        current_env()


def test_prod_requires_mysql_url(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (layered_env / ".env").write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    (layered_env / ".env.prod").write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    monkeypatch.setenv("APP_ENV", "prod")

    with pytest.raises(ValidationError, match="MYSQL_URL"):
        Settings()


def test_secret_is_masked_but_still_usable(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")

    settings = Settings()

    assert isinstance(settings.mysql_url, SecretStr)
    assert "shared%40pwd" not in repr(settings)  # SecretsStr 掩码，日志/报错都打不出密码
    assert "**********" in repr(settings)
    assert settings.sqlalchemy_url == SHARED_URL  # 配置层内部仍可拿到明文连接


def test_placeholder_raises_clear_error(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (layered_env / ".env").write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    (layered_env / ".env.dev").write_text("", encoding="utf-8")
    monkeypatch.setenv("APP_ENV", "dev")

    settings = Settings()

    assert settings.mysql_url.get_secret_value() == MYSQL_URL_PLACEHOLDER
    with pytest.raises(RuntimeError, match="MySQL 连接串未配置"):
        settings.sqlalchemy_url


def test_invalid_log_level_rejected(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        Settings()


def test_jwt_secret_is_masked_and_never_echoed_in_errors(
    layered_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """密钥只从环境变量取，且不会出现在 repr / 校验报错里。"""
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("JWT_SECRET_KEY", "short-secret")

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert "长度至少" in str(excinfo.value)
    assert "short-secret" not in str(excinfo.value)  # 报错不回显密钥原文


def test_jwt_secret_placeholder_is_rejected(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        Settings()


def test_jwt_secret_repr_is_masked(layered_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("JWT_SECRET_KEY", "unit-test-jwt-secret-key-0123456789")

    settings = Settings()

    assert "unit-test-jwt-secret" not in repr(settings)
    assert "**********" in repr(settings)
