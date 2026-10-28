"""Настройки ragkit: читаются один раз при старте.

Порядок старшинства: аргументы Settings(...) > переменные окружения > .env > умолчания.
Использование:
    from ragkit.config import get_settings
    settings = get_settings()
"""
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Явный путь от этого файла, а не от текущего каталога: .env в корне репозитория.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMSettings(BaseModel):
    """Группа настроек модели: APP_LLM__MODEL, APP_LLM__TIMEOUT."""
    model: str = "claude-sonnet-5-5"
    timeout: float = 30.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_ignore_empty=True,        # пустые строки из .env.example → умолчание
        extra="ignore",
        hide_input_in_errors=True,    # ключ не попадёт в текст ValidationError
    )

    api_key: SecretStr                        # обязательно: без него не стартуем
    database_url: str = "sqlite:///dev.db"
    max_retries: int = 3
    debug: bool = False
    llm: LLMSettings = LLMSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Один объект настроек на весь процесс. В тестах: get_settings.cache_clear()."""
    return Settings()
