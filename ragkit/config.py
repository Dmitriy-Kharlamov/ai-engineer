"""Настройки ragkit: читаются один раз при старте.

Порядок старшинства: аргументы Settings(...) > переменные окружения > .env > умолчания.
Использование:
    from ragkit.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel, SecretStr, model_validator
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
        env_ignore_empty=True,  # пустые строки из .env.example → умолчание
        extra="ignore",
        hide_input_in_errors=True,  # ключ не попадёт в текст ValidationError
    )

    api_key: SecretStr  # обязательно: без него не стартуем
    database_url: str = "sqlite:///dev.db"
    max_retries: int = 3
    debug: bool = False  # True включает учебный роутер /lab
    # None → случайный на процесс; допустимо только при debug=True (Ф3-11, задание 11)
    jwt_secret: SecretStr | None = None
    jwt_ttl_minutes: int = 15
    bcrypt_rounds: int = 12
    rate_limit_per_minute: int = 60
    max_documents_per_owner: int = 100
    llm: LLMSettings = LLMSettings()
    embedder_dim: int = 128  # размерность эмбеддера; 0 — сервис отвечает, но не готов

    @model_validator(mode="after")
    def _jwt_secret_is_safe(self) -> Self:
        """Без секрета в проде токены подписывались бы случайным ключом процесса: каждый
        рестарт и каждая реплика выдавали бы чужие друг другу токены. Отказываемся стартовать.
        """
        if self.jwt_secret is None:
            if not self.debug:
                raise ValueError("APP_JWT_SECRET обязателен при debug=False")
        elif len(self.jwt_secret.get_secret_value().encode()) < 32:
            raise ValueError("APP_JWT_SECRET короче 32 байт: для HS256 нужно не меньше")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Один объект настроек на весь процесс. В тестах: get_settings.cache_clear()."""
    return Settings()
