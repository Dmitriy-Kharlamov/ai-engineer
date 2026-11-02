from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.main import create_app
from ragkit.config import LLMSettings, Settings


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "api_key": SecretStr("test-key"),
        "debug": True,
        "max_retries": 7,
        "llm": LLMSettings(model="test-model"),
        "jwt_secret": SecretStr("test-jwt-secret-at-least-32-bytes-long!!"),
        "bcrypt_rounds": 4,  # тестам не нужна стоимость прода: 4 раунда ≈ 1 мс
        "rate_limit_per_minute": 3,
        "_env_file": None,  # не читать настоящий .env
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def app() -> FastAPI:
    return create_app(make_settings())


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:  # with — чтобы отработал lifespan
        yield c
