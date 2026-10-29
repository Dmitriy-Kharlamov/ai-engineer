"""Тесты API: pytest -q"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api import deps
from api.main import app
from ragkit.config import LLMSettings, Settings, get_settings


def fake_settings() -> Settings:
    return Settings(
        api_key=SecretStr("test-key"),
        debug=True,
        max_retries=7,
        llm=LLMSettings(model="test-model"),
        _env_file=None,  # не читать настоящий .env
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = fake_settings
    with TestClient(app) as c:  # with — чтобы отработал lifespan
        yield c
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_settings_override(client: TestClient) -> None:
    r = client.get("/health/config")
    assert r.json() == {"llm_model": "test-model", "debug": True, "max_retries": 7}


def test_literal_typo_is_422(client: TestClient) -> None:
    r = client.post("/chunks", json={"text": "a b c", "source": "s", "mode": "sentense"})
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["type"] == "literal_error" and err["loc"] == ["body", "mode"]


def test_password_not_echoed_but_error_useful(client: TestClient) -> None:
    r = client.post("/lab/login", json={"password": "secret123"})
    assert r.status_code == 422
    assert "secret123" not in r.text
    err = r.json()["detail"][0]
    assert err["loc"] == ["body", "password"] and err["type"] == "too_short"
    assert err["ctx"]["min_length"] == 12


def test_response_model_filters(client: TestClient) -> None:
    user = {"name": "d", "password": "pw"}
    assert "password_hash" in client.post("/lab/users-leak", json=user).json()
    assert client.post("/lab/users", json=user).json() == {"name": "d"}


def test_ragkit_errors_mapped(client: TestClient) -> None:
    too_big = client.post(
        "/chunks/documents", json={"name": "d", "max_chunk_tokens": 2, "texts": ["a b c"]}
    )
    assert too_big.status_code == 413 and too_big.json()["error"] == "ChunkTooLargeError"
    empty = client.post("/chunks/documents", json={"name": "d", "texts": []})
    assert empty.status_code == 422 and empty.json()["error"] == "EmptyIndexError"


def test_embedder_loaded_once() -> None:
    before = deps.EMBEDDER_LOADS
    with TestClient(app) as c:
        for _ in range(10):
            assert c.post("/chunks/embed", json={"texts": ["привет"]}).status_code == 200
    assert deps.EMBEDDER_LOADS - before == 1
