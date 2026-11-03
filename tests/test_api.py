"""Тесты API: pytest"""

import math
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from api import deps
from api.main import create_app
from api.routers import health
from ragkit import Embedder, HashEmbedder
from ragkit.config import Settings
from services.clock import FakeClock


def test_health(client: TestClient, clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    # Версию задаём сами: тест не зависит от того, установлен ли пакет в этом окружении,
    # и ловит и "unknown" вместо версии, и опечатку в имени пакета (KeyError -> 500).
    monkeypatch.setattr(health, "version", {"ragkit": "9.9.9"}.__getitem__)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ragkit_version"] == "9.9.9"
    assert datetime.fromisoformat(body["started_at"]) == clock.now()  # старт по часам app


def test_health_survives_missing_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_installed(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(health, "version", not_installed)  # как в свежем клоне без pip install
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ragkit_version"] == "unknown"


def test_settings_override(client: TestClient) -> None:
    r = client.get("/health/config")
    assert r.json() == {"llm_model": "test-model", "debug": True, "max_retries": 7}


def test_ready_when_embedder_works(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"ready": True, "problems": []}


def test_not_ready_but_alive_when_embedder_broken(
    settings_factory: Callable[..., Settings],
) -> None:
    with TestClient(create_app(settings_factory(embedder_dim=0))) as c:
        assert c.get("/health").status_code == 200  # живой
        r = c.get("/health/ready")
    assert r.status_code == 503  # но не готов
    assert r.json() == {"ready": False, "problems": ["эмбеддер не выдаёт векторы: dim=0"]}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"jwt_secret": None, "debug": False}, "APP_JWT_SECRET обязателен при debug=False"),
        ({"jwt_secret": SecretStr("короткий")}, "APP_JWT_SECRET короче 32 байт"),
    ],
    ids=["no-secret-in-prod", "short-secret"],
)
def test_unsafe_jwt_secret_refuses_to_start(
    settings_factory: Callable[..., Settings], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message) as info:
        settings_factory(**overrides)
    assert "короткий" not in str(info.value)  # hide_input_in_errors: значение не в тексте


def test_no_secret_allowed_in_debug(settings_factory: Callable[..., Settings]) -> None:
    with TestClient(create_app(settings_factory(jwt_secret=None, debug=True))) as c:
        assert c.get("/health").status_code == 200
        assert len(c.app.state.jwt_secret) >= 32  # type: ignore[attr-defined]


def test_lab_exists_only_in_debug(settings_factory: Callable[..., Settings]) -> None:
    with TestClient(create_app(settings_factory(debug=False))) as c:
        assert c.post("/lab/users", json={"name": "d", "password": "pw"}).status_code == 404


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


# ids латиницей: кириллические id pytest экранирует, и в выводе -v видно "\u0447\u0430..."
@pytest.mark.parametrize(
    ("path", "keys"),
    [
        ("/lab/users-leak", {"name", "password_hash", "internal_id"}),
        ("/lab/users", {"name"}),
    ],
    ids=[
        "leak-without-response-model",
        "response-model-filters",
    ],  # ASCII: кириллицу pytest выводит как \\u0431...
)
def test_response_model_filters(client: TestClient, path: str, keys: set[str]) -> None:
    body = client.post(path, json={"name": "d", "password": "pw"}).json()
    assert set(body) == keys
    assert body["name"] == "d"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"name": "d", "max_chunk_tokens": 2, "texts": ["a b c"]}, "chunk_too_large"),
        ({"name": "d", "texts": []}, "empty_index"),
    ],
    ids=["chunk-too-large", "empty-document"],
)
def test_ragkit_errors_mapped(client: TestClient, body: dict[str, Any], code: str) -> None:
    r = client.post("/chunks/documents", json=body)
    assert r.status_code == 422
    assert r.json()["code"] == code


def test_embedder_loaded_once(app: FastAPI) -> None:
    loads: list[int] = []  # счётчик живёт в тесте: другие тесты его не видят и не двигают

    def counting_loader(dim: int = 128) -> Embedder:
        loads.append(dim)
        return HashEmbedder(dim=3)  # метка: запросы обязаны получить именно этот объект

    app.dependency_overrides[deps.load_embedder] = counting_loader
    with TestClient(app) as c:
        for _ in range(10):
            r = c.post("/chunks/embed", json={"texts": ["привет"]})
            assert r.status_code == 200
            assert r.json()["dim"] == 3  # иначе эмбеддер создан в обход загрузчика
    assert loads == [128]  # ровно одна загрузка, и с размерностью из lifespan


def test_typo_in_field_name_is_422(client: TestClient) -> None:
    r = client.post("/chunks", json={"text": "a b", "source": "s", "max_token": 1})
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["type"] == "extra_forbidden" and err["loc"] == ["body", "max_token"]


def test_error_body_is_whitelisted(client: TestClient) -> None:
    r = client.post("/chunks", json={"text": "a b", "source": "../secret-path"})
    err = r.json()["detail"][0]
    assert set(err) <= {"type", "loc", "msg", "ctx"}
    assert err["type"] == "source_path" and "../secret-path" not in r.text


def test_embed_route_with_fake(client: TestClient) -> None:
    # Ф3-10, задание 10: вместо create_autospec — настоящий HashEmbedder из lifespan (фейк
    # модели). Проверяем не вызов, а результат: свойства, которые нужны поиску по векторам.
    r = client.post("/chunks/embed", json={"texts": ["привет", "мир", "привет"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == 128
    hello, world, hello_again = body["vectors"]  # по вектору на текст, в том же порядке
    assert hello == hello_again  # детерминированность: тот же текст — тот же вектор
    assert hello != world  # разные тексты — разные векторы
    for vector in body["vectors"]:
        assert len(vector) == 128
        assert math.isclose(math.fsum(x * x for x in vector), 1.0)  # единичная длина
