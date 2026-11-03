"""Тесты API: pytest"""

from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from typing import Any
from unittest.mock import create_autospec

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


def test_embed_route_with_mock(app: FastAPI) -> None:
    # Мок своего же кода: проверяет, что маршрут позвал embed и вернул его ответ как есть.
    embedder = create_autospec(HashEmbedder, instance=True)
    embedder.dim = 3
    embedder.embed.return_value = [[0.6, 0.8, 0.0]]
    app.dependency_overrides[deps.get_embedder] = lambda: embedder
    with TestClient(app) as c:
        r = c.post("/chunks/embed", json={"texts": ["привет"]})
    assert r.json() == {"dim": 3, "vectors": [[0.6, 0.8, 0.0]]}
    embedder.embed.assert_called_once_with(["привет"])
