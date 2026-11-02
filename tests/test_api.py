"""Тесты API: pytest -q"""

from fastapi.testclient import TestClient

from api import deps
from api.main import create_app
from tests.conftest import make_settings


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
    assert too_big.status_code == 422 and too_big.json()["code"] == "chunk_too_large"
    empty = client.post("/chunks/documents", json={"name": "d", "texts": []})
    assert empty.status_code == 422 and empty.json()["code"] == "empty_index"


def test_embedder_loaded_once() -> None:
    before = deps.EMBEDDER_LOADS
    with TestClient(create_app(make_settings())) as c:
        for _ in range(10):
            assert c.post("/chunks/embed", json={"texts": ["привет"]}).status_code == 200
    assert deps.EMBEDDER_LOADS - before == 1


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
