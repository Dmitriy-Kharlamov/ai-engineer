"""401 / 403 / 404, защита роутера, JWT, ограничение частоты."""

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.apikeys import ApiKeyStore


def keys(app: FastAPI) -> ApiKeyStore:
    store: ApiKeyStore = app.state.api_keys
    return store


def login(client: TestClient, user: str, password: str = "password-123") -> dict[str, str]:
    client.post("/auth/register", json={"username": user, "password": password})
    token = client.post("/auth/login", json={"username": user, "password": password})
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


def test_health_is_open(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_admin_router_protects_new_endpoint(client: TestClient, app: FastAPI) -> None:
    user_key = keys(app).issue("bob", frozenset({"user"}))
    admin_key = keys(app).issue("root", frozenset({"admin"}))
    no_key = client.get("/admin/new-report")
    assert no_key.status_code == 401 and no_key.headers["WWW-Authenticate"] == "ApiKey"
    assert client.get("/admin/new-report", headers={"X-API-Key": "rk_wrong"}).status_code == 401
    assert client.get("/admin/new-report", headers={"X-API-Key": user_key}).status_code == 403
    assert client.get("/admin/new-report", headers={"X-API-Key": admin_key}).status_code == 200


def test_foreign_document_is_404(client: TestClient) -> None:
    alice, bob = login(client, "alice"), login(client, "bob")
    doc = client.post("/documents", json={"name": "a.md", "text": "секрет"}, headers=alice)
    doc_id = doc.json()["id"]
    assert client.get(f"/documents/{doc_id}", headers=alice).status_code == 200
    foreign = client.get(f"/documents/{doc_id}", headers=bob)
    missing = client.get("/documents/999999", headers=bob)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()  # ответы неотличимы


def test_jwt_expired_and_tampered(client: TestClient) -> None:
    headers = login(client, "carol")
    token = headers["Authorization"].removeprefix("Bearer ")
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    assert client.get("/me", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401
    secret = client.app.state.jwt_secret  # type: ignore[attr-defined]
    old = jwt.encode(
        {"sub": "carol", "exp": datetime.now(UTC) - timedelta(minutes=1)}, secret, algorithm="HS256"
    )
    r = client.get("/me", headers={"Authorization": f"Bearer {old}"})
    assert r.status_code == 401 and r.json()["detail"] == "токен просрочен"


def test_rate_limit_429(client: TestClient, app: FastAPI) -> None:
    key = keys(app).issue("dave", frozenset())
    codes = [client.get("/me/key", headers={"X-API-Key": key}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]  # лимит в make_settings — 3 в минуту
    r = client.get("/me/key", headers={"X-API-Key": key})
    assert 1 <= int(r.headers["Retry-After"]) <= 60


def test_unknown_ctx_key_is_dropped() -> None:
    from api.errors import safe_error

    err = {"type": "x", "loc": ("a",), "msg": "m", "input": "s", "ctx": {"got": "s", "ge": 1}}
    assert safe_error(err) == {"type": "x", "loc": ("a",), "msg": "m", "ctx": {"ge": 1}}
