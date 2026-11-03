"""401 / 403 / 404, защита роутера, JWT, ограничение частоты."""

import secrets
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.security import create_access_token, decode_access_token
from services.apikeys import ApiKeyStore
from services.clock import FakeClock

OTHER_SECRET = "другой-секрет-той-же-длины-32-байта!"  # >= 32 байт: иначе InsecureKeyLengthWarning


def tamper(token: str) -> str:
    """Тот же заголовок и payload, подпись чужим ключом — отличается всегда, а не почти всегда."""
    payload = jwt.decode(token, options={"verify_signature": False})
    return jwt.encode(payload, OTHER_SECRET, algorithm="HS256")


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


def test_jwt_tampered_is_invalid(client: TestClient) -> None:
    headers = login(client, "carol")
    token = headers["Authorization"].removeprefix("Bearer ")
    r = client.get("/me", headers={"Authorization": f"Bearer {tamper(token)}"})
    assert r.status_code == 401 and r.json()["detail"] == "токен недействителен"


def test_jwt_expires_exactly_at_ttl(client: TestClient, clock: FakeClock) -> None:
    headers = login(client, "carol")  # токен выдан по часам приложения, ttl 15 минут
    clock.advance(15 * 60 - 1)
    assert client.get("/me", headers=headers).status_code == 200  # последняя живая секунда
    clock.advance(1)
    r = client.get("/me", headers=headers)
    assert r.status_code == 401 and r.json()["detail"] == "токен просрочен"


def test_jwt_from_the_future_is_invalid(client: TestClient, app: FastAPI, clock: FakeClock) -> None:
    secret: str = app.state.jwt_secret
    future = create_access_token(
        "carol", frozenset(), secret, 15, now=clock.now() + timedelta(hours=1)
    )
    r = client.get("/me", headers={"Authorization": f"Bearer {future}"})
    assert r.status_code == 401 and r.json()["detail"] == "токен недействителен"


def test_rate_limit_429(client: TestClient, app: FastAPI, clock: FakeClock) -> None:
    key = keys(app).issue("dave", frozenset())
    headers = {"X-API-Key": key}
    codes = [client.get("/me/key", headers=headers).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]  # лимит в settings_factory — 3 в минуту
    assert client.get("/me/key", headers=headers).headers["Retry-After"] == "60"  # часы стоят
    clock.advance(59)
    assert client.get("/me/key", headers=headers).headers["Retry-After"] == "1"
    clock.advance(1)  # первый запрос вышел из окна
    assert client.get("/me/key", headers=headers).status_code == 200


def test_unknown_ctx_key_is_dropped() -> None:
    from api.errors import safe_error

    err = {"type": "x", "loc": ("a",), "msg": "m", "input": "s", "ctx": {"got": "s", "ge": 1}}
    assert safe_error(err) == {"type": "x", "loc": ("a",), "msg": "m", "ctx": {"ge": 1}}


def test_tamper_always_breaks_signature() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(1000):  # каждый раз новый секрет, как новый процесс в разработке
        secret = secrets.token_urlsafe(32)
        token = create_access_token(f"user{i}", frozenset({"user"}), secret, 15, now=now)
        bad = tamper(token)
        assert bad.rsplit(".", 1)[0] == token.rsplit(".", 1)[0]  # меняется только подпись
        with pytest.raises(jwt.InvalidSignatureError, match="Signature verification failed"):
            decode_access_token(bad, secret, now=now)


@pytest.mark.parametrize(
    ("username", "password"),
    [("erin", "wrong-password-1"), ("nobody", "password-123")],
    ids=["wrong-password", "unknown-user"],
)
def test_login_failure_is_401_and_indistinguishable(
    client: TestClient, username: str, password: str
) -> None:
    client.post("/auth/register", json={"username": "erin", "password": "password-123"})
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 401
    assert r.json() == {"detail": "неверный логин или пароль"}  # один ответ на оба случая
    assert r.headers["WWW-Authenticate"] == "Bearer"
