"""Аутентификация (кто ты → 401) и авторизация (можно ли тебе → 403) для HTTP-слоя."""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from api.deps import get_api_keys, get_rate_limiter
from ragkit.config import Settings, get_settings
from services.apikeys import ApiKeyStore, Principal
from services.ratelimit import RateLimiter

ALGORITHM = "HS256"

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer = HTTPBearer(auto_error=False)


def unauthorized(scheme: str, detail: str = "нужна аутентификация") -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": scheme}
    )


# --- JWT -------------------------------------------------------------------------
def jwt_secret(request: Request) -> str:
    secret: str = request.app.state.jwt_secret
    return secret


def create_access_token(subject: str, roles: frozenset[str], secret: str, ttl_minutes: int) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "roles": sorted(roles),
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    # algorithms обязателен: алгоритм выбираем мы, а не заголовок токена
    data: dict[str, Any] = jwt.decode(token, secret, algorithms=[ALGORITHM])
    return data


def current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
    secret: Annotated[str, Depends(jwt_secret)],
) -> Principal:
    if creds is None:
        raise unauthorized("Bearer")
    try:
        data = decode_access_token(creds.credentials, secret)
    except jwt.ExpiredSignatureError:
        raise unauthorized("Bearer", "токен просрочен") from None
    except jwt.InvalidTokenError:
        raise unauthorized("Bearer", "токен недействителен") from None
    return Principal(name=data["sub"], roles=frozenset(data.get("roles", [])))


# --- API-ключ ----------------------------------------------------------------------
def require_api_key(
    key: Annotated[str | None, Security(api_key_header)],
    store: Annotated[ApiKeyStore, Depends(get_api_keys)],
) -> Principal:
    if key is None:
        raise unauthorized("ApiKey", "нужен заголовок X-API-Key")
    principal = store.verify(key)
    if principal is None:
        raise unauthorized("ApiKey", "неверный ключ")
    return principal


def require_admin(principal: Annotated[Principal, Depends(require_api_key)]) -> Principal:
    if "admin" not in principal.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="нужна роль admin")
    return principal


# --- Ограничение частоты -------------------------------------------------------------
def rate_limited(
    principal: Annotated[Principal, Depends(require_api_key)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> Principal:
    retry_after = limiter.hit(principal.name)
    if retry_after is not None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="слишком много запросов",
            headers={"Retry-After": str(retry_after)},
        )
    return principal


SettingsDep = Annotated[Settings, Depends(get_settings)]
