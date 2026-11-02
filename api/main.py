"""Сборка приложения — и больше ничего.

Запуск: uvicorn api.main:create_app --factory --reload
Фабрика вместо модульного `app`: настройки читаются при вызове, а не при импорте, поэтому
тесты и скрипты передают свои Settings, а учебный /lab включается только при debug=True.
"""

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from api import deps
from api.errors import install_error_handlers
from api.routers import admin, auth, chunks, documents, health, lab, me
from ragkit.config import Settings, get_settings
from services.apikeys import ApiKeyStore
from services.db import Database
from services.ratelimit import RateLimiter
from services.users import UserStore


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.started_at = datetime.now(UTC)
        app.state.loads_before = deps.EMBEDDER_LOADS
        app.state.embedder = deps.load_embedder(dim=128)  # один раз за запуск
        app.state.db = Database()
        app.state.users = UserStore(rounds=cfg.bcrypt_rounds)
        app.state.api_keys = ApiKeyStore()
        app.state.rate_limiter = RateLimiter(limit=cfg.rate_limit_per_minute, window=60.0)
        secret = cfg.jwt_secret.get_secret_value() if cfg.jwt_secret else None
        app.state.jwt_secret = secret or secrets.token_urlsafe(32)  # dev: новый на запуск
        yield

    app = FastAPI(title="ragkit API", version="0.1.0", lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: cfg  # приложение видит ровно эти настройки
    install_error_handlers(app)
    for router in (health.router, chunks.router, auth.router, me.router, documents.router):
        app.include_router(router)
    app.include_router(admin.router)
    if cfg.debug:
        app.include_router(lab.router)  # учебные эндпоинты — только в отладке
    return app
