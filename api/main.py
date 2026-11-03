"""Сборка приложения — и больше ничего.

Запуск: uvicorn api.main:create_app --factory --reload
Фабрика вместо модульного `app`: настройки читаются при вызове, а не при импорте, поэтому
тесты и скрипты передают свои Settings, а учебный /lab включается только при debug=True.
"""

import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import deps
from api.errors import install_error_handlers
from api.routers import admin, auth, chunks, documents, health, lab, me
from ragkit import Embedder
from ragkit.config import Settings, get_settings
from services.apikeys import ApiKeyStore
from services.clock import Clock, SystemClock
from services.db import Database
from services.ratelimit import RateLimiter
from services.users import UserStore


def create_app(settings: Settings | None = None, *, clock: Clock | None = None) -> FastAPI:
    cfg = settings if settings is not None else get_settings()
    time_source: Clock = clock if clock is not None else SystemClock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.clock = time_source  # все, кто читает время, берут его отсюда
        app.state.started_at = time_source.now()
        # Загрузчик подменяем тем же механизмом, что и зависимости: тест кладёт
        # свою функцию в app.dependency_overrides[load_embedder] и сам считает вызовы.
        load: Callable[..., Embedder] = app.dependency_overrides.get(
            deps.load_embedder, deps.load_embedder
        )
        app.state.embedder = load(dim=128)  # один раз за запуск
        app.state.db = Database()
        app.state.users = UserStore(rounds=cfg.bcrypt_rounds)
        app.state.api_keys = ApiKeyStore()
        app.state.rate_limiter = RateLimiter(
            limit=cfg.rate_limit_per_minute, window=60.0, clock=time_source.monotonic
        )
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
