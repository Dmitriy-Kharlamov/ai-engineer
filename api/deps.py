"""Зависимости: настройки, сессия «базы», репозитории, состояние из app.state.

Цепочка из задания 3:  get_settings ─┬─> get_db ──> get_repo ──> обработчик
                                     └──────────────^
get_settings нужен и get_db, и get_repo, но за запрос вызывается один раз (кеш Depends).
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from ragkit import Embedder, HashEmbedder
from ragkit.config import Settings, get_settings
from services.apikeys import ApiKeyStore
from services.db import Database, Session
from services.documents import DocumentRepo
from services.ratelimit import RateLimiter
from services.users import UserStore


def load_embedder(dim: int = 128) -> Embedder:
    """«Тяжёлая» загрузка модели. Вызывается в lifespan ровно один раз.

    Тесты считают загрузки, подменяя эту функцию через app.dependency_overrides:
    счётчик живёт в тесте, а не глобальной переменной в продакшн-коде.
    """
    return HashEmbedder(dim=dim)


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_database(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def get_api_keys(request: Request) -> ApiKeyStore:
    store: ApiKeyStore = request.app.state.api_keys
    return store


def get_users(request: Request) -> UserStore:
    users: UserStore = request.app.state.users
    return users


def get_rate_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


def get_db(
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[Session]:
    """Сессия = транзакция запроса.

    До yield — открыть; после — commit или rollback; в finally — всегда close.
    """
    session = database.session(url=settings.database_url)
    try:
        yield session
        session.commit()  # обработчик отработал без исключения
    except Exception:
        session.rollback()  # в том числе HTTPException
        raise
    finally:
        session.close()


def get_repo(
    session: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentRepo:
    return DocumentRepo(session, max_per_owner=settings.max_documents_per_owner)
