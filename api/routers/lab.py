"""Опыты урока Ф3-06: источники параметров, def против async def, утечки."""

import asyncio
import threading
import time
from typing import Annotated

from fastapi import APIRouter, Header, Path, Query

from api.schemas import Login, ParamsBody, ParamsOut, UserDB, UserIn, UserOut, UserOutStrict

router = APIRouter(prefix="/lab", tags=["lab"])

SLEEP = 1.0


# --- задание 2: откуда берётся параметр ------------------------------------------
@router.post("/params/{doc_id}")
async def params(
    doc_id: Annotated[int, Path(ge=1)],  # прогноз: есть в пути → path
    body: ParamsBody,  # прогноз: pydantic-модель → тело
    limit: Annotated[int, Query(ge=1, le=100)] = 10,  # прогноз: скаляр → query
    x_request_id: Annotated[str | None, Header()] = None,  # прогноз: заголовок X-Request-Id
) -> ParamsOut:
    return ParamsOut(doc_id=doc_id, limit=limit, x_request_id=x_request_id, body=body)


@router.get("/same-name/{item_id}")
async def same_name(item_id: int) -> dict[str, int]:
    """Скаляр с тем же именем, что часть пути: берётся из пути, ?item_id=… игнорируется."""
    return {"item_id": item_id}


# --- задание 3: def против async def ---------------------------------------------
def _thread() -> dict[str, int]:
    return {"thread": threading.get_ident()}


@router.get("/sleep/sync-def")
def sleep_sync_def() -> dict[str, int]:
    time.sleep(SLEEP)  # обычный def: FastAPI сам уводит в пул потоков
    return _thread()


@router.get("/sleep/async-blocking")
async def sleep_async_blocking() -> dict[str, int]:
    time.sleep(SLEEP)  # ПЛОХО: блокирует цикл событий
    return _thread()


@router.get("/sleep/async-sleep")
async def sleep_async_sleep() -> dict[str, int]:
    await asyncio.sleep(SLEEP)  # починка 1: неблокирующий аналог
    return _thread()


@router.get("/sleep/async-to-thread")
async def sleep_async_to_thread() -> dict[str, int]:
    await asyncio.to_thread(time.sleep, SLEEP)  # починка 2: блокирующее — в поток
    return _thread()


@router.get("/sleep/fixed-def")
def sleep_fixed_def() -> dict[str, int]:
    time.sleep(SLEEP)  # починка 3: просто убрать async
    return _thread()


# --- задание 4: утечка в 422 ---------------------------------------------------------
@router.post("/login")
async def login(body: Login) -> dict[str, bool]:
    return {"ok": True}


# --- задание 5: три модели -----------------------------------------------------------
def _load_user(user: UserIn) -> UserDB:
    """Имитация записи в базу: хеш вместо пароля и внутренний id."""
    fake_hash = "$2b$12$" + str(abs(hash(user.password.get_secret_value())))
    return UserDB(name=user.name, password_hash=fake_hash, internal_id=42)


@router.post("/users-leak")
async def create_user_leak(user: UserIn) -> UserDB:  # аннотация = схема ответа: утечка
    return _load_user(user)


@router.post("/users", response_model=UserOut)
async def create_user(user: UserIn) -> UserDB:  # response_model фильтрует до UserOut
    return _load_user(user)


@router.post("/users-missing", response_model=UserOutStrict)
async def create_user_missing(user: UserIn) -> UserDB:  # в UserDB нет email
    return _load_user(user)
