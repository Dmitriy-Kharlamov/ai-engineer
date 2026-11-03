"""Клиент к чужому API — PyPI JSON API (https://pypi.org/pypi/<пакет>/json).

Учебный «провайдер» для Ф3-10: настоящий, публичный, без ключа. Ответ разбираем
pydantic-моделью: если PyPI поменяет форму, упадёт разбор, а не код дальше по цепочке.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel

from services.http_retry import RetryLog, RetryPolicy, Sleep, request_with_retry

PYPI_URL = "https://pypi.org/pypi"


class PackageNotFoundOnPyPIError(Exception):
    """На PyPI нет такого пакета (404). Не повторяем: ответ окончательный."""


class PackageInfo(BaseModel):
    name: str
    version: str
    summary: str | None = None


class _Payload(BaseModel):
    info: PackageInfo


async def fetch_package(
    client: httpx.AsyncClient,
    name: str,
    *,
    policy: RetryPolicy | None = None,
    sleep: Sleep = asyncio.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: RetryLog | None = None,
) -> PackageInfo:
    try:
        response = await request_with_retry(
            client, "GET", f"{PYPI_URL}/{name}/json", policy=policy, sleep=sleep, now=now, log=log
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            raise PackageNotFoundOnPyPIError(name) from None
        raise
    return _Payload.model_validate(response.json()).info
