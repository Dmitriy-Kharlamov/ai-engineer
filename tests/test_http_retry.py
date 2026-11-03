"""Retry к чужому API без сети: respx отвечает вместо PyPI, пауза — фейк, который не ждёт."""

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from services.http_retry import RetryLog, RetryPolicy, retry_after_seconds
from services.pypi import PYPI_URL, PackageNotFoundOnPyPIError, fetch_package

FIXTURES = Path(__file__).parent / "fixtures"
URL = f"{PYPI_URL}/respx/json"
NOW = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
NO_JITTER = RetryPolicy(attempts=3, base=0.5, cap=8.0, jitter=0.0)


def recorded(name: str) -> httpx.Response:
    """Ответ, однажды записанный с живого PyPI (см. _note в файле)."""
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return httpx.Response(data["status"], headers=data["headers"], json=data["json"])


class SleepRecorder:
    """Фейк asyncio.sleep: запоминает паузу и не ждёт."""

    def __init__(self) -> None:
        self.pauses: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.pauses.append(seconds)


@pytest.fixture
def sleep() -> SleepRecorder:
    return SleepRecorder()


async def fetch(sleep: SleepRecorder, log: RetryLog | None = None) -> str:
    async with httpx.AsyncClient() as client:
        info = await fetch_package(
            client, "respx", policy=NO_JITTER, sleep=sleep, now=lambda: NOW, log=log
        )
    return info.version


async def test_retry_on_503_then_200(respx_mock: respx.MockRouter, sleep: SleepRecorder) -> None:
    route = respx_mock.get(URL).mock(
        side_effect=[httpx.Response(503), recorded("pypi_respx_200.json")]
    )
    assert await fetch(sleep) == "0.23.1"
    assert route.call_count == 2
    assert sleep.pauses == [0.5]  # base * 2**0, без джиттера


async def test_retry_after_seconds_is_respected(
    respx_mock: respx.MockRouter, sleep: SleepRecorder
) -> None:
    respx_mock.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            recorded("pypi_respx_200.json"),
        ]
    )
    log = RetryLog()
    assert await fetch(sleep, log) == "0.23.1"
    assert sleep.pauses == [7.0]  # а не 0.5 по формуле
    assert [a.source for a in log.attempts] == ["Retry-After", ""]


async def test_retry_after_http_date_is_respected(
    respx_mock: respx.MockRouter, sleep: SleepRecorder
) -> None:
    when = format_datetime(
        NOW + timedelta(seconds=30), usegmt=True
    )  # "Thu, 01 Jan 2026 09:00:30 GMT"
    respx_mock.get(URL).mock(
        side_effect=[
            httpx.Response(503, headers={"Retry-After": when}),
            recorded("pypi_respx_200.json"),
        ]
    )
    assert await fetch(sleep) == "0.23.1"
    assert sleep.pauses == [30.0]


async def test_404_is_final_not_retried(respx_mock: respx.MockRouter, sleep: SleepRecorder) -> None:
    route = respx_mock.get(f"{PYPI_URL}/no-such-package-xyz-123/json").mock(
        return_value=recorded("pypi_missing_404.json")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(PackageNotFoundOnPyPIError, match="no-such-package-xyz-123"):
            await fetch_package(client, "no-such-package-xyz-123", sleep=sleep)
    assert route.call_count == 1
    assert sleep.pauses == []


async def test_gives_up_after_last_attempt(
    respx_mock: respx.MockRouter, sleep: SleepRecorder
) -> None:
    route = respx_mock.get(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError, match="503 Service Unavailable"):
        await fetch(sleep)
    assert route.call_count == 3  # attempts=3
    assert sleep.pauses == [0.5, 1.0]  # пауз на одну меньше, чем попыток


async def test_connect_error_is_retried(respx_mock: respx.MockRouter, sleep: SleepRecorder) -> None:
    respx_mock.get(URL).mock(
        side_effect=[httpx.ConnectError("нет связи"), recorded("pypi_respx_200.json")]
    )
    assert await fetch(sleep) == "0.23.1"
    assert sleep.pauses == [0.5]


@pytest.mark.parametrize(
    ("header", "seconds"),
    [
        ("120", 120.0),
        ("Thu, 01 Jan 2026 09:02:00 GMT", 120.0),
        ("Thu, 01 Jan 2026 08:00:00 GMT", 0.0),  # дата в прошлом — повторять сразу
        ("скоро", None),  # мусор — пауза по формуле, а не падение
        (None, None),
    ],
    ids=["delta-seconds", "http-date", "date-in-past", "garbage", "absent"],
)
def test_retry_after_parsing(header: str | None, seconds: float | None) -> None:
    assert retry_after_seconds(header, NOW) == seconds
