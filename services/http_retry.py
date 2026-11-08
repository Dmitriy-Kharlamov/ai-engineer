"""Повтор HTTP-запросов к чужим API — retry из Ф3-03, перенесённый в пакет и типизированный.

Что изменилось относительно week-03/lesson_03.py:
- пауза и случайность внедряются (sleep=, rng=, now=) — тест не ждёт и знает каждую паузу;
- Retry-After понимается в обоих форматах RFC 9110: секунды и HTTP-дата;
- print заменён на возвращаемый журнал попыток;
- джиттер — аддитивный (как было) или полный, jitter_mode="full" (Ф4-07, задание 6):
  аддитивный не даёт паузе быть меньше экспоненты, полный тянет её от нуля до экспоненты.
  В Ф3-03 оба меряли симуляцией; полный в пакете появился, когда понадобилось мерить живьём.

Retry-After, если он есть и respect_retry_after=True, заменяет формулу целиком: без джиттера
и без cap. Это два открытых вопроса, не решения: одинаковый Retry-After синхронизирует всех
клиентов, которых задел один и тот же лимит, а огромный Retry-After клиент честно проспит.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

import httpx

IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"})
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
MAYBE_DONE = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.RemoteProtocolError)

Sleep = Callable[[float], Awaitable[None]]


def should_retry(outcome: int | type[Exception], method: str) -> bool:
    """outcome — код ответа или класс исключения httpx."""
    if isinstance(outcome, int):
        return outcome in RETRY_STATUSES
    if issubclass(outcome, NOT_SENT):  # запрос точно не ушёл на сервер
        return True
    if issubclass(outcome, MAYBE_DONE):  # мог выполниться — повторяем только идемпотентный
        return method.upper() in IDEMPOTENT
    return False


def retry_after_seconds(value: str | None, now: datetime) -> float | None:
    """Retry-After: "120" или "Wed, 21 Oct 2026 07:28:00 GMT"; мусор и прошлое → None/0."""
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None  # непонятный заголовок — пауза по формуле, а не падение
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - now).total_seconds())


@dataclass
class RetryPolicy:
    attempts: int = 5
    base: float = 0.5
    cap: float = 8.0
    jitter: float = 0.5  # аддитивный: сколько секунд случайности сверх экспоненты
    respect_retry_after: bool = True
    jitter_mode: Literal["additive", "full"] = "additive"

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts должно быть ≥ 1, получено {self.attempts}")


def backoff(policy: RetryPolicy, number: int, rng: Callable[[], float]) -> float:
    """Пауза после неудачной попытки number (с 1) по формуле, без Retry-After."""
    exp = min(policy.cap, policy.base * 2.0 ** (number - 1))
    if policy.jitter_mode == "full":
        return rng() * exp
    return exp + rng() * policy.jitter


@dataclass
class Attempt:
    number: int
    outcome: str  # "200", "503", "ConnectError"
    pause: float | None = None  # None — пауз больше не было
    source: str = ""  # "формула" или "Retry-After"


@dataclass
class RetryLog:
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def pauses(self) -> list[float]:
        return [a.pause for a in self.attempts if a.pause is not None]


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    policy: RetryPolicy | None = None,
    sleep: Sleep = asyncio.sleep,
    rng: Callable[[], float] = random.random,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: RetryLog | None = None,
) -> httpx.Response:
    """Ответ < 400 — вернуть; иначе повторить, если можно; не вышло — исключение httpx."""
    p = policy if policy is not None else RetryPolicy()
    journal = log if log is not None else RetryLog()
    for number in range(1, p.attempts + 1):
        response: httpx.Response | None = None
        outcome: int | type[Exception]
        try:
            response = await client.request(method, url)
        except httpx.TransportError as error:
            outcome = type(error)
            if number == p.attempts or not should_retry(outcome, method):
                journal.attempts.append(Attempt(number, outcome.__name__))
                raise
        else:
            if response.status_code < 400:
                journal.attempts.append(Attempt(number, str(response.status_code)))
                return response
            outcome = response.status_code
            if number == p.attempts or not should_retry(outcome, method):
                journal.attempts.append(Attempt(number, str(outcome)))
                response.raise_for_status()
        pause = backoff(p, number, rng)
        source = "формула"
        if p.respect_retry_after and response is not None:
            hinted = retry_after_seconds(response.headers.get("Retry-After"), now())
            if hinted is not None:
                pause, source = hinted, "Retry-After"
        name = str(outcome) if isinstance(outcome, int) else outcome.__name__
        journal.attempts.append(Attempt(number, name, pause, source))
        await sleep(pause)
    raise AssertionError("недостижимо: последний проход цикла возвращает или бросает")
