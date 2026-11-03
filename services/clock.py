"""Часы как зависимость (Ф3-10, задание 7) — по образцу CircuitBreaker(clock=...) из Ф3-03.

Кто читает время, получает Clock параметром: зависимость от времени видна в подписи,
а тест передаёт FakeClock и двигает время сам, без sleep и без patch.
"""

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Стенное время (UTC) — для exp в токенах и меток времени."""
        ...

    def monotonic(self) -> float:
        """Монотонные секунды — для интервалов (лимитер); не прыгают при смене часов ОС."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """Фейк для тестов: время стоит, пока его не сдвинут. Протокол Clock проверяет mypy."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock ждёт время с часовым поясом")
        self._now = start
        self._monotonic = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("время назад не идёт")
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds
