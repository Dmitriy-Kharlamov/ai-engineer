"""Источник: «настоящие» часы."""

from datetime import UTC, datetime


def now() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")
