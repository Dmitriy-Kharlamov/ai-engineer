"""Потребитель, поздняя привязка: модуль импортирован целиком, now ищется при каждом вызове."""

from clockdemo import clock


def stamp() -> str:
    return f"событие в {clock.now()}"
