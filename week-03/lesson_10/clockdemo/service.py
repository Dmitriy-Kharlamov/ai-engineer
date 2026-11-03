"""Потребитель, ранняя привязка: имя now копируется в этот модуль один раз, при импорте."""

from clockdemo.clock import now


def stamp() -> str:
    return f"событие в {now()}"
