"""Журнал событий: одна строка JSON на событие. Это столбец «что попадает в лог».

Что пишется: каждое обращение к модели (оценка входа, цена, итог счётчиков), каждый вызов
инструмента (что просила модель — как есть, чем кончилось, исходная ошибка), каждый повтор
после 429/503, каждая сработавшая защита, конец хода или отказ с типом исключения.

Чего нет: текстов документов и ответов целиком — только длины и первые LOG_TEXT символов.
Журнал читает тот, кто разбирает сбой, а не пользователь; копия папки ему не нужна.
"""

import json
import time
from pathlib import Path
from typing import Any, Protocol, TextIO

LOG_TEXT = 200


class Events(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


class NoEvents:
    def emit(self, event: str, **fields: Any) -> None:
        return None


class EventLog:
    def __init__(self, stream: TextIO, clock: Any = time.time) -> None:
        self.stream = stream
        self.clock = clock

    @classmethod
    def open(cls, path: Path) -> "EventLog":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path.open("a", encoding="utf-8"))

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": round(self.clock(), 3), "event": event, **fields}
        self.stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def clip(text: str, limit: int = LOG_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit})"
