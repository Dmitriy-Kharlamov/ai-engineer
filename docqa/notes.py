"""save_note — единственный инструмент, который что-то меняет: пишет заметку в папку заметок.

Подтверждение стоит внутри инструмента, а не в цикле: любой путь к записи — вызов модели,
вызов, разобранный из текста (№ 4), будущий код — проходит через confirm. Без «да» файла нет.
Пользователь видит ровно то, что будет записано: путь, заголовок и текст целиком.

Почему подтверждение: аргументы пишет модель, а модель выполняет инструкции из документов
(Ф4-08, замер № 10: явную — 10 из 10). Документ, который велит «сохрани заметку», получил бы
право писать на диск. Запись необратима в том смысле, что файл остаётся после сессии.

Чего инструмент не делает: не перезаписывает (файл с таким именем — отказ), не выходит из
папки заметок (имя файла строит код из заголовка, путь из модели не берётся), не пишет в папку
документов (заметка не должна становиться источником для следующих ответов).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragkit.tools import PublicToolError

MAX_TITLE = 80
MAX_TEXT = 2000


class SaveNoteArgs(BaseModel):
    """Сохранить заметку пользователя в его папку заметок. Только когда пользователь сам
    попросил сохранить или записать; пользователь подтвердит запись."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=MAX_TITLE, description="заголовок заметки")
    text: str = Field(min_length=1, max_length=MAX_TEXT, description="текст заметки целиком")


@dataclass(frozen=True)
class NoteRequest:
    """Что именно будет записано — это и показывается пользователю."""

    path: Path
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"# {self.title}\n\n{self.text}\n"


Confirm = Callable[[NoteRequest], bool]


class NotConfirmedError(PublicToolError):
    """Пользователь не сказал «да». Отдельный тип — чтобы код отличал отказ от сбоя записи."""


def slug(title: str) -> str:
    """Имя файла из заголовка: буквы, цифры, дефис; не длиннее 60 символов."""
    words = re.findall(r"[0-9a-zа-яё]+", title.casefold())
    return "-".join(words)[:60].strip("-") or "note"


@dataclass(frozen=True)
class NoteStore:
    root: Path
    confirm: Confirm

    def request(self, args: SaveNoteArgs) -> NoteRequest:
        path = self.root / f"{slug(args.title)}.md"
        if path.exists():
            raise PublicToolError(
                f"заметка {path.name} уже есть; перезаписывать нельзя — выбери другой заголовок"
            )
        return NoteRequest(path, args.title.strip(), args.text.strip())

    def save(self, args: SaveNoteArgs) -> dict[str, Any]:
        request = self.request(args)
        if not self.confirm(request):
            raise NotConfirmedError(
                "пользователь не подтвердил запись; заметка НЕ сохранена. Скажи об этом прямо"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        with request.path.open("x", encoding="utf-8") as f:  # «x»: не перезаписать гонкой
            f.write(request.content)
        return {"saved": True, "file": request.path.name, "chars": len(request.text)}
