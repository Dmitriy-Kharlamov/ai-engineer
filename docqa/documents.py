"""Папка с документами и три читающих инструмента: list_documents, search_documents, read_document.

Документ здесь — файл .md или .txt прямо в папке (без вложенных). Имя документа — имя файла;
по нему модель и читает, и ссылается. Поэтому имя проверяется по списку, а не собирается в
путь: «../secrets.txt» или «/etc/passwd» — не имя из списка, и до файловой системы такой
запрос не доходит.

Результаты урезаны структурой, а не обрезкой строки (Ф4-06, задание 9): у read_document —
целыми абзацами, с флагом truncated и полной длиной; у search_documents — не больше
MAX_HITS документов и MAX_LINES строк в каждом, со счётом совпадений. Модель видит, что
видит не всё.

Тексты отказов — из PublicToolError: в них только имена из своей папки, без того, что
написала модель.

quarantine=True (по умолчанию): строки, похожие на инструкцию для ИИ, до модели не доходят
ни через read_document, ни через search_documents (docqa.guards; Ф4-08, отказ 10).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragkit.tools import PublicToolError, ToolRegistry

from .guards import quarantine as clean_text
from .notes import NoteStore, SaveNoteArgs

SUFFIXES = (".md", ".txt")
MAX_HITS = 5
MAX_LINES = 3
LINE_CHARS = 200
WORD = re.compile(r"\w{3,}")


def stem(word: str) -> str:
    """Грубая основа русского слова: первые 5 букв. «доставка», «доставки» → «доста»."""
    return word.casefold().replace("ё", "е")[:5]


@dataclass(frozen=True)
class DocumentStore:
    root: Path
    max_read_chars: int = 4000
    quarantine: bool = True

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES
        )

    def _path(self, name: str) -> Path:
        names = self.names()
        if name not in names:
            raise PublicToolError(f"такого документа нет. Доступны: {', '.join(names) or 'ничего'}")
        return self.root / name

    def raw_text(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")

    def text(self, name: str) -> str:
        """Текст, который увидит модель: без строк-инструкций, если включён карантин."""
        raw = self.raw_text(name)
        return clean_text(raw)[0] if self.quarantine else raw

    def quarantined(self, name: str) -> list[str]:
        """Какие строки документа карантин не пропустил к модели (для журнала и CLI)."""
        return clean_text(self.raw_text(name))[1] if self.quarantine else []

    def list(self) -> dict[str, Any]:
        out = []
        for name in self.names():
            text = self.text(name)
            title = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
            out.append({"name": name, "title": title, "chars": len(text)})
        return {"documents": out, "count": len(out)}

    def read(self, name: str) -> dict[str, Any]:
        text = self.text(name)
        if len(text) <= self.max_read_chars:
            return {"name": name, "chars": len(text), "truncated": False, "text": text}
        # длинный: дальше — по абзацам
        kept: list[str] = []
        size = 0
        for paragraph in text.split("\n\n"):  # целыми абзацами
            if size + len(paragraph) + 2 > self.max_read_chars:
                break
            kept.append(paragraph)
            size += len(paragraph) + 2
        return {"name": name, "chars": len(text), "truncated": True, "text": "\n\n".join(kept)}

    def search(self, query: str) -> dict[str, Any]:
        stems = {stem(w) for w in WORD.findall(query)}
        if not stems:
            raise PublicToolError("в запросе нет слов длиной от трёх букв")
        hits: list[dict[str, Any]] = []
        for name in self.names():
            lines = self.text(name).splitlines()
            found: list[dict[str, Any]] = []
            matched: set[str] = set()
            for number, line in enumerate(lines, start=1):
                here = {stem(w) for w in WORD.findall(line)} & stems
                if here:
                    matched |= here
                    found.append({"line": number, "text": line.strip()[:LINE_CHARS]})
            if found:
                hits.append(
                    {
                        "name": name,
                        "matched_words": len(matched),
                        "matched_lines": len(found),
                        "lines": found[:MAX_LINES],
                    }
                )
        hits.sort(
            key=lambda h: (-int(h["matched_words"]), -int(h["matched_lines"]), str(h["name"]))
        )
        return {"found": len(hits), "shown": min(len(hits), MAX_HITS), "results": hits[:MAX_HITS]}


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=200, description="Слова для поиска, по-русски")


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Имя файла из list_documents, например delivery.md",
    )


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_registry(store: DocumentStore, notes: NoteStore | None = None) -> ToolRegistry:
    """Реестр трёх читающих инструментов над папкой и, если дана папка заметок, save_note.
    Схемы аргументов — pydantic."""
    registry = ToolRegistry()

    @registry.tool(NoArgs, name="list_documents")
    def _list(args: NoArgs) -> dict[str, Any]:
        """Список документов в папке: имя файла, заголовок, длина в символах."""
        return store.list()

    @registry.tool(SearchArgs, name="search_documents")
    def _search(args: SearchArgs) -> dict[str, Any]:
        """Поиск по словам во всех документах: какие документы и строки содержат слова запроса."""
        return store.search(args.query)

    @registry.tool(ReadArgs, name="read_document")
    def _read(args: ReadArgs) -> dict[str, Any]:
        """Текст документа по имени файла. Длинный документ приходит урезанным по абзацам."""
        return store.read(args.name)

    if notes is not None:

        @registry.tool(SaveNoteArgs, name="save_note")
        def _save(args: SaveNoteArgs) -> dict[str, Any]:
            """Сохранить заметку в папку заметок пользователя. Только когда пользователь сам
            попросил что-то сохранить или записать. Пользователь подтверждает запись; если
            не подтвердил — заметки нет, скажи об этом."""
            return notes.save(args)

    return registry
