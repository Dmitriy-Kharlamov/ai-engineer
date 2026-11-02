"""Учебная «база» в памяти с сессиями и транзакциями.

Сессия копит изменения и применяет их только на commit(); rollback() их выбрасывает.
Счётчики Database.stats доказывают, что открыто, зафиксировано, откачено и закрыто.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count


@dataclass(frozen=True)
class StoredDocument:
    id: int
    owner: str
    name: str
    text: str


@dataclass
class SessionStats:
    opened: int = 0
    committed: int = 0
    rolled_back: int = 0
    closed: int = 0


class SessionClosedError(RuntimeError):
    pass


@dataclass
class Database:
    documents: dict[int, StoredDocument] = field(default_factory=dict)
    stats: SessionStats = field(default_factory=SessionStats)
    _ids: Iterator[int] = field(default_factory=lambda: count(1))

    def session(self, url: str = "memory://") -> "Session":
        self.stats.opened += 1
        return Session(self, url)


class Session:
    def __init__(self, db: Database, url: str) -> None:
        self._db = db
        self.url = url  # в настоящей базе — строка подключения из настроек
        self._pending: dict[int, StoredDocument] = {}
        self.closed = False

    def _check_open(self) -> None:
        if self.closed:
            raise SessionClosedError("сессия уже закрыта")

    def add(self, owner: str, name: str, text: str) -> StoredDocument:
        self._check_open()
        doc = StoredDocument(id=next(self._db._ids), owner=owner, name=name, text=text)
        self._pending[doc.id] = doc
        return doc

    def get(self, doc_id: int) -> StoredDocument | None:
        self._check_open()
        return self._pending.get(doc_id) or self._db.documents.get(doc_id)

    def count_by_owner(self, owner: str) -> int:
        self._check_open()
        stored = [d for d in self._db.documents.values() if d.owner == owner]
        return len(stored) + sum(d.owner == owner for d in self._pending.values())

    def commit(self) -> None:
        self._check_open()
        self._db.documents.update(self._pending)
        self._pending.clear()
        self._db.stats.committed += 1

    def rollback(self) -> None:
        self._pending.clear()
        self._db.stats.rolled_back += 1

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._db.stats.closed += 1
