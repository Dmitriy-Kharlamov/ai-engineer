"""Пользователи в памяти: регистрация и вход с bcrypt."""

import threading
from dataclasses import dataclass

from services.passwords import check_password, dummy_hash, hash_password


class UserExistsError(Exception):
    """Логин уже занят."""


@dataclass(frozen=True)
class User:
    username: str
    roles: frozenset[str]


@dataclass(frozen=True)
class _Record:
    user: User
    password_hash: bytes


class UserStore:
    def __init__(self, rounds: int = 12) -> None:
        self.rounds = rounds
        self._users: dict[str, _Record] = {}
        self._lock = threading.Lock()  # register — обычный def, FastAPI зовёт его из пула потоков

    def register(self, username: str, password: str, roles: frozenset[str] = frozenset()) -> User:
        if username in self._users:  # быстрый отказ без bcrypt; окончательно решает блокировка
            raise UserExistsError(username)
        user = User(username, frozenset({"user"}) | roles)
        record = _Record(user, hash_password(password, self.rounds))  # дорогое — вне блокировки
        with self._lock:  # проверка и запись — одно действие (Ф3-10: найдено гонкой в async-тесте)
            if username in self._users:
                raise UserExistsError(username)
            self._users[username] = record
        return user

    def authenticate(
        self, username: str, password: str, *, equalize_timing: bool = True
    ) -> User | None:
        record = self._users.get(username)
        if record is None:
            if equalize_timing:
                # Нет пользователя — всё равно платим за bcrypt, иначе время ответа
                # подсказывает, какие логины существуют.
                check_password(password, dummy_hash(self.rounds))
            return None
        return record.user if check_password(password, record.password_hash) else None
