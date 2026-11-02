"""API-ключи: на сервере хранится только хеш, ключ показывается один раз при выпуске.

sha256, а не bcrypt: ключ — 256 случайных бит, перебирать его бессмысленно, и медленный
хеш нужен только для паролей, которые придумывают люди. Сравнение — compare_digest.
"""

import hashlib
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    name: str
    roles: frozenset[str]


def key_digest(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class ApiKeyStore:
    def __init__(self) -> None:
        self._by_digest: dict[str, Principal] = {}

    def issue(self, name: str, roles: frozenset[str]) -> str:
        raw = "rk_" + secrets.token_urlsafe(32)
        self._by_digest[key_digest(raw)] = Principal(name, roles)
        return raw  # единственный момент, когда ключ существует в открытом виде

    def verify(self, raw_key: str) -> Principal | None:
        digest = key_digest(raw_key)
        found: Principal | None = None
        for stored, principal in self._by_digest.items():
            if secrets.compare_digest(stored, digest):  # не ==: время не зависит от совпадения
                found = principal
        return found

    def stored_digests(self) -> list[str]:
        return list(self._by_digest)
