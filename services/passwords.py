"""Пароли: bcrypt напрямую (passlib не поддерживается).

Живёт в services, а не в api/security.py: им пользуется services/users.py, а сервис
не может импортировать api — направление зависимостей важнее «логичного» места.
"""

from functools import cache

import bcrypt


def hash_password(password: str, rounds: int = 12) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=rounds))  # соль — внутри хеша


def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)


@cache
def dummy_hash(rounds: int) -> bytes:
    """Фиктивный хеш той же стоимости: проверяем его, когда пользователя нет."""
    return hash_password("пароль-которого-нет", rounds)
