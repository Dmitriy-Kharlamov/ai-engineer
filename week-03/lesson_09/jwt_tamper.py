"""Задание 3: почему порча последних символов подписи иногда ничего не портит.

Запуск: python week-03/lesson_09/jwt_tamper.py
"""

import base64
import secrets
import string
import warnings
from datetime import UTC, datetime

import jwt

from api.security import create_access_token, decode_access_token

B64URL = string.ascii_uppercase + string.ascii_lowercase + string.digits + "-_"


def sig_bytes(token: str) -> bytes:
    """Подпись так, как её декодирует обычный base64: лишние 2 бита молча отбрасываются."""
    sig = token.rsplit(".", 1)[1]
    return base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))


def valid(token: str, secret: str) -> bool:
    try:
        decode_access_token(token, secret, now=datetime.now(UTC))
    except jwt.InvalidTokenError:
        return False
    return True


def old_tamper(token: str) -> str:
    return token[:-2] + ("AA" if not token.endswith("AA") else "BB")


def new_tamper(token: str) -> str:
    payload = jwt.decode(token, options={"verify_signature": False})  # payload читается без ключа
    return jwt.encode(payload, "другой-секрет-той-же-длины-32-байта!", algorithm="HS256")


def main() -> None:
    secret = secrets.token_urlsafe(32)
    token = create_access_token(
        "carol", frozenset({"user"}), secret, ttl_minutes=15, now=datetime.now(UTC)
    )
    sig = token.rsplit(".", 1)[1]
    raw = base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))

    print("--- 1. Откуда 43 символа")
    print(f"HMAC-SHA256: {len(raw)} байта = {len(raw) * 8} бит")
    print(
        f"base64url: 6 бит на символ, 256 / 6 = {256 / 6:.2f} -> {len(sig)} символа = {6 * 43} бит"
    )
    print(f"в последнем символе {256 - 42 * 6} бита данных и {43 * 6 - 256} бита-заполнителя")

    print("--- 2. Сколько вариантов последнего символа дают ту же подпись")
    same_bytes = [c for c in B64URL if sig_bytes(token[:-1] + c) == raw]
    print(
        f"base64: последний символ {sig[-1]!r}, те же 32 байта дают {len(same_bytes)} из 64 "
        f"-> {same_bytes}"
    )
    same_two = sum(sig_bytes(token[:-2] + a + b) == raw for a in B64URL for b in B64URL)
    print(f"base64: два последних символа, те же байты дают {same_two} из 4096")
    accepted = [c for c in B64URL if valid(token[:-1] + c, secret)]
    print(f"PyJWT {jwt.__version__}: принимает {len(accepted)} из 64 -> {accepted}")

    print("--- 3. Старая порча (замена двух последних на AA/BB) на 100 000 токенах")
    n = 100_000
    same = accepted_old = 0
    last_chars: set[str] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # как в pytest: любое предупреждение — ошибка
        for i in range(n):
            t = create_access_token(
                f"user{i}", frozenset(), secret, ttl_minutes=15, now=datetime.now(UTC)
            )
            bad = old_tamper(t)
            last_chars.add(t[-1])
            same += sig_bytes(bad) == sig_bytes(t)
            accepted_old += valid(bad, secret)
    print(
        f"подпись по байтам не изменилась: {same} из {n} = {same / n:.3%}; "
        f"наивная оценка для случайной пары: 3/4096 = {3 / 4096:.3%}"
    )
    print(f"PyJWT {jwt.__version__} принял испорченный токен: {accepted_old} из {n}")
    print(
        f"последний символ настоящих подписей: {len(last_chars)} разных -> "
        f"{''.join(sorted(last_chars, key=B64URL.index))}"
    )
    print("кодировщик пишет 2 лишних бита нулями: последний символ всегда A/E/I/...,")
    print("поэтому пары 'A'+B/C/D не бывает, а 'AA' меняется на 'BB' — старая порча портит всегда")

    print("--- 4. Новая порча (пересборка с другим секретом) на 1000 токенах")
    broken = 0
    for i in range(1000):
        s = secrets.token_urlsafe(32)  # каждый раз новый секрет, как на новом процессе
        t = create_access_token(
            f"user{i}", frozenset({"user"}), s, ttl_minutes=15, now=datetime.now(UTC)
        )
        bad = new_tamper(t)
        assert bad != t and bad.rsplit(".", 1)[0] == t.rsplit(".", 1)[0]  # меняется только подпись
        broken += not valid(bad, s)
    print(f"подпись испорчена в {broken} из 1000 токенов")


if __name__ == "__main__":
    main()
