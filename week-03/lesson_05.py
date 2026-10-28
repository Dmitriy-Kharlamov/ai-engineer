"""Ф3-05, задание 8: should_retry из Ф3-03 с честными типами.

Файл входит в [tool.mypy] files и проходит mypy --strict --warn-unreachable.
Неверные вызовы ниже помечены `# type: ignore[код]`, а в настройках включён
warn_unused_ignores: если бы mypy такой вызов НЕ поймал, комментарий стал бы
лишним и проверка упала. Так подавления сами доказывают, что ошибка ловится.
"""

from typing import TYPE_CHECKING, Literal, TypeAlias

import httpx

from ragkit import Chunk, HashEmbedder, embed_chunks

IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"})
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
MAYBE_DONE = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.RemoteProtocolError)
ALWAYS_RETRY_STATUS = frozenset({408, 425, 429, 502, 503, 504})


# ---------------------------------------------------------------------------
# Вариант 1: та же функция, просто честно аннотированная
# ---------------------------------------------------------------------------
Outcome: TypeAlias = int | type[httpx.TransportError]


def should_retry(outcome: Outcome, method: str, idempotency_key: str | None = None) -> bool:
    """outcome — код ответа или КЛАСС исключения; ключ разрешает повтор POST."""
    replay_safe = method.upper() in IDEMPOTENT or idempotency_key is not None
    if isinstance(outcome, int):
        return outcome in ALWAYS_RETRY_STATUS or (outcome == 500 and replay_safe)
    if issubclass(outcome, NOT_SENT):
        return True
    if issubclass(outcome, MAYBE_DONE):
        return replay_safe
    return False


# mypy проверяет вызовы. Блок под TYPE_CHECKING видит только mypy: в рантайме
# первый же вызов упал бы с TypeError: issubclass() arg 1 must be a class.
if TYPE_CHECKING:
    should_retry("503", "GET")  # type: ignore[arg-type]  # строка вместо кода
    should_retry(httpx.ReadTimeout("x"), "GET")  # type: ignore[arg-type]  # экземпляр, не класс
    should_retry(ValueError, "GET")  # type: ignore[arg-type]  # не исключение httpx
    should_retry(503, "GET", key="abc")  # type: ignore[call-arg]  # опечатка в имени аргумента
    # А вот это mypy пропускает, и это слабость подписи:
    should_retry(503, "GTE")  # метод — любая строка: опечатка доедет до рантайма


# ---------------------------------------------------------------------------
# Вариант 2: разделённый. Подпись перестала быть уродливой — значит, дело было в
# проектировании: одна функция отвечала на три разных вопроса.
# ---------------------------------------------------------------------------
Method: TypeAlias = Literal["GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE", "POST", "PATCH"]


def is_replay_safe(method: Method, idempotency_key: str | None) -> bool:
    """Безопасно ли выполнить запрос второй раз, если первый мог пройти."""
    return method in IDEMPOTENT or idempotency_key is not None


def retry_on_status(status: int, method: Method, idempotency_key: str | None = None) -> bool:
    """Сервер ответил: решаем по коду."""
    if status in ALWAYS_RETRY_STATUS:
        return True
    return status == 500 and is_replay_safe(method, idempotency_key)


def retry_on_error(
    error: httpx.TransportError, method: Method, idempotency_key: str | None = None
) -> bool:
    """Ответа не было: решаем по тому, ушёл ли запрос. Принимает само исключение."""
    if isinstance(error, NOT_SENT):
        return True
    if isinstance(error, MAYBE_DONE):
        return is_replay_safe(method, idempotency_key)
    return False


if TYPE_CHECKING:
    retry_on_status(503, "GTE")  # type: ignore[arg-type]  # опечатку в методе ловит Literal
    retry_on_error(httpx.ReadTimeout, "GET")  # type: ignore[arg-type]  # класс, не экземпляр


# ---------------------------------------------------------------------------
# Заодно к заданию 7: Protocol из ragkit проверяется структурно
# ---------------------------------------------------------------------------
class WrongEmbedder:
    """Похож на эмбеддер, но embed возвращает list[float], а не list[list[float]]."""

    dim = 3

    def embed(self, texts: list[str]) -> list[float]:
        return [0.0 for _ in texts]


if TYPE_CHECKING:
    embed_chunks(HashEmbedder(), [Chunk("текст", "doc")])  # подходит, хотя не наследует
    embed_chunks(WrongEmbedder(), [Chunk("текст", "doc")])  # type: ignore[arg-type]


def main() -> None:
    cases_v1: list[tuple[Outcome, str, str | None, bool]] = [
        (httpx.ConnectError, "POST", None, True),
        (httpx.ReadTimeout, "POST", None, False),
        (httpx.ReadTimeout, "POST", "order-42", True),  # ключ делает повтор безопасным
        (500, "POST", None, False),
        (500, "POST", "order-42", True),
        (503, "POST", None, True),
        (404, "GET", None, False),
    ]
    for outcome, method, key, expected in cases_v1:
        assert should_retry(outcome, method, key) is expected, (outcome, method, key)

    timeout = httpx.ReadTimeout("read timed out")
    assert retry_on_error(httpx.ConnectError("refused"), "POST") is True
    assert retry_on_error(timeout, "POST") is False
    assert retry_on_error(timeout, "POST", idempotency_key="order-42") is True
    assert retry_on_status(500, "POST") is False
    assert retry_on_status(500, "PUT") is True
    assert retry_on_status(429, "POST") is True
    print("should_retry и разделённый вариант: все проверки прошли")


if __name__ == "__main__":
    main()
