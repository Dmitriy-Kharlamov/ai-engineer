"""Вызов модели, который не принимает молча оборванный ответ (Ф4-01, задание 6).

Клиент — любой объект с методом chat(...), чей ответ похож на ollama.ChatResponse:
поля done_reason и message.content. Сам ragkit от библиотеки ollama не зависит.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .exceptions import RAGError


class IncompleteResponseError(RAGError):
    """Модель остановилась не потому, что закончила (done_reason != "stop")."""


class _Message(Protocol):
    @property
    def content(self) -> str | None: ...


class ChatReply(Protocol):
    @property
    def done_reason(self) -> str | None: ...

    @property
    def message(self) -> _Message: ...


class ChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> ChatReply: ...


def complete_chat(
    client: ChatClient,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> str:
    """Текст ответа — только если модель закончила сама; иначе IncompleteResponseError.

    done_reason == "length": кончился num_predict, текст оборван посреди слова.
    Любое другое значение, кроме "stop", тоже не считается законченным ответом.

    Сработавшая stop-строка даёт тот же done_reason == "stop", что и естественный конец
    (замер: week-04/lesson_02.py, задание 11, ollama 0.40.14), и по ответу их не различить.
    Для complete_chat это законченный ответ: обрезку по stop задал сам вызывающий.
    Кому нужно различать — проверяет текст.
    """
    reply = client.chat(model=model, messages=messages, options=options)
    content = reply.message.content or ""
    if reply.done_reason != "stop":
        tail = content[-40:]
        raise IncompleteResponseError(
            f"ответ неполный: done_reason={reply.done_reason!r}, "
            f"{len(content)} символов, конец: {tail!r}"
        )
    return content
