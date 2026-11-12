"""Инъектор отказов: каждый из таблицы присеста 2 воспроизводится флагом --fault.

FaultyClient стоит между сессией и настоящим клиентом ollama и подменяет одно или все
обращения; остальное идёт в настоящую модель. Так видно, что делает продукт целиком —
и код, и модель после отказа, — а не только отдельная функция.

    python -m docqa --fault unknown-tool -q "Сколько стоит доставка курьером?"

Отказы, которые от клиента не зависят, сделаны иначе: tool-crash — папка, где чтение падает
с OSError; переполнение окна — read-all плюс маленький --num-ctx; бюджет — --limit-usd и
--session-usd. Подмены выглядят ровно как куски потока ollama (message.content,
message.tool_calls, done, done_reason, счётчики), и ошибка сервера бросается при первом
чтении потока — так делает ollama.AsyncClient.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import ollama

from ragkit.streaming import AsyncToolStreamingChatClient, StreamChunk

from .documents import DocumentStore

FAULTS: dict[str, str] = {
    "unknown-tool": "1-е обращение: модель зовёт send_email — такого инструмента нет",
    "bad-args": "1-е обращение: read_document с аргументом file вместо name",
    "tool-crash": "чтение документа падает с PermissionError (папка недоступна)",
    "raw-call": "1-е обращение: вызов пришёл текстом в content, с мусорным токеном впереди",
    "raw-call-broken": "1-е обращение: вызов текстом, JSON оборван",
    "raw-call-bare": "1-е обращение: «search_documents {...}» текстом — форма из injection.txt",
    "loop": "каждое обращение: list_documents() — один и тот же вызов",
    "limit": "каждое обращение: search_documents с новым запросом — без повторов",
    "read-all": "1-е обращение: read_document для каждого документа сразу",
    "read-all-turns": "1-е обращение каждого хода: read_document для каждого документа",
    "no-tools": "1-е обращение: ответ без единого вызова, из головы",
    "stream-cut": "поток с текстом обрывается после 6 кусков, без последнего куска",
    "stream-stall": "поток с текстом замолкает после 6 кусков",
    "busy-503": "первые 2 обращения: 503 server busy, потом нормально",
    "busy-429": "первые 2 обращения: 429, потом нормально",
    "busy-forever": "каждое обращение: 503 server busy",
}
Fault = Literal[
    "unknown-tool",
    "bad-args",
    "tool-crash",
    "raw-call",
    "raw-call-broken",
    "raw-call-bare",
    "loop",
    "limit",
    "read-all",
    "read-all-turns",
    "no-tools",
    "stream-cut",
    "stream-stall",
    "busy-503",
    "busy-429",
    "busy-forever",
]
BUSY = "server busy, please try again.  maximum pending requests exceeded"
CUT_AFTER = 6


@dataclass
class _Fn:
    name: str
    arguments: Mapping[str, Any]


@dataclass
class _Call:
    function: _Fn


@dataclass
class _Msg:
    content: str
    tool_calls: list[_Call] | None = None


@dataclass
class _Chunk:
    message: _Msg
    done: bool = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None


def _final(prompt: int = 0, output: int = 0) -> _Chunk:
    """Последний кусок подставленного ответа. Счётчики — 0: сервер ничего не считал, и
    подставлять сюда правдоподобные числа нельзя — сессия считает от них вход и деньги."""
    return _Chunk(_Msg(""), True, "stop", prompt, 0, output, 0, 0, 0)


def _calls(*pairs: tuple[str, Mapping[str, Any]]) -> list[StreamChunk]:
    return [_Chunk(_Msg("", [_Call(_Fn(n, a)) for n, a in pairs])), _final()]


def _text(*parts: str) -> list[StreamChunk]:
    return [*(_Chunk(_Msg(p)) for p in parts), _final()]


async def _replay(chunks: Sequence[StreamChunk]) -> AsyncIterator[StreamChunk]:
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


class _Failing:
    """Поток, который падает при первом чтении — как ollama.AsyncClient при ошибке HTTP."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __aiter__(self) -> "_Failing":
        return self

    async def __anext__(self) -> StreamChunk:
        raise self.error


class FaultyClient:
    def __init__(
        self, inner: AsyncToolStreamingChatClient, fault: Fault, documents: Sequence[str] = ()
    ) -> None:
        self.inner = inner
        self.fault = fault
        self.documents = list(documents)
        self.requests = 0
        self.cut_done = False

    def _scripted(
        self, n: int, messages: Sequence[Mapping[str, Any]]
    ) -> list[StreamChunk] | Exception | None:
        f = self.fault
        if f == "read-all-turns" and messages and messages[-1].get("role") == "user":
            return _calls(*(("read_document", {"name": d}) for d in self.documents))
        if f == "unknown-tool" and n == 1:
            return _calls(("send_email", {"to": "client@example.com", "text": "Доставка 290 ₽"}))
        if f == "bad-args" and n == 1:
            return _calls(("read_document", {"file": "delivery.md"}))
        if f == "raw-call" and n == 1:
            return _text(
                "谤 ",
                '{"name": "read_document", ',
                '"arguments": {"name": "delivery.md"}}',
                " </tool_call>",
            )
        if f == "raw-call-broken" and n == 1:
            return _text("<tool_call>\n", '{"name": "read_document", "argum')
        if f == "raw-call-bare" and n == 1:
            return _text("search_", 'documents {"query": ', '"доставка курьер"}')
        if f == "loop":
            return _calls(("list_documents", {}))
        if f == "limit":
            return _calls(("search_documents", {"query": f"доставка вариант {n}"}))
        if f == "read-all" and n == 1:
            return _calls(*(("read_document", {"name": d}) for d in self.documents))
        if f == "no-tools" and n == 1:
            return _text("Доставка у нас ", "всегда бесплатная.")
        if f in ("busy-503", "busy-429") and n <= 2:
            return ollama.ResponseError(BUSY, 503 if f == "busy-503" else 429)
        if f == "busy-forever":
            return ollama.ResponseError(BUSY, 503)
        return None

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[StreamChunk]:
        self.requests += 1
        scripted = self._scripted(self.requests, messages)
        if isinstance(scripted, Exception):
            return _Failing(scripted)
        if scripted is not None:
            return _replay(scripted)
        real = await self.inner.chat(
            model=model, messages=messages, tools=tools, options=options, stream=True
        )
        if self.fault in ("stream-cut", "stream-stall") and not self.cut_done:
            return self._cut(real)
        return real

    async def _cut(self, real: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        texts = 0
        try:
            async for chunk in real:
                yield chunk
                if getattr(chunk.message, "content", None):
                    texts += 1
                if texts >= CUT_AFTER:
                    self.cut_done = True
                    if self.fault == "stream-stall":
                        await asyncio.sleep(3600)
                    return  # stream-cut: ни done, ни конверта
        finally:
            close = getattr(real, "aclose", None)
            if close is not None:
                await close()


class CrashingStore(DocumentStore):
    """Список и поиск работают, а чтение падает, как на файле без прав: ошибка с путём."""

    def read(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        raise PermissionError(13, "Permission denied", str(Path(path).resolve()))


class FirstCallClient:
    """Первое обращение заменено вызовом name(arguments), дальше — настоящая модель.

    Для замера инъекции при условии «документ прочитан»: модель сама читает pickup.md лишь в
    3 прогонах из 10 (injection.txt, первый прогон), и доля выполнения по трём прогонам
    ничего не говорит. Так в контексте всегда один и тот же результат чтения, а ответ пишет
    модель.
    """

    def __init__(
        self, inner: AsyncToolStreamingChatClient, name: str, arguments: Mapping[str, Any]
    ) -> None:
        self.inner = inner
        self.call = (name, arguments)
        self.requests = 0

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[StreamChunk]:
        self.requests += 1
        if self.requests == 1:
            return _replay(_calls(self.call))
        return await self.inner.chat(
            model=model, messages=messages, tools=tools, options=options, stream=True
        )
