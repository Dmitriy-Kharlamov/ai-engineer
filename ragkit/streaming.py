"""Потоковый ответ модели: дельты по пути, проверка — только в конце (Ф4-04).

Без стриминга порядок был «получил → проверил → показал». Со стримингом он
переворачивается: «показал, показал, показал → узнал, законченный ли ответ».
done_reason приходит в последнем куске. Поэтому здесь:

- куски превращаются в дельты текста (str), а конверт (Envelope) — отдельный тип,
  который существует только после последнего куска. У куска ollama и целого ответа один
  тип ChatResponse, и все поля Optional: mypy не отличит одно от другого и не заметит
  чтения done_reason «не там». Здесь отличит: у дельты нет поля done_reason, у конверта
  done_reason — str, а не str | None;
- обрыв (done_reason != "stop") — исключение StreamIncomplete ПОСЛЕ последней дельты,
  с уже показанным текстом в .partial. Вызывающий должен решить, что делать с тем, что
  пользователь уже видел (ragkit.conversation.PartialPolicy);
- поток, кончившийся без последнего куска (сервер упал, соединение порвалось), —
  StreamTruncated: такой ответ тоже нельзя считать законченным;
- соединение живёт, пока жив генератор клиента. ChatStream закрывает его в close() и в
  with; брошенный без close() поток держит соединение (замер: week-04/lesson_04.py,
  задание 6).

Три таймаута (только в AsyncChatStream — синхронное чтение нечем прервать):
первый кусок не пришёл; куски шли и перестали; поток идёт, но не кончается. Последний
таймаутом на кусок не ловится — нужен общий дедлайн, а надёжнее всего num_predict на
стороне сервера.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, Self

from .exceptions import RAGError
from .llm import IncompleteResponseError


class _Message(Protocol):
    @property
    def content(self) -> str | None: ...


class StreamChunk(Protocol):
    """Кусок потока — то, что отдаёт ollama.Client.chat(stream=True). Все поля Optional."""

    @property
    def done(self) -> bool | None: ...
    @property
    def done_reason(self) -> str | None: ...
    @property
    def message(self) -> _Message: ...
    @property
    def prompt_eval_count(self) -> int | None: ...
    @property
    def prompt_eval_duration(self) -> int | None: ...
    @property
    def eval_count(self) -> int | None: ...
    @property
    def eval_duration(self) -> int | None: ...
    @property
    def total_duration(self) -> int | None: ...
    @property
    def load_duration(self) -> int | None: ...


class StreamingChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> Iterator[StreamChunk]: ...


class AsyncStreamingChatClient(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[StreamChunk]: ...


@dataclass(frozen=True)
class Envelope:
    """Конверт ответа: есть только у законченного потока, и поля в нём не Optional."""

    done_reason: str
    prompt_eval_count: int
    eval_count: int
    prompt_eval_ns: int
    eval_ns: int
    total_ns: int
    load_ns: int
    chunks: int  # сколько кусков пришло, включая последний


class StreamIncomplete(IncompleteResponseError):
    """Поток кончился, но не потому, что модель закончила: done_reason != "stop"."""

    def __init__(self, envelope: Envelope, partial: str) -> None:
        super().__init__(
            f"ответ неполный: done_reason={envelope.done_reason!r}, "
            f"уже показано {len(partial)} символов, конец: {partial[-40:]!r}"
        )
        self.envelope = envelope
        self.partial = partial


class StreamTruncated(RAGError):
    """Поток кончился без последнего куска (done=True не пришёл)."""

    def __init__(self, partial: str, chunks: int) -> None:
        super().__init__(f"поток оборвался без конверта после {chunks} кусков")
        self.partial = partial


class StreamTimeout(RAGError):
    """Общий предок трёх таймаутов. partial — то, что уже успели показать."""

    def __init__(self, what: str, limit: float, elapsed: float, partial: str) -> None:
        super().__init__(
            f"{what}: {limit:g} с, прошло {elapsed:.2f} с, показано {len(partial)} симв."
        )
        self.limit = limit
        self.elapsed = elapsed
        self.partial = partial


class FirstChunkTimeout(StreamTimeout):
    """Первый кусок не пришёл: сервер мёртв, модель грузится, prefill слишком долгий."""


class StallTimeout(StreamTimeout):
    """Куски шли и перестали."""


class DeadlineExceeded(StreamTimeout):
    """Поток живой, но не укладывается в общий срок (зацикливание, слишком длинный ответ)."""


class _Assembler:
    """Складывает куски: копит текст, достаёт конверт из последнего. Без ввода-вывода."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self.chunks = 0
        self.envelope: Envelope | None = None

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def feed(self, chunk: StreamChunk) -> str:
        if self.envelope is not None:
            raise RAGError("кусок пришёл после последнего")
        self.chunks += 1
        delta = chunk.message.content or ""
        self._parts.append(delta)
        if chunk.done:
            self.envelope = Envelope(
                done_reason=chunk.done_reason or "",
                prompt_eval_count=chunk.prompt_eval_count or 0,
                eval_count=chunk.eval_count or 0,
                prompt_eval_ns=chunk.prompt_eval_duration or 0,
                eval_ns=chunk.eval_duration or 0,
                total_ns=chunk.total_duration or 0,
                load_ns=chunk.load_duration or 0,
                chunks=self.chunks,
            )
        return delta

    def finish(self) -> Envelope:
        """Проверка, которая раньше шла до показа, а теперь — после всего показанного."""
        if self.envelope is None:
            raise StreamTruncated(self.text, self.chunks)
        if self.envelope.done_reason != "stop":
            raise StreamIncomplete(self.envelope, self.text)
        return self.envelope


class ChatStream:
    """Итератор дельт (str). После исчерпания — .envelope; обрыв — исключение в конце.

    Пустые дельты (обычно последний кусок) не отдаются. Используй через with или зови
    close(): иначе соединение висит, пока жив генератор клиента.
    """

    def __init__(self, chunks: Iterator[StreamChunk]) -> None:
        self._chunks = chunks
        self._asm = _Assembler()
        self._finished = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        while True:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                self._finished = True
                self._asm.finish()
                raise
            delta = self._asm.feed(chunk)
            if delta:
                return delta

    @property
    def text(self) -> str:
        """Всё, что пришло к этому моменту."""
        return self._asm.text

    @property
    def envelope(self) -> Envelope:
        if not self._finished or self._asm.envelope is None:
            raise RuntimeError("поток ещё не дочитан: конверт приходит в последнем куске")
        return self._asm.envelope

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def stream_chat(
    client: StreamingChatClient,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> ChatStream:
    """Потоковая версия complete_chat: дельты по пути, проверка законченности — в конце."""
    return ChatStream(client.chat(model=model, messages=messages, options=options, stream=True))


@dataclass(frozen=True)
class StreamTimeouts:
    """first — до первого куска; between — между кусками; total — на весь поток.

    None — предохранителя нет. httpx-таймаут чтения похож на between: он считает одну
    операцию чтения, а не весь поток, и живой медленный поток не ловит никогда.
    """

    first: float | None = None
    between: float | None = None
    total: float | None = None


class AsyncChatStream:
    """Асинхронный поток дельт с тремя таймаутами. После исчерпания — .envelope."""

    def __init__(
        self,
        chunks: AsyncIterator[StreamChunk],
        timeouts: StreamTimeouts,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._chunks = chunks
        self._timeouts = timeouts
        self._clock = clock
        self._asm = _Assembler()
        self._started = clock()
        self._finished = False

    def __aiter__(self) -> Self:
        return self

    def _limit(self) -> tuple[type[StreamTimeout], str, float] | None:
        """Какой предохранитель ближе всего и сколько до него секунд."""
        t = self._timeouts
        per = t.first if self._asm.chunks == 0 else t.between
        kind: type[StreamTimeout] = FirstChunkTimeout if self._asm.chunks == 0 else StallTimeout
        what = "нет первого куска" if self._asm.chunks == 0 else "куски перестали приходить"
        if t.total is not None:
            left = t.total - (self._clock() - self._started)
            if per is None or left <= per:
                return DeadlineExceeded, "общий дедлайн", left
        if per is None:
            return None
        return kind, what, per

    async def __anext__(self) -> str:
        while True:
            limit = self._limit()
            try:
                if limit is None:
                    chunk = await anext(self._chunks)
                else:
                    kind, what, seconds = limit
                    if seconds <= 0:
                        raise TimeoutError
                    chunk = await asyncio.wait_for(anext(self._chunks), seconds)
            except StopAsyncIteration:
                self._finished = True
                self._asm.finish()
                raise
            except TimeoutError:
                assert limit is not None
                kind, what, _ = limit
                full = {
                    FirstChunkTimeout: self._timeouts.first,
                    StallTimeout: self._timeouts.between,
                    DeadlineExceeded: self._timeouts.total,
                }[kind]
                await self.aclose()
                raise kind(
                    what, full or 0.0, self._clock() - self._started, self._asm.text
                ) from None
            delta = self._asm.feed(chunk)
            if delta:
                return delta

    @property
    def text(self) -> str:
        return self._asm.text

    @property
    def envelope(self) -> Envelope:
        if not self._finished or self._asm.envelope is None:
            raise RuntimeError("поток ещё не дочитан: конверт приходит в последнем куске")
        return self._asm.envelope

    async def aclose(self) -> None:
        aclose = getattr(self._chunks, "aclose", None)
        if aclose is not None:
            await aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


async def astream_chat(
    client: AsyncStreamingChatClient,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
    *,
    timeouts: StreamTimeouts | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncChatStream:
    """Асинхронный поток с таймаутами. Запрос уходит при первом чтении, не здесь."""
    chunks = await client.chat(model=model, messages=messages, options=options, stream=True)
    return AsyncChatStream(chunks, timeouts or StreamTimeouts(), clock)
