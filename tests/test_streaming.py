"""Потоки и потоковые ходы: на фейках, без сети и без модели."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest

from ragkit.conversation import (
    PARTIAL_MARK,
    Conversation,
    PartialPolicy,
    validate_alternation,
)
from ragkit.prompts import Prompt
from ragkit.streaming import (
    DeadlineExceeded,
    FirstChunkTimeout,
    StallTimeout,
    StreamChunk,
    StreamIncomplete,
    StreamTimeouts,
    StreamTruncated,
    astream_chat,
    stream_chat,
)

CHAT = Prompt("test_chat", "1", "Отвечай коротко.", "{question}")


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChunk:
    """Кусок как у ollama: у всех, кроме последнего, конверт пустой."""

    message: FakeMessage
    done: bool | None = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None


def chunks(*parts: str, done_reason: str | None = "stop") -> list[FakeChunk]:
    """Куски с дельтами parts; done_reason=None — поток без последнего куска."""
    out = [FakeChunk(FakeMessage(p)) for p in parts]
    if done_reason is not None:
        out.append(
            FakeChunk(FakeMessage(""), True, done_reason, 12, 3_000_000, len(parts), 9, 99, 1)
        )
    return out


@dataclass
class FakeStreamingClient:
    """Отдаёт заранее заданные куски; error — исключение после этих кусков."""

    script: list[FakeChunk]
    error: Exception | None = None
    calls: list[list[Mapping[str, Any]]] = field(default_factory=list)
    closed: list[bool] = field(default_factory=list)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> Iterator[StreamChunk]:
        self.calls.append(list(messages))
        return self._gen()

    def _gen(self) -> Iterator[StreamChunk]:
        try:
            yield from self.script
            if self.error is not None:
                raise self.error
        finally:
            self.closed.append(True)  # здесь у ollama закрывается соединение


# ---------------------------------------------------------------------------
# ChatStream
# ---------------------------------------------------------------------------
def test_deltas_join_into_answer_and_envelope_comes_last() -> None:
    client = FakeStreamingClient(chunks("Ра", "сте", "ния."))
    stream = stream_chat(client, "m", [{"role": "user", "content": "?"}])
    with pytest.raises(RuntimeError, match="не дочитан"):
        _ = stream.envelope
    assert list(stream) == ["Ра", "сте", "ния."]  # пустой последний кусок не отдаётся
    assert stream.text == "Растения."
    assert (stream.envelope.done_reason, stream.envelope.eval_count) == ("stop", 3)
    assert stream.envelope.chunks == 4


def test_cut_stream_raises_after_everything_was_shown() -> None:
    client = FakeStreamingClient(chunks("Расте", "ния дел", done_reason="length"))
    shown: list[str] = []
    with pytest.raises(StreamIncomplete, match="length") as error:
        for delta in stream_chat(client, "m", []):
            shown.append(delta)
    assert shown == ["Расте", "ния дел"]  # проверка — после показа, не до
    assert error.value.partial == "Растения дел"


def test_stream_without_final_chunk_is_truncated() -> None:
    client = FakeStreamingClient(chunks("Расте", done_reason=None))
    with pytest.raises(StreamTruncated) as error:
        list(stream_chat(client, "m", []))
    assert error.value.partial == "Расте"


def test_break_does_not_close_connection_but_close_does() -> None:
    client = FakeStreamingClient(chunks("а", "б", "в"))
    stream = stream_chat(client, "m", [])
    for _ in stream:
        break
    assert client.closed == []  # ссылка жива — генератор, а с ним и соединение, открыт
    stream.close()
    assert client.closed == [True]


def test_with_closes_on_break() -> None:
    client = FakeStreamingClient(chunks("а", "б", "в"))
    with stream_chat(client, "m", []) as stream:
        next(stream)
    assert client.closed == [True]


# ---------------------------------------------------------------------------
# AsyncChatStream: три таймаута
# ---------------------------------------------------------------------------
@dataclass
class TimedClient:
    """Асинхронный поток: перед каждым куском пауза из delays (секунды)."""

    delays: list[float]
    closed: list[bool] = field(default_factory=list)

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[StreamChunk]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[StreamChunk]:
        try:
            for i, delay in enumerate(self.delays):
                await asyncio.sleep(delay)
                yield FakeChunk(FakeMessage(f"т{i} "))
            yield chunks(done_reason="stop")[0]
        finally:
            self.closed.append(True)


async def read_all(client: TimedClient, timeouts: StreamTimeouts) -> str:
    stream = await astream_chat(client, "m", [], timeouts=timeouts)
    return "".join([d async for d in stream])


async def test_first_chunk_timeout() -> None:
    client = TimedClient([0.3, 0.0])
    with pytest.raises(FirstChunkTimeout) as error:
        await read_all(client, StreamTimeouts(first=0.05, between=1.0, total=5.0))
    assert error.value.partial == ""
    assert client.closed == [True]


async def test_stall_timeout_after_some_text() -> None:
    client = TimedClient([0.0, 0.0, 0.3])
    with pytest.raises(StallTimeout) as error:
        await read_all(client, StreamTimeouts(first=1.0, between=0.05, total=5.0))
    assert error.value.partial == "т0 т1 "


async def test_slow_but_alive_stream_is_caught_only_by_deadline() -> None:
    client = TimedClient([0.03] * 20)  # 20 кусков по 30 мс — меньше таймаута на кусок
    no_deadline = await read_all(client, StreamTimeouts(first=0.2, between=0.1))
    assert no_deadline.count("т") == 20  # таймаут на кусок не сработал ни разу
    with pytest.raises(DeadlineExceeded):
        await read_all(TimedClient([0.03] * 20), StreamTimeouts(first=0.2, between=0.1, total=0.2))


# ---------------------------------------------------------------------------
# Conversation.ask_stream: что попадает в историю
# ---------------------------------------------------------------------------
def conversation() -> Conversation:
    return Conversation(CHAT, budget=10_000, count_tokens=lambda m: len(m))


def drain(it: Iterator[str]) -> str:
    return "".join(it)


def test_full_stream_is_recorded_once() -> None:
    conv = conversation()
    client = FakeStreamingClient(chunks("Сахар ", "из света."))
    assert (
        drain(conv.ask_stream(client, "m", question="Что такое фотосинтез?")) == "Сахар из света."
    )
    assert [m["content"] for m in conv.history] == ["Что такое фотосинтез?", "Сахар из света."]
    assert (conv.records[-1].outcome, conv.records[-1].eval_count) == ("stop", 2)


def test_discarded_partial_is_not_in_history() -> None:
    conv = conversation()
    client = FakeStreamingClient(chunks("Сахар ", "из", done_reason="length"))
    with pytest.raises(StreamIncomplete):
        drain(conv.ask_stream(client, "m", question="?"))
    assert conv.history == ()
    assert conv.records == ()


def test_marked_partial_is_in_history_with_mark() -> None:
    conv = conversation()
    client = FakeStreamingClient(chunks("Сахар ", "из", done_reason="length"))
    with pytest.raises(StreamIncomplete):
        drain(conv.ask_stream(client, "m", partial=PartialPolicy.MARK, question="?"))
    expected = "Сахар из" + PARTIAL_MARK.format(reason="StreamIncomplete")
    assert conv.history[-1]["content"] == expected
    assert conv.records[-1].outcome == "partial:mark"


def test_partial_as_complete_looks_like_a_normal_answer() -> None:
    conv = conversation()
    client = FakeStreamingClient(chunks("Сахар ", "из"), error=ConnectionError("сервер упал"))
    with pytest.raises(ConnectionError):
        drain(conv.ask_stream(client, "m", partial=PartialPolicy.AS_COMPLETE, question="?"))
    assert conv.history[-1]["content"] == "Сахар из"  # по истории не отличить от ответа


@pytest.mark.parametrize("policy", list(PartialPolicy), ids=lambda p: p.value)
@pytest.mark.parametrize("ending", ["cut", "server-error", "reader-left"])
def test_history_stays_valid_whatever_happens(policy: PartialPolicy, ending: str) -> None:
    conv = conversation()
    drain(conv.ask_stream(FakeStreamingClient(chunks("Первый ответ.")), "m", question="1"))
    if ending == "cut":
        client = FakeStreamingClient(chunks("обры", done_reason="length"))
    else:
        error = RuntimeError("ошибка посреди потока") if ending == "server-error" else None
        client = FakeStreamingClient(chunks("обры", "вок", done_reason=None), error=error)
    turn = conv.ask_stream(client, "m", partial=policy, question="2")
    with contextlib.suppress(Exception), contextlib.closing(turn):
        for _ in turn:
            if ending == "reader-left":
                break
    assert client.closed == [True]  # соединение закрыто в любом исходе
    validate_alternation(
        [{"role": "system", "content": CHAT.system}, *conv.history], outgoing=False
    )
    after = FakeStreamingClient(chunks("Третий."))
    drain(conv.ask_stream(after, "m", question="3"))
    validate_alternation(after.calls[0])  # следующий ход уходит в правильной форме


# ---------------------------------------------------------------------------
# Вызовы инструментов в потоке (Ф4-08)
# ---------------------------------------------------------------------------
@dataclass
class _Fn:
    name: str
    arguments: Mapping[str, Any]


@dataclass
class _Call:
    function: _Fn


@dataclass
class ToolMessage:
    content: str | None
    tool_calls: list[_Call] | None = None


@dataclass
class FakeToolClient:
    script: list[FakeChunk]
    tools_seen: list[Sequence[Mapping[str, Any]]] = field(default_factory=list)

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[StreamChunk]:
        self.tools_seen.append(tools)

        async def gen() -> AsyncIterator[StreamChunk]:
            for c in self.script:
                yield c

        return gen()


async def test_tool_calls_from_stream_are_collected_in_order() -> None:
    from ragkit.streaming import StreamToolCall, astream_tool_chat

    calls = [_Call(_Fn("search", {"q": "a"})), _Call(_Fn("read", {"name": "b.md"}))]
    script: list[FakeChunk] = [
        FakeChunk(ToolMessage("", calls[:1])),  # type: ignore[arg-type]
        FakeChunk(ToolMessage("", calls[1:])),  # type: ignore[arg-type]
        *chunks(),
    ]
    client = FakeToolClient(script)
    stream = await astream_tool_chat(client, "m", [], [{"type": "function"}])
    async with stream:
        text = "".join([d async for d in stream])
    assert text == ""
    assert stream.tool_calls == (
        StreamToolCall("search", {"q": "a"}),
        StreamToolCall("read", {"name": "b.md"}),
    )
    assert client.tools_seen == [[{"type": "function"}]]


async def test_plain_stream_has_no_tool_calls() -> None:
    from ragkit.streaming import astream_tool_chat

    stream = await astream_tool_chat(FakeToolClient(chunks("а", "б")), "m", [], [])
    async with stream:
        assert "".join([d async for d in stream]) == "аб"
    assert stream.tool_calls == ()
