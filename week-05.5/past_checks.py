"""Ф5.5-02, задание 1: проверки на три настоящие прошлые поломки — без pytest и без модели.

Файл запускается и из текущего дерева, и из worktree старого коммита (lesson_02.py, задание 1):
каждая проверка обращается только к API, который был и тогда, — иначе красный цвет на старом
коммите значил бы «нет функции», а не «есть поломка».

    python week-05.5/past_checks.py          # печатает по строке на проверку, код 1 — есть красные
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. streaming.py не собирал tool_calls (Ф4-08, присест 1; 13 тестов были зелёными)
#    Слой: фейковый клиент — но только если фейк знает настоящий формат куска: вызовы
#    приходят в message.tool_calls куска потока, а не в тексте. Тринадцать тестов были
#    зелёными, потому что их фейки этого не умели.
# ---------------------------------------------------------------------------
@dataclass
class _Function:
    name: str
    arguments: dict[str, Any]


@dataclass
class _Call:
    function: _Function


@dataclass
class _Msg:
    content: str | None
    tool_calls: list[_Call] | None = None


@dataclass
class _Chunk:
    message: _Msg
    done: bool | None = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None


@dataclass
class _StreamClient:
    """Как ollama.AsyncClient(stream=True): вызов приходит в куске, текста нет вовсе."""

    sent: list[Any] = field(default_factory=list)

    async def chat(self, **kwargs: Any) -> AsyncIterator[_Chunk]:
        self.sent.append(kwargs)

        async def gen() -> AsyncIterator[_Chunk]:
            call = _Call(_Function("read_document", {"name": "delivery.md"}))
            yield _Chunk(_Msg("", [call]))
            yield _Chunk(_Msg(""), True, "stop", 400, 1, 20, 1, 2, 0)

        return gen()


async def _stream_tool_calls() -> tuple[bool, str]:
    from ragkit import streaming

    stream = await streaming.astream_chat(_StreamClient(), "m", [{"role": "user", "content": "?"}])
    async for _ in stream:
        pass
    calls = getattr(stream.envelope, "tool_calls", ())
    names = [getattr(c, "name", None) for c in calls]
    ok = names == ["read_document"]
    return ok, f"вызовы в конверте: {names or 'нет'}"


def check_stream_tool_calls() -> tuple[bool, str]:
    return asyncio.run(_stream_tool_calls())


# ---------------------------------------------------------------------------
# 2. Chunk.__eq__ по числу токенов (Ф2-03): разные чанки с одинаковым числом слов «равны»
#    Слой: фейковый — модели тут нет вовсе.
# ---------------------------------------------------------------------------
def check_chunk_equality() -> tuple[bool, str]:
    from ragkit import Chunk

    a = Chunk("курьер 290 рублей", "delivery.md")
    b = Chunk("возврат 14 дней", "returns.md")
    distinct = len({a, b})
    ok = a != b and distinct == 2 and b not in [a]
    return ok, f"a == b: {a == b}; в множестве {distinct} из 2; b in [a]: {b in [a]}"


CHECKS = {
    "streaming собирает tool_calls": check_stream_tool_calls,
    "Chunk: разные чанки не равны": check_chunk_equality,
}


def main() -> int:
    red = 0
    for name, check in CHECKS.items():
        try:
            ok, detail = check()
        except Exception as error:  # noqa: BLE001 — на старом коммите упасть может что угодно
            ok, detail = False, f"{type(error).__name__}: {error}"
        red += not ok
        print(f"{'зелёная' if ok else 'КРАСНАЯ'}  {name}: {detail}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
