"""Ф4-04. Streaming — замеры.

Нужно: запущенный Ollama и `ollama pull qwen2.5:7b`. Из корня репозитория, по частям:
    python -u week-04/lesson_04.py 1 2 3   2>&1 | tee week-04/lesson_04_part1.txt
    python -u week-04/lesson_04.py 4       2>&1 | tee week-04/lesson_04_part2.txt
    python -u week-04/lesson_04.py 5 6 7   2>&1 | tee week-04/lesson_04_part3.txt
Другая модель: MODEL=llama3.2:3b python -u week-04/lesson_04.py 1

Модель всегда выдаёт ответ по одному токену. Streaming меняет только одно — кому отдают
токены по пути, — и покупает время до первого слова (TTFT). Плату за это мерим здесь же:
проверка законченности уезжает в конец, ошибка может прийти посреди текста, а брошенный
поток держит соединение.

В задании 4 часть отказов воспроизводится локальными серверами внутри скрипта: «молчащий»
сервер и сервер, который отдаёт записанные куски настоящего ответа с заданными паузами.
Настоящий ollama не умеет застрять по заказу. Прогнозы — в PREDICTIONS, до первого запуска.
"""

import asyncio
import gc
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
import ollama

from ragkit.conversation import CharRatioEstimate, Conversation, PartialPolicy
from ragkit.llm import IncompleteResponseError, complete_chat
from ragkit.prompts import BOOKSHOP_CHAT
from ragkit.streaming import (
    StreamIncomplete,
    StreamTimeout,
    StreamTimeouts,
    astream_chat,
    stream_chat,
)
from services.http_retry import should_retry

MODEL = os.environ.get("MODEL", "qwen2.5:7b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=HOST)
SEED = 42

Messages = list[dict[str, str]]

LONG = [
    {
        "role": "user",
        "content": "Напиши длинный подробный рассказ о путешествии на поезде через Сибирь: "
        "станции, попутчики, еда, пейзажи за окном. Не меньше 1500 слов.",
    }
]
SHORT = [{"role": "user", "content": "Назови три столицы Европы и по одному факту о каждой."}]

# Прогнозы записаны ДО первого запуска. Меняй на свои — но до запуска.
PREDICTIONS = {
    1: "общее время одинаковое в пределах 5 % на всех трёх длинах; TTFT со стримингом — "
    "0,1–0,3 с на всех длинах, без стриминга TTFT = общее время ≈ N × 50 мс (Ф4-01): "
    "около 2,5 с, 10 с и 30 с; TTFT падает почти ровно на eval_duration",
    2: "кусков столько же, сколько eval_count, плюс один последний с пустым content; "
    "в каждом куске — один токен, 1–4 символа",
    4: "первый кусок: молчащий сервер ловится таймаутом до первого куска, выгруженная модель "
    "тоже (загрузка ~4 с, Ф4-01); застрявший поток — таймаутом между кусками; медленный "
    "живой поток и зацикливание таймаутом на кусок не ловятся ни нашим, ни httpx — только "
    "дедлайном и num_predict",
    5: "медиана около 50 мс, 95-й процентиль на 20–30 % выше, максимум — 2–3 медианы; "
    "к концу 600 токенов медиана растёт на 10–20 % (длиннее контекст)",
    6: "пока генератор жив и соединение открыто, ollama досчитывает ответ (куски помещаются "
    "в буферы сокета), и пробный запрос ждёт; после close() ollama замечает разрыв и "
    "бросает генерацию, пробный запрос отвечает сразу; после break с живой ссылкой "
    "соединение открыто, после close() — закрыто",
}


# ===========================================================================
# Общее
# ===========================================================================
def commit() -> str:
    """Хеш коммита, на котором запущен замер, и есть ли незакоммиченные правки."""
    root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "? (git недоступен)"
    return head + (" + незакоммиченные правки" if dirty else "")


def header() -> None:
    version = httpx.get(f"{HOST}/api/version", timeout=10).json()["version"]
    d = client.show(MODEL).details
    digest = next((m.digest for m in client.list().models if m.model == MODEL), None) or "?"
    print(f"коммит {commit()}")
    print(f"ollama {version}; модель {MODEL}, digest {digest[:12]}; библиотека ollama ", end="")
    print(_library_version())
    if d is not None:
        print(f"   {d.family}, {d.parameter_size}, {d.quantization_level}, {d.format}")


def _library_version() -> str:
    from importlib.metadata import version

    return version("ollama")


def prediction(n: int) -> None:
    print(f"   прогноз (до запуска): {PREDICTIONS[n]}")


def opts(num_predict: int, **extra: Any) -> dict[str, Any]:
    return {"temperature": 0, "seed": SEED, "num_predict": num_predict, **extra}


def ms(ns: int | None) -> float:
    return (ns or 0) / 1e6


def close(gen: Iterator[Any]) -> None:
    """Закрыть генератор клиента ollama: в его finally закрывается соединение."""
    cast(Generator[Any, None, None], gen).close()


def warm_up() -> None:
    client.chat(model=MODEL, messages=SHORT, options=opts(1))


def percentile(values: Sequence[float], q: float) -> float:
    """q-й процентиль, линейная интерполяция (как numpy по умолчанию)."""
    data = sorted(values)
    pos = (len(data) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


# ===========================================================================
# 1. TTFT против общего времени
# ===========================================================================
LENGTHS = (50, 200, 600)
RUNS_1 = 3


def timed_whole(num_predict: int) -> tuple[float, float, ollama.ChatResponse]:
    """Без стриминга: первое слово видно, только когда пришёл весь ответ."""
    started = time.perf_counter()
    r = client.chat(model=MODEL, messages=LONG, options=opts(num_predict))
    total = time.perf_counter() - started
    return total, total, r


def timed_stream(num_predict: int) -> tuple[float, float, ollama.ChatResponse, str]:
    """Со стримингом: TTFT — первый кусок с непустым текстом."""
    started = time.perf_counter()
    first: float | None = None
    last: ollama.ChatResponse | None = None
    parts: list[str] = []
    for chunk in client.chat(model=MODEL, messages=LONG, options=opts(num_predict), stream=True):
        if first is None and chunk.message.content:
            first = time.perf_counter() - started
        parts.append(chunk.message.content or "")
        last = chunk
    total = time.perf_counter() - started
    assert last is not None
    return first or total, total, last, "".join(parts)


def task1() -> None:
    print("\n1. TTFT против общего времени: тот же запрос с stream=False и stream=True")
    prediction(1)
    warm_up()
    print(f"   по {RUNS_1} прогона на режим и длину, режимы чередуются; в таблице медианы")
    print(
        f"   {'N':>4} │ {'без потока: TTFT = всё':>22} │ {'поток: TTFT':>11} {'всё':>8} │ "
        f"{'разница всего':>13} │ {'TTFT упал на':>12} {'eval_duration':>13} │ текст"
    )
    for n in LENGTHS:
        whole_total: list[float] = []
        stream_ttft: list[float] = []
        stream_total: list[float] = []
        eval_s: list[float] = []
        same_text = True
        for _ in range(RUNS_1):
            total, _, r = timed_whole(n)
            whole_total.append(total)
            ttft, s_total, last, text = timed_stream(n)
            stream_ttft.append(ttft)
            stream_total.append(s_total)
            eval_s.append(ms(last.eval_duration) / 1000)
            same_text = same_text and text == (r.message.content or "")
        wt, st, sf = (statistics.median(x) for x in (whole_total, stream_total, stream_ttft))
        drop = wt - sf
        ev = statistics.median(eval_s)
        print(
            f"   {n:>4} │ {wt:>20.2f} с │ {sf:>9.2f} с {st:>6.2f} с │ {(st - wt) / wt:>+12.1%} │ "
            f"{drop:>10.2f} с {ev:>11.2f} с │ {'одинаковый' if same_text else 'РАЗНЫЙ'}"
        )
    print("   «TTFT упал на» = TTFT без потока − TTFT с потоком; eval_duration — время декода")


# ===========================================================================
# 2. Анатомия куска
# ===========================================================================
ENVELOPE_FIELDS = (
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


def show_chunk(label: str, i: int, chunk: ollama.ChatResponse) -> None:
    data = chunk.model_dump()
    message = data.pop("message")
    print(f"   {label} (кусок {i}):")
    print(f"      message: {message}")
    print(f"      остальное: {data}")


def task2() -> None:
    print("\n2. Анатомия куска")
    prediction(2)
    whole = client.chat(model=MODEL, messages=SHORT, options=opts(60))
    got = list(client.chat(model=MODEL, messages=SHORT, options=opts(60), stream=True))
    middle = len(got) // 2
    show_chunk("первый", 0, got[0])
    show_chunk("из середины", middle, got[middle])
    show_chunk("последний", len(got) - 1, got[-1])

    contents = [c.message.content or "" for c in got]
    joined = "".join(contents)
    text = whole.message.content or ""
    print("   content — дельта, а не накопленное:")
    print(
        f"      склейка всех content == ответ без стриминга (тот же seed, T=0): {joined == text} "
        f"({len(joined)} и {len(text)} символов)"
    )
    grows = sum(
        1
        for prev, cur in zip(contents, contents[1:], strict=False)
        if prev and cur.startswith(prev) and len(cur) > len(prev)
    )
    longest = max(contents, key=len)
    print(
        f"      кусков, которые начинаются с предыдущего и длиннее его (так выглядело бы "
        f"накопленное): {grows} из {len(got) - 1}"
    )
    print(
        f"      самый длинный content — {len(longest)} симв. {longest!r}; будь content "
        f"накопленным, последний непустой был бы всем ответом, {len(text)} симв."
    )
    print(f"      первые 12 дельт: {contents[:12]}")
    print("   конверт — только в последнем куске (номера кусков, где поле не None):")
    for name in ENVELOPE_FIELDS:
        where = [i for i, c in enumerate(got) if getattr(c, name) is not None]
        print(f"      {name:<21} {where}")
    done_at = [i for i, c in enumerate(got) if c.done]
    print(
        f"      done=True — в кусках {done_at}; done=False — в остальных {len(got) - len(done_at)}"
    )
    nonempty = sum(1 for c in contents if c)
    print(
        f"   кусков всего {len(got)}, с непустым текстом {nonempty}, eval_count {got[-1].eval_count} "
        f"(без стриминга {whole.eval_count})"
    )


# ===========================================================================
# 3. Инверсия проверки
# ===========================================================================
class StreamingAsWhole:
    """Клиент, который на chat() отдаёт поток, — как если бы complete_chat позвали со stream."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        return client.chat(model=model, messages=messages, options=options, stream=True)


class FirstChunkOnly:
    """Клиент, который отдаёт первый кусок потока, — проверка «как раньше», но рано."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> ollama.ChatResponse:
        stream = client.chat(model=model, messages=messages, options=options, stream=True)
        first = next(stream)
        close(stream)
        return first


def task3() -> None:
    print("\n3. Инверсия проверки")
    print("   complete_chat с потоком:")
    try:
        complete_chat(StreamingAsWhole(), MODEL, SHORT, opts(60))
    except AttributeError as error:
        print(f"      AttributeError: {error} — у потока нет ни message, ни done_reason")
    try:
        complete_chat(FirstChunkOnly(), MODEL, SHORT, opts(60))
    except IncompleteResponseError as error:
        print(f"      по первому куску: IncompleteResponseError: {error}")
        print("      — хотя ответ нормальный: done_reason у первого куска всегда None")

    print("   ragkit.streaming.stream_chat, законченный ответ:")
    with stream_chat(client, MODEL, SHORT, opts(200)) as stream:
        deltas = list(stream)
    e = stream.envelope
    print(
        f"      {len(deltas)} дельт, {len(stream.text)} симв.; конверт: done_reason={e.done_reason!r}, "
        f"eval_count={e.eval_count}, кусков {e.chunks}"
    )

    print("   ragkit.streaming.stream_chat, обрыв по num_predict=12:")
    shown: list[str] = []
    started = time.perf_counter()
    try:
        with stream_chat(client, MODEL, LONG, opts(12)) as stream:
            for delta in stream:
                shown.append(delta)
    except StreamIncomplete as error:
        late = time.perf_counter() - started
        print(f"      показано {len(shown)} дельт: {''.join(shown)!r}")
        print(f"      и только потом, через {late:.2f} с от начала: StreamIncomplete: {error}")


# ===========================================================================
# 4. Три таймаута — и локальные серверы, которые умеют застревать
# ===========================================================================
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class SilentServer:
    """Принимает соединение, читает запрос и молчит: «сервер жив, но не отвечает»."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = int(self.sock.getsockname()[1])
        self.held: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.held.append(conn)

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class ReplayServer:
    """Отдаёт записанные куски настоящего ответа; delay(i) — пауза перед i-м куском."""

    def __init__(self, lines: list[dict[str, Any]], delay: Callable[[int], float]) -> None:
        def handler_factory(*args: Any) -> BaseHTTPRequestHandler:
            return _ReplayHandler(lines, delay, *args)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


class _ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # без Content-Length: конец ответа — закрытие соединения

    def __init__(
        self, lines: list[dict[str, Any]], delay: Callable[[int], float], *args: Any
    ) -> None:
        self.lines = lines
        self.delay = delay
        super().__init__(*args)

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            for i, line in enumerate(self.lines):
                time.sleep(self.delay(i))
                self.wfile.write(json.dumps(line, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # клиент ушёл — так и задумано


def record_chunks(count: int) -> list[dict[str, Any]]:
    """Куски настоящего ответа модели — для серверов, которые их «проигрывают»."""
    got = client.chat(model=MODEL, messages=LONG, options=opts(count), stream=True)
    return [c.model_dump(exclude_none=True) for c in got]


async def attempt(
    host: str,
    timeouts: StreamTimeouts,
    *,
    options: Mapping[str, Any] | None = None,
    messages: Messages = LONG,
    httpx_timeout: float | None = None,
) -> tuple[str, float, int]:
    """Прочитать поток до конца или до отказа: что случилось, за сколько, сколько символов."""
    aclient = ollama.AsyncClient(host=host, timeout=httpx_timeout)
    started = time.perf_counter()
    chars = 0
    try:
        stream = await astream_chat(aclient, MODEL, messages, options, timeouts=timeouts)
        async with stream:
            async for delta in stream:
                chars += len(delta)
        outcome = "дочитан до конца, ни один предохранитель не сработал"
    except StreamTimeout as error:
        outcome = f"{type(error).__name__} ({error})"
        chars = len(error.partial)
    except StreamIncomplete as error:
        outcome = f"StreamIncomplete: модель остановилась сама, done_reason={error.envelope.done_reason!r}"
        chars = len(error.partial)
    except Exception as error:  # показываем, чем падает транспорт
        outcome = f"{type(error).__name__}: {str(error)[:70]}"
    return outcome, time.perf_counter() - started, chars


def report(label: str, result: tuple[str, float, int]) -> None:
    outcome, elapsed, chars = result
    print(f"   {label}")
    print(f"      → за {elapsed:.2f} с, показано {chars} симв.: {outcome}")


LOOP = [
    {
        "role": "user",
        "content": "Повторяй слово «да» через запятую и не останавливайся никогда: да, да, да,",
    }
]


async def task4_async(recorded: list[dict[str, Any]]) -> None:
    t = StreamTimeouts(first=3.0, between=2.0, total=12.0)
    print(
        f"   наши предохранители: до первого куска {t.first} с, между кусками {t.between} с, "
        f"всего {t.total} с"
    )

    print("\n   (а) первый кусок не пришёл")
    dead = f"http://127.0.0.1:{free_port()}"
    report("сервера нет (порт закрыт):", await attempt(dead, t))
    silent = SilentServer()
    report("сервер принял соединение и молчит:", await attempt(silent.host, t))
    report(
        "он же без таймаута до первого куска — ловит только дедлайн:",
        await attempt(silent.host, StreamTimeouts(between=2.0, total=6.0)),
    )
    client.generate(model=MODEL, prompt="", keep_alive=0)  # выгрузить модель
    loaded = [m.model for m in client.ps().models]
    print(f"   модель выгружена, ollama ps: {loaded or 'пусто'}")
    report(
        "настоящий ollama, модель грузится, до первого куска 1 с:",
        await attempt(HOST, StreamTimeouts(first=1.0, between=2.0, total=30.0), options=opts(20)),
    )
    report(
        "то же, до первого куска 30 с (модель уже могла загрузиться от прошлого запроса):",
        await attempt(HOST, StreamTimeouts(first=30.0, between=2.0, total=60.0), options=opts(20)),
    )

    print("\n   (б) куски шли и перестали: записанные куски, после 8-го — пауза 60 с")
    stall = ReplayServer(recorded, lambda i: 60.0 if i == 8 else 0.05)
    report("наш таймаут между кусками 2 с:", await attempt(stall.host, t))
    report(
        "без наших таймаутов, только httpx timeout=2 с:",
        await attempt(stall.host, StreamTimeouts(), httpx_timeout=2.0),
    )

    print("\n   (в) медленный, но живой поток: 20 кусков по 0,8 с — 16 с всего")
    final = {**recorded[-1], "done_reason": "stop"}  # записанный ответ кончился по длине
    slow = ReplayServer([*recorded[:20], final], lambda i: 0.8)
    report(
        "таймаут между кусками 2 с, дедлайна нет:",
        await attempt(slow.host, StreamTimeouts(first=3.0, between=2.0)),
    )
    report(
        "только httpx timeout=2 с:",
        await attempt(slow.host, StreamTimeouts(), httpx_timeout=2.0),
    )
    report("наши три предохранителя, дедлайн 12 с:", await attempt(slow.host, t))

    print("\n   (г) поток не кончается: настоящая модель, просьба повторять «да» бесконечно")
    report(
        "num_predict=1500 (почти бесконечно), между кусками 2 с, дедлайн 12 с:",
        await attempt(HOST, t, options=opts(1500), messages=LOOP),
    )
    report(
        "num_predict=100 — предохранитель на стороне сервера:",
        await attempt(HOST, t, options=opts(100), messages=LOOP),
    )


def task4() -> None:
    print("\n4. Три таймаута")
    prediction(4)
    warm_up()
    recorded = record_chunks(40)
    print(f"   записано {len(recorded)} кусков настоящего ответа для локальных серверов")
    asyncio.run(task4_async(recorded))

    print("\n   можно ли повторять — по services/http_retry.should_retry (метод POST /api/chat):")
    cases = [
        ("сервера нет, соединение не установлено", httpx.ConnectError),
        ("первый кусок не пришёл (запрос ушёл, ответа нет)", httpx.ReadTimeout),
        ("куски перестали приходить посреди ответа", httpx.ReadTimeout),
        ("соединение порвалось посреди ответа", httpx.RemoteProtocolError),
    ]
    for label, error in cases:
        print(f"      {label:<48} {error.__name__:<20} повторять: {should_retry(error, 'POST')}")
    print("      поток не кончается — не ошибка транспорта, should_retry её не знает")


# ===========================================================================
# 5. Межтокенная задержка
# ===========================================================================
RUNS_5 = 3


def gaps_of(seed: int) -> tuple[list[float], list[str], float, ollama.ChatResponse]:
    """Паузы между кусками (мс), тексты кусков, TTFT (мс) и последний кусок."""
    started = time.perf_counter()
    stamps: list[float] = []
    texts: list[str] = []
    last: ollama.ChatResponse | None = None
    options = {"temperature": 0.7, "seed": seed, "num_predict": 600}
    for chunk in client.chat(model=MODEL, messages=LONG, options=options, stream=True):
        stamps.append(time.perf_counter())
        texts.append(chunk.message.content or "")
        last = chunk
    assert last is not None
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)]
    return gaps, texts, (stamps[0] - started) * 1000, last


def histogram(gaps: Sequence[float], median: float) -> None:
    edges = [0, 0.5, 0.8, 0.9, 1.1, 1.25, 1.5, 2, 3, float("inf")]
    print("      гистограмма (в медианах; счёт и доля):")
    for lo, hi in zip(edges, edges[1:], strict=False):
        count = sum(1 for g in gaps if lo * median <= g < hi * median)
        top = "∞" if hi == float("inf") else f"{hi * median:.0f}"
        bar = "#" * round(60 * count / len(gaps))
        print(
            f"        {lo:>4g}–{hi:<4g} ({lo * median:>4.0f}–{top:>4} мс): {count:>4} "
            f"{count / len(gaps):>5.1%} {bar}"
        )


def task5() -> None:
    print(f"\n5. Межтокенная задержка: ответ в 600 токенов, {RUNS_5} прогона (temperature 0,7)")
    prediction(5)
    warm_up()
    for seed in range(1, RUNS_5 + 1):
        gaps, texts, ttft, last = gaps_of(seed)
        median = statistics.median(gaps)
        print(f"   seed {seed}: кусков {len(texts)}, пауз {len(gaps)}, TTFT {ttft:.0f} мс")
        print(
            f"      медиана {median:.1f} мс, среднее {statistics.fmean(gaps):.1f}, "
            f"95-й процентиль {percentile(gaps, 95):.1f}, 99-й {percentile(gaps, 99):.1f}, "
            f"максимум {max(gaps):.1f}, ст. отклонение {statistics.pstdev(gaps):.1f}"
        )
        first, tail = gaps[:100], gaps[-100:]
        print(
            f"      медиана первых 100 пауз {statistics.median(first):.1f} мс, "
            f"последних 100 — {statistics.median(tail):.1f} мс"
        )
        wall = ttft + sum(gaps)
        print(
            f"      по часам клиента {wall / 1000:.2f} с; total_duration {ms(last.total_duration) / 1000:.2f} с; "
            f"eval_duration {ms(last.eval_duration) / 1000:.2f} с на {last.eval_count} токенов "
            f"({ms(last.eval_duration) / max(last.eval_count or 1, 1):.1f} мс/токен)"
        )
        histogram(gaps, median)
        stalls = [(i, g) for i, g in enumerate(gaps) if g > 3 * median]
        if not stalls:
            print("      замираний больше трёх медиан: нет")
        for i, g in stalls[:10]:
            before = "".join(texts[max(0, i - 6) : i + 1])
            print(
                f"      замирание: перед куском {i + 1} из {len(texts)}, {g:.0f} мс "
                f"({g / median:.1f} медианы), после {before[-30:]!r}"
            )
        if len(stalls) > 10:
            print(f"      … и ещё {len(stalls) - 10}")


# ===========================================================================
# 6. Брошенный поток
# ===========================================================================
PORT = urlparse(HOST).port or 11434
LOG = Path.home() / ".ollama" / "logs" / "server.log"


def connections() -> str:
    """Открытые TCP-соединения этого процесса к ollama — по данным ОС."""
    pid = str(os.getpid())
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-a", "-p", pid, f"-iTCP:{PORT}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return "lsof нет"
    states = re.findall(r"\((\w+)\)", out)
    return f"{len(states)} ({', '.join(states)})" if states else "0"


def pool(c: ollama.Client) -> str:
    """Что думает пул httpx внутри клиента ollama (внутренности — только для отладки)."""
    try:
        conns = c._client._transport._pool.connections
    except AttributeError:
        return "?"
    return ", ".join(repr(x).split("[", 1)[-1].rstrip("]>") for x in conns) or "пусто"


def probe() -> tuple[float, float]:
    """Короткий запрос сразу после: если сервер ещё занят брошенным, он подождёт."""
    started = time.perf_counter()
    r = ollama.Client(host=HOST).chat(model=MODEL, messages=SHORT, options=opts(5))
    return time.perf_counter() - started, ms(r.total_duration) / 1000


def log_size() -> int:
    return LOG.stat().st_size if LOG.exists() else 0


def log_since(offset: int) -> list[str]:
    """Новые строки лога ollama, где он пишет об отменах и разрывах."""
    if not LOG.exists():
        return []
    with LOG.open("rb") as f:
        f.seek(offset)
        lines = f.read().decode(errors="replace").splitlines()
    keys = ("abort", "cancel", "closed", "client", "disconnect", "context")
    return [ln for ln in lines if any(k in ln.lower() for k in keys)][-5:]


def task6() -> None:
    print("\n6. Брошенный поток: чтение прервано на 10-м куске из 600")
    prediction(6)
    warm_up()
    c = ollama.Client(host=HOST)
    started = time.perf_counter()
    got = 0
    for _ in c.chat(model=MODEL, messages=LONG, options=opts(600), stream=True):
        got += 1
    full = time.perf_counter() - started
    print(f"   полный прогон: {got} кусков за {full:.2f} с")
    print(f"      соединения после: ОС {connections()}; пул httpx: {pool(c)}")
    took, server = probe()
    print(f"      пробный запрос: {took:.2f} с по часам, {server:.2f} с на сервере")

    print("   break на 10-м куске, ссылка на генератор жива:")
    c2 = ollama.Client(host=HOST)
    t0 = log_size()
    started = time.perf_counter()
    gen = c2.chat(model=MODEL, messages=LONG, options=opts(600), stream=True)
    for i, _ in enumerate(gen, start=1):
        if i == 10:
            break
    print(f"      прочитано 10 кусков за {time.perf_counter() - started:.2f} с")
    print(f"      соединения: ОС {connections()}; пул httpx: {pool(c2)}")
    took, server = probe()
    print(
        f"      пробный запрос: {took:.2f} с по часам, {server:.2f} с на сервере; "
        f"от начала брошенного запроса до свободного сервера — {time.perf_counter() - started:.2f} с "
        f"(полный прогон {full:.2f} с)"
    )
    for line in log_since(t0):
        print(f"      лог ollama: {line[:150]}")
    close(gen)
    print(f"      после gen.close(): ОС {connections()}; пул httpx: {pool(c2)}")

    print("   break на 10-м куске и сразу close():")
    c3 = ollama.Client(host=HOST)
    t0 = log_size()
    started = time.perf_counter()
    with stream_chat(c3, MODEL, LONG, opts(600)) as stream:
        for i, _ in enumerate(stream, start=1):
            if i == 10:
                break
    print(f"      соединения после with: ОС {connections()}; пул httpx: {pool(c3)}")
    took, server = probe()
    print(
        f"      пробный запрос: {took:.2f} с по часам, {server:.2f} с на сервере; "
        f"от начала до свободного сервера — {time.perf_counter() - started:.2f} с"
    )
    for line in log_since(t0):
        print(f"      лог ollama: {line[:150]}")

    print("   break и del gen (последняя ссылка) без close():")
    c4 = ollama.Client(host=HOST)
    gen4 = c4.chat(model=MODEL, messages=LONG, options=opts(600), stream=True)
    for i, _ in enumerate(gen4, start=1):
        if i == 10:
            break
    print(f"      до del: ОС {connections()}")
    del gen4
    gc.collect()
    print(f"      после del: ОС {connections()}; пул httpx: {pool(c4)}")
    took, server = probe()
    print(f"      пробный запрос: {took:.2f} с по часам")
    if not LOG.exists():
        print(f"   лога {LOG} нет — строки об отмене смотри в логе ollama вручную")


# ===========================================================================
# 7. Conversation и поток
# ===========================================================================
def task7() -> None:
    print("\n7. Conversation.ask_stream: что попадает в историю, когда поток оборвался")
    for policy in PartialPolicy:
        conv = Conversation(BOOKSHOP_CHAT, budget=16_000, count_tokens=CharRatioEstimate(2.43))
        first = "".join(
            conv.ask_stream(client, MODEL, opts(1000), question="Посоветуй одну книгу на выходные.")
        )
        cut = "(не оборвался)"
        try:
            "".join(
                conv.ask_stream(
                    client,
                    MODEL,
                    opts(15),
                    partial=policy,
                    question="Почему именно её? Расскажи подробно.",
                )
            )
        except StreamIncomplete as error:
            cut = error.partial
        third = "".join(
            conv.ask_stream(
                client,
                MODEL,
                opts(1000),
                question="Повтори одной фразой, что ты сказал в своём прошлом ответе.",
            )
        )
        print(f"   {policy.name}:")
        print(f"      ход 1: {first[:70]!r}…")
        print(f"      ход 2 оборван, пользователь видел: {cut!r}")
        print(
            f"      в истории {len(conv.history)} сообщений; исходы ходов: "
            f"{[r.outcome for r in conv.records]}"
        )
        if policy is not PartialPolicy.DISCARD:
            print(f"      ход 2 в истории: {conv.history[3]['content']!r}")
        print(f"      ход 3 «повтори прошлый ответ»: {third[:120]!r}")


# ===========================================================================
# 8. Письменно
# ===========================================================================
# Что покупает streaming и чего не покупает. Покупает время до первого слова: человек видит,
#   что ответ пишется, через доли секунды, а не через всё время декода. Не покупает
#   ничего другого: модель генерирует те же токены с той же скоростью, общее время то же,
#   деньги и нагрузка те же. И берёт плату: законченность ответа известна только в конце.
#
# Почему mypy не поможет отличить кусок от ответа. В библиотеке ollama кусок и целый ответ —
#   один и тот же ChatResponse, и все поля в нём Optional. done_reason куска — None, ответа —
#   "stop", а тип у обоих str | None. Для mypy чтение done_reason из первого куска ничем не
#   хуже чтения из последнего. Различие появляется, только если завести свои типы: дельта —
#   str, конверт — Envelope с done_reason: str, и конверт существует только после
#   последнего куска (ragkit.streaming).
#
# Почему таймаут на чанк не защищает от бесконечного потока. Он считает паузу между двумя
#   кусками, а не длительность всего потока. Поток, который отдаёт кусок каждые 50 мс,
#   никогда не простоит 2 с — и будет идти хоть час. Так же устроен таймаут чтения httpx:
#   он на одну операцию чтения. От бесконечного потока защищают общий дедлайн на клиенте и
#   num_predict на сервере; num_predict надёжнее — он останавливает генерацию там, где она
#   идёт, а не только чтение.
#
# Почему ошибка посреди потока опаснее ошибки до него. Ошибка до первого куска — ничего не
#   показано: можно повторить запрос, и пользователь ничего не заметит. Ошибка после двухсот
#   показанных токенов — пользователь уже прочитал половину ответа. Повтор даст другой текст
#   (или тот же — но его надо уметь склеить), молчание оставит обрывок на экране, а в
#   истории надо решить, что лежит. И обрывок выглядит как текст: без пометки его не отличить
#   от законченного.
#
# Почему «записать обрывок как полный ответ» — решение про безопасность. История — это то,
#   что модель на следующих ходах считает уже сказанным и защищает. Ф4-03, задание 5:
#   ложь в собственном прошлом ответе модель повторила 20 раз из 20, и инструкция «исправь
#   ошибку» не помогла ни разу. Обрывок без пометки — такой же непроверенный текст: «не
#   рекомендуется при…» вместо «не рекомендуется при беременности» — уже другое
#   утверждение. Записав его как полный, ты кладёшь в контекст ложь, которую потом ничем не
#   вылечить.


TASKS: dict[int, Callable[[], None]] = {
    1: task1,
    2: task2,
    3: task3,
    4: task4,
    5: task5,
    6: task6,
    7: task7,
}


def main(argv: list[str]) -> None:
    header()
    for n in [int(a) for a in argv] or list(TASKS):
        started = time.perf_counter()
        TASKS[n]()
        print(f"   [задание {n}: {time.perf_counter() - started:.1f} с]")


if __name__ == "__main__":
    main(sys.argv[1:])
