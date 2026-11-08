"""Ф4-07. Токены, стоимость, rate limits — модель цены и замеры очереди.

Нужно: запущенный Ollama и `ollama pull qwen2.5:7b`. Из корня репозитория, по частям.
Задания 1–3, 6, 7 — при любом сервере:
    python -u week-04/lesson_07.py 1 2 3 6 7   2>&1 | tee week-04/lesson_07_part1.txt
Задание 4 — дважды, с перезапуском сервера (как перезапустить — в README_RUN ниже):
    SERVER_PARALLEL=1 python -u week-04/lesson_07.py 4   2>&1 | tee week-04/lesson_07_part2.txt
    SERVER_PARALLEL=4 python -u week-04/lesson_07.py 4   2>&1 | tee week-04/lesson_07_part3.txt
Задание 5 — сервер с маленькой очередью:
    SERVER_PARALLEL=1 SERVER_MAX_QUEUE=4 python -u week-04/lesson_07.py 5 \\
        2>&1 | tee week-04/lesson_07_part4.txt

SERVER_PARALLEL и SERVER_MAX_QUEUE — только подписи в выводе: клиент не может узнать
настройки сервера. Поэтому задание 4 само считает наблюдаемый параллелизм по времени
обслуживания запросов и сверяет его с подписью.

README_RUN — как перезапустить ollama на macOS с нужными переменными:
    1) выйти из приложения Ollama (значок в строке меню → Quit Ollama);
    2) в отдельном терминале: OLLAMA_NUM_PARALLEL=1 ollama serve
       (для задания 5: OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_QUEUE=4 ollama serve);
       в первых строках лога сервер печатает свою конфигурацию — проверь там
       OLLAMA_NUM_PARALLEL и OLLAMA_MAX_QUEUE;
    3) в другом терминале — команда из списка выше; после замеров — Ctrl+C в первом
       терминале и снова запустить приложение.

Денег здесь никто не платит: цены — модель на своих замерах токенов и чужом прайсе с датой.
Зато очередь меряется полностью: при переполнении параллелизма локальный сервер не
отвечает ошибкой, а держит запрос, и в каждом ответе честно рапортует нормальное время
генерации. Прогнозы — в PREDICTIONS, до запуска.
"""

import asyncio
import inspect
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import formatdate
from pathlib import Path
from typing import Any

import httpx
import lesson_06 as l6  # week-04/lesson_06.py: сценарий цикла с заказами и общие помощники
import ollama
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ragkit.conversation import CharRatioEstimate, split_turns, validate_alternation
from ragkit.cost import CostGuard, CostLimitExceededError, Price, conservative_ratio
from ragkit.structured import parse_structured
from ragkit.tools import IterationLimitError, LoopResult, run_tools, validate_tool_results
from services.http_retry import (
    RETRY_STATUSES,
    RetryLog,
    RetryPolicy,
    Sleep,
    backoff,
    request_with_retry,
    retry_after_seconds,
    should_retry,
)

MODEL = l6.MODEL
HOST = l6.HOST
client = l6.client
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
Msg = dict[str, Any]

# Прогнозы записаны ДО первого запуска. Меняй на свои — но до запуска.
PREDICTIONS = {
    1: "на 1000 запросов дороже всех диалог на 20 ходов (вход ~48 000 токенов на диалог "
    "из-за переотправки истории) — ~150 $ у Sonnet 5.5; RAG — ~15 $, цикл агента — ~6 $, "
    "промпт с инструментами — ~1,5 $, один вопрос-ответ — меньше 1 $; три главные статьи: "
    "переотправка истории диалога, большой результат инструмента в RAG, выход диалога; с "
    "кешем вход диалога подешевеет в ~10 раз, и главным станет выход",
    2: "по деньгам без кеша: чат — вход ~60 %, RAG — вход ~90 %, агент — вход ~75 %; с кешем "
    "у чата и агента главным станет выход; по времени: чат и агент — выход >90 % (токен "
    "ответа в 9,4 раза дороже, а prefill кешируется), RAG — вход (холодный prefill ~6000 "
    "токенов ~30 с против ~5 с ответа); «по деньгам» и «по времени» совпадут только у RAG",
    3: "символов на токен: русская проза ~2,4, английская ~4, Python ~3, JSON с кириллицей "
    "~2 (с ensure_ascii=True ~1,3 — \\uXXXX); разброс внутри типа ±10 %, между типами — в "
    "2–3 раза; оценка одного RAG-запроса по разным курсам разъедется в ~2 раза; калибровать "
    "на смеси фрагментов из корпуса Ф5, а для бюджета брать худший курс с запасом",
    4: "OLLAMA_NUM_PARALLEL=1: медиана TTFT при N=8 в 4–5 раз больше, чем при N=1, p95 — в "
    "~8 раз (последний ждёт семерых); клиентское общее время растёт так же; prompt_eval + "
    "eval на сервере не меняются (±10 %), а total_duration сервера ВКЛЮЧАЕТ ожидание и "
    "растёт вместе с клиентом; наблюдаемый параллелизм — 1. При 4: на N ≤ 4 TTFT почти как "
    "при N=1, на N=8 — вдвое больше; токенов в секунду на запрос меньше в 1,5–2 раза, а "
    "суммарно больше в ~2 раза; оптимум — параллелизм, равный ожидаемой нагрузке, пока "
    "токены в секунду на запрос выше скорости чтения (~10 ток/с)",
    5: "по умолчанию OLLAMA_MAX_QUEUE = 512 (FAQ ollama). При MAX_QUEUE=4 и загруженной модели "
    "12 одновременных запросов 503 не дадут: MAX_QUEUE — очередь планировщика до выдачи "
    "модели, а дальше запросы ждут слот внутри runner'а; при выгруженной модели запросы "
    "копятся в очереди планировщика, пока модель грузится, и лишние получат 503 «server "
    "busy, please try again.  maximum pending requests exceeded» за миллисекунды; "
    "Retry-After не будет; should_retry(503, 'POST') = True — и это верно: запрос не "
    "начинал выполняться",
    6: "Retry-After в секундах уважается точно (паузы 2,0 и 2,0, всего ~4 с, 3 попытки); в "
    "формате даты — пауза на долю секунды меньше (дата с точностью до секунды); "
    "Retry-After: 3600 клиент проспит час — cap на него не действует. Буря (20 клиентов, 5 "
    "в секунду): без джиттера — больше всех 429 (~60) и дольше всех; полный джиттер — 429 "
    "вдвое меньше, время ~5–6 с; Retry-After=1 — все отвергнутые вернутся ровно через "
    "секунду, всплеск в одном окне, но общее время близко к минимуму ~3 с",
    7: "запрос с полным списком заказов (~6000 токенов) за порогом 0,005 $ не уйдёт, "
    "урезанный уйдёт; оценка входа больше факта на 20–60 % — ошибается вверх, как задумано",
}


# ===========================================================================
# Общее
# ===========================================================================
def header() -> None:
    l6.header()


def prediction(n: int) -> None:
    print(f"   прогноз (до запуска): {PREDICTIONS[n]}")


def user(text: str, system: str | None = None) -> list[Msg]:
    return l6.user(text, system)


def share(k: int, n: int) -> str:
    return l6.share(k, n)


def pct(part: float, whole: float) -> str:
    return f"{part / whole:4.0%}" if whole else "  —"


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else math.nan


def p95(xs: Sequence[float]) -> float:
    """95-й процентиль по ближайшему рангу: на малых выборках это максимум, честно."""
    if not xs:
        return math.nan
    ordered = sorted(xs)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def ms(ns: int | None) -> float:
    return (ns or 0) / 1e6


# ===========================================================================
# Прайс: взят 8 ноября 2026 года со страницы цен Anthropic
# ===========================================================================
PRICE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
SONNET = Price(
    provider="Anthropic",
    model="Claude Sonnet 5.5",
    as_of="2026-11-08",
    source=PRICE_URL,
    input_per_mtok=2.0,
    output_per_mtok=10.0,
    cache_read_per_mtok=0.10,
)
SONNET_CACHE_WRITE = 2.50  # $ за миллион токенов, запись в кеш на 5 минут
HAIKU = Price(
    provider="Anthropic",
    model="Claude Haiku 5.5 (промпт ≤ 100 000 токенов)",
    as_of="2026-11-08",
    source=PRICE_URL,
    input_per_mtok=0.10,
    output_per_mtok=0.50,
    cache_read_per_mtok=0.01,
)
DECODE_MS = 45.9  # мс на токен ответа, Ф4-01 («короткий → длинный»): там, где время не снято


# ===========================================================================
# 1. Модель стоимости из своих замеров
# ===========================================================================
@dataclass(frozen=True)
class Req:
    """Один запрос к модели: токены и, если сняты, время prefill и decode."""

    prompt: int
    output: int
    prompt_ms: float | None = None
    output_ms: float | None = None
    cached: int = 0  # сколько токенов промпта — префикс прошлого запроса (кешируемо)


@dataclass(frozen=True)
class Scenario:
    name: str
    source: str
    requests: tuple[Req, ...]

    @property
    def prompt(self) -> int:
        return sum(r.prompt for r in self.requests)

    @property
    def output(self) -> int:
        return sum(r.output for r in self.requests)

    @property
    def cached(self) -> int:
        return sum(r.cached for r in self.requests)

    def cost(self, price: Price) -> float:
        return sum(price.cost(r.prompt, r.output) for r in self.requests)

    def cost_cached(self) -> tuple[float, float, float]:
        """Sonnet с кешем: (запись нового входа, чтение кеша, выход) в $."""
        write = sum((r.prompt - r.cached) for r in self.requests) * SONNET_CACHE_WRITE / 1e6
        read = self.cached * (SONNET.cache_read_per_mtok or 0) / 1e6
        out = self.output * SONNET.output_per_mtok / 1e6
        return write, read, out

    def seconds(self) -> tuple[float, float, bool]:
        """(prefill, decode, оценено ли хоть что-то) в секундах."""
        estimated = False
        pre = dec = 0.0
        for r in self.requests:
            pre += (r.prompt_ms or 0) / 1000
            if r.output_ms is None:
                estimated = True
                dec += r.output * DECODE_MS / 1000
            else:
                dec += r.output_ms / 1000
        return pre, dec, estimated


def chained(prompts: Sequence[int], outputs: Sequence[int], prompt_ms: Sequence[float],
            output_ms: Sequence[float | None]) -> tuple[Req, ...]:  # fmt: skip
    """Запросы одного разговора: префикс k-го — промпт и ответ (k−1)-го, если он короче."""
    out = []
    for k, (p, o) in enumerate(zip(prompts, outputs, strict=True)):
        cached = 0 if k == 0 else min(prompts[k - 1] + outputs[k - 1], p)
        out.append(Req(p, o, prompt_ms[k], output_ms[k], cached))
    return tuple(out)


# Ф4-03, week-04/lesson_03_part1.txt, строки 7–26: лестница «всё» на 20 ходов
DIALOG_PROMPT = [80, 189, 291, 392, 613, 841, 1170, 1306, 1549, 1930,
                 2203, 2363, 2739, 3170, 3567, 4063, 4624, 5104, 5703, 6269]  # fmt: skip
DIALOG_OUTPUT = [76, 75, 69, 196, 201, 305, 106, 217, 358, 248,
                 140, 346, 408, 369, 461, 537, 450, 573, 535, 206]  # fmt: skip
DIALOG_PREFILL_MS = [442, 324, 168, 321, 175, 179, 185, 187, 212, 191,
                     195, 192, 216, 236, 321, 628, 325, 340, 347, 439]  # fmt: skip


def loop_scenario(name: str, compact: bool) -> tuple[Scenario, LoopResult | None]:
    """Повтор сценария Ф4-06, задание 9 (temperature 0, seed 42): теперь с eval_count."""
    registry = l6.shop_registry(compact)
    result: LoopResult | None = None
    try:
        result = run_tools(client, MODEL, user(l6.COST_QUESTION), registry, max_iterations=6,
                           options={"seed": SEED, **l6.COST_OPTIONS})  # fmt: skip
        steps = result.steps
    except IterationLimitError as error:
        steps = error.steps
    prompts = [s.prompt_eval_count or 0 for s in steps]
    outputs = [s.eval_count or 0 for s in steps]
    req = chained(prompts, outputs, [ms(s.prompt_eval_ns) for s in steps],
                  [ms(s.eval_ns) for s in steps])  # fmt: skip
    source = "живой повтор Ф4-06, задание 9 (" + ("выжимка" if compact else "полный список") + ")"
    return Scenario(name, source, req), result


_SCENARIOS: list[Scenario] = []


def scenarios() -> list[Scenario]:
    """Пять форм из шести уроков. Живые повторы — один раз за запуск."""
    if _SCENARIOS:
        return _SCENARIOS
    qa = Scenario("один вопрос-ответ", "Ф4-03, part1.txt:7 (ход 1)",
                  (Req(80, 76, 442.0, None),))  # fmt: skip
    dialog = Scenario("диалог на 20 ходов", "Ф4-03, part1.txt:7–26",
                      chained(DIALOG_PROMPT, DIALOG_OUTPUT, DIALOG_PREFILL_MS,
                              [None] * 20))  # fmt: skip
    tools = Scenario("промпт с инструментами", "Ф4-06, part1.txt:16–18 (вызов и финал)",
                     chained([194, 252], [23, 33], [1008, 304], [1206, 1524]))  # fmt: skip
    agent, agent_result = loop_scenario("цикл агента, 3 итерации", compact=True)
    rag, rag_result = loop_scenario("RAG-подобный запрос", compact=False)
    for label, result in (("агент", agent_result), ("RAG", rag_result)):
        if result is not None:
            validate_tool_results(result.messages)
            print(f"      повтор [{label}]: ответ {l6.short(result.answer, 100)}")
    _SCENARIOS.extend([qa, dialog, tools, agent, rag])
    return _SCENARIOS


def task1() -> None:
    print(f"\n1. Модель стоимости: {SONNET.provider} {SONNET.model}, прайс на {SONNET.as_of}")
    prediction(1)
    print(f"   прайс: вход {SONNET.input_per_mtok} $, выход {SONNET.output_per_mtok} $, чтение кеша "
          f"{SONNET.cache_read_per_mtok} $, запись кеша {SONNET_CACHE_WRITE} $ за миллион; "
          f"для сравнения {HAIKU.model}: {HAIKU.input_per_mtok} / {HAIKU.output_per_mtok} $")  # fmt: skip
    print(f"   источник: {PRICE_URL}")
    items = scenarios()
    print(f"   {'сценарий':<26} {'запр.':>5} {'вход':>7} {'выход':>6} │ {'$ за запрос':>11} "
          f"{'$ за 1000':>9} {'с кешем':>8} │ {'Haiku, 1000':>11}")  # fmt: skip
    for s in items:
        cost = s.cost(SONNET)
        cached = sum(s.cost_cached())
        print(f"   {s.name:<26} {len(s.requests):>5} {s.prompt:>7} {s.output:>6} │ {cost:>11.5f} "
              f"{cost * 1000:>9.2f} {cached * 1000:>8.2f} │ {s.cost(HAIKU) * 1000:>11.3f}")  # fmt: skip
        print(f"      источник: {s.source}")
    tools_overhead = scenarios()[3].requests[0].prompt - bare_prompt()
    print(
        f"   описания трёх инструментов магазина: {tools_overhead} токенов в каждом обращении "
        "(первый промпт цикла минус тот же вопрос без tools)"
    )
    print("   статьи расхода, Sonnet 5.5 без кеша, $ на 1000 сценариев:")
    lines: list[tuple[float, str]] = []
    for s in items:
        fresh = sum(r.prompt - r.cached for r in s.requests)
        lines.append(
            (fresh * SONNET.input_per_mtok / 1e3, f"{s.name}: вход впервые ({fresh} ток.)")
        )
        lines.append((s.cached * SONNET.input_per_mtok / 1e3,
                      f"{s.name}: вход повторно — префикс прошлых запросов ({s.cached} ток.)"))  # fmt: skip
        lines.append(
            (s.output * SONNET.output_per_mtok / 1e3, f"{s.name}: выход ({s.output} ток.)")
        )
        if s.name.startswith(("цикл", "RAG")):
            n = len(s.requests)
            lines.append((tools_overhead * n * SONNET.input_per_mtok / 1e3,
                          f"{s.name}: описания инструментов ({tools_overhead} × {n}) — "
                          "внутри строк входа выше"))  # fmt: skip
    for rank, (usd, text) in enumerate(sorted(lines, reverse=True)[:6], start=1):
        print(f"      {rank}. {usd:8.2f} $  {text}")


_BARE: list[int] = []


def bare_prompt() -> int:
    if not _BARE:
        r = l6.chat(user(l6.COST_QUESTION), num_predict=1, **l6.COST_OPTIONS)
        _BARE.append(r.prompt_eval_count or 0)
    return _BARE[0]


# ===========================================================================
# 2. Что побеждает: цена или объём
# ===========================================================================
def task2() -> None:
    print("\n2. Доли входа и выхода: в деньгах (Sonnet 5.5) и в секундах (этот ноутбук)")
    prediction(2)
    forms = {"чат (диалог 20 ходов)": 1, "RAG": 4, "цикл агента": 3}
    items = scenarios()
    print(f"   {'форма':<22} │ {'$ вход':>8} {'$ выход':>8} {'вход %':>6} │ {'с кешем: вход':>13} "
          f"{'выход':>7} {'вход %':>6} │ {'с prefill':>9} {'с decode':>8} {'вход %':>6}")  # fmt: skip
    for label, index in forms.items():
        s = items[index]
        money_in = s.prompt * SONNET.input_per_mtok / 1e6
        money_out = s.output * SONNET.output_per_mtok / 1e6
        write, read, out = s.cost_cached()
        pre, dec, estimated = s.seconds()
        print(f"   {label:<22} │ {money_in:>8.4f} {money_out:>8.4f} {pct(money_in, money_in + money_out)} │ "
              f"{write + read:>13.4f} {out:>7.4f} {pct(write + read, write + read + out)} │ "
              f"{pre:>9.1f} {dec:>8.1f}{'*' if estimated else ' '}{pct(pre, pre + dec)}")  # fmt: skip
    print(f"   * decode оценён по {DECODE_MS} мс на токен (Ф4-01): в Ф4-03 время ответа не "
          "печаталось; prefill — фактический, с кешем префикса сервера")  # fmt: skip
    print(
        "   секунды в prefill — это время сервера с кешем префикса: в диалоге и цикле заново "
        "считается только новое"
    )


# ===========================================================================
# 3. Курс обмена — не константа
# ===========================================================================
RU_PROSE = [
    "Утром на станции было пусто: только дворник сметал с платформы мокрые листья, а в "
    "кассе дремала женщина в синем платке. Поезд опаздывал на сорок минут, и никто этому не "
    "удивлялся. Я купил в буфете чай в стакане с подстаканником, сел на скамейку и стал "
    "смотреть, как над лесом поднимается туман. Где-то далеко прогудел товарный состав, "
    "собака у шлагбаума подняла голову и снова уснула. Когда наконец показались огни, "
    "дворник отставил метлу и пошёл встречать почту.",
    "Бабушка хранила рецепты не в тетради, а на обороте старых открыток. На одной было "
    "написано «пирог с капустой, тесто как у Нины», и больше ничего: ни муки, ни масла, ни "
    "времени в духовке. Мама долго пыталась восстановить пропорции, звонила тёте Нине, но та "
    "уверяла, что всегда делала на глаз. В конце концов пирог получился, хотя и не такой. "
    "Открытку мы так и оставили на холодильнике под магнитом из Суздаля.",
    "Отчёт о продажах за третий квартал показал рост выручки на двенадцать процентов, но "
    "маржа снизилась: подорожала доставка, а скидки для крупных клиентов съели часть прибыли. "
    "Отдел закупок предлагает пересмотреть договоры с перевозчиками до конца года, отдел "
    "продаж — ограничить скидки суммой заказа. Решение отложили до совещания в понедельник; "
    "к нему финансовая служба подготовит расчёт по трём сценариям.",
]
EN_PROSE = [
    "The station was empty in the morning: only a caretaker sweeping wet leaves off the "
    "platform, and a woman in a blue scarf dozing in the ticket office. The train was forty "
    "minutes late, and nobody seemed surprised. I bought tea in a glass with a metal holder, "
    "sat on a bench and watched the fog rise over the forest. Somewhere far off a freight "
    "train sounded its horn; a dog by the barrier lifted its head and went back to sleep.",
    "Grandmother kept her recipes not in a notebook but on the backs of old postcards. One "
    "said only 'cabbage pie, dough like Nina's', with nothing about flour, butter or oven "
    "time. Mother spent weeks trying to work out the proportions and phoned aunt Nina, who "
    "insisted she had always done it by eye. In the end the pie turned out well, though not "
    "the same. We left the postcard on the fridge under a magnet.",
    "The third-quarter sales report showed revenue up twelve percent, but margins fell: "
    "shipping costs rose and discounts for large customers ate part of the profit. "
    "Purchasing proposes renegotiating carrier contracts before the end of the year; sales "
    "wants to cap discounts by order size. The decision was postponed to Monday's meeting, "
    "for which finance will prepare estimates under three scenarios.",
]


def code_samples() -> list[str]:
    """Настоящий код этого репозитория: три функции."""
    return [inspect.getsource(f) for f in (split_turns, validate_alternation, parse_structured)]


def json_samples() -> list[str]:
    return [
        json.dumps(l6.orders()[:2], ensure_ascii=False, indent=2),
        json.dumps(l6.shop_registry(False).schemas(), ensure_ascii=False),
        json.dumps({c[0]: l6.weather_of(c[0]) for c in l6.CITIES}, ensure_ascii=False),
    ]


def tokens_of(text: str) -> int:
    return l6.tokens_of(text)


@dataclass(frozen=True)
class Sample:
    kind: str
    chars: int
    tokens: int

    @property
    def ratio(self) -> float:
        return self.chars / self.tokens


_SAMPLES: list[Sample] = []


def measure_samples() -> list[Sample]:
    if _SAMPLES:
        return _SAMPLES
    kinds = {
        "русская проза": RU_PROSE,
        "английская проза": EN_PROSE,
        "Python": code_samples(),
        "JSON (ensure_ascii=False)": json_samples(),
    }
    for kind, texts in kinds.items():
        _SAMPLES.extend(Sample(kind, len(t), tokens_of(t)) for t in texts)
    return _SAMPLES


def rag_request() -> list[Msg]:
    """Один RAG-подобный запрос: русский и английский фрагменты, JSON и вопрос."""
    fragments = "\n\n".join([RU_PROSE[2], EN_PROSE[2], json_samples()[0]])
    return user(
        f"Найденные фрагменты:\n{fragments}\n\nВопрос: когда создан последний заказ и как "
        "изменилась маржа в третьем квартале?",
        system="Отвечай только по найденным фрагментам. Если ответа в них нет — так и скажи.",
    )


def task3() -> None:
    print("\n3. Курс обмена: символов на токен на четырёх типах текста, по 3 образца")
    prediction(3)
    samples = measure_samples()
    by_kind: dict[str, list[Sample]] = {}
    for s in samples:
        by_kind.setdefault(s.kind, []).append(s)
    for kind, group in by_kind.items():
        ratios = [s.ratio for s in group]
        listed = ", ".join(f"{s.chars}/{s.tokens}" for s in group)
        print(f"   {kind:<26} {min(ratios):.2f}–{max(ratios):.2f} символа на токен "
              f"(разброс {max(ratios) / min(ratios) - 1:+.0%}; символов/токенов: {listed})")  # fmt: skip
    all_ratios = [s.ratio for s in samples]
    print(f"   по всем 12: {min(all_ratios):.2f}–{max(all_ratios):.2f}, "
          f"в {max(all_ratios) / min(all_ratios):.1f} раза")  # fmt: skip
    escaped = json.dumps(l6.orders()[:2], ensure_ascii=True, indent=2)
    plain = json_samples()[0]
    t_esc, t_plain = tokens_of(escaped), tokens_of(plain)
    print(f"   тот же JSON с ensure_ascii=True (\\uXXXX вместо кириллицы): {len(escaped)} символов, "
          f"{t_esc} токенов против {len(plain)} и {t_plain} — в {t_esc / t_plain:.1f} раза больше токенов")  # fmt: skip

    request = rag_request()
    actual = l6.chat(request, num_predict=1, **l6.COST_OPTIONS).prompt_eval_count or 0
    chars = sum(len(m["content"]) for m in request)
    print(
        f"   один RAG-подобный запрос ({chars} символов в двух сообщениях): факт {actual} токенов"
    )
    print(f"   {'курс из замера':<34} {'симв/ток':>8} {'оценка':>7} {'ошибка':>7}")
    candidates = [(f"{s.kind} #{i % 3 + 1}", s.ratio) for i, s in enumerate(samples)]
    candidates += [("Ф4-03, откалиброван на заметках", 2.43), ("Ф4-02, примеры", 1.65)]
    pairs = [(s.chars, s.tokens) for s in samples]
    candidates.append(("худший из 12 × 0,9 (для бюджета)", conservative_ratio(pairs)))
    estimates = []
    for label, ratio in candidates:
        e = CharRatioEstimate(chars_per_token=ratio)(request)
        estimates.append(e)
        print(f"   {label:<34} {ratio:>8.2f} {e:>7} {e / actual - 1:>+7.0%}")
    print(f"   разброс оценок одного запроса: {min(estimates)}–{max(estimates)} токенов, "
          f"в {max(estimates) / min(estimates):.1f} раза")  # fmt: skip
    print("   новые пары для tests/fixtures/token_counts.json:")
    for s in samples:
        print(
            f"      {json.dumps({'kind': s.kind, 'chars': s.chars, 'tokens': s.tokens}, ensure_ascii=False)}"
        )


# ===========================================================================
# 4. Сервер выглядит здоровым, пока люди ждут
# ===========================================================================
NUM_CTX = 4096  # одно окно на все запросы: иначе ollama держит отдельную копию модели
CONC_OPTIONS: dict[str, Any] = {"temperature": 0, "num_predict": 80, "num_ctx": NUM_CTX}
CONCURRENCY = [1, 2, 4, 8]
ROUNDS = 3


@dataclass
class Timed:
    start: float
    first: float | None
    end: float
    prompt_tokens: int
    output_tokens: int
    load_ns: int
    prompt_ns: int
    eval_ns: int
    total_ns: int

    @property
    def ttft(self) -> float:
        return (self.first or self.end) - self.start

    @property
    def client_total(self) -> float:
        return self.end - self.start

    @property
    def server_work(self) -> float:
        return (self.prompt_ns + self.eval_ns) / 1e9

    @property
    def server_total(self) -> float:
        return self.total_ns / 1e9

    @property
    def hidden(self) -> float:
        """total_duration минус три фазы: что сервер насчитал сверх работы."""
        return (self.total_ns - self.load_ns - self.prompt_ns - self.eval_ns) / 1e9

    @property
    def tok_s(self) -> float:
        return self.output_tokens / (self.eval_ns / 1e9) if self.eval_ns else math.nan


def story(k: int) -> list[Msg]:
    city = l6.CITIES[k % len(l6.CITIES)][0]
    return user(
        f"Запрос {k}. {RU_PROSE[0]}\n\nНапиши продолжение этого рассказа примерно на 150 слов: "
        f"герой садится в поезд до города {city}. Без списков и заголовков."
    )


async def timed_chat(aclient: ollama.AsyncClient, messages: list[Msg]) -> Timed:
    start = time.perf_counter()
    first: float | None = None
    final: ollama.ChatResponse | None = None
    stream = await aclient.chat(model=MODEL, messages=messages, stream=True, options=CONC_OPTIONS)
    async for part in stream:
        if first is None and part.message.content:
            first = time.perf_counter()
        if part.done:
            final = part
    end = time.perf_counter()
    if final is None:
        raise RuntimeError("поток кончился без финального куска")
    return Timed(start, first, end, final.prompt_eval_count or 0, final.eval_count or 0,
                 final.load_duration or 0, final.prompt_eval_duration or 0,
                 final.eval_duration or 0, final.total_duration or 0)  # fmt: skip


def observed_parallelism(rows: Sequence[Timed]) -> int:
    """Сколько запросов обслуживалось одновременно: интервалы [конец − работа, конец]."""
    events = []
    for r in rows:
        events.append((r.end - r.server_work + 0.02, 1))  # 20 мс допуска на стыке
        events.append((r.end, -1))
    best = now = 0
    for _, delta in sorted(events):
        now += delta
        best = max(best, now)
    return best


async def concurrency_rounds() -> dict[int, tuple[list[Timed], list[float], list[float], int]]:
    aclient = ollama.AsyncClient(host=HOST)
    await timed_chat(aclient, story(999))  # прогрев: загрузка модели с этим окном
    out: dict[int, tuple[list[Timed], list[float], list[float], int]] = {}
    counter = 0
    for n in CONCURRENCY:
        rows: list[Timed] = []
        aggregate: list[float] = []
        walls: list[float] = []
        parallel = 0
        for _ in range(ROUNDS):
            batch = [story(counter + i) for i in range(n)]
            counter += n
            got = await asyncio.gather(*(timed_chat(aclient, m) for m in batch))
            wall = max(r.end for r in got) - min(r.start for r in got)
            walls.append(wall)
            aggregate.append(sum(r.output_tokens for r in got) / wall)
            parallel = max(parallel, observed_parallelism(got))
            rows.extend(got)
        out[n] = (rows, aggregate, walls, parallel)
    return out


def task4() -> None:
    label = os.environ.get("SERVER_PARALLEL", "?")
    print(f"\n4. Очередь глазами клиента и сервера: OLLAMA_NUM_PARALLEL={label} (подпись запуска)")
    prediction(4)
    print(f"   N одновременных = {CONCURRENCY}, по {ROUNDS} раунда; num_predict=80, окно {NUM_CTX}, "
          "temperature 0; у каждого запроса своё начало — кеш префикса не помогает")  # fmt: skip
    data = asyncio.run(concurrency_rounds())
    print(f"   {'N':>2} │ {'TTFT мед':>8} {'p95':>6} │ {'клиент мед':>10} {'p95':>6} │ "
          f"{'сервер: prefill+decode':>22} {'total_duration':>14} {'сверх фаз':>9} │ "
          f"{'ток/с на запрос':>15} {'всего ток/с':>11} │ {'паралл.':>7}")  # fmt: skip
    base_ttft = base_total = base_work = math.nan
    for n, (rows, aggregate, _, parallel) in data.items():
        ttft = [r.ttft for r in rows]
        total = [r.client_total for r in rows]
        work = [r.server_work for r in rows]
        if n == CONCURRENCY[0]:
            base_ttft, base_total, base_work = median(ttft), median(total), median(work)
        print(f"   {n:>2} │ {median(ttft):>8.2f} {p95(ttft):>6.2f} │ {median(total):>10.2f} "
              f"{p95(total):>6.2f} │ {median(work):>22.2f} {median([r.server_total for r in rows]):>14.2f} "
              f"{median([r.hidden for r in rows]):>9.2f} │ {median([r.tok_s for r in rows]):>15.1f} "
              f"{median(aggregate):>11.1f} │ {parallel:>7}")  # fmt: skip
    print("   во сколько раз выросло против N=1 (медианы):")
    for n, (rows, _, _, _) in data.items():
        print(f"      N={n}: TTFT ×{median([r.ttft for r in rows]) / base_ttft:.1f}, клиент "
              f"×{median([r.client_total for r in rows]) / base_total:.1f}, сервер prefill+decode "
              f"×{median([r.server_work for r in rows]) / base_work:.2f}")  # fmt: skip
    rows8 = data[CONCURRENCY[-1]][0]
    worst = max(rows8, key=lambda r: r.ttft)
    print(f"   худший запрос при N={CONCURRENCY[-1]}: TTFT {worst.ttft:.2f} с, клиент "
          f"{worst.client_total:.2f} с; сервер о нём: prompt_eval {ms(worst.prompt_ns):.0f} мс, "
          f"eval {ms(worst.eval_ns):.0f} мс, total_duration {worst.server_total:.2f} с, "
          f"load {ms(worst.load_ns):.0f} мс")  # fmt: skip
    print(
        "   «сверх фаз» = total_duration − load − prompt_eval − eval: если сервер считает "
        "ожидание, оно здесь"
    )


# ===========================================================================
# 5. Где очередь кончается
# ===========================================================================
OVERFLOW = 12


async def raw_chat(http: httpx.AsyncClient, k: int) -> tuple[int, float, str, str | None]:
    body = {"model": MODEL, "messages": story(500 + k), "stream": False,
            "options": {**CONC_OPTIONS, "num_predict": 16}}  # fmt: skip
    start = time.perf_counter()
    response = await http.post("/api/chat", json=body)
    elapsed = time.perf_counter() - start
    text = response.text if response.status_code != 200 else ""
    return response.status_code, elapsed, text[:140], response.headers.get("Retry-After")


async def unload() -> None:
    async with httpx.AsyncClient(base_url=HOST, timeout=120, trust_env=False) as http:
        await http.post("/api/generate", json={"model": MODEL, "keep_alive": 0})
        for _ in range(100):
            loaded = (await http.get("/api/ps")).json().get("models", [])
            if not any(m.get("model") == MODEL or m.get("name") == MODEL for m in loaded):
                return
            await asyncio.sleep(0.2)


async def overflow(loaded: bool) -> list[tuple[int, float, str, str | None]]:
    if loaded:
        aclient = ollama.AsyncClient(host=HOST)
        await timed_chat(aclient, story(998))
    else:
        await unload()
    limits = httpx.Limits(max_connections=OVERFLOW * 2)
    async with httpx.AsyncClient(
        base_url=HOST, timeout=600, trust_env=False, limits=limits
    ) as http:
        return list(await asyncio.gather(*(raw_chat(http, k) for k in range(OVERFLOW))))


def task5() -> None:
    queue = os.environ.get("SERVER_MAX_QUEUE", "?")
    parallel = os.environ.get("SERVER_PARALLEL", "?")
    print(f"\n5. Где очередь кончается: OLLAMA_MAX_QUEUE={queue}, OLLAMA_NUM_PARALLEL={parallel} "
          f"(подписи запуска); {OVERFLOW} одновременных запросов")  # fmt: skip
    prediction(5)
    print("   по умолчанию OLLAMA_MAX_QUEUE = 512 — FAQ ollama, docs.ollama.com/faq")
    for loaded in (True, False):
        title = "модель загружена" if loaded else "модель выгружена (keep_alive=0), грузится"
        rows = asyncio.run(overflow(loaded))
        codes = Counter(code for code, _, _, _ in rows)
        print(f"   [{title}] коды: {dict(codes)}")
        for code in sorted(codes):
            times = [t for c, t, _, _ in rows if c == code]
            example = next((txt for c, _, txt, _ in rows if c == code), "")
            hint = next((ra for c, _, _, ra in rows if c == code), None)
            print(f"      {code}: {len(times)} шт., время ответа {min(times):.3f}–{max(times):.3f} с; "
                  f"Retry-After: {hint!r}" + (f"; тело: {example!r}" if example else ""))  # fmt: skip
    print("   классификация services/http_retry.py:")
    for code in (503, 429, 500):
        print(f"      should_retry({code}, 'POST') = {should_retry(code, 'POST')}")
    print(f"      RETRY_STATUSES = {sorted(RETRY_STATUSES)}; метод для кода не смотрится")
    pause = backoff(RetryPolicy(), 1, lambda: 0.0)
    print(f"      без Retry-After пауза по формуле: после 1-й попытки {pause} с + джиттер")


# ===========================================================================
# 6. Настоящий 429 и буря повторов
# ===========================================================================
class TokenBucket:
    """rate запросов в секунду, запас capacity; без блокировок — сервер однопоточный."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate, self.capacity = rate, capacity
        self.tokens = capacity
        self.at = time.monotonic()

    def take(self) -> float | None:
        """None — пропустил; иначе через сколько секунд появится токен."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if self.tokens >= 1:
            self.tokens -= 1
            return None
        return (1 - self.tokens) / self.rate


@dataclass
class FakeState:
    counts: Counter[str] = field(default_factory=Counter)
    bucket: TokenBucket = field(default_factory=lambda: TokenBucket(5, 5))
    arrivals: list[tuple[float, int]] = field(default_factory=list)  # (время, код)


STATE = FakeState()


async def seconds_route(request: Request) -> Response:
    key = request.path_params["key"]
    STATE.counts[key] += 1
    if STATE.counts[key] <= 2:
        return Response(status_code=429, headers={"Retry-After": "2"})
    return JSONResponse({"ok": True})


async def date_route(request: Request) -> Response:
    key = request.path_params["key"]
    STATE.counts[key] += 1
    if STATE.counts[key] <= 1:
        return Response(
            status_code=429, headers={"Retry-After": formatdate(time.time() + 3, usegmt=True)}
        )
    return JSONResponse({"ok": True})


async def huge_route(request: Request) -> Response:
    return Response(status_code=429, headers={"Retry-After": "3600"})


async def limited_route(request: Request) -> Response:
    wait = STATE.bucket.take()
    if wait is None:
        STATE.arrivals.append((time.monotonic(), 200))
        return JSONResponse({"ok": True})
    STATE.arrivals.append((time.monotonic(), 429))
    return Response(status_code=429, headers={"Retry-After": str(math.ceil(wait))})


APP = Starlette(routes=[
    Route("/seconds/{key}", seconds_route),
    Route("/date/{key}", date_route),
    Route("/huge", huge_route),
    Route("/limited", limited_route),
])  # fmt: skip


class FakeServer:
    """Настоящий HTTP-сервер на 127.0.0.1 в отдельном потоке: клиенту — как чужой API."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(APP, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]},
                                       daemon=True)  # fmt: skip

    def __enter__(self) -> "FakeServer":
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


async def one_retry(
    url: str, policy: RetryPolicy, sleep: Sleep = asyncio.sleep
) -> tuple[RetryLog, float, str]:
    log = RetryLog()
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
        try:
            response = await request_with_retry(
                http, "GET", url, policy=policy, sleep=sleep, log=log
            )
            result = str(response.status_code)
        except httpx.HTTPStatusError as error:
            result = f"сдался: {error.response.status_code}"
    return log, time.perf_counter() - start, result


STRATEGIES: dict[str, RetryPolicy] = {
    "без джиттера": RetryPolicy(attempts=10, base=0.5, cap=8.0, jitter=0.0, respect_retry_after=False),
    "аддитивный 0,5 с": RetryPolicy(attempts=10, base=0.5, cap=8.0, jitter=0.5, respect_retry_after=False),
    "полный джиттер": RetryPolicy(attempts=10, base=0.5, cap=8.0, jitter_mode="full", respect_retry_after=False),
    "Retry-After (как по умолчанию)": RetryPolicy(attempts=10, base=0.5, cap=8.0),
}  # fmt: skip
CLIENTS = 20
STORM_SEEDS = [1, 2, 3]


@dataclass(frozen=True)
class Storm:
    rejected: int
    makespan: float
    gave_up: int
    peak: int  # повторов в худшем окне 100 мс, без первой волны


async def storm(url: str, policy: RetryPolicy, seed: int) -> Storm:
    STATE.bucket = TokenBucket(5, 5)
    STATE.arrivals = []
    master = random.Random(seed)
    rngs = [random.Random(master.random()).random for _ in range(CLIENTS)]
    limits = httpx.Limits(max_connections=CLIENTS * 2)
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30, trust_env=False, limits=limits) as http:

        async def worker(i: int) -> bool:
            try:
                await request_with_retry(http, "GET", url, policy=policy, rng=rngs[i])
            except httpx.HTTPStatusError:
                return False
            return True

        done = await asyncio.gather(*(worker(i) for i in range(CLIENTS)))
    makespan = time.perf_counter() - start
    gave_up = sum(not ok for ok in done)
    rejected = sum(code == 429 for _, code in STATE.arrivals)
    t0 = STATE.arrivals[0][0] if STATE.arrivals else 0.0
    windows = Counter(int((t - t0) / 0.1) for t, _ in STATE.arrivals)
    windows.pop(0, None)  # первая волна — все 20 разом при любой стратегии
    return Storm(rejected, makespan, gave_up, max(windows.values(), default=0))


def task6() -> None:
    print("\n6. Настоящий 429: фейковый сервер на 127.0.0.1 и services/http_retry.py из Ф3-03")
    prediction(6)
    with FakeServer() as fake:
        policy = RetryPolicy()  # как в пакете: 5 попыток, Retry-After уважается
        for label, path in (("Retry-After: 2 (секунды), два раза", "/seconds/a"),
                            ("Retry-After: дата через 3 с, один раз", "/date/b")):  # fmt: skip
            log, elapsed, result = asyncio.run(one_retry(fake.url + path, policy))
            pauses = ", ".join(
                f"{a.pause:.3f} ({a.source})" for a in log.attempts if a.pause is not None
            )
            print(f"   [{label}] итог {result}, попыток {len(log.attempts)}, паузы: {pauses}; "
                  f"всего {elapsed:.2f} с")  # fmt: skip
        recorded: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            recorded.append(seconds)

        log, _, result = asyncio.run(
            one_retry(fake.url + "/huge", RetryPolicy(attempts=2), fake_sleep)
        )
        print(f"   [Retry-After: 3600] итог {result}; клиент лёг бы спать на {recorded} с при cap="
              f"{RetryPolicy().cap} — cap на Retry-After не действует (sleep подменён, не ждём)")  # fmt: skip
        print(f"   разбор даты: {retry_after_seconds(formatdate(time.time() + 3, usegmt=True), datetime_now())!r} с "
              "из «через 3 с» — HTTP-дата без долей секунды")  # fmt: skip

        print(f"\n   буря: {CLIENTS} клиентов разом, лимит 5 запросов в секунду (корзина, запас 5), "
              f"до 10 попыток; base 0,5, cap 8; по {len(STORM_SEEDS)} seed")  # fmt: skip
        print(f"   нижняя граница: {CLIENTS} запросов при 5 в секунду и запасе 5 — "
              f"{(CLIENTS - 5) / 5:.0f} с и {CLIENTS - 5} отказов 429 в первой волне")  # fmt: skip
        print(f"   {'стратегия':<32} {'429 (мед)':>9} {'время, с (мед)':>14} {'пик повторов':>13} "
              f"{'сдались':>7}   по seed: 429 / время")  # fmt: skip
        results: dict[str, list[Storm]] = {}
        for name, strategy in STRATEGIES.items():
            runs = [
                asyncio.run(storm(fake.url + "/limited", strategy, seed)) for seed in STORM_SEEDS
            ]
            results[name] = runs
            per_seed = "; ".join(f"{r.rejected} / {r.makespan:.1f}" for r in runs)
            print(f"   {name:<32} {median([r.rejected for r in runs]):>9.0f} "
                  f"{median([r.makespan for r in runs]):>14.2f} {median([r.peak for r in runs]):>13.0f} "
                  f"{sum(r.gave_up for r in runs):>7}   {per_seed}")  # fmt: skip


def datetime_now() -> datetime:
    return datetime.now(UTC)


# ===========================================================================
# 7. Бюджет в коде
# ===========================================================================
LIMIT_USD = 0.005  # порог на один запрос, $


def fixture_pairs() -> list[tuple[int, int]]:
    data = json.loads((ROOT / "tests" / "fixtures" / "token_counts.json").read_text())
    return [(s["chars"], s["tokens"]) for s in data["text_samples"]]


def orders_request(compact: bool) -> list[Msg]:
    data, _ = l6.shop_registry(compact).call("list_orders", {"customer_id": l6.CUSTOMER_ID})
    return user(f"Данные о заказах клиента:\n{json.dumps(data, ensure_ascii=False)}\n\n"
                f"Вопрос: {l6.COST_QUESTION}")  # fmt: skip


def task7() -> None:
    print(
        f"\n7. Бюджет в коде: ragkit.cost.CostGuard, порог {LIMIT_USD} $ на запрос, {SONNET.model}"
    )
    prediction(7)
    pairs = fixture_pairs()
    ratio = conservative_ratio(pairs)
    estimator = CharRatioEstimate(chars_per_token=ratio, per_message=5, base=30)
    print(f"   оценщик: худший курс из {len(pairs)} замеров × 0,9 = {ratio:.2f} символа на токен, "
          "база 30 токенов на разметку шаблона")  # fmt: skip
    guard = CostGuard(SONNET, LIMIT_USD, estimator)
    for label, compact in (("полный список заказов", False), ("выжимка", True)):
        messages = orders_request(compact)
        actual = l6.chat(messages, num_predict=1, **l6.COST_OPTIONS).prompt_eval_count or 0
        try:
            estimate = guard.check(messages, max_output_tokens=200)
        except CostLimitExceededError as error:
            print(f"   [{label}] НЕ отправлен: {error}")
            estimate = error.estimate
        else:
            reply = guard.chat(client, MODEL, messages, {"num_predict": 200, **l6.COST_OPTIONS})
            print(
                f"   [{label}] отправлен: оценка {estimate}; ответ {l6.short(reply.message.content, 80)}"
            )
        print(f"      вход: оценка {estimate.input_tokens}, факт {actual} — "
              f"{'вверх' if estimate.input_tokens >= actual else 'ВНИЗ'} на "
              f"{estimate.input_tokens / actual - 1:+.0%}; цена по факту входа и 200 токенов выхода "
              f"{SONNET.cost(actual, 200):.6f} $")  # fmt: skip
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/test_cost.py", "-p", "no:randomly"],
                           cwd=ROOT, capture_output=True, text=True, check=False)  # fmt: skip
    print(f"   pytest tests/test_cost.py: {tests.stdout.strip().splitlines()[-1]}")


# ===========================================================================
# 8. Письменно — после замеров
# ===========================================================================


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
