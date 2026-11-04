"""Ф4-01. Первый вызов API и параметры генерации.

Нужно: запущенный Ollama (приложение или `ollama serve`) и модель:
    ollama pull qwen2.5:7b
Запуск из корня репозитория:
    python week-04/lesson_01.py           # все задания (~10–20 минут)
    python week-04/lesson_01.py 2 7       # только задания 2 и 7
Другая модель: MODEL=llama3.2:3b python week-04/lesson_01.py

Все замеры — с фиксированным seed. Шапка вывода печатает модель, тег, квантование
и версию ollama: без них числа ничего не значат.
"""

import ast
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import ollama

from ragkit.llm import IncompleteResponseError, complete_chat

MODEL = os.environ.get("MODEL", "qwen2.5:7b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=HOST)
SEED = 42


def sec(ns: int | None) -> float:
    return (ns or 0) / 1e9


def ask(prompt: str, **options: Any) -> ollama.ChatResponse:
    return client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options=options)


def complete_raw(prompt: str, **kwargs: Any) -> ollama.GenerateResponse:
    """Продолжение текста без шаблона чата: модель видит ровно prompt."""
    r: ollama.GenerateResponse = client.generate(
        model=MODEL, prompt=prompt, raw=True, stream=False, **kwargs
    )
    return r


def raw_text(prompt: str, **options: Any) -> str:
    return complete_raw(prompt, options=options).response or ""


def header() -> None:
    version = httpx.get(f"{HOST}/api/version", timeout=10).json()["version"]
    show = client.show(MODEL)
    d = show.details
    digest = next((m.digest for m in client.list().models if m.model == MODEL), None) or "?"
    print(f"ollama {version}; модель {MODEL}, digest {digest[:12]}")
    if d is not None:
        print(
            f"   семейство {d.family}, параметров {d.parameter_size}, "
            f"квантование {d.quantization_level}, формат {d.format}"
        )


# ===========================================================================
# 1. Конверт ответа
# ===========================================================================
def envelope(r: ollama.ChatResponse) -> None:
    rest = sec(r.total_duration) - sec(r.load_duration) - sec(r.prompt_eval_duration)
    rest -= sec(r.eval_duration)
    print(f"   model                : {r.model}")
    print(f"   created_at           : {r.created_at}")
    print(f"   done                 : {r.done}")
    print(f"   done_reason          : {r.done_reason!r}  (stop — закончила сама, length — лимит)")
    print(f"   total_duration       : {sec(r.total_duration):.3f} с — весь вызов на сервере")
    print(f"   load_duration        : {sec(r.load_duration):.3f} с — загрузка модели в память")
    print(f"   prompt_eval_count    : {r.prompt_eval_count} токенов запроса")
    print(
        f"   prompt_eval_duration : {sec(r.prompt_eval_duration):.3f} с — чтение запроса (prefill)"
    )
    print(f"   eval_count           : {r.eval_count} токенов ответа")
    print(f"   eval_duration        : {sec(r.eval_duration):.3f} с — генерация (decode)")
    print(f"   message.role         : {r.message.role}")
    print(f"   logprobs             : {r.logprobs!r}  (None — не запрашивали)")
    print(f"   прочее (total − load − prefill − decode): {rest:.3f} с")


def task1() -> None:
    print("\n1. Конверт ответа")
    client.generate(model=MODEL, prompt="", keep_alive=0)  # выгрузить модель из памяти
    time.sleep(1)
    for n in (1, 2):
        r = ask(
            "Назови столицу Франции одним словом.",
            temperature=0,
            seed=SEED,
            num_predict=50,
        )
        print(
            f"  вызов {n} (модель {'выгружена' if n == 1 else 'уже в памяти'}): "
            f"content={r.message.content!r}"
        )
        envelope(r)


# ===========================================================================
# 2. Prefill против decode
# ===========================================================================
def long_text() -> str:
    """Около двух страниц: 60 разных пунктов, около 2000 токенов."""
    goods = ["муки", "сахара", "соли", "гречки", "риса", "масла", "чая", "кофе", "мёда", "овса"]
    cities = ["Казань", "Пермь", "Самару", "Тверь", "Омск", "Тулу"]
    lines = [
        f"Пункт {i}. В понедельник склад номер {i % 7 + 1} получил {i * 13 % 97 + 3} мешков "
        f"{goods[i % len(goods)]}, а во вторник отправил {i * 7 % 41 + 1} из них в "
        f"{cities[i % len(cities)]}; кладовщик отметил, что остаток проверен дважды."
        for i in range(1, 61)
    ]
    return "\n".join(lines)


CASES = [
    ("короткий → короткий", False, "Назови столицу Франции одним словом.", 20),
    ("короткий → длинный", False, "Напиши подробный рассказ о море, примерно 500 слов.", 600),
    ("длинный → короткий", True, "Сколько пунктов в тексте выше? Ответь одним числом.", 20),
    ("длинный → длинный", True, "Подробно перескажи текст выше, пункт за пунктом.", 600),
]


def task2() -> None:
    print("\n2. Prefill против decode (num_ctx=8192 у всех, чтобы модель не перезагружалась)")
    common = {"temperature": 0, "seed": SEED, "num_ctx": 8192}
    ask("Привет", num_predict=5, **common)  # прогрев: загрузка модели не войдёт в замеры
    text = long_text()
    rows = []
    for n, (name, is_long, question, limit) in enumerate(CASES, 1):
        # Уникальное начало: иначе ollama переиспользует кеш префикса от прошлого вызова,
        # и prefill длинного текста «бесплатный».
        prompt = f"Замер {n}.\n" + (f"{text}\n\n{question}" if is_long else question)
        r = ask(prompt, num_predict=limit, **common)
        pin, pdur = r.prompt_eval_count or 0, sec(r.prompt_eval_duration)
        out, odur = r.eval_count or 0, sec(r.eval_duration)
        rows.append((name, pin, pdur, out, odur, r.done_reason))
    print(
        f"   {'случай':<22}{'токенов запр.':>14}{'prefill с':>11}{'ток/с':>9}"
        f"{'токенов отв.':>14}{'decode с':>10}{'ток/с':>8}  done"
    )
    for name, pin, pdur, out, odur, done in rows:
        print(
            f"   {name:<22}{pin:>14}{pdur:>11.3f}{pin / pdur:>9.1f}"
            f"{out:>14}{odur:>10.3f}{out / odur:>8.1f}  {done}"
        )
    _, pin, pdur, out, odur, _ = rows[3]  # длинный → длинный: обе фазы большие
    prefill_ms, decode_ms = pdur / pin * 1000, odur / out * 1000
    print(
        f"   длинный → длинный: {prefill_ms:.2f} мс на токен запроса, "
        f"{decode_ms:.2f} мс на токен ответа -> токен ответа дороже в "
        f"{decode_ms / prefill_ms:.1f} раза"
    )


# ===========================================================================
# 3. Детерминизм и его границы
# ===========================================================================
# Прогнозы записаны ДО запуска. Поменяй на свои, если думаешь иначе, — но до запуска.
PREDICTIONS = {
    "temperature=0, seed=42": "1 вариант из 10: при нулевой температуре берётся самый "
    "вероятный токен, seed не участвует",
    "temperature=0, без seed": "1 вариант из 10: та же жадная выборка, seed ей не нужен",
    "temperature=0.8, seed=42": "1 вариант из 10: случайность есть, но генератор каждый "
    "раз стартует с одного seed",
    "temperature=0.8, без seed (контроль)": "несколько разных вариантов: случайность без "
    "фиксированного seed",
}


def task3() -> None:
    print("\n3. Детерминизм: 10 одинаковых запросов в каждой серии")
    prompt = "Придумай название для кафе у моря. Ответь одной строкой, без пояснений."
    series: list[tuple[str, dict[str, Any]]] = [
        ("temperature=0, seed=42", {"temperature": 0, "seed": SEED}),
        ("temperature=0, без seed", {"temperature": 0}),
        ("temperature=0.8, seed=42", {"temperature": 0.8, "seed": SEED}),
        ("temperature=0.8, без seed (контроль)", {"temperature": 0.8}),
    ]
    for name, opts in series:
        answers = [ask(prompt, num_predict=30, **opts).message.content or "" for _ in range(10)]
        counts = Counter(answers)
        print(f"   {name}")
        print(f"      прогноз : {PREDICTIONS[name]}")
        print(
            f"      факт    : {len(counts)} разных из 10, побайтово "
            f"{'совпали все' if len(counts) == 1 else 'НЕ совпали'}"
        )
        for text, k in counts.most_common(4):
            print(f"         {k} × {text!r}")


# ===========================================================================
# 4. Температура глазами
# ===========================================================================
TEMPERATURES = [0, 0.3, 0.7, 1.2, 2.0]


def top_tokens(prompt: str, k: int, **options: Any) -> list[tuple[str, float]]:
    """k самых вероятных первых токенов продолжения: (токен, вероятность)."""
    r = complete_raw(
        prompt, logprobs=True, top_logprobs=k, options={"num_predict": 1, "seed": SEED, **options}
    )
    if not r.logprobs:
        raise RuntimeError("ollama не вернул logprobs: нужна версия сервера с поддержкой logprobs")
    first = r.logprobs[0]
    return [(t.token, math.exp(t.logprob)) for t in first.top_logprobs or []]


def scaled(probs: Sequence[float], t: float) -> list[float]:
    """Как температура меняет распределение: p_i^(1/T), нормировано (по показанным токенам)."""
    if t == 0:
        return [1.0] + [0.0] * (len(probs) - 1)
    powered = [p ** (1 / t) for p in probs]
    total = sum(powered)
    return [p / total for p in powered]


def task4() -> None:
    print("\n4. Пять самых вероятных токенов при разной температуре")
    for prompt in ("Москва — столица", "Мой любимый цвет —"):
        print(f"   запрос {prompt!r}")
        base: list[tuple[str, float]] = []
        for t in TEMPERATURES:
            tops = top_tokens(prompt, 5, temperature=t)
            base = base or tops
            line = "  ".join(f"{tok!r}={p:.3f}" for tok, p in tops)
            gap = tops[0][1] - tops[1][1] if len(tops) > 1 else float("nan")
            print(f"      T={t:<4} ollama: {line}   разрыв 1−2: {gap:.3f}")
        tokens, probs = [tok for tok, _ in base], [p for _, p in base]
        print("      пересчёт p^(1/T) по тем же пяти токенам (что делает температура при выборке):")
        for t in TEMPERATURES:
            ps = scaled(probs, t)
            line = "  ".join(f"{tok!r}={p:.3f}" for tok, p in zip(tokens, ps, strict=True))
            print(f"      T={t:<4} {line}   разрыв 1−2: {ps[0] - ps[1]:.3f}")


# ===========================================================================
# 5. top_k и top_p
# ===========================================================================
def task5() -> None:
    print("\n5. Сколько вариантов оставляют top_k=5 и top_p=0.9")
    for prompt in ("Москва — столица", "Мой любимый цвет —"):
        tops = top_tokens(prompt, 20, temperature=1.0)
        probs = [p for _, p in tops]
        cumulative, kept_p = 0.0, None
        for i, p in enumerate(probs, 1):
            cumulative += p
            if cumulative >= 0.9:
                kept_p = i
                break
        print(f"   запрос {prompt!r}: первые токены {[t for t, _ in tops[:6]]}")
        print(f"      top_k=5   оставляет 5 токенов, их суммарная вероятность {sum(probs[:5]):.3f}")
        if kept_p is None:
            print(f"      top_p=0.9 оставляет больше 20 токенов: сумма 20 лучших {sum(probs):.3f}")
        else:
            print(f"      top_p=0.9 оставляет {kept_p} ток., сумма {sum(probs[:kept_p]):.3f}")
    print("   при temperature=0 top_k и top_p не влияют (запрос без единственного ответа):")
    variants = [
        {"top_k": 1, "top_p": 1.0},
        {"top_k": 5, "top_p": 0.5},
        {"top_k": 40, "top_p": 0.95},
        {"top_k": 100, "top_p": 0.1},
    ]
    for t in (0, 1.0):
        texts = [
            complete_raw(
                "Мой любимый цвет —",
                options={"temperature": t, "seed": SEED, "num_predict": 25, **v},
            ).response
            for v in variants
        ]
        print(f"      T={t}: {len(set(texts))} разных ответа из {len(texts)} настроек")
        for v, text in zip(variants, texts, strict=True):
            print(f"         {v} -> {text!r}")


# ===========================================================================
# 6. Обрыв по длине
# ===========================================================================
def task6() -> None:
    print("\n6. Обрыв по длине")
    messages = [{"role": "user", "content": "Напиши подробный рассказ о маяке на острове."}]
    r = client.chat(
        model=MODEL, messages=messages, options={"temperature": 0, "seed": SEED, "num_predict": 15}
    )
    print(f"   num_predict=15: done_reason={r.done_reason!r}, eval_count={r.eval_count}")
    print(f"   текст: {r.message.content!r}")
    try:
        complete_chat(client, MODEL, messages, {"temperature": 0, "seed": SEED, "num_predict": 15})
    except IncompleteResponseError as e:
        print(f"   complete_chat: IncompleteResponseError: {e}")
    else:
        print(
            "   complete_chat: ответ уместился в 15 токенов — увеличь задачу, чтобы увидеть обрыв"
        )
    full = complete_chat(
        client, MODEL, messages, {"temperature": 0, "seed": SEED, "num_predict": 1000}
    )
    print(f"   complete_chat с num_predict=1000: законченный ответ, {len(full)} символов")


# ===========================================================================
# 7. Окно контекста, которое молчит
# ===========================================================================
MARKER = "ФИОЛЕТОВЫЙ-ЖИРАФ-17"


def context_probe(num_ctx: int | None) -> tuple[int, str]:
    filler = "\n".join(
        f"Строка {i}. Поезд номер {i} отправился с платформы {i % 9 + 1} в {i % 24:02d}:{i % 60:02d} "
        f"и прибыл по расписанию, пассажиров было {i * 37 % 500 + 20}."
        for i in range(1, 251)
    )
    prompt = (
        f"Кодовое слово: {MARKER}. Запомни его.\n\n{filler}\n\n"
        "Какое кодовое слово было названо в самом начале этого текста? "
        "Ответь только кодовым словом."
    )
    options: dict[str, Any] = {"temperature": 0, "seed": SEED, "num_predict": 30}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    r = ask(prompt, **options)
    return r.prompt_eval_count or 0, r.message.content or ""


def task7() -> None:
    print("\n7. Окно контекста")
    show = client.show(MODEL)
    trained = {k: v for k, v in (show.modelinfo or {}).items() if k.endswith("context_length")}
    print(f"   модель умеет (modelinfo): {trained}")
    params = show.parameters or ""
    in_modelfile = [line for line in params.splitlines() if "num_ctx" in line]
    print(
        f"   num_ctx в параметрах модели: {in_modelfile or 'не задан — берётся умолчание сервера'}"
    )
    full, answer = context_probe(16384)
    print(
        f"   num_ctx=16384: прочитано {full} токенов, ответ {answer!r}, "
        f"маркер {'найден' if MARKER in answer else 'НЕ найден'}"
    )
    for ctx in (None, 2048):
        got, answer = context_probe(ctx)
        label = "по умолчанию" if ctx is None else f"num_ctx={ctx}"
        print(
            f"   {label}: прочитано {got} токенов из {full}, потеряно {full - got}; "
            f"ответ {answer!r}, маркер {'найден' if MARKER in answer else 'НЕ найден'}"
        )


# ===========================================================================
# 8. stop и repeat_penalty
# ===========================================================================
DIALOG = (
    "Пользователь: Привет! Подскажи, как сварить гречку?\n"
    "Ассистент: Промой крупу, залей водой 1:2 и вари 15 минут под крышкой.\n"
    "Пользователь: А сколько соли?\n"
    "Ассистент:"
)


def numbered_ok(text: str) -> int:
    return len(re.findall(r"^\s*\d+\.\s+\S", text, flags=re.MULTILINE))


def code_ok(text: str) -> bool:
    block = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    try:
        ast.parse(block.group(1) if block else text)
    except SyntaxError:
        return False
    return True


def task8() -> None:
    print("\n8. stop и repeat_penalty")
    base = {"temperature": 0, "seed": SEED, "num_predict": 120, "repeat_penalty": 1.0}
    free = raw_text(DIALOG, **base)
    print(f"   без stop: {free!r}")
    print(f"      модель дописала за пользователя: {'Пользователь:' in free}")
    cut = raw_text(DIALOG, **base, stop=["\nПользователь:"])
    print(f"   stop=['\\nПользователь:']: {cut!r}")

    loop_prompt = "Список покупок: молоко, хлеб, молоко, хлеб, молоко, хлеб,"
    for rp in (1.0, 1.5):
        looped = raw_text(loop_prompt, **{**base, "num_predict": 60, "repeat_penalty": rp})
        words = re.findall(r"\w+", looped.lower())
        share = sum(w in {"молоко", "хлеб"} for w in words) / max(len(words), 1)
        print(
            f"   зацикливание, repeat_penalty={rp}: доля 'молоко'/'хлеб' {share:.0%}, "
            f"{looped[:80]!r}"
        )

    checks: list[tuple[str, str, Callable[[str], object]]] = [
        (
            "список",
            "Напиши нумерованный список из 15 крупнейших городов России, "
            "по одному в строке, в формате '1. Москва'. Без пояснений.",
            numbered_ok,
        ),
        (
            "код",
            "Напиши на Python функцию sum_even(numbers), которая возвращает сумму чётных "
            "чисел списка, с docstring и тремя assert-проверками. Только код.",
            code_ok,
        ),
    ]
    for name, prompt, check in checks:
        for rp in (1.0, 1.8):
            r = ask(prompt, temperature=0, seed=SEED, num_predict=400, repeat_penalty=rp)
            text = r.message.content or ""
            print(f"   {name}, repeat_penalty={rp}: проверка -> {check(text)!r}")
            print("      " + text[:300].replace("\n", "\n      "))


TASKS = {
    1: task1,
    2: task2,
    3: task3,
    4: task4,
    5: task5,
    6: task6,
    7: task7,
    8: task8,
}


def main(argv: list[str]) -> None:
    header()
    chosen = [int(a) for a in argv] or list(TASKS)
    for n in chosen:
        started = time.perf_counter()
        TASKS[n]()
        print(f"   [задание {n}: {time.perf_counter() - started:.1f} с]")


if __name__ == "__main__":
    main(sys.argv[1:])
