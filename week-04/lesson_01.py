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


# Замеры: ollama 0.40.14, qwen2.5:7b (digest 845dbda0ea48, 7.6B, Q4_K_M, gguf), MacBook Air
#   на Apple Silicon. seed=42 во всех заданиях.
#
# Первый вызов, модель выгружена: total_duration 4,500 с, из них load_duration 4,050 с — 90 %
#   времени ушло на загрузку весов в память. Второй вызов с тем же запросом: load 0,002 с,
#   total 0,185 с — в 24 раза быстрее. «Первый запрос медленный» — это load_duration.
# Зачем каждое поле:
#   model, created_at — какая модель и когда ответила; без тега модели числа не сравнить.
#   done — ответ пришёл целиком (при стриминге куски идут с done=False).
#   done_reason — 'stop' в обоих вызовах: модель закончила сама. 'length' — упёрлась в
#     num_predict; это единственный признак оборванного ответа (задание 6).
#   total_duration — всё время на сервере; если оно заметно больше суммы фаз, время ушло
#     на загрузку или очередь.
#   load_duration — 4,050 с холодный старт, 0,002 с потом.
#   prompt_eval_count — 43 токена, хотя в вопросе 6 слов: шаблон чата qwen2.5 добавляет
#     системное сообщение по умолчанию и служебные токены ролей (видно в
#     `ollama show qwen2.5:7b --template`). Эти токены тоже занимают окно и время.
#   prompt_eval_duration — prefill: 0,298 с в первом вызове, 0,045 с во втором. Первый
#     в 6,6 раза медленнее — прогрев сразу после загрузки; точную причину я не измерял.
#   eval_count, eval_duration — 4 токена «Париж» за 0,136–0,138 с: около 29 токенов/с decode.
#   message.role — 'assistant'.
#   logprobs — None, пока не попросили logprobs=True (задание 4).
#   прочее — 0,013 и 0,001 с: накладные расходы сервера вне трёх фаз.


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


# Результат (num_ctx=8192, после прогревочного вызова):
#   случай               запрос, ток.  prefill, с  ток/с   ответ, ток.  decode, с  ток/с
#   короткий → короткий        49        0,160    306,5        4         0,135     29,7
#   короткий → длинный         55        0,160    344,6      600        27,554     21,8
#   длинный → короткий       3605       18,339    196,6        3         0,131     23,0
#   длинный → длинный        3602       19,364    186,0      600        30,310     19,8
# Токен ответа дороже токена запроса в 9,4 раза: 50,52 мс против 5,38 мс в случае
#   «длинный → длинный». Перекрёстная проверка даёт то же: decode в «коротком → длинном» —
#   45,9 мс на токен, prefill в «длинном → коротком» — 5,09 мс на токен, в 9,0 раза.
# Обе фазы медленнеют с длиной контекста: prefill 344,6 ток/с на 55 токенах против
#   186,0 на 3602; decode 29,7 ток/с при коротком запросе против 19,8 при длинном запросе и
#   длинном ответе — каждый новый токен смотрит на всё, что накоплено до него. Prefill на 49–55
#   токенах — это 0,16 с, в основном накладные расходы, поэтому 306–345 ток/с там завышены.
# Для Ф5. Сократить ответ на 1000 токенов — это около 50 с на этом ноутбуке, сократить
#   найденный контекст на 1000 токенов — около 5,4 с: длинный запрос дешевле длинного ответа
#   в 9 раз. Но prefill не бесплатный: 3600 токенов контекста — это 18–19 с до первого токена
#   ответа. Время до первого слова растёт с каждым найденным фрагментом, поэтому в контекст
#   кладут нужное, а не всё подряд.


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


# Результат: все четыре прогноза сошлись.
#   temperature=0, seed=42      — 1 вариант из 10: 'Ветер Соленой Волны' ×10.
#   temperature=0, без seed     — 1 вариант из 10, и тот же самый 'Ветер Соленой Волны':
#     при нулевой температуре берётся самый вероятный токен, seed не участвует.
#   temperature=0.8, seed=42    — 1 вариант из 10: 'Солнечный Песок' ×10. Случайность есть,
#     но генератор каждый раз начинает с того же seed — выборка повторяется.
#   temperature=0.8, без seed   — 9 разных из 10. Среди них 'Золотой ПесокCAFÉ': при 0,8
#     выборка иногда берёт маловероятный токен, и в ответ попадает мусор.
# Расхождений с прогнозом нет. Границы детерминизма, о которых говорит урок, этот опыт не
#   проверял: все 40 запросов шли по одному, на одной машине, с одной моделью и одним
#   num_ctx. Параллельные запросы, другой num_ctx, другое квантование или обновление ollama
#   могут дать другой ответ при том же seed — это я не измерял.


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


# Находка: ollama возвращает logprobs ДО температуры. Вероятность ' России' — 0,557 при
#   всех пяти температурах (0, 0,3, 0,7, 1,2, 2,0), остальные отличаются не больше чем на
#   0,001. Возвращённые числа — исходное распределение модели; температура применяется уже
#   при выборке, и увидеть её в этих полях нельзя. Поэтому вторая таблица — пересчёт
#   p^(1/T), нормированный по тем же пяти токенам. Это приближение: хвост распределения не
#   учтён (пять лучших дают 0,697 и 0,438 вероятности), а при T > 1 хвост растёт сильнее всего.
#
# Разрыв между первым и вторым токеном по пересчёту:
#   T                       0      0,3    0,7    1,2    2,0
#   «Москва — столица»    1,000  0,999  0,902  0,628  0,359   (исходно 0,557 против 0,045)
#   «Мой любимый цвет —»  1,000  0,288  0,095  0,047  0,025   (исходно 0,141 против 0,115)
# Когда ответ один, первый токен держится до T≈1: при 0,7 у ' России' всё ещё 0,927. Когда
#   правильного ответа нет, распределение плоское с самого начала: уже при T=0,3 второй
#   вариант получает 0,295, а при T=2,0 пять вариантов почти равны — от 0,259 до 0,156.
# И две детали, которые видно только по токенам. У ' России' всего 0,557, потому что
#   ' Российской' (0,034) и ',' (0,041) — тоже верные продолжения: уверенность в смысле
#   делится между разными токенами. А «цвета» — это обрывки слов ' с', ' з', ' сер', ' крас':
#   русское слово часто состоит из нескольких токенов, и выбор цвета решается на первом куске.


# ===========================================================================
# 5. top_k и top_p
# ===========================================================================
def kept_by_top_p(probs: Sequence[float], p: float) -> int | None:
    """Сколько первых токенов нужно, чтобы сумма достигла p; None — не хватило показанных."""
    total = 0.0
    for i, prob in enumerate(probs, 1):
        total += prob
        if total >= p:
            return i
    return None


def task5() -> None:
    print("\n5. Сколько вариантов оставляют top_k=5 и top_p")
    prompts = ("Столица Франции — город", "Москва — столица", "Мой любимый цвет —")
    for prompt in prompts:
        tops = top_tokens(prompt, 20, temperature=1.0)
        probs = [p for _, p in tops]
        print(f"   запрос {prompt!r}: первые токены {[(t, round(p, 3)) for t, p in tops[:4]]}")
        print(f"      top_k=5   оставляет 5 токенов, их суммарная вероятность {sum(probs[:5]):.3f}")
        for p in (0.5, 0.9):
            kept = kept_by_top_p(probs, p)
            if kept is None:
                print(
                    f"      top_p={p} оставляет больше 20 токенов: сумма 20 лучших {sum(probs):.3f}"
                )
            else:
                print(f"      top_p={p} оставляет {kept} ток., их сумма {sum(probs[:kept]):.3f}")
    print("   при temperature=0 top_k и top_p не влияют (запрос без единственного ответа):")
    variants = [
        {"top_k": 1, "top_p": 1.0},
        {"top_k": 5, "top_p": 0.5},
        {"top_k": 40, "top_p": 0.95},
        {"top_k": 100, "top_p": 0.1},
    ]
    for t in (0, 1.0):
        texts = [
            raw_text("Мой любимый цвет —", temperature=t, seed=SEED, num_predict=25, **v)
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
    opts: dict[str, Any] = {"temperature": 0, "seed": SEED}
    full = complete_chat(client, MODEL, messages, {**opts, "num_predict": 1000})
    print(f"   полный ответ (num_predict=1000): законченный, {len(full)} символов")
    # При temperature=0 оборванный ответ — начало полного: по следующему символу полного
    # видно, оборвалось ли слово посередине.
    mid_word = None
    for limit in range(10, 31):
        r = client.chat(model=MODEL, messages=messages, options={**opts, "num_predict": limit})
        text = r.message.content or ""
        nxt = full[len(text)] if full.startswith(text) and len(full) > len(text) else ""
        if r.done_reason == "length" and text[-1:].isalpha() and nxt.isalpha():
            mid_word = (limit, r, text, nxt)
            break
    if mid_word is None:
        print("   за 10–30 токенов обрыв посреди слова не встретился")
        return
    limit, r, text, nxt = mid_word
    print(f"   num_predict={limit}: done_reason={r.done_reason!r}, eval_count={r.eval_count}")
    print(f"   текст: {text!r}")
    print(
        f"   в полном ответе дальше идёт {full[len(text) - 12 : len(text) + 12]!r}: "
        f"слово оборвано перед буквой {nxt!r}"
    )
    try:
        complete_chat(client, MODEL, messages, {**opts, "num_predict": limit})
    except IncompleteResponseError as e:
        print(f"   complete_chat: IncompleteResponseError: {e}")


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


def loaded_context() -> int | None:
    """С каким окном модель сейчас загружена — по `ollama ps`, а не по догадке."""
    for m in client.ps().models:
        if m.model == MODEL:
            return m.context_length
    return None


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
        f"   num_ctx=16384 (окно в памяти {loaded_context()}): прочитано {full} токенов, "
        f"ответ {answer!r}, маркер {'найден' if MARKER in answer else 'НЕ найден'}"
    )
    for ctx in (None, 2048):
        got, answer = context_probe(ctx)
        label = "по умолчанию" if ctx is None else f"num_ctx={ctx}"
        print(
            f"   {label} (окно в памяти {loaded_context()}): прочитано {got} токенов из {full}, "
            f"потеряно {full - got}; ответ {answer!r}, "
            f"маркер {'найден' if MARKER in answer else 'НЕ найден'}"
        )


# ===========================================================================
# 8. stop и repeat_penalty
# ===========================================================================
QA = (
    "Вопрос: Сколько будет 2+2?\n"
    "Ответ: 4.\n"
    "Вопрос: Какая столица Франции?\n"
    "Ответ: Париж.\n"
    "Вопрос: Сколько дней в неделе?\n"
    "Ответ:"
)


def numbered_ok(text: str) -> int:
    return len(re.findall(r"^\s*\d+\.\s+\S", text, flags=re.MULTILINE))


def script_mess(text: str) -> tuple[int, int]:
    """(слов, где кириллица смешана с латиницей; иероглифов) — следы того, что модель
    уходит от уже использованных токенов к токенам другой письменности."""
    words = re.findall(r"[\w-]+", text)
    mixed = sum(
        bool(re.search("[а-яё]", w, re.I)) and bool(re.search("[a-z]", w, re.I)) for w in words
    )
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return mixed, cjk


def code_ok(text: str) -> bool:
    block = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    try:
        ast.parse(block.group(1) if block else text)
    except SyntaxError:
        return False
    return True


def task8() -> None:
    print("\n8. stop и repeat_penalty")
    base = {"temperature": 0, "seed": SEED, "num_predict": 80, "repeat_penalty": 1.0}
    free = raw_text(QA, **base)
    print(f"   без stop: {free!r}")
    print(f"      модель задала следующий вопрос сама: {'Вопрос:' in free}")
    cut = raw_text(QA, **base, stop=["\nВопрос:"])
    print(f"   stop=['\\nВопрос:']: {cut!r}")

    loop_prompt = "Список покупок: молоко, хлеб, молоко, хлеб, молоко, хлеб,"
    for rp in (1.0, 1.5):
        looped = raw_text(loop_prompt, **{**base, "num_predict": 60, "repeat_penalty": rp})
        words = re.findall(r"\w+", looped.lower())
        top, count = Counter(words).most_common(1)[0] if words else ("", 0)
        print(
            f"   зацикливание, repeat_penalty={rp}: самое частое слово {top!r} — "
            f"{count} из {len(words)} слов; {looped[:80]!r}"
        )

    checks: list[tuple[str, str, Callable[[str], object]]] = [
        (
            "список",
            "Напиши нумерованный список из 15 крупнейших городов России, "
            "по одному в строке, в формате '1. Москва'. Без пояснений.",
            lambda t: f"пунктов {numbered_ok(t)}, смешанных слов и иероглифов {script_mess(t)}",
        ),
        (
            "код",
            "Напиши на Python функцию sum_even(numbers), которая возвращает сумму чётных "
            "чисел списка, с docstring и тремя assert-проверками. Только код.",
            lambda t: f"код разбирается Python: {code_ok(t)}",
        ),
    ]
    for name, prompt, check in checks:
        for rp in (1.0, 1.8):
            r = ask(prompt, temperature=0, seed=SEED, num_predict=400, repeat_penalty=rp)
            text = r.message.content or ""
            print(f"   {name}, repeat_penalty={rp}: {check(text)}")
            print("      " + text[:300].replace("\n", "\n      "))


# ===========================================================================
# 9. Устно, потом письменно
# ===========================================================================
# Почему temperature: 0 обнуляет смысл top_k. При нулевой температуре всегда берётся самый
#   вероятный токен. top_k и top_p только отрезают хвост, а самый вероятный токен в него не
#   попадает никогда. Задание 5: при T=0 четыре разных сочетания top_k и top_p дали один и тот
#   же ответ.
#
# Чем токен ответа дороже токена запроса и почему. Prefill читает все токены запроса за
#   один проход, параллельно. Decode делает отдельный проход на каждый новый токен и каждый
#   раз смотрит на всё накопленное. У меня токен ответа дороже в 9,4 раза: 50,52 мс против
#   5,38 мс (задание 2).
#
# Как узнать, что ответ оборван. По done_reason: 'length' — кончился num_predict, 'stop' —
#   модель закончила сама. По тексту этого не понять: обрыв приходится на любое место, без
#   многоточия. complete_chat поднимает IncompleteResponseError на всё, кроме 'stop'.
#
# Что происходит при переполнении окна и почему это опаснее ошибки. Ollama молча
#   отбрасывает часть запроса и отвечает так, будто её не было. Ни исключения, ни особого
#   done_reason. У меня из 12 082 токенов при окне по умолчанию дошло 2050, и модель вместо
#   кодового слова ответила 'пассажиров' (задание 7). Ошибку видно и её чинят; молчаливая
#   обрезка даёт уверенный ответ не по тем данным, и его никто не заметит.
#
# Зачем seed, если в проде его не ставят. Для измерений и отладки: с ним прогон
#   повторяется побайтово (задание 3), и разницу между двумя настройками можно приписать
#   настройке, а не случайности. Без seed «температура сделала лучше» — впечатление, с
#   seed — замер. И он позволяет воспроизвести у себя ответ, на который пожаловались.


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
