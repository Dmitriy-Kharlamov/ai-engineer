"""Ф4-05. Structured output и JSON mode — замеры.

Нужно: запущенный Ollama и `ollama pull qwen2.5:7b`. Из корня репозитория, по частям:
    python -u week-04/lesson_05.py 1 2 3   2>&1 | tee week-04/lesson_05_part1.txt
    python -u week-04/lesson_05.py 4 5     2>&1 | tee week-04/lesson_05_part2.txt
    python -u week-04/lesson_05.py 6 7     2>&1 | tee week-04/lesson_05_part3.txt
Другая модель: MODEL=llama3.2:3b python -u week-04/lesson_05.py 1

Промпт просит, грамматика запрещает. Три уровня — просьба, format='json', format=<схема> —
меряются на одной задаче извлечения: 20 разных текстов, по одному прогону на текст
(temperature 0,7, seed = номер текста). Верность значений считается отдельно от
валидности: схема гарантирует форму, а не правду. Прогнозы — в PREDICTIONS, до запуска.
"""

import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import httpx
import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragkit.structured import ArrayItemParser, parse_structured

MODEL = os.environ.get("MODEL", "qwen2.5:7b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=HOST)
SEED = 42
RUNS = 20

Format = dict[str, Any] | Literal["json"] | None

# Прогнозы записаны ДО первого запуска. Меняй на свои — но до запуска.
PREDICTIONS = {
    1: "просьба: JSON разберётся в ~85 % (остальное — обёртка ```json), валидно ~80 %, "
    "верно ~70 %; format='json': разберётся 100 %, валидно ~90 % (ключи могут уплыть), "
    "верно ~70 %; схема: 100 % и 100 %, но верно столько же ~70 % — грамматика значения "
    "не чинит; Фишер различит «разобралось» и «валидно», но не «верно»; eval_count у "
    "просьбы больше всех (обёртка и пояснения)",
    2: "грамматика укорачивает вывод (нет обёртки и пояснений) и немного замедляет токен — "
    "на 5–15 % больше мс на токен: проверка маски на каждом шаге",
    3: "logprobs — до грамматики, как и до температуры: у вынужденного токена будет малая "
    "вероятность, а в top_logprobs — запрещённые схемой токены, те же, что без схемы",
    4: "русские и английские имена полей — почти одинаково; description не изменит ничего: "
    "грамматика берёт из схемы форму, а описания модель не видит, если не положить схему "
    "в промпт; со схемой в промпте — плюс на падежах города",
    5: "без выхода: город выдуман в ~60 % (остальное — отписки вроде «не указан»), возраст "
    "выдуман почти всегда — у int нет отписки; с None или found=false — выдумок меньше "
    "10 %; Фишер различит обе пары",
    6: "грамматика скобки не дописывает: ответ оборван посреди строки или числа, не "
    "разбирается, скобки не закрыты; done_reason='length'",
    7: "из N префиксов валиден ровно один — последний; объект из массива готов через "
    "~1/10 общего времени: первая книга на ~10 % времени, а не на 100 %",
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
    print(f"ollama {version}; модель {MODEL}, digest {digest[:12]}")
    if d is not None:
        print(f"   {d.family}, {d.parameter_size}, {d.quantization_level}, {d.format}")


# Ко второму запуску задания 4: записано ПОСЛЕ первого, ДО второго.
PREDICTIONS_RERUN = {
    4: "с русскими ключами неверно поле «имя»: модель пишет имя без фамилии; странные "
    "description без схемы в промпте модель не видит — возраст в месяцах 0/20; со схемой в "
    "промпте — больше половины в месяцах, Фишер различит",
}


def prediction(n: int) -> None:
    print(f"   прогноз (до запуска): {PREDICTIONS[n]}")
    if n in PREDICTIONS_RERUN:
        print(f"   прогноз ко второму запуску: {PREDICTIONS_RERUN[n]}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k:>2}/{n} = {k / n:4.0%} [{lo:3.0%}–{hi:4.0%}]"


def fisher(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Точный тест Фишера, двусторонний."""
    (k1, n1), (k2, n2) = a, b
    total_k, total_n = k1 + k2, n1 + n2

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, total_k - x) / math.comb(total_n, total_k)

    observed = prob(k1)
    xs = range(max(0, total_k - n2), min(n1, total_k) + 1)
    return min(1.0, sum(prob(x) for x in xs if prob(x) <= observed * (1 + 1e-9)))


def verdict(p: float) -> str:
    return "различимы" if p < 0.05 else "неразличимы"


def ms(ns: int | None) -> float:
    return (ns or 0) / 1e6


def chat(messages: Sequence[Mapping[str, Any]], fmt: Format, **options: Any) -> ollama.ChatResponse:
    r: ollama.ChatResponse = client.chat(
        model=MODEL, messages=messages, options=options, format=fmt
    )
    return r


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def norm(s: object) -> str:
    return str(s).strip().casefold().replace("ё", "е")


# ===========================================================================
# Данные: 20 текстов о людях с известными ответами
# ===========================================================================
NAMES = [
    "Анна Петрова", "Иван Соколов", "Мария Кузнецова", "Дмитрий Орлов", "Елена Смирнова",
    "Сергей Волков", "Ольга Морозова", "Павел Лебедев", "Наталья Козлова", "Алексей Новиков",
]  # fmt: skip
CITIES = [  # именительный, предложный, родительный
    ("Казань", "Казани", "Казани"),
    ("Новосибирск", "Новосибирске", "Новосибирска"),
    ("Самара", "Самаре", "Самары"),
    ("Екатеринбург", "Екатеринбурге", "Екатеринбурга"),
    ("Пермь", "Перми", "Перми"),
    ("Томск", "Томске", "Томска"),
    ("Воронеж", "Воронеже", "Воронежа"),
    ("Тверь", "Твери", "Твери"),
    ("Калуга", "Калуге", "Калуги"),
    ("Иркутск", "Иркутске", "Иркутска"),
]
AGES = [32, 45, 27, 51, 38, 23, 64, 41, 29, 56, 33, 47, 22, 61, 35, 44, 26, 58, 39, 31]
UNITS = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
TENS = {2: "двадцать", 3: "тридцать", 4: "сорок", 5: "пятьдесят", 6: "шестьдесят"}


def years(n: int) -> str:
    last, last2 = n % 10, n % 100
    if last == 1 and last2 != 11:
        return "год"
    if 2 <= last <= 4 and not 12 <= last2 <= 14:
        return "года"
    return "лет"


def in_words(n: int) -> str:
    return f"{TENS[n // 10]} {UNITS[n % 10]}".strip() + f" {years(n)}"


TEMPLATES: list[Callable[[str, tuple[str, str, str], int], str]] = [
    lambda n, c, a: f"{n}, {a} {years(a)}, живёт в {c[1]} и работает инженером.",
    lambda n, c, a: (
        f"Знакомьтесь: {n}. Родом из {c[2]} и до сих пор там живёт. "
        f"В этом году исполнилось {in_words(a)}."
    ),
    lambda n, c, a: f"На конференцию приехал участник из {c[2]}: {n}, возраст — {a} {years(a)}.",
    lambda n, c, a: (
        f"В {c[1]} мы встретили человека по имени {n}. Недавно ему исполнилось {in_words(a)}."
    ),
    lambda n, c, a: f"Анкета. Город: {c[0]}. Имя: {n}. Полных лет: {a}.",
]


@dataclass(frozen=True)
class Case:
    text: str
    name: str
    city: str
    age: int


CASES = [
    Case(TEMPLATES[i % 5](NAMES[i % 10], CITIES[(i * 3) % 10], AGES[i]), NAMES[i % 10],
         CITIES[(i * 3) % 10][0], AGES[i])
    for i in range(RUNS)
]  # fmt: skip


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    city: str
    age: int = Field(ge=0, le=130)


EXTRACT = (
    "Извлеки из текста имя, город и возраст человека. Ответь в формате JSON "
    "с полями name, city, age.\n\nТекст: {text}"
)


def correct(obj: Mapping[str, Any], case: Case) -> bool:
    return (
        norm(obj.get("name")) == norm(case.name)
        and norm(obj.get("city")) == norm(case.city)
        and obj.get("age") == case.age
    )


# ===========================================================================
# 1. Три уровня
# ===========================================================================
LEVELS: list[tuple[str, Format]] = [
    ("просьба в промпте", None),
    ("format='json'", "json"),
    ("format=<схема>", Person.model_json_schema()),
]


@dataclass(frozen=True)
class Run:
    content: str
    parsed: bool
    fenced: bool
    valid: bool
    right: bool
    eval_count: int
    eval_ms: float


def judge(content: str, case: Case) -> Run:
    """Разобралось ли, прошло ли схему, верны ли значения — три разные вещи."""
    fenced = content.lstrip().startswith("```")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return Run(content, False, fenced, False, False, 0, 0.0)
    try:
        person = Person.model_validate(data)
    except ValidationError:
        right = isinstance(data, dict) and correct(data, case)
        return Run(content, True, fenced, False, right, 0, 0.0)
    return Run(content, True, fenced, True, correct(person.model_dump(), case), 0, 0.0)


@cache
def level_runs(level: int) -> tuple[Run, ...]:
    """20 текстов на уровень; общий прогон для заданий 1 и 2."""
    _, fmt = LEVELS[level]
    out = []
    for i, case in enumerate(CASES, start=1):
        r = chat(
            user(EXTRACT.format(text=case.text)), fmt, temperature=0.7, seed=i, num_predict=300
        )
        run = judge(r.message.content or "", case)
        out.append(
            Run(**{**run.__dict__, "eval_count": r.eval_count or 0, "eval_ms": ms(r.eval_duration)})
        )
    return tuple(out)


def task1() -> None:
    print(f"\n1. Три уровня: извлечь имя, город и возраст из {RUNS} текстов")
    prediction(1)
    counts: dict[str, dict[str, tuple[int, int]]] = {}
    print(f"   {'уровень':<20} {'разобралось как JSON':<22} {'прошло pydantic':<22} "
          f"{'значения верные':<22} eval_count")  # fmt: skip
    for level, (label, _) in enumerate(LEVELS):
        runs = level_runs(level)
        n = len(runs)
        c = {
            "разобралось": (sum(r.parsed for r in runs), n),
            "валидно": (sum(r.valid for r in runs), n),
            "верно": (sum(r.right for r in runs), n),
        }
        counts[label] = c
        tokens = statistics.fmean(r.eval_count for r in runs)
        print(f"   {label:<20} {share(*c['разобралось']):<22} {share(*c['валидно']):<22} "
              f"{share(*c['верно']):<22} {tokens:.1f}")  # fmt: skip
        fenced = sum(r.fenced for r in runs)
        if fenced:
            print(f"      в обёртке ```: {fenced} из {n}")
        bad = next((r for r in runs if not r.valid), None)
        if bad is not None:
            print(f"      пример невалидного: {bad.content[:110]!r}")
        wrong = next((r for r in runs if r.valid and not r.right), None)
        if wrong is not None:
            i = runs.index(wrong)
            print(f"      пример валидного, но неверного: {wrong.content[:90]!r}")
            print(f"         а правильно: {CASES[i].name}, {CASES[i].city}, {CASES[i].age}; "
                  f"текст: {CASES[i].text[:80]!r}")  # fmt: skip
    print("   тест Фишера:")
    labels = [label for label, _ in LEVELS]
    for a, b in ((0, 1), (1, 2), (0, 2)):
        parts = []
        for col in ("разобралось", "валидно", "верно"):
            p = fisher(counts[labels[a]][col], counts[labels[b]][col])
            parts.append(f"{col} p = {p:.3f} ({verdict(p)})")
        print(f"      {labels[a]} против {labels[b]}: " + "; ".join(parts))


# Результат (20 текстов, temperature 0,7, seed = номер текста):
#   уровень              разобралось как JSON   прошло pydantic   значения верные   eval_count
#   просьба в промпте      0/20 [0–16 %]          0/20              0/20              37,8
#   format='json'         20/20 [84–100 %]       19/20 [76–99 %]   19/20 [76–99 %]   33,9
#   format=<схема>        20/20 [84–100 %]       20/20 [84–100 %]  20/20 [84–100 %]  33,8
#   просьба: все 20 ответов в обёртке ```json; format='json': один невалидный — "age": "23 года"
#   Фишер: просьба против json и против схемы — p < 0,001 по всем трём столбцам;
#          json против схемы — p = 1,0 по всем трём
#
# Просьба в промпте не дала ни одного разбираемого JSON: все 20 ответов модель завернула в
#   ```json … ```. Внутри обёртки JSON правильный — пример: '```json\n{"name": "Анна Петрова",
#   "city": "Казань", "age": 32}\n```', — но json.loads строгий, и столбец «значения верные»
#   здесь меряет не знание модели, а то, дошли ли значения до кода. Снимать обёртку регуляркой
#   — уже ремонт вывода, а не разбор. В Ф4-02 было 95 % — там в промпте был пример ответа без
#   обёртки; здесь примера нет, и модель пишет так, как чаще всего видела в обучении.
# format='json' гарантировал синтаксис (20/20), но не поля: в одном ответе возраст пришёл
#   строкой «23 года». Схема убрала и это: 20/20 по всем столбцам. Но на 20 прогонах 19/20 и
#   20/20 неразличимы (p = 1,0) — что схема надёжнее json, этот замер не доказывает.
# Значения верные 19–20 из 20 — выше моего прогноза (~70 %): задача лёгкая, падежи города
#   («из Екатеринбурга») и возраст словами модель разобрала. Поэтому «форма ≠ правда» здесь не
#   видно — она видна в заданиях 5 и 7, где правильного ответа нет или модель его не знает.
# Прогноз: по «разобралось» ошибся сильно (ждал 85 %, вышло 0 %); по «верно» — тоже (ждал
#   ~70 %, вышло 95–100 %); eval_count у просьбы больше всех — сошлось: обёртка стоит 4 токена.


# ===========================================================================
# 2. Что стоит грамматика
# ===========================================================================
class Book(BaseModel):
    title: str
    author: str
    year: int


class Library(BaseModel):
    books: list[Book] = Field(min_length=10, max_length=10)


BOOKS_PROMPT = (
    "Перечисли 10 известных книг русской классики: название, автор, год первой публикации. "
    "Ответь в формате JSON: объект с полем books — массивом объектов с полями title, author, year."
)


def speed(label: str, runs: Sequence[tuple[int, float]]) -> tuple[float, float]:
    tokens = [t for t, _ in runs]
    per = [m / t for t, m in runs if t]
    print(
        f"      {label:<20} токенов {statistics.fmean(tokens):>6.1f} (от {min(tokens)} до "
        f"{max(tokens)}), мс на токен: медиана {statistics.median(per):.1f}, "
        f"{1000 / statistics.median(per):.1f} ток/с"
    )
    return statistics.fmean(tokens), statistics.median(per)


def task2() -> None:
    print("\n2. Что стоит грамматика: тот же запрос и seed со схемой и без")
    prediction(2)
    print(f"   короткий вывод — извлечение из задания 1 ({RUNS} текстов, одни и те же seed):")
    base: dict[str, tuple[float, float]] = {}
    for level, (label, _) in enumerate(LEVELS):
        runs = level_runs(level)
        base[label] = speed(label, [(r.eval_count, r.eval_ms) for r in runs])
    print("   длинный вывод — 10 книг, 5 прогонов на вариант (temperature 0,7, seed 1–5):")
    long: dict[str, tuple[float, float]] = {}
    formats: list[tuple[str, Format]] = [
        ("без format", None),
        ("format='json'", "json"),
        ("format=<схема>", Library.model_json_schema()),
    ]
    for label, fmt in formats:
        measured: list[tuple[int, float]] = []
        for seed in range(1, 6):
            r = chat(user(BOOKS_PROMPT), fmt, temperature=0.7, seed=seed, num_predict=1500)
            measured.append((r.eval_count or 0, ms(r.eval_duration)))
        long[label] = speed(label, measured)
    t0, m0 = long["без format"]
    t2, m2 = long["format=<схема>"]
    print(
        f"   схема против «без format» на длинном выводе: длина {t2 / t0 - 1:+.0%}, "
        f"мс на токен {m2 / m0 - 1:+.1%}"
    )


# Результат:
#   короткий вывод (извлечение, 20 прогонов): токенов 37,8 → 33,9 (json) → 33,8 (схема);
#     мс на токен 45,8 → 44,5 → 44,8 — разница в пределах шума
#   длинный вывод (10 книг, 5 прогонов): токенов 442,4 → 389,8 → 389,8;
#     мс на токен 45,6 → 47,5 (+4 %) → 53,0 (+16 %); ток/с 21,9 → 21,0 → 18,9
#   всего декода на длинном: без format 20,2 с, json 18,5 с, схема 20,7 с
#
# Грамматика меняет и то, и другое, — но по-разному в зависимости от схемы. Длину — всегда:
#   минус 11–12 % токенов, это обёртка ``` и текст вокруг JSON, которых под format больше нет.
#   Скорость на токен — только на сложной схеме: массив из 10 объектов с обязательными полями
#   стоил +16 % на токен, а плоская схема из трёх полей — ничего. format='json' (только
#   синтаксис) — посередине, +4 %. Проверка маски на каждом шаге дороже, чем больше в
#   грамматике состояний.
# Итог по времени на длинном выводе: схема короче на 12 % и медленнее на 16 % — в сумме +2 %
#   (20,7 с против 20,2 с). Грамматика почти бесплатна по общему времени, но не по скорости.
# format='json' и схема на длинном выводе дали одинаковое число токенов во всех пяти прогонах
#   (389,8, от 335 до 434): при тех же seed модель написала тот же текст, который подходит под
#   обе грамматики, — а время на токен разное. Это чистый замер цены схемы поверх синтаксиса.
# Прогноз сошёлся по направлению: короче — да; медленнее на 5–15 % — да, на длинной схеме
#   +16 %, на короткой — нет.

# ===========================================================================
# 3. logprobs под грамматикой: до или после вычёркивания?
# ===========================================================================
CAPITAL = user("Какой город — столица Франции? Ответь в формате JSON с полем capital.")
FORCED = {
    "type": "object",
    "properties": {"capital": {"type": "string", "enum": ["Лондон"]}},
    "required": ["capital"],
}
FREE = {
    "type": "object",
    "properties": {"capital": {"type": "string"}},
    "required": ["capital"],
}


@dataclass(frozen=True)
class Step:
    index: int
    token: str
    prob: float
    top: list[tuple[str, float]]


def value_start(r: ollama.ChatResponse) -> tuple[list[Step], int | None]:
    """Шаги генерации и номер шага, на котором начинается значение capital."""
    steps: list[Step] = []
    text = ""
    start: int | None = None
    for i, lp in enumerate(r.logprobs or []):
        # первый токен с буквой после "capital": — кавычка может прилипнуть к нему
        if (
            start is None
            and re.search(r'"capital"\s*:\s*"?\s*$', text)
            and re.search(r"[^\W\d_]", lp.token)
        ):
            start = i
        top = [(t.token, math.exp(t.logprob)) for t in lp.top_logprobs or []]
        steps.append(Step(i, lp.token, math.exp(lp.logprob), top))
        text += lp.token
    return steps, start


def show_step(label: str, step: Step, allowed: str | None) -> None:
    print(f"   {label}: шаг {step.index}, выбран {step.token!r} с вероятностью {step.prob:.4f}")
    for token, p in step.top:
        mark = ""
        if allowed is not None:
            core = token.lstrip(' "')
            ok = core == "" or allowed.startswith(core)
            mark = "  разрешён схемой" if ok else "  ЗАПРЕЩЁН схемой"
        print(f"      {token!r:<12} {p:.4f}{mark}")


def task3() -> None:
    print("\n3. logprobs под грамматикой: до или после вычёркивания запрещённых токенов?")
    prediction(3)
    print("   опыт: схема разрешает в capital ровно одно значение — «Лондон», а модель знает,")
    print("   что столица Франции — Париж. На первом токене значения грамматика оставляет")
    print("   только начала слова «Лондон». Если logprobs — после грамматики, в top_logprobs")
    print("   будут только они, и у выбранного вероятность около 1. Если до — там будут «Пар…»")
    print("   с большой вероятностью, у выбранного — малая, и top совпадёт с опытом без enum.")
    results = {}
    for label, fmt, allowed in (
        ("схема без enum", FREE, None),
        ("схема enum=[Лондон]", FORCED, "Лондон"),
    ):
        r = client.chat(
            model=MODEL,
            messages=CAPITAL,
            format=fmt,
            options={"temperature": 0, "seed": SEED, "num_predict": 40},
            logprobs=True,
            top_logprobs=10,
        )
        if not r.logprobs:
            print("   ollama не вернул logprobs — опыт на этой версии не провести")
            return
        steps, start = value_start(r)
        print(f"   {label}: ответ {r.message.content!r}")
        if start is None:
            print("      начало значения не найдено")
            continue
        results[label] = steps[start]
        show_step(label, steps[start], allowed)
    if len(results) < 2:
        return
    free, forced = results["схема без enum"], results["схема enum=[Лондон]"]
    illegal = sum(
        p for t, p in forced.top if (core := t.lstrip(' "')) and not "Лондон".startswith(core)
    )
    same_top = [t for t, _ in free.top] == [t for t, _ in forced.top]
    print(
        f"   под enum: суммарная вероятность запрещённых токенов в top — {illegal:.4f}; "
        f"выбранный токен — {forced.prob:.4f}; top совпадает с опытом без enum: {same_top}"
    )
    if illegal > 0.01:
        print("   вывод: logprobs — ДО грамматики; малая вероятность выбранного — признак")
        print("   «модель заставили»: она хотела другое, а грамматика не дала")
    else:
        print("   вывод: logprobs — ПОСЛЕ грамматики: запрещённых токенов в top нет")


# Результат (вопрос о столице Франции, temperature 0, top_logprobs 10):
#   схема без enum:      ответ {"capital": "Париж"}; на первом токене значения выбран 'П'
#                        с вероятностью 0,9993; дальше 'Paris' 0,0004, ' П' 0,0002, '巴黎'…
#   схема enum=[Лондон]: ответ {"capital": "Лондон"}; на том же шаге выбран 'Л' с вероятностью
#                        0,0000; в top_logprobs — те же десять токенов с теми же вероятностями:
#                        'П' 0,9993, 'Paris' 0,0004… — и все десять запрещены схемой
#   суммарная вероятность запрещённых токенов в top под enum — 1,0000; top совпадает с опытом
#   без enum: True
#
# Опыт различает гипотезы однозначно: logprobs ollama отдаёт ДО грамматики. Будь они после
#   вычёркивания, в top были бы только начала слова «Лондон», и у выбранного 'Л' вероятность
#   была бы около 1. А вышло наоборот: распределение то же, что без enum, грамматика выбрала
#   токен, которому модель давала меньше 0,00005, — его даже нет в первой десятке.
#   Прогноз сошёлся.
# Вместе с Ф4-01 (logprobs — до температуры) получается: logprobs показывают, что модель
#   хотела, а не что ей разрешили. Отсюда признак «модель заставили»: вероятность выбранного
#   токена много меньше вероятности лучшего. Под схемой это значит, что грамматика выбрала за
#   модель — enum без правильного варианта, обязательное поле, которого нет в тексте. Такие
#   значения стоит помечать как сомнительные (в Ф5 — не показывать как факт).
# Чего опыт не показывает: один вопрос и одна модель. Что «до грамматики» — свойство этой
#   версии ollama (0.40.14), а не закон; на другом сервере стоит проверить тем же опытом.


# ===========================================================================
# 4. Схема — это тоже промпт
# ===========================================================================
class PersonRu(BaseModel):
    """Русские имена полей: в схеме и в выводе — имя, город, возраст."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str = Field(alias="имя")
    city: str = Field(alias="город")
    age: int = Field(alias="возраст")


class PersonDescribed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Имя и фамилия человека в именительном падеже")
    city: str = Field(description="Название города в именительном падеже: «Казань», а не «Казани»")
    age: int = Field(description="Возраст в полных годах, числом")


class PersonOdd(BaseModel):
    """Описания, которые меняют ответ: если модель их видит, значения станут другими."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Только фамилия человека, без имени")
    city: str = Field(description="Название города в именительном падеже")
    age: int = Field(description="Возраст в МЕСЯЦАХ: число полных лет, умноженное на 12")


NEUTRAL = "Извлеки из текста сведения о человеке. Ответь в формате JSON.\n\nТекст: {text}"
WITH_SCHEMA = NEUTRAL + "\n\nСхема ответа: {schema}"
VARIANTS: list[tuple[str, type[BaseModel], str]] = [
    ("русские имена полей", PersonRu, NEUTRAL),
    ("английские имена", Person, NEUTRAL),
    ("английские + description", PersonDescribed, NEUTRAL),
    ("то же + схема в промпте (контроль)", PersonDescribed, WITH_SCHEMA),
]
# Второй запуск: в первом description и схема в промпте упёрлись в потолок (19/20 и 20/20) и
# ничего не различили. Различающий опыт — описания, которые требуют другого ответа: фамилия
# без имени и возраст в месяцах. Видит модель описание — значения изменятся; не видит — нет.
ODD_VARIANTS: list[tuple[str, str]] = [
    ("странные description", NEUTRAL),
    ("странные description + схема в промпте", WITH_SCHEMA),
]


def mismatches(data: Mapping[str, Any], case: Case) -> list[str]:
    """Какие поля не совпали — чтобы видеть, в чём ошибка, а не только что она есть."""
    out = []
    if norm(data.get("name")) != norm(case.name):
        out.append(f"имя {data.get('name')!r}≠{case.name!r}")
    if norm(data.get("city")) != norm(case.city):
        out.append(f"город {data.get('city')!r}≠{case.city!r}")
    if data.get("age") != case.age:
        out.append(f"возраст {data.get('age')!r}≠{case.age}")
    return out


def extract(model: type[BaseModel], prompt: str, i: int, case: Case) -> BaseModel | str:
    """Объект или, если не прошёл схему, сырой вывод."""
    schema = model.model_json_schema()
    text = prompt.format(text=case.text, schema=json.dumps(schema, ensure_ascii=False))
    r = chat(user(text), schema, temperature=0.7, seed=i, num_predict=300)
    content = r.message.content or ""
    try:
        return model.model_validate_json(content)
    except ValidationError:
        return content


def task4() -> None:
    print(f"\n4. Схема — это тоже промпт: {RUNS} текстов, промпт поля не называет")
    prediction(4)
    got: dict[str, tuple[int, int]] = {}
    for label, model, prompt in VARIANTS:
        right = valid = 0
        example = ""
        wrong: dict[str, int] = {}
        shown: list[str] = []
        for i, case in enumerate(CASES, start=1):
            obj = extract(model, prompt, i, case)
            if isinstance(obj, str):
                example = example or obj
                continue
            valid += 1
            data = obj.model_dump()
            right += correct(data, case)
            for m in mismatches(data, case):
                field_name = m.split()[0]
                wrong[field_name] = wrong.get(field_name, 0) + 1
                if len(shown) < 4:
                    shown.append(m)
        got[label] = (right, RUNS)
        print(f"   {label:<36} валидно {valid}/{RUNS}; значения верные {share(right, RUNS)}")
        if wrong:
            print(f"      неверных полей: {wrong}; примеры: {'; '.join(shown)}")
        if example:
            print(f"      невалидный: {example[:90]!r}")
    labels = [label for label, _, _ in VARIANTS]
    print("   тест Фишера по верным значениям:")
    for a, b in ((0, 1), (1, 2), (2, 3)):
        p = fisher(got[labels[a]], got[labels[b]])
        print(f"      {labels[a]} против {labels[b]}: p = {p:.3f} — {verdict(p)}")

    print("   различающий опыт: description требует фамилию без имени и возраст в месяцах")
    followed: dict[str, tuple[int, int]] = {}
    for label, prompt in ODD_VARIANTS:
        surname = months = 0
        examples: list[str] = []
        for i, case in enumerate(CASES, start=1):
            obj = extract(PersonOdd, prompt, i, case)
            if isinstance(obj, str):
                continue
            data = obj.model_dump()
            surname += norm(data["name"]) == norm(case.name.split()[-1])
            months += data["age"] == case.age * 12
            if len(examples) < 2:
                examples.append(f"{data['name']!r}, {data['age']} (было {case.name}, {case.age})")
        followed[label] = (months, RUNS)
        print(f"   {label:<40} фамилия без имени {share(surname, RUNS)}; "
              f"возраст в месяцах {share(months, RUNS)}")  # fmt: skip
        print(f"      примеры: {'; '.join(examples)}")
    plain, in_prompt = (followed[label] for label, _ in ODD_VARIANTS)
    p = fisher(plain, in_prompt)
    print(f"   возраст в месяцах, без схемы в промпте против со схемой: p = {p:.4f} — {verdict(p)}")


# Результат первого запуска (20 текстов, промпт поля не называет; второй запуск — с
# разбором по полям и различающим опытом про description):
#   русские имена полей (имя, город, возраст)   валидно 20/20; верно  7/20 = 35 % [18–57 %]
#   английские имена (name, city, age)           валидно 20/20; верно 19/20 = 95 % [76–99 %]
#   английские + description на русском          валидно 20/20; верно 19/20 = 95 %
#   то же + схема в промпте (контроль)            валидно 20/20; верно 20/20 = 100 %
#   Фишер: русские против английских — p < 0,001; английские против description — p = 1,0;
#   description против схемы в промпте — p = 1,0
#
# Имена полей — промпт, и сильный: русские ключи уронили верность с 95 % до 35 %. Модель сама
#   пишет ключ (грамматика его навязывает) и дальше продолжает уже после него — смысл ключа
#   становится частью вопроса. Город с русскими ключами совпал во всех 20 ответах (строки
#   «город не тот» нет), значит ошибки в имени или возрасте. Моя гипотеза: «имя» по-русски —
#   имя без фамилии, и модель пишет «Анна» вместо «Анна Петрова». Второй запуск печатает
#   неверные поля — проверит.
# description и схема в промпте упёрлись в потолок: 19/20 и 20/20, и на этом опыте не видно,
#   читает ли модель описания. Второй запуск — описания, которые требуют другого ответа
#   (фамилия без имени, возраст в месяцах): увидит модель их — значения изменятся.

# ===========================================================================
# 5. Выхода нет — значит выдумает
# ===========================================================================
PROFESSIONS = ["врач", "программист", "учитель", "повар", "архитектор"]
HOBBIES = ["шахматы", "бег", "фотографию", "рыбалку", "акварель"]
NO_INFO = [
    (lambda n, p, h: f"{n} работает, как говорят, хорошим специалистом: {p}. Любит {h}."),
    (lambda n, p, h: f"Коллеги ценят {n} за аккуратность. По профессии — {p}, по выходным — {h}."),
    (
        lambda n, p, h: (
            f"{n} — {p}. Мечтает когда-нибудь съездить в Рим и увлекается, кроме того, "
            f"занятиями: {h}."
        )
    ),  # fmt: skip
    (
        lambda n, p, h: (
            f"В клубе любителей хобби «{h}» появился новый участник: {n}, {p}. "
            f"Встречи клуба проходят онлайн."
        )
    ),  # fmt: skip
]
NO_INFO_CASES = [
    NO_INFO[i % 4](NAMES[i % 10], PROFESSIONS[i % 5], HOBBIES[(i * 2) % 5]) for i in range(RUNS)
]


class PersonNoExit(BaseModel):
    name: str
    city: str
    age: int


class PersonOptional(BaseModel):
    name: str
    city: str | None
    age: int | None


class Found(BaseModel):
    found: Literal[True]
    value: str


class NotFound(BaseModel):
    found: Literal[False]


class PersonFlags(BaseModel):
    name: str
    city: Found | NotFound
    age: int | NotFound


EXCUSE = re.compile(
    r"^\s*$|не\s*(указан|известн|сказан|упомян)|неизвест|unknown|n/?a|^\W+$", re.IGNORECASE
)


def classify_city(value: Any) -> str:
    if value is None or (isinstance(value, dict) and value.get("found") is False):
        return "честно: нет"
    if isinstance(value, dict):
        value = value.get("value")
    return "отписка" if EXCUSE.search(str(value)) else "выдумка"


def classify_age(value: Any) -> str:
    if value is None or (isinstance(value, dict) and value.get("found") is False):
        return "честно: нет"
    if isinstance(value, int) and value <= 0:
        return "отписка"
    return "выдумка"


def task5() -> None:
    print(f"\n5. Выхода нет — значит выдумает: {RUNS} текстов без города и возраста")
    prediction(5)
    print(f"   пример текста: {NO_INFO_CASES[2]!r}")
    variants: list[tuple[str, type[BaseModel]]] = [
        ("без выхода (str, int)", PersonNoExit),
        ("None (str | None, int | None)", PersonOptional),
        ('вариант {"found": false}', PersonFlags),
    ]
    invented: dict[str, tuple[int, int]] = {}
    prompt = (
        "Извлеки из текста имя, город и возраст человека. Ответь в формате JSON.\n\nТекст: {text}"
    )
    for label, model in variants:
        schema = model.model_json_schema()
        city_kinds: dict[str, int] = {}
        age_kinds: dict[str, int] = {}
        examples: list[str] = []
        any_invented = 0
        for i, text in enumerate(NO_INFO_CASES, start=1):
            r = chat(
                user(prompt.format(text=text)), schema, temperature=0.7, seed=i, num_predict=200
            )
            content = r.message.content or ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                city_kinds["не JSON"] = city_kinds.get("не JSON", 0) + 1
                continue
            ck, ak = classify_city(data.get("city")), classify_age(data.get("age"))
            city_kinds[ck] = city_kinds.get(ck, 0) + 1
            age_kinds[ak] = age_kinds.get(ak, 0) + 1
            if "выдумка" in (ck, ak):
                any_invented += 1
                if len(examples) < 3:
                    examples.append(f"{data.get('city')!r}, {data.get('age')!r}")
        invented[label] = (any_invented, RUNS)
        print(f"   {label}:")
        print(f"      город:   {city_kinds}")
        print(f"      возраст: {age_kinds}")
        print(f"      хоть что-то выдумано: {share(any_invented, RUNS)}")
        if examples:
            print(f"      выдуманное (город, возраст): {'; '.join(examples)}")
    labels = [label for label, _ in variants]
    print("   тест Фишера по доле ответов с выдумкой:")
    for b in (1, 2):
        p = fisher(invented[labels[0]], invented[labels[b]])
        print(f"      {labels[0]} против {labels[b]}: p = {p:.4f} — {verdict(p)}")
    p = fisher(invented[labels[1]], invented[labels[2]])
    print(f"      {labels[1]} против {labels[2]}: p = {p:.3f} — {verdict(p)}")


# Результат (20 текстов без города и возраста; промпт просит имя, город и возраст):
#   схема                         город                                возраст
#   без выхода (str, int)         отписка 18, выдумка 2 («Рим»)        отписка 20 (0 или −1)
#   None (str | None, int | None)  null 13, отписка 5, выдумка 2 («Рим») null 20
#   вариант {"found": false}       found=false 20                        found=false 20
#   хоть одна выдумка: 2/20, 2/20, 0/20 — p = 1,0 и 0,49, неразличимы
#   город сказан честно («нет»): 0/20, 13/20, 20/20 — p < 0,001 (без выхода против None),
#   p = 0,008 (None против found), p < 0,001 (без выхода против found)
#
# Прогноз ошибся в главном: без выхода модель почти не выдумывала — 2 из 20, — а писала
#   отписки: город «не указан», возраст 0 или −1. Но отписка — та же ложь, только хуже
#   спрятанная: она проходит схему и валидацию, и код примет «Не указан» за название города,
#   а 0 — за возраст. Схема без выхода заставила модель изобретать свой способ сказать «нет»
#   внутри значения — и каждый такой способ надо потом угадывать в коде (здесь пришлось
#   писать регулярку для «не указан»).
# None помог наполовину: возраст — null во всех 20, а город — null только в 13: в пяти модель
#   и при разрешённом null написала строку «не указан», в двух — «Рим» («мечтает съездить в
#   Рим»). Явный флаг {"found": false} дал 20 из 20 честных «нет» по обоим полям — и ни одного
#   «Рима»: чтобы вписать город, модели пришлось бы сначала написать "found": true.
# Самый важный вывод урока подтвердился, но точнее, чем в уроке: без выхода модель не столько
#   выдумывает, сколько маскирует «нет» под значение. Выход в схеме нужен явный и отдельный
#   от значения — флаг, а не только допустимый null.
# Чего замер не показывает: тексты одного вида, одна модель. Доля «Рима» — 2 из 20 —
#   держится на одной формулировке («мечтает съездить в Рим»), и на 20 прогонах 2/20 и 0/20
#   неразличимы.

# ===========================================================================
# 6. Обрыв под грамматикой
# ===========================================================================
EXAMPLE_BOOK = {"title": "Война и мир", "author": "Лев Толстой", "year": 1869}


def prompt_tokens(text: str) -> int:
    r = chat(user(text), None, temperature=0, num_predict=1)
    return r.prompt_eval_count or 0


def brackets(text: str) -> str:
    opened = text.count("{") + text.count("[")
    closed = text.count("}") + text.count("]")
    return f"открыто {{ и [: {opened}, закрыто: {closed}"


def task6() -> None:
    print("\n6. Обрыв под грамматикой: 10 книг, схема с вложенным массивом")
    prediction(6)
    schema = Library.model_json_schema()
    full = chat(user(BOOKS_PROMPT), schema, temperature=0, seed=SEED, num_predict=2000)
    total = full.eval_count or 0
    print(f"   без лимита: {total} токенов, done_reason={full.done_reason!r}")
    limit = max(10, int(total * 0.4))
    cut = chat(user(BOOKS_PROMPT), schema, temperature=0, seed=SEED, num_predict=limit)
    raw = cut.message.content or ""
    print(f"   num_predict={limit}: done_reason={cut.done_reason!r}, eval_count={cut.eval_count}")
    print(f"   пришло ровно это ({len(raw)} символов):")
    print(f"      {raw!r}")
    try:
        json.loads(raw)
        print("   json.loads: разобралось")
    except json.JSONDecodeError as error:
        print(f"   json.loads: не разобралось — {error.msg}, позиция {error.pos}")
    print(f"   скобки: {brackets(raw)}; кончается на {raw[-12:]!r}")
    try:
        parse_structured(raw, cut.done_reason, Library)
    except Exception as error:
        print(f"   ragkit.structured: {type(error).__name__}: {error}")

    print("   сколько нужно num_predict — из схемы и примера:")
    example = json.dumps(EXAMPLE_BOOK, ensure_ascii=False)
    base = prompt_tokens("Пример:")
    per_item = prompt_tokens(f"Пример: {example}") - base
    wrapper = prompt_tokens('Пример: {"books": []}') - base
    needed = wrapper + 10 * (per_item + 2)  # +2 на запятую и пробел между объектами
    margin = math.ceil(needed * 1.5)
    print(
        f"      пример объекта {example!r}: {per_item} токенов; обёртка {wrapper}; "
        f"10 объектов ≈ {needed} токенов; с запасом ×1,5 — {margin}"
    )
    print(f"      (факт без лимита при temperature 0: {total}; оценка ошиблась на "
          f"{(needed - total) / max(total, 1):+.0%})")  # fmt: skip
    for seed in range(1, 6):
        r = chat(user(BOOKS_PROMPT), schema, temperature=0.7, seed=seed, num_predict=margin)
        try:
            lib = parse_structured(r.message.content or "", r.done_reason, Library)
            status = f"разобралось, книг {len(lib.books)}"
        except Exception as error:
            status = f"{type(error).__name__}"
        print(
            f"      seed {seed}: done_reason={r.done_reason!r}, eval_count {r.eval_count} "
            f"из {margin} — {status}"
        )


# Результат (10 книг, схема Library: books — ровно 10 объектов):
#   без лимита: 435 токенов, done_reason='stop'
#   num_predict=174 (40 %): done_reason='length', пришло 446 символов, кончается на
#     '"year": 1831' — посреди четвёртого объекта; json.loads: «Expecting ',' delimiter»;
#     скобок открыто 6, закрыто 3; ragkit.structured: TruncatedOutputError
#   оценка из схемы и примера: пример объекта — 29 токенов, обёртка 5, 10 объектов ≈ 315;
#     с запасом ×1,5 — 473; факт без лимита — 435, оценка ошиблась на −28 %
#   с лимитом 473, 5 прогонов: done_reason='stop' во всех, книг 10; eval_count 335, 434, 385,
#     420, 375 из 473
#
# Грамматика скобки не дописывает: прогноз сошёлся. Обрыв пришёл ровно там, где кончился
#   лимит, — после числа, без закрывающих скобок, и под схемой такой вывод так же не
#   разбирается, как без неё. Лимит у грамматики приоритетнее: она запрещает лишнее, но не
#   заставляет закончить.
# Оценка по примеру занизила нужное на 28 %: настоящий вывод под format — с отступами и
#   переводами строк ('\n      "author"'), и с полными именами («Лев Николаевич Толстой»
#   против «Лев Толстой» в примере). Запас ×1,5 спас, но впритык: в прогоне с seed 2 ушло 434
#   токена из 473 — 92 % лимита. Считать num_predict надо по настоящему выводу (максимум по
#   нескольким прогонам), а не по аккуратному примеру, — и с запасом от максимума.
# Попутно — «форма не правда» вживую: все объекты прошли схему, а в них «Алексandr Сергеевич
#   Пушкин» (латиница посреди имени) и «Колыбельная для младенца» Пушкина 1831 года — такой
#   книги нет.


# ===========================================================================
# 7. Стриминг структуры
# ===========================================================================
def task7() -> None:
    print("\n7. Стриминг структуры: 10 книг со схемой, поток")
    prediction(7)
    schema = Library.model_json_schema()
    started = time.perf_counter()
    text = ""
    prefixes = valid = 0
    first_valid: float | None = None
    parser = ArrayItemParser()
    ready: list[tuple[float, dict[str, Any]]] = []
    first_chunk: float | None = None
    stream = client.chat(
        model=MODEL,
        messages=user(BOOKS_PROMPT),
        format=schema,
        options={"temperature": 0, "seed": SEED, "num_predict": 2000},
        stream=True,
    )
    for chunk in stream:
        delta = chunk.message.content or ""
        now = time.perf_counter() - started
        if delta and first_chunk is None:
            first_chunk = now
        for item in parser.feed(delta):
            ready.append((now, item))
        if not delta:
            continue
        text += delta
        prefixes += 1
        try:
            json.loads(text)
            valid += 1
            first_valid = first_valid if first_valid is not None else now
        except json.JSONDecodeError:
            pass
    total = time.perf_counter() - started
    print(f"   префиксов (после каждого куска) {prefixes}, из них валидный JSON — {valid}")
    print(f"   первый валидный — на {first_valid or 0:.2f} с из {total:.2f} с "
          f"(последний кусок с текстом)")  # fmt: skip
    print(f"   первый кусок пришёл на {first_chunk or 0:.2f} с — но показать из него нечего")
    print("   приём из урока — объект из массива, как только он закрылся:")
    for i, (at, item) in enumerate(ready, start=1):
        try:
            book = Book.model_validate(item)
            what = f"{book.title} — {book.author}, {book.year}"
        except ValidationError:
            what = f"не прошёл Book: {str(item)[:60]}"
        print(f"      {i:>2}. на {at:5.2f} с ({at / total:4.0%} времени): {what}")
    if ready:
        print(
            f"   TTFT первого готового объекта {ready[0][0]:.2f} с против {total:.2f} с всего — "
            f"в {total / ready[0][0]:.1f} раза раньше; без приёма показать что-то можно "
            f"только через {first_valid or total:.2f} с"
        )


# Результат (10 книг со схемой, поток, temperature 0):
#   префиксов (после каждого куска с текстом) 434, валидный JSON — 1, последний: на 20,43 с из
#   20,48 с; первый кусок пришёл на 0,08 с, но показать из него нечего
#   объекты из массива по мере закрытия: 1-й на 2,20 с (11 % времени), дальше примерно каждые
#   2 с: 4,13 / 6,08 / 8,23 / 10,49 / 12,27 / 14,38 / 16,37 / 18,26 / 20,28 с
#   TTFT первого готового объекта 2,20 с против 20,48 с всего — в 9,3 раза раньше
#
# Префикс валидного JSON невалиден: из 434 префиксов разобрался один — весь ответ. Значит,
#   стриминг структуры «как есть» ничего не покупает: первое, что можно показать, появляется
#   через 20,43 с, на 0,05 с раньше конца. Прогноз сошёлся.
# Массив объектов возвращает выигрыш: первая книга готова через 2,20 с — первый объект
#   закрылся на 11 % времени, и дальше по одной каждые 2 с. Это тот же приём, что у продуктов,
#   где список «наполняется». TTFT здесь — не первый токен (0,08 с), а первый законченный
#   объект: раньше показывать нечего, даже со стримингом.
# Цена приёма: каждый объект надо проверять отдельно (Book.model_validate), а законченность
#   всего ответа — по-прежнему в конце. Показанные книги — до того, как известно, что весь
#   ответ не оборван (Ф4-04): если поток оборвётся на седьмой книге, шесть уже на экране.
#   И «форма не правда» и здесь: «Колыбельная для младенца» и «Белкинovo собрание рассказов»
#   прошли схему.

# ===========================================================================
# 9. Письменно
# ===========================================================================
# Чем грамматика отличается от просьбы в промпте. Просьба — текст, который делает JSON
#   вероятнее; модель может его не выполнить (обёртка ```json, пояснение, лишнее поле).
#   Грамматика действует не на вероятности, а на выбор: сервер на каждом шаге вычёркивает
#   токены, после которых вывод перестал бы подходить под схему. Невалидный JSON становится
#   невозможным, а не маловероятным. Но только синтаксис и форма: ключи, типы, enum,
#   required — то, что выражается в JSON Schema.
#
# Почему схема гарантирует форму и не гарантирует правду. Грамматика разрешает любое значение
#   подходящего типа: {"city": "Париж"} и {"city": "Лондон"} для неё одинаковы. Что в поле
#   окажется, решает модель — с той же вероятностью ошибки, что и без схемы. Значения
#   проверяются отдельно: валидаторами (согласованность полей — model_validator) и
#   сверкой с источником. Замер (задание 7): «Колыбельная для младенца» Пушкина 1831 года и
#   «Алексandr Сергеевич Пушкин» прошли схему Book без единой ошибки валидации.
#
# Почему отсутствие «не нашёл» в схеме — ошибка проектирования. Схема, где у каждого поля
#   есть только значение, не даёт модели сказать «этого нет в тексте». Грамматика не
#   пропустит ни null, ни пропуск, и модель заполнит поле тем, что есть: выдумкой или
#   отпиской вроде «не указан», которую потом примут за значение. Это не модель врёт —
#   это схема её заставила. Явный выход — None или вариант {"found": false} — часть
#   контракта, как Optional в сигнатуре функции. Замер (задание 5): без выхода — «не указан»
#   в 18 городах из 20 и возраст 0 или −1 во всех 20, и всё это прошло валидацию; с явным
#   {"found": false} — 20 из 20 честных «нет» по обоим полям.
#
# Почему битый JSON нельзя починять. Дописать скобку — значит решить за модель, чем кончаются
#   данные: был ли это последний элемент, полное ли название, то ли число. Это догадка, и
#   после разбора её не отличить от настоящего значения. А догадка, попавшая в контекст или
#   базу, становится фактом (Ф4-03, задание 5: 20 из 20). done_reason != "stop" на
#   структурированном выводе — жёсткий отказ: нужен запас num_predict, а не ремонт. Замер
#   (задание 6): под грамматикой обрыв пришёл после '"year": 1831' — посреди четвёртого
#   объекта, с тремя незакрытыми скобками; грамматика закончить не заставляет.
#
# Почему структурированный вывод и стриминг плохо совместимы. Префикс валидного JSON сам
#   невалиден: показать пользователю нечего, пока не закрылась последняя скобка, и выигрыш
#   стриминга — время до первого слова — пропадает. Что делают: не стримят; стримят, но
#   показывают в конце (смысла мало); или проектируют схему как массив объектов и показывают
#   каждый объект, как только он закрылся (ragkit.structured.ArrayItemParser). Замер
#   (задание 7): из 434 префиксов валиден один, последний; объект из массива — через 2,20 с
#   вместо 20,48 с.
#
# Почему ValidationError на выводе модели опасна в Ф5. В тексте ValidationError — input_value,
#   то есть вывод модели. В RAG этот вывод строится по найденным документам, а инструкция из
#   документа выполняется 20 раз из 20 (Ф4-02, задание 10). Значит, чужой текст из документа
#   может уехать в лог, в ответ API, в сообщение пользователю — внутри сообщения об ошибке.
#   Лечение — белый список, как в Ф3-06: тип ошибки, путь и известные схеме имена полей, без
#   input и без исходного исключения в цепочке (ragkit.structured.safe_errors).


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
