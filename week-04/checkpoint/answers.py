"""Ф4-08, присест 3: итог свободным текстом против итога-объекта с проверкой источников.

    python -u week-04/checkpoint/answers.py 2>&1 | tee week-04/checkpoint/answers.txt
    python -u week-04/checkpoint/answers.py --seeds 1      # быстрая проверка, ~5 мин
    python -u week-04/checkpoint/answers.py --dump week-05.5/data/answers60.jsonl   # Ф5.5-01

--dump (Ф5.5-01): каждый ответ целиком — строкой JSON в файл, для ручной разметки и судьи
(в выводе выше от ответа только хвост в 70 знаков, а «факт» — регулярка, не человек).

10 вопросов по sample_docs (9 с ответом в одном документе, 1 — без ответа в папке) × seed 1…3
× два режима, temperature 0.7, свежая сессия на каждый вопрос:
  текст  — как в присесте 2: итог — свободный ответ модели потоком (--no-structured);
  объект — итог DocAnswer {answer, sources, not_found} через ragkit.structured, код проверяет
           источники до показа (по умолчанию).

Что считается для каждого ответа:
  факт     — в показанном ответе есть ключевой факт документа (регулярка; для вопроса без
             ответа — признание, что ответа нет);
  прочитан — нужный документ прочитан в ходе (журнал исполнения);
  ссылка   — нужный документ назван источником (текст: имя файла в ответе; объект: sources);
  плохая   — назван документ, которого нет в папке или который в ходе не читался;
  поймано  — плохая ссылка, о которой код сказал строкой перед ответом (только у объекта);
  время    — от вопроса до конца ответа, с.

Прогноз (до запуска): факт — одинаково, 25–29 из 30 в обоих режимах (схема гарантирует форму,
а не правду: в Ф4-05 верно ~70 % и с format, и без); плохие ссылки — 1–4 из 30 в обоих (в
присесте 2 «…undry.md» — 4 из 40), у объекта все пойманы кодом, у текста — ни одна; вопрос
без ответа — not_found=true в 2–3 из 3; время — объект медленнее на 3–6 с на ответ (ещё одно
обращение с ответом заново), то есть на 30–60 %.

# Результат (коммит d1f4ccc90c08, answers.txt; 10 вопросов × 3 seed, temperature 0.7)
#   режим   факт         нужный прочитан  нужный назван        плохая ссылка  без ответа: признал  время, медиана
#   текст   28/30        26/27            5/27 [8–37 %]        1/30           1/3                  11,0 с
#   объект  28/30        25/27            25/27 [77–98 %]      0/30           3/3                  16,8 с
# Факт — одинаково (p = 1,000), как и предсказано: схема гарантирует форму, а не правду.
# Главное отличие — источник назван: 25 из 27 против 5 из 27 (p < 0,001). В свободном тексте
#   модель обычно не называет файл, хотя системный промпт просит; в объекте поле sources
#   заполняется почти всегда.
# Вопрос без ответа (доставка дронами): объект — not_found=true 3 из 3, текст признал 1 из 3,
#   в двух других перечислил другие способы доставки и не сказал «нет».
# Плохие ссылки редки в обоих режимах: 1 из 30 у текста (delivery.md без чтения) и 0 у
#   объекта — меньше прогноза 1–4; ловить коду здесь было нечего (проверку источников
#   показывают строки 14 и 3 таблицы отказов).
# Промахи объекта — честные: в3 seed 1 прочитан delivery.md вместо gift_cards.md, ответ «не
#   указано»; в5 seed 2 — «ответа в документах нет». Промахи текста — ответ мимо вопроса.
# Цена: +5,9 с на ответ (+54 %; прогноз 3–6 с, 30–60 %) — ещё одно обращение, ответ пишется
#   заново. Медианы: 11,0 и 16,8 с.
# Слитные слова в ответах («290рублей», «накарту») — проверить в answers.txt: такие же слияния
#   есть в строках, которые код печатает с пробелом («окне3584»), то есть это, вероятно,
#   перенос строки терминала при копировании, а не модель.
"""

import argparse
import asyncio
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import ollama

from docqa.documents import DocumentStore
from docqa.session import Session, Settings
from ragkit.exceptions import RAGError
from ragkit.tools import ToolResult

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docqa" / "sample_docs"
NOT_FOUND = (
    r"нет (информации|сведений|данных|ответа)|не (нашл|найден|указан|упомина|сказано)|отсутств"
)
# (вопрос, документ с ответом или None, регулярка факта)
QUESTIONS: list[tuple[str, str | None, str]] = [
    ("Сколько стоит курьерская доставка по Казани?", "delivery.md", r"290"),
    ("Можно ли вернуть электронную книгу?", "returns.md", r"не подлеж|нельзя|не возвра|невозможно"),
    ("Можно ли оплатить доставку подарочным сертификатом?", "gift_cards.md", r"нельзя|невозможно|не (можно|принима|распростран)"),
    ("До скольки работает магазин на Пушкина в субботу?", "hours.md", r"20[:.]00|до 20\b"),
    ("Сколько дней заказ хранится в пункте выдачи СДЭК?", "delivery.md", r"\b7\b|семь"),
    ("Сколько процентов бонусами получает участник «Друг Полки»?", "loyalty.md", r"\b5\s*%|пять процент"),
    ("В какой день недели собирается книжный клуб?", "events.md", r"четверг"),
    ("Сколько действует подарочный сертификат?", "gift_cards.md", r"\bгод|12 месяц"),
    ("За сколько дней приходят деньги за возврат на карту?", "returns.md", r"\b10\b|десят"),
    ("Есть ли у вас доставка дронами?", None, NOT_FOUND),
]  # fmt: skip
MODES = {"текст": False, "объект": True}
CITED = re.compile(r"[\w-]+\.(?:md|txt)\b")


class Quiet:
    def __init__(self) -> None:
        self.notices: list[str] = []

    def delta(self, text: str) -> None:
        pass

    def tool(self, result: ToolResult) -> None:
        pass

    def debug(self, line: str) -> None:
        pass

    def notice(self, text: str) -> None:
        self.notices.append(text)


@dataclass
class Tally:
    runs: int = 0
    fact: int = 0
    read: int = 0
    cited: int = 0
    bad: int = 0
    caught: int = 0
    answerable: int = 0  # ответов на вопросы, у которых ответ в папке есть
    not_found_flag: int = 0  # вопрос без ответа: флаг (объект) или признание (текст)
    unanswerable: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k:>2}/{n:<2} [{lo:3.0%}–{hi:4.0%}]"


def fisher(a: tuple[int, int], b: tuple[int, int]) -> float:
    (k1, n1), (k2, n2) = a, b
    total_k, total_n = k1 + k2, n1 + n2
    if total_n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, total_k - x) / math.comb(total_n, total_k)

    observed = prob(k1)
    xs = range(max(0, total_k - n2), min(n1, total_k) + 1)
    return min(1.0, sum(prob(x) for x in xs if prob(x) <= observed * (1 + 1e-9)))


def commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return head + (" + незакоммиченные правки" if dirty else "")


async def one(
    mode: str, index: int, seed: int, tally: Tally, host: str | None, dump: Path | None
) -> None:
    question, doc, fact_re = QUESTIONS[index]
    store = DocumentStore(DOCS)
    client = ollama.AsyncClient(host=host)
    out = Quiet()
    session = Session(
        store,
        client,
        out,
        Settings(temperature=0.7, seed=seed, structured=MODES[mode]),
        structured=client,
    )
    tally.runs += 1
    started = time.monotonic()
    try:
        report = await session.ask(question)
    except (RAGError, ollama.ResponseError, httpx.HTTPError) as error:
        tally.errors.append(f"в{index + 1} seed {seed}: {type(error).__name__}")
        print(f"      в{index + 1:<2} seed {seed}: отказ {type(error).__name__}: {str(error)[:90]}")
        return
    tally.seconds.append(time.monotonic() - started)
    shown = report.answer
    names = store.names()
    if report.structured is not None:
        cited = list(dict.fromkeys(report.structured.sources))
    else:
        cited = list(dict.fromkeys(CITED.findall(shown)))
    bad = [c for c in cited if c not in names or c not in report.documents_read]
    fact = bool(re.search(fact_re, shown, re.IGNORECASE))
    caught = bool(bad) and any(i.kind in ("nonexistent", "unread") for i in report.issues)
    tally.fact += fact
    tally.bad += bool(bad)
    tally.caught += caught
    if doc is None:
        tally.unanswerable += 1
        flag = report.structured.not_found if report.structured is not None else fact
        tally.not_found_flag += flag
        mark = f"not_found={'да' if flag else 'НЕТ'}"
    else:
        tally.answerable += 1
        tally.read += doc in report.documents_read
        tally.cited += doc in cited
        mark = f"{'прочитал' if doc in report.documents_read else 'НЕ читал'} {doc}"
    if dump is not None:
        record = {
            "id": f"{mode}-в{index + 1}-s{seed}",
            "mode": mode,
            "seed": seed,
            "q": index + 1,
            "question": question,
            "doc": doc,
            "answer": shown,
            "documents_read": report.documents_read,
            "sources": cited,
            "regex_fact": fact,
        }
        with dump.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"      в{index + 1:<2} seed {seed}: факт {'да ' if fact else 'НЕТ'}; {mark}; "
        f"ссылки {cited or '—'}"
        + (f"; ПЛОХИЕ {bad}{' (код сказал)' if caught else ''}" if bad else "")
        + f"; {tally.seconds[-1]:.1f} с; …{shown.strip()[-70:]!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--host", default=None)
    parser.add_argument("--dump", type=Path, default=None, help="JSONL: ответы целиком")
    args = parser.parse_args()
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text("", encoding="utf-8")
    print(f"коммит {commit()}; модель {Settings().model}; temperature 0.7, seed 1…{args.seeds}")
    print(f"   прогноз (до запуска): {(__doc__ or '').split('Прогноз (до запуска): ')[1].strip()}")
    tallies = {mode: Tally() for mode in MODES}
    for mode in MODES:
        print(f"\n[{mode}]")
        for index in range(len(QUESTIONS)):
            for seed in range(1, args.seeds + 1):
                asyncio.run(one(mode, index, seed, tallies[mode], args.host, args.dump))
    print("\nитог: [95 % Wilson]")
    print(
        "   режим   факт                   нужный прочитан        нужный назван          "
        "плохая ссылка          из них поймано  без ответа: признал  время, с (медиана)  отказы"
    )
    for mode, t in tallies.items():
        answered = t.runs - len(t.errors)
        print(
            f"   {mode:<7} {share(t.fact, answered):<22} "
            f"{share(t.read, t.answerable):<22} "
            f"{share(t.cited, t.answerable):<22} "
            f"{share(t.bad, answered):<22} {t.caught:>2}/{t.bad:<12} "
            f"{t.not_found_flag:>2}/{t.unanswerable:<17} "
            f"{statistics.median(t.seconds) if t.seconds else 0:>6.1f}             {len(t.errors)}"
        )
    text, obj = tallies["текст"], tallies["объект"]
    print(
        f"\nсравнения (Фишер): факт p={fisher((text.fact, text.runs), (obj.fact, obj.runs)):.3f}; "
        f"плохая ссылка p={fisher((text.bad, text.runs), (obj.bad, obj.runs)):.3f}"
    )
    if text.seconds and obj.seconds:
        extra = statistics.median(obj.seconds) - statistics.median(text.seconds)
        print(
            f"время: объект − текст = {extra:+.1f} с на ответ "
            f"({extra / statistics.median(text.seconds):+.0%})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
