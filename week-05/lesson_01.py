"""Ф5-01: архитектура RAG — из чего состоит пайплайн.

Корпус — docqa/sample_docs, вопросы — q01–q20 из regress/suite.json (Ф5.5-02): у 18 есть
документ с ответом (метрика «источник»), у q10 и q20 ответа в корпусе нет. Судья — fact@v1
из Ф5.5-01 (κ +0,95 на 40 пунктах, слепое пятно — ложное «информации нет»), с его кешем.

    python -u week-05/lesson_01.py --task 1 2>&1 | tee week-05/lesson_01_task1.txt   # без модели
    python -u week-05/lesson_01.py --task 2 2>&1 | tee week-05/lesson_01_task2.txt   # ~10 мин
    python -u week-05/lesson_01.py --task 3 2>&1 | tee week-05/lesson_01_task3.txt   # по прогону 2
    python -u week-05/lesson_01.py --task 4 2>&1 | tee week-05/lesson_01_task4.txt   # по прогону 2
    python -u week-05/lesson_01.py --task 5 2>&1 | tee week-05/lesson_01_task5.txt   # 4 обращения
    python -u week-05/lesson_01.py --task 6 2>&1 | tee week-05/lesson_01_task6.txt   # блок в наборе

Три ветки задания 2, все на одних и тех же 20 вопросах, temperature 0, seed 42 (как продукт):
* оракул — пайплайн, в контекст кладётся документ из золотой разметки;
* как есть — docqa целиком (агент: модель сама решает, что искать и читать; подстраховка
  кода — если не прочитала ничего), запуск `python -m docqa -q …` с журналом событий;
* поиск кода — пайплайн с поиском docqa без модели (DocumentStore.search по тексту вопроса,
  первый найденный документ), то есть та ветка, что останется, когда поиск уйдёт из агента.
"""

import argparse
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import ollama

from docqa.documents import DocumentStore
from docqa.pipeline import StoreSearchRetriever
from docqa.regress import last_answer
from ragkit.judge import wilson
from ragkit.pipeline import (
    BlockContext,
    FolderLoader,
    OllamaGenerator,
    OracleRetriever,
    Pipeline,
    WholeDocument,
)
from ragkit.regress import Suite, paired

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = ROOT / "docqa" / "sample_docs"
SUITE = ROOT / "regress" / "suite.json"
OUT = HERE / "out"
DATA = HERE / "data"
RESULTS = DATA / "branches.json"
MODEL = "qwen2.5:7b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}
JUDGE_FLOOR = 0.13  # верхняя граница ошибки судьи на 40 пунктах (Ф5.5-02, задание 6)
# Судить по документу нужно и там, где ответа в корпусе нет: берём ближайший по теме
# (так в Ф5.5-01 судили вопрос про дроны — по delivery.md).
JUDGE_DOC_FOR_NONE = {"q10": "delivery.md", "q20": "gift_cards.md"}

PREDICTIONS = {
    "2": "сквозная доля (судья): оракул 17–19 из 20, как есть (docqa) 16–19, поиск кода 13–17; "
    "разрыв оракул − как есть 0–10 п. п. — на 20 вопросах «не различимо» (порог ≥ 13 п. п.); "
    "нужный документ в контексте: оракул 18/18 по построению, как есть 17–18/18 (в Ф5.5-02 "
    "источник 97,8 %), поиск кода 13–16/18 (поиск по основам слов всего вопроса путает "
    "доставку и сертификаты)",
    "3": "как есть: A 0,94–1,00, B 0,90–0,95, A×B ≈ 0,88, сквозная по тем же 18 — 0,85–0,95; "
    "произведение сходится в пределах 1 вопроса; отдельно q10 и q20 — верны без нужного "
    "документа (его нет), их нельзя засчитать ни в A, ни в B",
    "4": "как есть на запрос: вход 2500–4000 токенов, выход 60–150, обращений 2–3, 10–20 с, "
    "0,005–0,008 $ по прайсу; из этого на поиск силами модели (обращения, кончившиеся вызовом "
    "инструмента) — 30–45 % токенов входа и 30–40 % секунд; поиск кода: 600–900 токенов, "
    "1 обращение, 3–6 с",
    "5": "docqa сейчас: новый ответ с первого же запроса (читает файл на каждом); пайплайн с "
    "индексом без проверки — никогда, пока индекс не пересобран; с версией документа в чанке — "
    "расхождение видно на первом же запросе",
    "6": "чтобы подменить ретривер, тронуть 1 стадию из 5",
}


# Результат (коммит c4e3809ba1c9; lesson_01_task1…6.txt, data/branches.json; qwen2.5:7b,
#   temperature 0, seed 42; судья fact@v1 — 31 новое обращение, 29 из кеша)
#
# ПОТОЛОК ПО ОРАКУЛУ: 19 из 20 = 95 % [76–99 %]. Единственная ошибка (q06) — судьи, а не
#   генерации: ответ «5 % от суммы покупки; со статусом „Постоянный читатель“ — 10 %» верен, но
#   fact@v1 штрафует лишнее (его сдвиг из Ф5.5-01, задание 5). По-человечески — 20 из 20.
#
# Задание 1. Офлайна в docqa нет вовсе: загрузка, карантин и «представление» (основы слов)
#   делаются на каждом вызове инструмента; нарезки, хранилища и переупорядочивания нет; на месте
#   поиска — модель, которая вызывает search_documents и read_document, и подстраховка кода.
#   Письменно: на каждом запросе выполняется то, что могло бы выполняться один раз: (1) перечисление
#   папки и чтение файлов с карантином — documents.py:50–69; (2) основы слов всех строк всех
#   документов для поиска — documents.py:97–124 (замер: 1,4 мс на запрос — пока дёшево, 6 файлов);
#   (3) заголовки документов для list_documents — documents.py:75; (4) системный промпт и схемы
#   инструментов — около 500 одинаковых токенов, которые уходят заново в каждом обращении (задание
#   4). Кандидаты на перенос в офлайн для Ф5-04 — первые три; четвёртое офлайн не уносится, но
#   уходит вместе с поиском силами модели.
#
# Задание 2. Сквозная доля по судье: оракул 19/20 (прогноз 17–19 ✓), docqa как есть 20/20 (прогноз
#   16–19 — выше), поиск кода 19/20 (прогноз 13–17 — выше). Нужный документ в контексте: оракул
#   18/18, как есть 17/18, поиск кода 18/18 (прогноз 13–16 — поиск по основам слов вопроса оказался
#   точнее, чем я ждал: вопросы набора сформулированы словами документов). Разрыв оракул − как есть:
#   −5 п. п., одна несовпавшая пара, Макнемар p = 1 — «не различимо» при пороге 16 п. п. (прогноз
#   0–10 и «не различимо» ✓).
#   Письменно: на этом корпусе систему не ограничивает ни поиск, ни генерация — по крайней мере в
#   пределах того, что видно на 20 вопросах. Поиск кода приносит нужный документ 18 из 18 раз, и
#   сквозная доля у него та же, что у оракула (19/20), — значит, поиск не ограничивает. Генерация
#   ошиблась один раз, и тот спорный (судья). Потолок 95 % с интервалом 76–99 % — ориентир, но
#   слабый: корпус из 6 коротких файлов и вопросы словами документов слишком лёгкие, чтобы показать
#   разницу между ретриверами. Это и есть первый вывод фазы: прежде чем менять стадии, нужен
#   корпус, на котором замер способен различать. И оговорка: оракул собран на генераторе
#   пайплайна (свой промпт, один ответ), а docqa — агент со своим генератором и итогом DocAnswer;
#   поэтому docqa может быть «выше потолка» — потолок относится к генератору пайплайна.
#
# Задание 3. docqa как есть, 18 вопросов с документом: A = 17/18 (0,944), B = 17/17 (1,000),
#   A × B = 0,944, сквозная по тем же 18 — 18/18 (1,000): расхождение +5,6 п. п., ровно один
#   вопрос — q14 (прогноз «сходится в пределах одного вопроса» ✓). Поиск кода и оракул: A × B =
#   0,944 = сквозная, сходится.
#   Письменно: в модели «поиск × генерация» нет стадии «ответ без опоры». q14 по журналу: модель
#   попросила несуществующий organisation_meetings.md, получила «такого документа нет», и ответ
#   ушёл без единого прочитанного документа (issues: unverified, источников нет); судья засчитал
#   его верным. Такой ответ не должен идти в плюс ни A, ни B: верность без документа — удача или
#   память модели, а не работа пайплайна. Нашлась и причина в коде: отказ «нет такого документа»
#   попадал в read_failed как падение чтения, и подстраховка не включалась («упрётся в то же
#   падение»), хотя докстринг read_failed говорит обратное. Исправлено (session.py, read_failed;
#   тест test_wrong_document_name_is_not_a_failed_read); кассеты поймали изменение ровно на q14 —
#   её запись перезаписана.
#
# Задание 4. docqa как есть на запрос: вход 2499 токенов (прогноз 2500–4000 ✓, на границе), выход
#   135 (60–150 ✓), обращений 3,0 (2–3 ✓), 18,7 с (10–20 ✓), 8,9 тысячных $ по прайсу (5–8 — чуть
#   выше). На поиск силами модели — 30 % токенов входа (30–45 ✓) и 21 % секунд (30–40 — меньше).
#   Первое обращение хода — около 500 токенов: system и схемы инструментов, которые уходят заново в
#   каждом из трёх обращений, то есть около 60 % входа — повтор одного и того же. Подстраховка
#   кода работала в 13 запросах из 20: документ чаще находит код, а не модель. Поиск кода
#   (пайплайн): 327 токенов, 1 обращение, 0,9 с — в 7,6 раза меньше входа и в 20 раз быстрее (прогноз
#   600–900 токенов и 3–6 с — дешевле, чем я ждал: документы короткие). Чтение и поиск по папке —
#   1,4 мс на запрос.
#
# Задание 5. Прогноз сбылся по всем трём: docqa видит правку с первого же запроса (310); пайплайн
#   с индексом, снятым до правки, отвечает по старому (290) и будет отвечать, пока индекс не
#   пересобран; метка версии в чанке (sha256 текста, metadata["version"]) показывает расхождение на
#   первом же запросе, без обращения к модели; после пересборки — 310.
#   Метка, которая это ловит: версия документа (хеш содержимого) в каждом чанке и в индексе, и
#   сверка с диском (Index.stale) перед ответом; дата сборки индекса — для отчёта, но не для проверки.
#   Письменно: всё, что сохранено с внешнего источника (кассета, индекс, кеш), хранит отпечаток
#   источника и сверяется с ним перед использованием: не совпало — пересобрать или громко отказать,
#   а не отвечать по старому.
#
# Задание 6. Подмена ретривера оракулом тронула 1 стадию из 5 (прогноз 1 ✓): with_stage(retriever=…)
#   возвращает ту же сборку, остальные четыре стадии — те же объекты (тест
#   test_oracle_replaces_only_the_retriever). Блок «pipeline» в regress/suite.json: по каждой ветке A,
#   B, сквозная с интервалом Уилсона и пределом чувствительности (16–19 п. п. на 18–20 вопросах),
#   токенов входа и секунд на запрос. Базовая линия — один прогон при temperature 0: погоду этих
#   метрик ещё не мерили.


# ---------------------------------------------------------------------------
# Вопросы и ветки
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    id: str
    text: str
    gold: str | None  # документ с ответом; None — ответа в корпусе нет
    fact_re: str

    @property
    def judge_doc(self) -> str:
        return self.gold or JUDGE_DOC_FOR_NONE[self.id]


def questions() -> list[Question]:
    out = []
    for item in Suite.load(SUITE).items:
        if item["kind"] != "вопрос":
            continue
        source = item["metrics"].get("источник", {}).get("source")
        out.append(
            Question(item["id"], item["title"], source, item["metrics"]["факт"]["answer_re"])
        )
    return out


@dataclass
class Run:
    """Один ответ одной ветки на один вопрос."""

    branch: str
    id: str
    answer: str
    documents: list[str]
    prompt_tokens: int
    output_tokens: int
    calls: int
    seconds: float
    cost_usd: float = 0.0
    search_prompt_tokens: int = 0  # обращения, кончившиеся вызовом инструмента (поиск моделью)
    search_seconds: float = 0.0
    judge: bool | None = None
    regex: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def load_lesson_01() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "f55_lesson_01", ROOT / "week-05.5" / "lesson_01.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["f55_lesson_01"] = module
    spec.loader.exec_module(module)
    return module


def pipeline(retriever: Any) -> Pipeline:
    generator = OllamaGenerator(ollama.Client(), MODEL, OPTIONS)
    return Pipeline(FolderLoader(DOCS), WholeDocument(), retriever, BlockContext(), generator)


def run_pipeline(branch: str, pipe: Pipeline, qs: Sequence[Question]) -> list[Run]:
    pipe.build_index()
    out = []
    for q in qs:
        started = time.monotonic()
        a = pipe.answer(q.text)
        out.append(Run(branch, q.id, a.generation.text, a.documents, a.generation.prompt_tokens,
                       a.generation.output_tokens, a.generation.calls, time.monotonic() - started,
                       extra={"retrieve_ms": round(a.retrieve_seconds * 1000, 2)}))  # fmt: skip
        print(f"   {branch:<10} {q.id}: {', '.join(a.documents) or '—'}; "
              f"{a.generation.prompt_tokens}+{a.generation.output_tokens} т; "
              f"{out[-1].seconds:.1f} с", flush=True)  # fmt: skip
    return out


def run_agent(qs: Sequence[Question]) -> list[Run]:
    """docqa как есть: подпроцесс, как у пользователя, с журналом событий."""
    folder = OUT / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for q in qs:
        events_path, log_path = folder / f"{q.id}.events.jsonl", folder / f"{q.id}.txt"
        events_path.unlink(missing_ok=True)
        started = time.monotonic()
        done = subprocess.run(
            [sys.executable, "-m", "docqa", "-q", q.text, "--events", str(events_path.relative_to(ROOT)),
             "--log", str(log_path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )  # fmt: skip
        seconds = time.monotonic() - started
        events = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines()]
        responses = [e for e in events if e["event"] == "response"]
        turn: dict[str, Any] = next((e for e in reversed(events) if e["event"] == "turn_end"), {})
        searching = [r for r in responses if r.get("calls", 0) > 0]
        # время обращения — между событием request и его response
        stamps = {
            (e["turn"], e["request"]): e.get("ts", 0.0) for e in events if e["event"] == "request"
        }
        search_s = sum(r.get("ts", 0.0) - stamps.get((r["turn"], r["request"]), r.get("ts", 0.0))
                       for r in searching)  # fmt: skip
        run = Run("как есть", q.id, last_answer(done.stdout), list(turn.get("documents_read") or []),
                  sum(r.get("prompt_eval_count") or 0 for r in responses),
                  sum(r.get("eval_count") or 0 for r in responses), len(responses), seconds,
                  float(turn.get("spent_usd") or 0.0),
                  sum(r.get("prompt_eval_count") or 0 for r in searching), max(0.0, search_s),
                  extra={"fallback": any(e["event"] == "fallback" for e in events),
                         "first_prompt": responses[0].get("prompt_eval_count") if responses else 0,
                         "code": done.returncode})  # fmt: skip
        out.append(run)
        print(f"   как есть   {q.id}: {', '.join(run.documents) or '—'}; {run.prompt_tokens}+"
              f"{run.output_tokens} т, обращений {run.calls}; {seconds:.1f} с"
              + ("; подстраховка" if run.extra["fallback"] else ""), flush=True)  # fmt: skip
    return out


def judge_all(runs: list[Run], qs: Sequence[Question]) -> None:
    l1 = load_lesson_01()
    runner = l1.Runner(None)
    by_id = {q.id: q for q in qs}
    for r in runs:
        q = by_id[r.id]
        item = l1.Item(
            f"{r.branch}-{r.id}", int(q.id[1:]), q.text, q.judge_doc, r.answer.strip(), True
        )
        r.judge = runner.verdict(item)
        r.regex = re.search(q.fact_re, r.answer, re.IGNORECASE) is not None
    print(
        f"   судья fact@v1: обращений {runner.calls}, из кеша {runner.hits}, ошибок разбора {runner.failures}"
    )


def collect() -> tuple[list[Question], dict[str, list[Run]]]:
    qs = questions()
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        return qs, {b: [Run(**r) for r in rows] for b, rows in data.items()}
    gold = {q.text: q.gold for q in qs}
    branches = {
        "оракул": run_pipeline("оракул", pipeline(OracleRetriever(gold)), qs),
        "как есть": run_agent(qs),
        "поиск кода": run_pipeline(
            "поиск кода", pipeline(StoreSearchRetriever(DocumentStore(DOCS))), qs
        ),
    }
    judge_all([r for rows in branches.values() for r in rows], qs)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({b: [asdict(r) for r in rows] for b, rows in branches.items()},
                                  ensure_ascii=False, indent=1) + "\n", encoding="utf-8")  # fmt: skip
    print(f"   записано: {RESULTS.relative_to(ROOT)}")
    return qs, branches


def predict(task: str) -> None:
    print(f"   прогноз (до запуска): {PREDICTIONS[task]}")


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]" if n else "—"


# ---------------------------------------------------------------------------
# Задание 1. docqa по стадиям
# ---------------------------------------------------------------------------
def where(path: str, pattern: str) -> str:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if re.search(pattern, line):
            return f"{path}:{number}"
    return f"{path}:?"


def task1() -> None:
    print("\n== Задание 1. docqa по стадиям (строки — по текущему коду)")
    rows = [
        ("ОФЛАЙН", "загрузка и парсинг", where("docqa/documents.py", r"def names\("),
         "нет офлайна: names() перечисляет папку, raw_text()/text() читают файл и чистят карантином "
         f"({where('docqa/documents.py', r'def text')}) — на каждом вызове инструмента"),
        ("ОФЛАЙН", "нарезка", where("docqa/documents.py", r"def read\("),
         "нет: единица — документ целиком; read() урезает длинный до 4000 знаков целыми абзацами"),
        ("ОФЛАЙН", "представление", "—", "нет: основы слов (stem, первые 5 букв) считаются на запросе "
         f"({where('docqa/documents.py', r'def stem')})"),
        ("ОФЛАЙН", "запись в хранилище", "—", "нет: хранилище — сами файлы, перечитываются каждый раз"),
        ("ОНЛАЙН", "понимание запроса", where("docqa/session.py", r"async def _ask"),
         "модель: сама решает, искать ли и с какими словами (аргументы search_documents); "
         f"подстраховка — вопрос целиком, до 200 знаков ({where('docqa/session.py', r'query = .*FALLBACK_QUERY_CHARS')})"),
        ("ОНЛАЙН", "поиск кандидатов", where("docqa/documents.py", r"def search\("),
         "пересечение основ слов запроса и строк всех документов, полный просмотр папки; вызывает "
         f"модель (инструмент, {where('docqa/documents.py', r'def _search')}) или код "
         f"({where('docqa/session.py', r'def _fallback')})"),
        ("ОНЛАЙН", "переупорядочивание", where("docqa/documents.py", r"hits.sort\("),
         "нет как стадии: сортировка по числу совпавших слов и строк, затем по имени"),
        ("ОНЛАЙН", "сборка контекста", where("docqa/session.py", r"produced.append\(result.message\(\)\)"),
         "результат инструмента — сообщением tool в историю хода; system — _SYSTEM "
         f"({where('docqa/session.py', r'^_SYSTEM = ')}); история — LastTurns "
         f"({where('docqa/session.py', r'policy=LastTurns')})"),
        ("ОНЛАЙН", "генерация ответа", where("docqa/session.py", r"got = await self._request"),
         f"поток с инструментами; итог — DocAnswer по схеме ({where('docqa/session.py', r'async def _final')})"),
        ("ОНЛАЙН", "проверка и цитаты", where("docqa/answer.py", r"def check_answer"),
         "источники: есть в папке и прочитаны в этом ходе; script_mess; фильтр адресов (guards)"),
    ]  # fmt: skip
    for side, stage, line, what in rows:
        print(f"   {side:<7} {stage:<20} {line:<28} {what}")
    print("\n   на месте поиска сейчас: модель, которая вызывает search_documents (основы слов по всем "
          "строкам всех файлов) и read_document; если она не прочитала ничего — тот же поиск "
          "вызывает код по тексту вопроса")  # fmt: skip


# ---------------------------------------------------------------------------
# Задания 2–3
# ---------------------------------------------------------------------------
def task2() -> None:
    print("\n== Задание 2. Потолок: оракул против docqa как есть (и поиска кода)")
    predict("2")
    qs, branches = collect()
    with_doc = [q for q in qs if q.gold]
    print(
        f"\n   {'ветка':<11} {'сквозная (судья)':<26} {'сквозная (регулярка)':<26} нужный документ в контексте"
    )
    for name, rows in branches.items():
        by = {r.id: r for r in rows}
        judge = sum(r.judge is True for r in rows)
        regex = sum(r.regex for r in rows)
        hit = sum(q.gold in by[q.id].documents for q in with_doc)
        print(
            f"   {name:<11} {share(judge, len(rows)):<26} {share(regex, len(rows)):<26} {share(hit, len(with_doc))}"
        )
    order = [q.id for q in qs]
    vec = {
        b: [next(r for r in rows if r.id == i).judge is True for i in order]
        for b, rows in branches.items()
    }
    print("\n   попунктно (правило фазы: разница ниже max(предел чувствительности, ошибка судьи 13 %) — "
          "«не различимо»):")  # fmt: skip
    for a, b in (("как есть", "оракул"), ("поиск кода", "оракул"), ("поиск кода", "как есть")):
        print(f"      {paired(vec[a], vec[b], JUDGE_FLOOR).text(a, b)}")
    for name, rows in branches.items():
        wrong = [r for r in rows if r.judge is not True]
        for r in wrong:
            q = next(q for q in qs if q.id == r.id)
            print(f"   {name}: {r.id} неверно — документы {r.documents or '—'} (нужен {q.gold or 'нет'}); "
                  f"«{' '.join(r.answer.split())[:110]}»")  # fmt: skip
    oracle = sum(r.judge is True for r in branches["оракул"])
    print(f"\n   ПОТОЛОК ПО ОРАКУЛУ: {share(oracle, len(qs))} — выше этого не поднимет никакой ретривер "
          "при этой генерации")  # fmt: skip


def task3() -> None:
    print("\n== Задание 3. Сходится ли произведение: A × B против сквозной")
    predict("3")
    qs, branches = collect()
    with_doc = [q for q in qs if q.gold]
    for name in ("как есть", "поиск кода", "оракул"):
        by = {r.id: r for r in branches[name]}
        hit = [q for q in with_doc if q.gold in by[q.id].documents]
        right_hit = [q for q in hit if by[q.id].judge is True]
        e2e = sum(by[q.id].judge is True for q in with_doc)
        a, b = len(hit) / len(with_doc), (len(right_hit) / len(hit) if hit else 0.0)
        lucky = [
            q.id for q in with_doc if q.gold not in by[q.id].documents and by[q.id].judge is True
        ]
        print(f"\n   {name}: на 18 вопросах с документом")
        print(f"      A = нужный документ в контексте: {share(len(hit), len(with_doc))}")
        print(f"      B = верно среди тех, где он был: {share(len(right_hit), len(hit))}")
        print(
            f"      A × B = {a * b:.3f}; сквозная по тем же 18: {share(e2e, len(with_doc))} "
            f"({e2e / len(with_doc):.3f}); расхождение {(e2e / len(with_doc) - a * b) * 100:+.1f} п. п."
        )
        print(f"      верно без нужного документа (не идёт ни в A, ни в B): {lucky or 'нет'}")
        none = [q for q in qs if not q.gold]
        print("      вопросы без документа в корпусе: "
              + ", ".join(f"{q.id} {'верно' if by[q.id].judge else 'неверно'}" for q in none))  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 4. Цена запроса
# ---------------------------------------------------------------------------
def stats(values: Sequence[float]) -> str:
    return f"среднее {statistics.mean(values):.1f}, медиана {statistics.median(values):.1f}, {min(values):.0f}–{max(values):.0f}"


def task4() -> None:
    print("\n== Задание 4. Цена одного запроса сегодня (базовая линия фазы)")
    predict("4")
    qs, branches = collect()
    for name in ("как есть", "поиск кода", "оракул"):
        rows = branches[name]
        print(f"\n   {name} ({len(rows)} запросов):")
        print(f"      токенов входа: {stats([r.prompt_tokens for r in rows])}")
        print(f"      токенов выхода: {stats([r.output_tokens for r in rows])}")
        print(f"      обращений к модели: {stats([r.calls for r in rows])}")
        print(f"      секунд: {stats([r.seconds for r in rows])}")
        if name == "как есть":
            print(
                f"      цена по прайсу: {stats([r.cost_usd * 1000 for r in rows])} (тысячные доли $)"
            )
            tin = sum(r.prompt_tokens for r in rows)
            tsearch = sum(r.search_prompt_tokens for r in rows)
            sec = sum(r.seconds for r in rows)
            ssec = sum(r.search_seconds for r in rows)
            print(f"      из них на поиск силами модели (обращения, кончившиеся вызовом инструмента): "
                  f"{tsearch / tin:.0%} токенов входа, {ssec / sec:.0%} секунд")  # fmt: skip
            first = [r.extra.get("first_prompt") or 0 for r in rows]
            print(
                f"      первое обращение хода (system + схемы инструментов + вопрос): {stats(first)} токенов"
            )
            print(
                f"      подстраховка сработала в {sum(bool(r.extra.get('fallback')) for r in rows)} из {len(rows)}"
            )
    store = DocumentStore(DOCS)
    started = time.monotonic()
    for q in qs:
        hits = store.search(" ".join(q.text.split())[:200])["results"]
        if hits:
            store.read(hits[0]["name"])
    per = (time.monotonic() - started) / len(qs) * 1000
    print(f"\n   чтение файлов, карантин, основы слов и поиск по всей папке на каждом запросе: {per:.1f} мс "
          "на запрос — та часть, что уходит в офлайн при индексе (сейчас дёшево: 6 коротких файлов)")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 5. Документ изменился
# ---------------------------------------------------------------------------
def task5() -> None:
    print("\n== Задание 5. Документ изменился: через сколько запросов ответ станет новым")
    predict("5")
    q = next(q for q in questions() if q.id == "q01")
    changed = OUT / "changed_docs"
    if changed.exists():
        shutil.rmtree(changed)
    shutil.copytree(DOCS, changed)
    pipe = pipeline(StoreSearchRetriever(DocumentStore(changed))).with_stage(
        loader=FolderLoader(changed)
    )
    index = pipe.build_index()  # индекс снят ДО правки
    before = pipe.answer(q.text).generation.text
    path = changed / "delivery.md"
    path.write_text(path.read_text(encoding="utf-8").replace("290", "310"), encoding="utf-8")
    print(f"   правка: delivery.md, 290 → 310; вопрос {q.id}: {q.text}")
    done = subprocess.run([sys.executable, "-m", "docqa", "--docs", str(changed.relative_to(ROOT)), "-q", q.text],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)  # fmt: skip
    now = last_answer(done.stdout)
    print(
        f"   docqa сейчас, запрос 1 после правки: {'новый' if '310' in now else 'СТАРЫЙ'} — «{' '.join(now.split())[:120]}»"
    )
    after = pipe.answer(q.text).generation.text
    print(f"   пайплайн, индекс до правки: до правки «{' '.join(before.split())[:80]}»; после правки — "
          f"{'новый' if '310' in after else 'СТАРЫЙ'}: «{' '.join(after.split())[:80]}»")  # fmt: skip
    stale = index.stale(pipe.loader.load())
    print(f"   метка версии в чанке (sha256 текста, metadata['version']): разошлись {stale} — видно на первом "
          "же запросе, без обращения к модели")  # fmt: skip
    pipe.build_index()
    rebuilt = pipe.answer(q.text).generation.text
    print(
        f"   после пересборки индекса: {'новый' if '310' in rebuilt else 'старый'} — «{' '.join(rebuilt.split())[:80]}»"
    )


# ---------------------------------------------------------------------------
# Задание 6. Блок в наборе
# ---------------------------------------------------------------------------
def task6() -> None:
    print("\n== Задание 6. Подменяемость и блок «по стадиям» в regress/suite.json")
    predict("6")
    stages = ("loader", "chunker", "retriever", "context", "generator")
    base = pipeline(StoreSearchRetriever(DocumentStore(DOCS)))
    oracle = base.with_stage(retriever=OracleRetriever({}))
    touched = [s for s in stages if getattr(base, s) is not getattr(oracle, s)]
    print(f"   подмена ретривера оракулом тронула стадий: {len(touched)} из {len(stages)} ({', '.join(touched)}); "
          "тест — tests/test_pipeline.py::test_oracle_replaces_only_the_retriever")  # fmt: skip
    qs, branches = collect()
    with_doc = [q for q in qs if q.gold]
    suite = Suite.load(SUITE)
    block: dict[str, Any] = {"about": "Ф5-01: по стадиям на q01–q20, temperature 0, seed 42; A — нужный "
                             "документ в контексте (18 вопросов с документом), B — верно (судья fact@v1) "
                             "среди них; предел чувствительности — точка минус нижняя граница Уилсона, "
                             "порог различимости — max(предел, 13 % ошибки судьи)"}  # fmt: skip
    for name, rows in branches.items():
        by = {r.id: r for r in rows}
        hit = [q for q in with_doc if q.gold in by[q.id].documents]
        right = [q for q in hit if by[q.id].judge is True]
        e2e = sum(r.judge is True for r in rows)

        def entry(k: int, n: int) -> dict[str, Any]:
            lo, hi = wilson(k, n)
            return {"k": k, "n": n, "point": round(k / n, 3) if n else None, "low": round(lo, 3),
                    "high": round(hi, 3), "sensitivity": round(k / n - lo, 3) if n else None}  # fmt: skip

        block[name] = {
            "A": entry(len(hit), len(with_doc)),
            "B": entry(len(right), len(hit)),
            "сквозная": entry(e2e, len(rows)),
            "токенов входа на запрос": round(statistics.mean(r.prompt_tokens for r in rows), 1),
            "секунд на запрос": round(statistics.mean(r.seconds for r in rows), 2),
        }
        print(f"   {name}: A {block[name]['A']['k']}/{block[name]['A']['n']}, B {block[name]['B']['k']}/"
              f"{block[name]['B']['n']}, сквозная {e2e}/{len(rows)} (предел "
              f"{block[name]['сквозная']['sensitivity'] * 100:.0f} п. п.), токенов входа "
              f"{block[name]['токенов входа на запрос']}")  # fmt: skip
    suite.extra["pipeline"] = block
    suite.save(SUITE)
    print(f"   блок «pipeline» записан в {SUITE.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=range(1, 7), required=True)
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip
    print(f"коммит {head}; модель {MODEL}, {OPTIONS}; задание {args.task}")
    tasks: dict[int, Callable[[], None]] = {
        1: task1,
        2: task2,
        3: task3,
        4: task4,
        5: task5,
        6: task6,
    }
    tasks[args.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
