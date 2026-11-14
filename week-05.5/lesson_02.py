"""Ф5.5-02: регрессионный набор в CI.

Набор — regress/suite.json (49 пунктов: q01–q20 — вопросы, 01–15e — таблица отказов Ф4-08);
бегун — python -m docqa.regress; машинерия — ragkit/regress.py. Этот скрипт — задания урока:

    python -u week-05.5/lesson_02.py --task 1 2>&1 | tee week-05.5/lesson_02_task1.txt   # без модели
    python -u week-05.5/lesson_02.py --task 3 2>&1 | tee week-05.5/lesson_02_task3.txt   # без модели
    # запись кассет (живая модель, ~15–20 мин) — до заданий 2, 5, 7:
    python -u -m docqa.regress --layer record --accept --save regress/runs/record.json 2>&1 | tee week-05.5/lesson_02_record.txt
    python -u week-05.5/lesson_02.py --task 2 2>&1 | tee week-05.5/lesson_02_task2.txt   # кассеты
    python -u week-05.5/lesson_02.py --task 4 2>&1 | tee week-05.5/lesson_02_task4.txt   # живая, ~30 мин
    python -u week-05.5/lesson_02.py --task 5 2>&1 | tee week-05.5/lesson_02_task5.txt   # судья, несколько обращений
    python -u week-05.5/lesson_02.py --task 6 2>&1 | tee week-05.5/lesson_02_task6.txt   # судья, ~120 обращений
    python -u week-05.5/lesson_02.py --task 7 2>&1 | tee week-05.5/lesson_02_task7.txt   # кассеты
    python -u week-05.5/lesson_02.py --task 8 2>&1 | tee week-05.5/lesson_02_task8.txt   # время шагов CI

Старый код (задания 1, 2, 7) запускается в git worktree во временной папке: рабочее дерево не
трогается, мутации не могут остаться в репозитории.
"""

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from docqa.regress import CASSETTES, REGRESS, SUITE, later_requests_error, run_suite
from ragkit.judge import Calibration, wilson
from ragkit.quality import script_mess
from ragkit.regress import (
    Baseline,
    RunResult,
    Suite,
    compare,
    min_detectable,
    sample_size,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = REGRESS / "runs"
DOCS = ROOT / "docqa" / "sample_docs"

PREDICTIONS = {
    "1": "streaming (tool_calls): ловит самый дешёвый слой — фейковый клиент, но только если "
    "фейк знает настоящий формат куска (вызов в message.tool_calls); на 1359f0a красная, на HEAD "
    "зелёная. Chunk.__eq__: фейковый слой (модели нет вовсе); красная на 1359f0a и на 5ba2d30 (баг "
    "жив), зелёная после починки. Оценщик токенов: самый дешёвый слой НЕ ловит — правда (сколько "
    "токенов насчитал сервер) живёт вне кода; ловит кассета с записанным prompt_eval_count: "
    "запись старого кода (debug1b.txt) +102…157 % — красная, воспроизведение HEAD ≤ +50 % — зелёная",
    "2": "запись всего набора 12–20 мин (у пунктов с моделью 15–25 с), воспроизведение 30–60 с — "
    "быстрее в 15–30 раз; мутация в ragkit/quarantine.py краснеет на 10a (промах кассеты: "
    "документ уходит модели со строкой-инструкцией, и ожидание «карантин сработал»), остальные "
    "зелёные; подменённый итог q01 (290 → 390) в вердикт не попадает — ворота зелёные, «факт» "
    "падает на 1 только в блоке метрик, которые не ворота; подменённый вызов посреди хода (q02, "
    "другой документ) — промах на следующем обращении, то есть красный как «не совпало с записью»",
    "3": "n ≈ 150 (149,1) для 0,95 → 0,90; при n = 40 заметно падение до ≈ 0,85, то есть ≈ 10 п. п.",
    "4": "живой слой, 20 вопросов, temperature 0,7, seed 1–5: «факт» от 17 до 20 из 20; планка А "
    "(точка ≈ 19/20) краснеет в 2–3 чистых прогонах из 5, планка Б (нижняя граница ≈ 76 % → "
    "16/20) — в 0 из 5; предел чувствительности Б ≈ 19 п. п.",
    "5": "точное совпадение: ложных тревог 18–20 из 20, поймано 20 из 20; регулярка: ложных 0–2, "
    "поймано 8–10 (числа и ложные отказы; реклама, приписка и структура мимо), наивная подстрока "
    "«Казань» краснеет на «по Казани»; структура: ложных 8–10 (у ответов режима «текст» источник "
    "не назван — 5 из 27 в Ф4-08), поймано 12–14; судья: ложных 0–1, поймано 12–15 (текст да, "
    "структура нет). Ложное «информации нет» ловят структура (not_found против ожидания) и "
    "регулярка (факта нет), судья — нет (Ф5.5-01: 2 из 2 пропущено). Без обращения к модели "
    "проверяются 9–12 строк таблицы отказов из 29",
    "6": "судья temperature 0,7, seed 1 против seed 2: разойдутся 1–4 вердикта из 40; temperature 0, "
    "seed 1, свежий прогон против кеша Ф5.5-01: 0 из 40",
    "7": "отключённый counter.observe в docqa/session.py (поломка Ф4-08, присест 1, возвращена): "
    "вердикт изменят 15–30 пунктов из 49 — почти везде ожидание «оценка входа от факта ≤ +50 %», в "
    "07a/07b/09b ещё и окно с бюджетом раньше времени",
    "8": "блокирующий поток на GitHub 3–5 мин (установка 1–2 мин, mypy ~40 с, pytest ~30 с, "
    "кассеты ~1 мин); ночной на Mac 15–25 мин, цена по прайсу 0,3–0,6 $",
}


# Результат, часть 1 (коммит e3491f79533f; lesson_02_record.txt, lesson_02_task1/3/4/5/6.txt)
#
# Запись кассет: 49 пунктов за 659 с (11 мин), цена по счётчикам 0,31 $. Вопросы — 20 из 20 прошли,
#   таблица отказов — 12 из 29 красные, и все 12 — на неявной проверке «оценка входа ≤ +50 %».
#   Ошибалась проверка, а не код: после ответа, подставленного инъектором (счёт 0), факта у кода
#   нет, и следующее обращение законно идёт по чистой оценке (+112…165 %); у 05 и 06 (+175 200 %)
#   завышение посчитано на самом подставленном ответе со счётом 0. Проверка исправлена: смотрит
#   только обращения после первого настоящего ответа хода. Перезаписывать кассеты не нужно —
#   ожидания считаются по журналу; задание 2 пересчитывает запись по её журналам.
#
# Задание 1. Прогноз сбылся по всем трём.
#   Поток без tool_calls: на 1359f0a красная, на 5ba2d30 и HEAD зелёная. Слой — фейковый клиент,
#   но фейк обязан выдавать вызов так, как ollama, — в message.tool_calls куска; 13 старых тестов
#   были зелёными, потому что их фейки этого не умели.
#   Chunk.__eq__: красная на 1359f0a и 5ba2d30 (из двух разных чанков в множестве остаётся 1),
#   зелёная после починки. Слой фейков — модели там нет вовсе.
#   Оценщик токенов: запись старого кода — +102…126 % на 6 обращениях, красная; воспроизведение
#   HEAD — +10…47 % на 39, зелёная. Слой кассет: самый дешёвый слой её не поймает, потому что
#   «правильно» здесь задаёт токенизатор сервера.
#   Письменно: у поломок, которые фейк не ловит, правильный ответ лежит вне кода — в поведении
#   сервера или модели: сколько токенов насчитано, в каком виде приходит кусок потока, что модель
#   ответит. Фейк знает о сервере ровно то, что знает его автор. Граница — там, где нужна правда о
#   внешней системе: за формат и счёт платят один раз, записью (кассета хранит prompt_eval_count и
#   куски потока), а за поведение модели — живыми обращениями каждую ночь.
#
# Задание 3. n = 150 (прогноз 150 ✓); при n = 40 видно падение до 0,845, то есть на 10,5 п. п.
#   (прогноз ≈ 0,85 ✓); 20 вопросов набора видят только падение на 15,6 п. п.
#   Письменно: кроме n в формуле три рычага. (1) Знаменатель (p₀ − p₁)²: мерить там, где
#   ожидаемая разница больше, — на трудных и пограничных пунктах, а не на тех, что проходят всегда:
#   0,95 → 0,85 требует 44 пункта вместо 150. (2) Дисперсия p(1−p) под корнем: у долей около 0 или 1
#   она мала, у «монеток» с p ≈ 0,5 велика (0,50 → 0,45 требует 616), — и парное сравнение (тот же
#   пункт в варианте A и в варианте B) убирает разброс между пунктами. (3) z_α и z_β: согласиться на
#   α = 10 % (113 пунктов) или мощность 70 % (107) — дешевле, но чаще ложная тревога или пропуск.
#
# Задание 4. Живой слой (20 вопросов, temperature 0,7, seed 1–5): «факт» 18, 19, 20, 17, 19 из 20
#   (прогноз 17–20 ✓), точка 93 %, интервал одного прогона 76–99 %. Планка А (точка 93 %) краснеет в
#   2 из 5 чистых прогонов (прогноз 2–3 ✓): ложная тревога 40 %. Планка Б (нижняя граница 76 %,
#   то есть ниже 15,3 из 20) — в 0 из 5 (прогноз ✓). «Источник» 18, 18, 18, 16, 18 из 18: А
#   краснеет в 1 из 5, Б — в 0. «Честное „нет“» — 2 пункта, интервал 9–91 %: метрика из двух
#   пунктов не сторожит ничего. Прогон — 6–8 мин и 0,15–0,19 $ по прайсу.
#   Письменно: предел чувствительности ворот Б — 17 п. п. для «факта» (93 % → ниже 76 %) и 15 п. п.
#   для «источника» (прогноз ≈ 19). Регрессия меньше этого — например, 93 % → 80 %, три вопроса из
#   20 — проходит зелёной. Поэтому отчёт печатает предел рядом с каждой метрикой («ворота не видят
#   падения меньше 17 п. п.»): без этой строки зелёный цвет значит «не хуже», а на деле — только
#   «не хуже больше чем на 17 п. п.».
#
# Задание 5. 20 верных и 20 испорченных:
#   точное совпадение — ложные тревоги 17 из 20, поймано 18 из 20 (прогноз 18–20 и 20; пропустило
#     две структурные порчи, где текст не менялся). Бракует любую перефразировку — непригодно;
#   регулярка по факту — 0 и 7 (прогноз 0–2 и 8–10): числа 4/4, ложные отказы 2/2, «другой
#     документ» 1/1; рекламу, приписку и всю структуру — нет. Наивная подстрока «Казань» на «по
#     Казани» — ложная тревога, регулярка «Казан» — нет;
#   структура — 7 и 12 (прогноз 8–10 и 12–14): все 7 ложных — ответы режима «текст», где источник
#     не назван; ловит все 10 структурных порч и оба ложных отказа, ни одной текстовой;
#   судья fact@v1 — 0 и 10 (прогноз 0–1 и 12–15): текстовые 8 из 8, структурные 2 из 10 (те, где
#     испорчен текст: иероглифы, слитая латиница), ложные отказы 0 из 2.
#   Ложное «информации нет» поймали все способы, кроме судьи: точное совпадение, регулярка и
#   структура — 2 из 2, судья — 0 из 2, ровно его слепое пятно из Ф5.5-01. Способы не заменяют
#   друг друга: регулярка и судья видят текст, структура — то, чего в тексте нет; судья вместе со
#   структурой ловят 20 из 20 при 7 ложных тревогах, и все 7 — от режима без источников.
#   Письменно: без единого обращения к модели проверяются 7 из 29 строк таблицы отказов (прогноз
#   9–12): 04b, 05, 06, 07a, 09a, 11c, 11d — там ответы целиком подставляет инъектор или модель не
#   нужна. На кассетах (одна запись, дальше — сколько угодно воспроизведений) — все 29 и все 20
#   вопросов: они и стоят воротами, а 7 из 49 пунктов не зависят даже от записи.
#
# Задание 6. Судья fact@v1 на 40 пунктах: temperature 0,7, seed 1 против seed 2 — разошлись 0 из 40
#   (прогноз 1–4 — не сбылся); temperature 0, seed 1, свежий прогон против кеша Ф5.5-01 — 0 из 40
#   (прогноз ✓). Каппа во всех трёх +0,95, пропущен один и тот же ложный отказ. Шум от seed —
#   0 [0–9 %]: вердикт судьи почти детерминирован и при 0,7, а его ошибка систематическая — тот же
#   пункт каждый раз.
#   Письменно: судья расходится с эталоном в 1 случае из 40 (2,5 %), но по 40 пунктам верхняя
#   граница его ошибки — 13 % (Уилсон). Два варианта (два чанкинга в Ф5), разница между которыми
#   меньше этой границы, он различить не может в принципе: разница в 1–2 вердикта из 40 может
#   целиком быть его ошибкой. Правило: набор не сообщает разницу меньше шума своего измерителя —
#   разница ниже max(предел чувствительности метрики, верхняя граница ошибки судьи) печатается
#   «не различимо», а не «B лучше». Систематичность ошибки и помогает, и вредит: на одних и тех же
#   пунктах она вычитается при сравнении A с B, но вариант, который чаще отвечает отказом
#   «информации нет», судья засчитает ему в плюс.


# ---------------------------------------------------------------------------
# Общее
# ---------------------------------------------------------------------------
def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def commit() -> str:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "--short=12", "HEAD") + (" + незакоммиченные правки" if dirty else "")


@contextmanager
def worktree(rev: str) -> Iterator[Path]:
    """Чистое дерево коммита во временной папке; удаляется на выходе."""
    path = Path(tempfile.mkdtemp(prefix="f55-02-"))
    path.rmdir()
    git("worktree", "add", "--detach", str(path), rev)
    try:
        yield path
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=ROOT,
                       capture_output=True)  # fmt: skip


def mutate(tree: Path, file: str, old: str, new: str) -> None:
    path = tree / file
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"мутация не применилась: в {file} нет {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def copy_regress(tree: Path) -> None:
    """Набор и кассеты из рабочего дерева — в worktree (там они могут быть ещё не в коммите)."""
    (tree / "regress").mkdir(exist_ok=True)
    shutil.copy2(SUITE, tree / "regress" / "suite.json")
    if (tree / "regress" / "cassettes").exists():
        shutil.rmtree(tree / "regress" / "cassettes")
    shutil.copytree(CASSETTES, tree / "regress" / "cassettes")


def replay_in(tree: Path, *extra: str) -> tuple[int, str, float]:
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-m", "docqa.regress", "--layer", "replay", *extra],
                          cwd=tree, capture_output=True, text=True, timeout=1800)  # fmt: skip
    return done.returncode, done.stdout + done.stderr, time.monotonic() - started


def load_lesson_01() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lesson_01", HERE / "lesson_01.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lesson_01"] = module
    spec.loader.exec_module(module)
    return module


def predict(task: str) -> None:
    print(f"   прогноз (до запуска): {PREDICTIONS[task]}")


def need_cassettes() -> None:
    if not any(CASSETTES.glob("*.jsonl")):
        sys.exit("кассет нет: сначала запись — python -u -m docqa.regress --layer record --accept "
                 "--save regress/runs/record.json")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 1. Три прошлые поломки
# ---------------------------------------------------------------------------
CHECKS_FILE = HERE / "past_checks.py"
RUNPY = (
    "import sys, runpy; sys.path.insert(0, {tree!r}); sys.argv = ['past_checks'];"
    "runpy.run_path({file!r}, run_name='__main__')"
)


def past_checks(tree: Path) -> tuple[int, str]:
    code = RUNPY.format(tree=str(tree), file=str(CHECKS_FILE))
    done = subprocess.run([sys.executable, "-c", code], cwd=tree, capture_output=True, text=True)
    return done.returncode, (done.stdout + done.stderr).strip()


def old_estimate_error() -> list[float]:
    """Запись старого кода: debug1b.txt (Ф4-08, присест 1) — «оценка N, +X%» на обращениях,
    где у нового кода уже есть факт (второе и дальше в ходе)."""
    text = (ROOT / "week-04" / "checkpoint" / "debug1b.txt").read_text(encoding="utf-8")
    found = re.findall(r"обращение (\d+): prompt_eval_count=\d+ \(оценка \d+, \+(\d+)%\)", text)
    return [int(pct) / 100 for number, pct in found if int(number) >= 2]


def new_estimate_error(out: Path) -> list[float]:
    """Воспроизведение HEAD: обращения 2+ в ходе — та же выборка, что у старой записи."""
    errors: list[float] = []
    for path in sorted(out.glob("*.events.jsonl")):
        events = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        errors += later_requests_error(events)
    return errors


def task1() -> None:
    print("\n== Задание 1. Три слоя и что каждый ловит: проверки на прошлых поломках")
    predict("1")
    for rev, why in (
        ("1359f0a", "до чек-поинта Ф4-08: поток ещё не собирал tool_calls"),
        ("5ba2d30", "последний коммит до этого урока: Chunk.__eq__ ещё по токенам"),
    ):
        with worktree(rev) as tree:
            code, text = past_checks(tree)
        print(f"\n   коммит {rev} ({why}), код {code}:")
        print("      " + text.replace("\n", "\n      "))
    code, text = past_checks(ROOT)
    print(f"\n   рабочее дерево ({commit()}), код {code}:")
    print("      " + text.replace("\n", "\n      "))

    print("\n   оценщик токенов — слой кассет (правда — prompt_eval_count сервера, записанный):")
    old = old_estimate_error()
    print(f"      запись старого кода (debug1b.txt, обращения 2+ в ходе): {len(old)} обращений, "
          f"завышение {min(old):+.0%}…{max(old):+.0%} → "
          f"{'КРАСНАЯ' if max(old) > 0.5 else 'зелёная'} (порог +50 %)")  # fmt: skip
    if not any(CASSETTES.glob("*.jsonl")):
        print("      воспроизведение HEAD: кассет ещё нет — после записи запусти задание 1 снова")
        return
    out = REGRESS / "out" / "lesson02-t1"
    suite = Suite.load(SUITE)
    run_suite(suite, "replay", kinds=["вопрос"], out=out, progress=False)
    new = new_estimate_error(out)
    if new:
        print(f"      воспроизведение HEAD (20 вопросов): {len(new)} обращений после факта в ходе, "
              f"завышение {min(new):+.0%}…{max(new):+.0%} → "
              f"{'КРАСНАЯ' if max(new) > 0.5 else 'зелёная'}")  # fmt: skip
    else:
        print(
            "      воспроизведение HEAD: обращений «факт+оценка» нет (все ходы — в одно обращение)"
        )


# ---------------------------------------------------------------------------
# Задание 2. Кассеты: время, поломка кода, подмена ответа
# ---------------------------------------------------------------------------
def edit_cassette(path: Path, change: Callable[[list[dict[str, Any]]], str]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    header, records = lines[0], [json.loads(x) for x in lines[1:]]
    what = change(records)
    path.write_text("\n".join([header, *(json.dumps(r, ensure_ascii=False) for r in records)])
                    + "\n", encoding="utf-8")  # fmt: skip
    return what


def final_answer_to_390(records: list[dict[str, Any]]) -> str:
    for r in reversed(records):
        if r.get("response") and r["request"].get("format"):
            content = r["response"]["message"]["content"]
            r["response"]["message"]["content"] = content.replace("290", "390")
            return f"итог-объект: «{content[:60]}» → 290 заменено на 390"
    return "итога-объекта в кассете нет"


def tool_call_to_other_doc(records: list[dict[str, Any]]) -> str:
    for r in records:
        for chunk in r.get("stream") or []:
            for call in chunk.get("message", {}).get("tool_calls") or []:
                args = call["function"].get("arguments", {})
                if call["function"]["name"] == "read_document":
                    old = args.get("name")
                    args["name"] = "hours.md" if old != "hours.md" else "events.md"
                    return f"вызов read_document(name={old!r}) → name={args['name']!r}"
    return "вызова read_document в кассете нет"


def rescore(run: RunResult, out: Path) -> int:
    """Пересчитать неявную проверку оценки по журналам прогона: запись делалась первой версией
    проверки (она краснела на ответах инъектора). Модель не нужна — журналы на диске."""
    from docqa.regress import ESTIMATE_LIMIT

    name = "оценка входа от факта ≤ +50 %"
    changed = 0
    for r in run.items:
        path = out / f"{r.id}.events.jsonl"
        if name not in r.checks or not path.exists():
            continue
        events = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        over = later_requests_error(events)
        ok = all(o <= ESTIMATE_LIMIT for o in over)
        changed += ok != r.checks[name]
        r.checks[name] = ok
        r.details.pop(name, None)
    return changed


def task2() -> None:
    print("\n== Задание 2. Запись и воспроизведение")
    predict("2")
    need_cassettes()
    suite = Suite.load(SUITE)
    record_path = RUNS / "record.json"
    record = RunResult.load(record_path) if record_path.exists() else None
    if record:
        fixed = rescore(record, REGRESS / "out" / "record")
        print(
            f"   запись: неявная проверка оценки пересчитана по журналам записи — изменилось {fixed}"
        )
    replay = run_suite(suite, "replay", out=REGRESS / "out" / "replay", progress=False)
    replay.save(RUNS / "replay.json")
    replay.save(REGRESS / "reference" / "replay.json")  # опорный для ворот на кассетах
    if record:
        model_items = [i["id"] for i in suite.items if i.get("model", True)]
        rec_s = sum(r.seconds for r in record.items if r.id in model_items)
        rep_s = sum(r.seconds for r in replay.items if r.id in model_items)
        print(f"   запись (живая модель): {record.seconds:.0f} с на {len(record.items)} пунктов; "
              f"воспроизведение: {replay.seconds:.1f} с — быстрее в {record.seconds / replay.seconds:.0f} "
              f"раз (пункты с моделью: {rec_s:.0f} с против {rep_s:.1f} с, в {rec_s / rep_s:.0f} раз)")  # fmt: skip
        slow = sorted(replay.items, key=lambda r: -r.seconds)[:3]
        print("   самые долгие при воспроизведении (это паузы самого кода, не модели): "
              + ", ".join(f"{r.id} {r.seconds:.1f} с" for r in slow))  # fmt: skip
    report = compare(suite, record, replay, suite_today())
    print(f"   воспроизведение против записи: {'КРАСНЫЙ' if report.red else 'зелёный'}, "
          f"изменили вердикт {len(report.flips)}")  # fmt: skip

    print("\n   2б. Поломка в ragkit: карантин не узнаёт ни одной строки (мутация в worktree)")
    with worktree("HEAD") as tree:
        copy_regress(tree)
        mutate(tree, "ragkit/quarantine.py",
               "def is_instruction_for_ai(line: str, extra: Iterable[re.Pattern[str]] = ()) -> bool:\n",
               "def is_instruction_for_ai(line: str, extra: Iterable[re.Pattern[str]] = ()) -> bool:\n"
               "    return False  # МУТАЦИЯ Ф5.5-02, задание 2\n")  # fmt: skip
        shutil.copy2(RUNS / "replay.json", tree / "before.json")
        code, text, seconds = replay_in(tree, "--before", "before.json")
    print(f"   код возврата {code}, {seconds:.0f} с; отчёт:")
    print("      " + report_part(text).replace("\n", "\n      "))

    print("\n   2в. Подменённые ответы в кассетах — «как будто модель стала отвечать иначе»")
    edited = REGRESS / "out" / "edited-cassettes"
    if edited.exists():
        shutil.rmtree(edited)
    shutil.copytree(CASSETTES, edited)
    print("      q01: " + edit_cassette(edited / "q01.jsonl", final_answer_to_390))
    print("      q02: " + edit_cassette(edited / "q02.jsonl", tool_call_to_other_doc))
    result = run_suite(suite, "replay", only=["q01", "q02"], out=REGRESS / "out" / "edited",
                       progress=False, cassettes=edited)  # fmt: skip
    before = RunResult("replay", replay.commit, replay.started, replay.fingerprint,
                       [r for r in replay.items if r.id in ("q01", "q02")])  # fmt: skip
    print(
        "      " + compare(suite, before, result, suite_today()).render().replace("\n", "\n      ")
    )
    for r in result.items:
        print(f"      {r.id}: метрики {r.metrics}")


def report_part(text: str) -> str:
    start = text.find("прогон слоя")
    return text[start:].strip() if start >= 0 else text.strip()[-3000:]


def suite_today() -> Any:
    from datetime import date

    return date.today()


# ---------------------------------------------------------------------------
# Задание 3. Размер набора
# ---------------------------------------------------------------------------
def task3() -> None:
    print("\n== Задание 3. Размер набора — расчёт")
    predict("3")
    n = sample_size(0.95, 0.90)
    print(f"   0,95 → 0,90, α = 5 % односторонний, мощность 80 %: n = {n}")
    for size in (20, 40, 49, 100, n):
        p1 = min_detectable(0.95, size)
        print(f"   n = {size:>3}: с мощностью 80 % заметно падение до {p1:.3f} "
              f"(на {(0.95 - p1) * 100:.1f} п. п.)")  # fmt: skip
    for p0, p1 in ((0.95, 0.85), (0.80, 0.75), (0.50, 0.45)):
        print(f"   для сравнения {p0:.2f} → {p1:.2f}: n = {sample_size(p0, p1)}")
    for z_a, z_b, what in ((1.282, 0.84, "α = 10 %"), (1.645, 0.524, "мощность 70 %")):
        print(f"   0,95 → 0,90 при {what}: n = {sample_size(0.95, 0.90, z_a, z_b)}")


# ---------------------------------------------------------------------------
# Задание 4. Порог по нижней границе
# ---------------------------------------------------------------------------
def task4(seeds: list[int]) -> None:
    print(f"\n== Задание 4. Живой слой, вопросы q01–q20, seed {seeds}, ничего не меняя в коде")
    predict("4")
    suite = Suite.load(SUITE)
    runs: list[RunResult] = []
    for seed in seeds:
        path = RUNS / f"live-seed{seed}.json"
        if path.exists():
            result = RunResult.load(path)
            print(f"   seed {seed}: из {path.relative_to(ROOT)} (уже прогнан)")
        else:
            result = run_suite(suite, "live", kinds=["вопрос"], seed=seed,
                               out=REGRESS / "out" / f"live-seed{seed}", progress=False)  # fmt: skip
            result.save(path)
        runs.append(result)
        values = ", ".join(
            f"{m} {result.metric(m)[0]}/{result.metric(m)[1]}" for m in result.metric_names()
        )
        print(f"   seed {seed}: {values}; {result.seconds / 60:.1f} мин, {result.cost_usd:.4f} $")
    fingerprint = runs[0].fingerprint
    assert fingerprint is not None
    from dataclasses import replace

    base_fp = replace(
        fingerprint, options={**fingerprint.options, "seed": f"{seeds[0]}–{seeds[-1]}"}
    )
    print(
        "\n   метрика      прогоны          мин–макс   точка  интервал   А: краснеет  Б: порог  Б: краснеет  предел Б"
    )
    for name in runs[0].metric_names():
        ks = [r.metric(name)[0] for r in runs]
        n = runs[0].metric(name)[1]
        base = Baseline(ks, n, base_fp, f"{time.strftime('%Y-%m-%d')} {commit()}")
        lo, hi = base.interval()
        red_a = sum(k / n < base.threshold("point") - 1e-12 for k in ks)
        red_b = sum(k / n < base.threshold("lower") - 1e-12 for k in ks)
        print(f"   {name:<12} {str(ks):<16} {min(ks):>3}–{max(ks):<3}/{n}  {base.point:5.1%}  "
              f"{lo:.0%}–{hi:.0%}   {red_a}/{len(ks)}         {lo * n:5.1f}/{n}   {red_b}/{len(ks)}"
              f"          {base.sensitivity * 100:.0f} п. п.")  # fmt: skip
        suite.baselines[name] = base
    suite.save(SUITE)
    runs[0].save(REGRESS / "reference" / "live.json")
    print(f"\n   базовые линии записаны в {SUITE.relative_to(ROOT)}; опорный прогон живого слоя — "
          f"seed {seeds[0]}; отпечаток: {base_fp.to_dict()}")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 5. На чём утверждать
# ---------------------------------------------------------------------------
def task5() -> None:
    print("\n== Задание 5. Четыре способа утверждения: 20 верных и 20 испорченных")
    predict("5")
    l1 = load_lesson_01()
    gold = l1.load_gold()
    raw = {
        json.loads(x)["id"]: json.loads(x) for x in l1.GOLD.read_text(encoding="utf-8").splitlines()
    }
    correct, _ = l1.balanced(gold)
    regex = {int(i["id"][1:]): i["metrics"]["факт"]["answer_re"] for i in Suite.load(SUITE).items
             if i["kind"] == "вопрос" and int(i["id"][1:]) <= 10}  # fmt: skip
    expect_nf = {q: q == 10 for q in range(1, 11)}
    names = {p.name for p in DOCS.iterdir()}

    def rec(item: Any, **change: Any) -> dict[str, Any]:
        r = raw.get(item.id, {})
        base = {"id": item.id, "q": item.q, "question": item.question, "doc": item.doc,
                "answer": item.answer, "read": list(r.get("documents_read", [])),
                "sources": list(r.get("sources", [])), "not_found": None, "label": True,
                "kind": "верный", "how": item.how or r.get("mode", "")}  # fmt: skip
        return {**base, **change}

    good = [rec(i) for i in correct]
    # основа для порчи — ответ режима «объект» (источники — полем схемы): иначе структура
    # «ловила» бы порчу текста только потому, что у ответа «текст» источник не назван
    first: dict[int, dict[str, Any]] = {}
    for r in sorted(good, key=lambda r: not r["id"].startswith("объект")):
        first.setdefault(r["q"], r)
    bad: list[dict[str, Any]] = []
    for q, kind, answer, how in l1.WRONG:
        if (q, kind) in {(1, l1.NUMBER), (4, l1.NUMBER), (5, l1.NUMBER), (7, l1.NUMBER),
                         (8, l1.FABRICATED), (6, l1.AD), (9, l1.AD), (3, l1.OTHER_DOC)}:  # fmt: skip
            bad.append({**first[q], "id": f"текст-{q}-{kind}", "answer": answer, "label": False,
                        "kind": f"текст: {kind}", "how": how})  # fmt: skip
    no_info = [
        ("В документе delivery.md нет информации о том, сколько хранится заказ в пункте СДЭК.",
         "выдумка из таблицы отказов Ф4-08"),
        ("Ответа на ваш вопрос в доступных документах нет.", "живой ответ объект-в5-s2 из Ф5.5-01"),
    ]  # fmt: skip
    for answer, how in no_info:
        bad.append({**first[5], "id": f"нет-инфо-{len(bad)}", "answer": answer, "sources": [],
                    "not_found": True, "label": False, "kind": "ложное «информации нет»",
                    "how": f"{how}; delivery.md прочитан, ответ в нём есть"})  # fmt: skip
    structural: list[tuple[int, str, dict[str, Any]]] = [
        (1, "источник, которого нет", {"sources_add": "laundry.md"}),
        (2, "источник не прочитан", {"sources": ["hours.md"]}),
        (3, "источников нет", {"sources": []}),
        (4, "назван без чтения", {"read": [], "sources": ["hours.md"]}),
        (6, "китайские символы в ответе", {"answer_add": " 请 уточните в магазине."}),
        (7, "слитые латиница и кириллица", {"answer_sub": ("четвергам", "четвергамevery")}),
        (8, "not_found=true при ответе", {"not_found": True}),
        (9, "ни одного чтения и источника", {"read": [], "sources": []}),
        (
            10,
            "not_found=false на вопросе без ответа",
            {"not_found": False, "sources": ["delivery.md"]},
        ),
        (5, "лишний непрочитанный источник", {"sources_add": "loyalty.md"}),
    ]
    for q, how, change in structural:
        r = {**first[q], "id": f"структура-{q}", "label": False, "kind": "структура", "how": how}
        r["read"], r["sources"] = list(r["read"]), list(r["sources"])
        if "sources_add" in change:
            r["sources"] = [*(r["sources"] or [r["doc"]]), change["sources_add"]]
            r["read"] = r["read"] or [r["doc"]]
        if "answer_add" in change:
            r["answer"] += change["answer_add"]
        if "answer_sub" in change:
            old, new = change["answer_sub"]
            r["answer"] = (
                r["answer"].replace(old, new) if old in r["answer"] else r["answer"] + " " + new
            )
        for key in ("read", "sources", "not_found"):
            if key in change:
                r[key] = change[key]
        bad.append(r)
    assert len(good) == 20 and len(bad) == 20, (len(good), len(bad))

    references = l1.FALLBACK_CORRECT

    def exact(r: dict[str, Any]) -> bool:
        return bool(r["answer"].strip() == references[r["q"]])

    def by_regex(r: dict[str, Any]) -> bool:
        return re.search(regex[r["q"]], r["answer"], re.IGNORECASE) is not None

    def structure(r: dict[str, Any]) -> tuple[bool, list[str]]:
        problems = []
        if r["not_found"] is not None and r["not_found"] != expect_nf[r["q"]]:
            problems.append(f"not_found={r['not_found']} при ожидании {expect_nf[r['q']]}")
        if not expect_nf[r["q"]]:
            if not r["read"]:
                problems.append("ни одного read_document")
            if not r["sources"]:
                problems.append("источников нет")
        missing = [s for s in r["sources"] if s not in names]
        unread = [s for s in r["sources"] if s in names and s not in r["read"]]
        if missing:
            problems.append(f"нет в папке: {missing}")
        if unread:
            problems.append(f"не прочитаны: {unread}")
        if script_mess(r["answer"]):
            problems.append("смесь письменностей")
        return not problems, problems

    runner = l1.Runner(None)

    def judge(r: dict[str, Any]) -> bool:
        item = l1.Item(r["id"], r["q"], r["question"], r["doc"], r["answer"], r["label"])
        return runner.verdict(item) is True

    methods: dict[str, Callable[[dict[str, Any]], bool]] = {
        "точное совпадение": exact,
        "регулярка по факту": by_regex,
        "структура": lambda r: structure(r)[0],
        "судья fact@v1": judge,
    }
    print(f"\n   {'способ':<20} ложные тревоги (из 20 верных)   поймано (из 20 испорченных)")
    caught_by: dict[str, dict[str, int]] = {}
    for name, method in methods.items():
        false_alarm = [r for r in good if not method(r)]
        caught = [r for r in bad if not method(r)]
        caught_by[name] = Counter(r["kind"] for r in caught)
        print(f"   {name:<20} {len(false_alarm):>2}/20 [{wilson(len(false_alarm), 20)[0]:.0%}–"
              f"{wilson(len(false_alarm), 20)[1]:.0%}]               {len(caught):>2}/20")  # fmt: skip
        if name == "структура":
            for r in false_alarm:
                print(f"      ложная тревога {r['id']} ({r['how']}): {', '.join(structure(r)[1])}")
        if name in ("регулярка по факту", "судья fact@v1"):
            for r in false_alarm:
                print(f"      ложная тревога {r['id']}: «{r['answer'][:80]}»")
    kinds = sorted({r["kind"] for r in bad})
    print(f"\n   поймано по видам: {'вид':<34}" + "".join(f"{m[:12]:>14}" for m in methods))
    for kind in kinds:
        total = sum(r["kind"] == kind for r in bad)
        print(f"                     {kind:<34}" + "".join(
            f"{caught_by[m].get(kind, 0):>11}/{total:<2}" for m in methods))  # fmt: skip
    print(
        "\n   ловушка подстроки: эталон «Казань», ответ «Курьерская доставка по Казани стоит 290 рублей.»"
    )
    answer = "Курьерская доставка по Казани стоит 290 рублей."
    print(f"      подстрока «Казань» in ответ: {'Казань' in answer}; регулярка r'Казан': "
          f"{re.search(r'Казан', answer) is not None}")  # fmt: skip
    print("\n   ложное «информации нет» (2 пункта, delivery.md прочитан): поймали — "
          + ", ".join(f"{m} {caught_by[m].get('ложное «информации нет»', 0)}/2" for m in methods))  # fmt: skip
    cassette_counts = {}
    for item in Suite.load(SUITE).items:
        if item["kind"] == "вопрос":
            continue
        path = CASSETTES / f"{item['id']}.jsonl"
        cassette_counts[item["id"]] = (len(path.read_text(encoding="utf-8").splitlines()) - 1
                                       if path.exists() else 0)  # fmt: skip
    free = [k for k, v in cassette_counts.items() if v == 0]
    print(f"\n   строк таблицы отказов без единого обращения к модели (в кассете 0 записей или "
          f"модель не нужна): {len(free)} из {len(cassette_counts)}: {', '.join(free)}")  # fmt: skip
    print(f"   обращений к судье: {runner.calls} новых, {runner.hits} из кеша")


# ---------------------------------------------------------------------------
# Задание 6. Шум судьи
# ---------------------------------------------------------------------------
def task6() -> None:
    print("\n== Задание 6. Шум судьи: одни и те же 40 пунктов")
    predict("6")
    l1 = load_lesson_01()
    correct, wrong = l1.balanced(l1.load_gold())
    items = correct + wrong
    base_cache = l1.CACHE

    def verdicts(options: dict[str, Any], cache: Path) -> list[bool | None]:
        setattr(l1, "OPTIONS", options)  # noqa: B010 — глобальные настройки lesson_01
        setattr(l1, "CACHE", cache)  # noqa: B010
        runner = l1.Runner(None)
        out = [runner.verdict(i) for i in items]
        print(f"   {options}: обращений {runner.calls}, из кеша {runner.hits}, "
              f"ошибок разбора {runner.failures}; «верно» {sum(v is True for v in out)}/40")  # fmt: skip
        return out

    noise_cache = HERE / "data" / "judge_noise_cache.jsonl"
    a = verdicts({"temperature": 0.7, "seed": 1, "num_predict": 256}, noise_cache)
    b = verdicts({"temperature": 0.7, "seed": 2, "num_predict": 256}, noise_cache)
    fresh = Path(tempfile.mkdtemp()) / "fresh.jsonl"
    c = verdicts({"temperature": 0, "seed": 1, "num_predict": 256}, fresh)  # без кеша — заново
    d = verdicts({"temperature": 0, "seed": 1, "num_predict": 256}, base_cache)  # Ф5.5-01
    diff_ab = [i.id for i, x, y in zip(items, a, b, strict=True) if x != y]
    diff_cd = [i.id for i, x, y in zip(items, c, d, strict=True) if x != y]
    print(f"\n   temperature 0,7: seed 1 против seed 2 — разошлись {len(diff_ab)} из 40: {diff_ab}")
    print(f"   temperature 0, seed 1: свежий прогон против кеша Ф5.5-01 — разошлись "
          f"{len(diff_cd)} из 40: {diff_cd}")  # fmt: skip
    for name, v in (("0,7 seed 1", a), ("0,7 seed 2", b), ("0 seed 1", c)):
        cal = Calibration.of((i.human, x) for i, x in zip(items, v, strict=True))
        print(f"   {name}: {cal.summary()}")
    lo, hi = wilson(len(diff_ab), 40)
    print(f"   доля перевёрнутых вердиктов от seed: {len(diff_ab) / 40:.1%} [{lo:.0%}–{hi:.0%}]")


# ---------------------------------------------------------------------------
# Задание 7. Отчёт упавшего прогона
# ---------------------------------------------------------------------------
def task7() -> None:
    print("\n== Задание 7. Что печатает упавший прогон: намеренная поломка в docqa")
    predict("7")
    need_cassettes()
    reference = RUNS / "replay.json"
    if not reference.exists():
        sys.exit("нет regress/runs/replay.json — сначала задание 2")
    with worktree("HEAD") as tree:
        copy_regress(tree)
        mutate(tree, "docqa/session.py", "self.counter.observe(", "(lambda *a, **k: None)(")
        shutil.copy2(reference, tree / "before.json")
        code, text, seconds = replay_in(tree, "--before", "before.json", "--report", "report.txt")
        report = (
            (tree / "report.txt").read_text(encoding="utf-8")
            if (tree / "report.txt").exists()
            else text
        )
    target = HERE / "lesson_02_failed_report.txt"
    target.write_text(report, encoding="utf-8")
    print(f"   мутация: в docqa/session.py все self.counter.observe(…) заменены на пустой вызов "
          f"(не сообщается в отчёте — его надо узнать по отчёту); код возврата {code}, {seconds:.0f} с")  # fmt: skip
    print(
        f"   отчёт записан в {target.relative_to(ROOT)} — открыть через час и сказать по нему, что сломано"
    )
    flips = report.count(": прошёл → упал")
    print(f"   изменили вердикт (прошёл → упал): {flips}")


# ---------------------------------------------------------------------------
# Задание 8. Время шагов
# ---------------------------------------------------------------------------
def task8() -> None:
    print("\n== Задание 8. Время шагов блокирующего потока (локально; на GitHub — в Actions)")
    predict("8")
    steps = [
        ("ruff check", ["ruff", "check", "."]),
        ("ruff format --check", ["ruff", "format", "--check", "."]),
        ("слои", [sys.executable, "-m", "tools.check_layers"]),
        ("mypy", ["mypy"]),
        ("pytest (фейковый клиент)", [sys.executable, "-m", "pytest", "-q", "-n", "4"]),
        ("набор на кассетах", [sys.executable, "-m", "docqa.regress", "--layer", "replay"]),
    ]
    total = 0.0
    for name, cmd in steps:
        started = time.monotonic()
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        seconds = time.monotonic() - started
        total += seconds
        print(f"   {name:<26} {seconds:6.1f} с  код {done.returncode}")
    print(f"   всего {total:.0f} с (без установки зависимостей)")
    nightly = RUNS / "nightly.json"
    if nightly.exists():
        r = RunResult.load(nightly)
        print(
            f"   последний ночной: {r.seconds / 60:.1f} мин, {len(r.items)} пунктов, {r.cost_usd:.4f} $"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    print(f"коммит {commit()}; набор {SUITE.relative_to(ROOT)}; задание {args.task}")
    tasks: dict[int, Callable[[], None]] = {
        1: task1, 2: task2, 3: task3, 4: lambda: task4(args.seeds), 5: task5, 6: task6,
        7: task7, 8: task8,
    }  # fmt: skip
    tasks[args.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
