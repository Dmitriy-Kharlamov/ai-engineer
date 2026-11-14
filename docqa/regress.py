"""Регрессионный набор docqa (Ф5.5-02): пункты из regress/suite.json, три слоя.

    python -m docqa.regress --layer replay                 # кассеты: ворота на каждый push
    python -m docqa.regress --layer record                 # живая модель, запись кассет
    python -m docqa.regress --layer live --seed 3          # живая модель, ночной прогон
    python -m docqa.regress --layer replay --only 05 q01   # выбранные пункты

Каждый пункт — запуск `python -m docqa …` в подпроцессе, как у пользователя, с журналом
событий. Ожидания (expect) — про код: какие события, какой код возврата, сколько файлов;
текст ответа в них не участвует. Метрики (metrics) — про модель: факт в ответе, источник,
честное «нет». В слое кассет метрики — копия записи, а ворота — только ожидания кода.

Каждому пункту добавляются два ожидания для всех:
* «журнал без сырых ошибок» — ragkit.regress.scan_secrets по журналу событий;
* «оценка входа от факта ≤ +50 %» — обращения хода после первого настоящего ответа (у кода
  уже есть факт сервера) не завышены больше чем наполовину (поломка Ф4-08, присест 1: чистая
  оценка — +102…157 %).

Отчёт сравнивает прогон с опорным (regress/reference/<слой>.json): что изменило вердикт,
метрики против порогов, отпечаток среды, карантин. Код возврата 1 — ворота красные.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from docqa.session import Settings
from ragkit.quality import script_mess
from ragkit.regress import (
    Fingerprint,
    ItemResult,
    RunBudget,
    RunBudgetExceeded,
    RunResult,
    Suite,
    compare,
    probe_fingerprint,
    scan_secrets,
    why,
)

ROOT = Path(__file__).resolve().parent.parent
REGRESS = ROOT / "regress"
SUITE = REGRESS / "suite.json"
OUT = REGRESS / "out"
CASSETTES = REGRESS / "cassettes"
ESTIMATE_LIMIT = 0.5  # «факт+оценка» не выше факта больше чем на 50 %


# ---------------------------------------------------------------------------
# Подготовка пункта
# ---------------------------------------------------------------------------
def rel(path: Path) -> str:
    """Путь от корня репозитория: абсолютный попал бы в журнал с именем пользователя."""
    return str(path.resolve().relative_to(ROOT))


def prepare(suite: Suite, item: Mapping[str, Any]) -> dict[str, str]:
    """Папки пункта: копия документов с добавками (fixture), пустая папка заметок."""
    places: dict[str, str] = {}
    name = item.get("fixture")
    if name:
        spec = suite.extra["fixtures"][name]
        folder = OUT / "fixtures" / name
        if folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(ROOT / spec["copy"], folder)
        for file, text in spec.get("add", {}).items():
            (folder / file).write_text(text, encoding="utf-8")
        places["@fixture"] = rel(folder)
    notes = OUT / "notes" / item["id"]
    if notes.exists():
        shutil.rmtree(notes)
    places["@notes"] = rel(notes)
    return places


def argv_for(
    suite: Suite,
    item: Mapping[str, Any],
    layer: str,
    places: Mapping[str, str],
    *,
    seed: int | None,
    budget_usd: float | None,
    cassettes: Path | None = None,
) -> list[str]:
    """Кассеты — продукт как есть (настройки docqa по умолчанию): запись и воспроизведение
    обязаны слать одинаковые запросы. Живой слой — temperature из набора и seed прогона."""
    out = [places.get(a, a) for a in item["argv"]]
    uses_model = item.get("model", True)
    if uses_model and layer in ("record", "replay"):
        cassette = (cassettes or CASSETTES) / f"{item['id']}.jsonl"
        out += ["--cassette", rel(cassette), "--cassette-mode", layer]
    if uses_model and layer == "live":
        temperature = suite.extra.get("options", {}).get("temperature")
        if temperature is not None and "--temperature" not in out:
            out += ["--temperature", str(temperature)]
        if seed is not None and "--seed" not in out:
            out += ["--seed", str(seed)]
    if budget_usd is not None and "--session-usd" not in out:
        out += ["--session-usd", f"{budget_usd:.6f}"]
    return out


# ---------------------------------------------------------------------------
# Ожидания и метрики
# ---------------------------------------------------------------------------
def matches(event: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    for field, want in where.items():
        have = event.get(field)
        if want == []:
            if have != []:
                return False
        elif isinstance(have, list):
            wanted = want if isinstance(want, list) else [want]
            if not all(w in have for w in wanted):
                return False
        elif have != want:
            return False
    return True


def last_answer(transcript: str) -> str:
    """Текст последнего «модель> …» до строки кода или следующего вопроса."""
    parts = transcript.split("модель> ")
    if len(parts) < 2:
        return ""
    tail = parts[-1]
    end = re.search(r"\n  \[|\nвы> ", tail)
    return (tail[: end.start()] if end else tail).strip()


class Observed:
    """Что видно после запуска: код возврата, терминал, журнал событий, папка заметок."""

    def __init__(self, code: int, stdout: str, events: list[dict[str, Any]], log: str,
                 notes: Path) -> None:  # fmt: skip
        self.code = code
        self.stdout = stdout
        self.events = events
        self.log = log
        self.notes = notes
        self.answer = last_answer(stdout)

    def of(self, name: str, where: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == name and matches(e, where or {})]

    def files(self) -> int:
        return len(list(self.notes.glob("*"))) if self.notes.exists() else 0

    def last(self, name: str) -> dict[str, Any] | None:
        found = self.of(name)
        return found[-1] if found else None


def check(spec: Mapping[str, Any], seen: Observed) -> bool:
    """Одно ожидание кода. Виды: exit, event (+where, min, max), stdout_re, stdout_not_re, files."""
    if "exit" in spec:
        return seen.code == int(spec["exit"])
    if "event" in spec:
        count = len(seen.of(spec["event"], spec.get("where")))
        low = int(spec.get("min", 1))
        high = spec.get("max")
        return count >= low and (high is None or count <= int(high))
    if "stdout_re" in spec:
        return re.search(spec["stdout_re"], seen.stdout) is not None
    if "stdout_not_re" in spec:
        return re.search(spec["stdout_not_re"], seen.stdout) is None
    if "files" in spec:
        return seen.files() == int(spec["files"])
    raise ValueError(f"неизвестное ожидание: {dict(spec)}")


def metric(spec: Mapping[str, Any], seen: Observed) -> bool:
    """Метрика модели: все условия спецификации разом."""
    answer = seen.last("answer") or {}
    turn = seen.last("turn_end") or {}
    for key, want in spec.items():
        if key == "answer_re":
            ok = re.search(want, seen.answer, re.IGNORECASE) is not None
        elif key == "source":
            ok = want in (answer.get("sources") or [])
        elif key == "read":
            ok = want in (turn.get("documents_read") or [])
        elif key == "not_found":
            ok = answer.get("not_found") is want
        elif key == "clean":
            ok = bool(seen.answer) and not script_mess(seen.answer)
        elif key == "files":
            ok = seen.files() == want
        else:
            raise ValueError(f"неизвестная метрика: {key}")
        if not ok:
            return False
    return True


def later_requests_error(events: Sequence[Mapping[str, Any]]) -> list[float]:
    """Завышение оценки входа там, где у кода уже был факт сервера: на обращениях хода после
    первого НАСТОЯЩЕГО ответа (prompt_eval_count > 0).

    Не «помеченные факт+оценка»: если код перестал брать факт, пометок не будет вовсе, и
    проверка по пометке молча пройдёт (так вышло в задании 7 на первой версии). И не «все
    обращения 2+»: ответ, подставленный инъектором отказов, факта не несёт (счёт 0), и
    следующее обращение законно идёт по чистой оценке — так первая живая запись покраснела на
    12 строках таблицы отказов, где ошибалась проверка, а не код (+112…165 % после подмены,
    +175 200 % — сам подставленный ответ со счётом 0).
    """
    real: set[Any] = set()  # ходы, где уже был настоящий ответ
    out: list[float] = []
    for e in events:
        if e["event"] != "response":
            continue
        count = e.get("prompt_eval_count") or 0
        if count <= 0:
            continue  # ответ инъектора: факта нет, оценивать нечего
        if e.get("turn") in real and isinstance(e.get("request"), int):
            out.append(float(e.get("estimate_error", 0.0)))
        real.add(e.get("turn"))
    return out


def implicit(seen: Observed, uses_model: bool) -> tuple[dict[str, bool], dict[str, str]]:
    """Ожидания для всех пунктов и что увидено (печатается рядом с невыполненным)."""
    found = scan_secrets(seen.log)
    checks = {"журнал без сырых ошибок": not found}
    details = {"журнал без сырых ошибок": found[0][:160]} if found else {}
    if uses_model:
        over = later_requests_error(seen.events)
        name = "оценка входа от факта ≤ +50 %"
        checks[name] = all(o <= ESTIMATE_LIMIT for o in over)
        if over:
            details[name] = f"после факта в ходе: {min(over):+.0%}…{max(over):+.0%}"
    return checks, details


def spent(seen: Observed) -> float:
    values = [
        e.get("spent_usd", 0.0) for e in seen.events if e["event"] in ("turn_end", "turn_failed")
    ]
    return max(values, default=0.0)


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------
def run_item(
    suite: Suite,
    item: Mapping[str, Any],
    layer: str,
    *,
    seed: int | None,
    budget_usd: float | None,
    out: Path,
    cassettes: Path | None = None,
) -> ItemResult:
    uses_model = item.get("model", True)
    cassette = (cassettes or CASSETTES) / f"{item['id']}.jsonl"
    if layer == "replay" and uses_model and not cassette.exists():
        return ItemResult(item["id"], item["kind"], {}, {}, 0.0,
                          f"кассеты {rel(cassette)} нет — запиши: python -m docqa.regress "
                          f"--layer record --only {item['id']}")  # fmt: skip
    places = prepare(suite, item)
    events_path = out / f"{item['id']}.events.jsonl"
    transcript = out / f"{item['id']}.txt"
    events_path.unlink(missing_ok=True)
    argv = argv_for(
        suite, item, layer, places, seed=seed, budget_usd=budget_usd, cassettes=cassettes
    )
    argv += ["--log", rel(transcript), "--events", rel(events_path)]
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-m", "docqa", *argv], cwd=ROOT, capture_output=True,
                          input=item.get("stdin", ""), text=True, timeout=900)  # fmt: skip
    seconds = time.monotonic() - started
    log = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
    events = [json.loads(line) for line in log.splitlines()]
    seen = Observed(done.returncode, done.stdout, events, log, ROOT / places["@notes"])
    error = ""
    miss = seen.of("turn_failed", {"error": "CassetteMiss"})
    if miss:
        error = f"промах кассеты: {miss[0].get('message', '')}"
    elif done.returncode not in (0, 1) or "Traceback" in done.stderr:
        last = done.stderr.strip().splitlines()[-1:] or [""]
        error = f"docqa упал (код {done.returncode}): {last[0][:300]}"
    checks = {spec["name"]: check(spec, seen) for spec in item.get("expect", [])}
    extra, details = implicit(seen, uses_model)
    checks.update(extra)
    metrics = {name: metric(spec, seen) for name, spec in item.get("metrics", {}).items()}
    return ItemResult(item["id"], item["kind"], checks, metrics, round(seconds, 2), error,
                      round(spent(seen), 6), details)  # fmt: skip


def commit() -> str:
    try:
        git = ["git", "-C", str(ROOT)]
        head = subprocess.run([*git, "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
        status = [*git, "status", "--porcelain", "--untracked-files=no"]
        dirty = subprocess.run(status, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return head + ("+правки" if dirty else "")


def layer_fingerprint(
    suite: Suite, layer: str, seed: int | None, cassettes: Path | None = None
) -> Fingerprint | None:
    """Отпечаток среды прогона: в живых слоях — у ollama, в кассетах — из записи.

    Запись и воспроизведение — продукт как есть (настройки docqa по умолчанию), живой слой —
    temperature из набора и seed прогона.
    """
    product = Settings()
    options: dict[str, Any] = {
        "temperature": product.temperature,
        "seed": product.seed,
        "num_ctx": product.num_ctx,
        "num_predict": product.num_predict,
    }
    if layer == "live":
        options.update(suite.extra.get("options", {}))
        if seed is not None:
            options["seed"] = seed
    if layer == "replay":  # в кассетах среда — та, что была при записи
        files = sorted((cassettes or CASSETTES).glob("*.jsonl"))
        if not files:
            return None
        header = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        return Fingerprint.from_dict(header["fingerprint"])
    return probe_fingerprint(suite.extra.get("model", "qwen2.5:7b"), options)


def run_suite(
    suite: Suite,
    layer: str,
    *,
    only: Sequence[str] = (),
    kinds: Sequence[str] = (),
    seed: int | None = None,
    budget_usd: float | None = None,
    out: Path | None = None,
    progress: bool = True,
    cassettes: Path | None = None,
) -> RunResult:
    items = [
        i
        for i in suite.items
        if (not only or i["id"] in only) and (not kinds or i["kind"] in kinds)
    ]
    out = out or OUT / layer
    out.mkdir(parents=True, exist_ok=True)
    budget = RunBudget(budget_usd) if budget_usd is not None else None
    fingerprint = layer_fingerprint(suite, layer, seed, cassettes)
    started = time.monotonic()
    results = []
    for item in items:
        limit = None
        if budget is not None and layer in ("record", "live") and item.get("model", True):
            try:
                limit = budget.before(item["id"], suite.extra.get("item_min_usd", 0.02))
            except RunBudgetExceeded as error:
                results.append(ItemResult(item["id"], item["kind"], {}, {}, 0.0, str(error)))
                continue
        result = run_item(
            suite, item, layer, seed=seed, budget_usd=limit, out=out, cassettes=cassettes
        )
        if budget is not None:
            budget.add(result.cost_usd)
        results.append(result)
        if progress:
            mark = "ок" if result.passed else "УПАЛ"
            reason = "" if result.passed else f" — {why(result)}"
            print(f"   {result.id:<5} {mark:<4} {result.seconds:6.1f} с  {item['title']}{reason}",
                  flush=True)  # fmt: skip
    return RunResult(layer, commit(), datetime.now().isoformat(timespec="seconds"), fingerprint,
                     results, time.monotonic() - started)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docqa.regress", description="регрессионный набор docqa")
    parser.add_argument("--layer", choices=["replay", "record", "live"], required=True)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--only", nargs="*", default=[], help="id пунктов")
    parser.add_argument("--kind", nargs="*", default=[], help="виды пунктов, например «вопрос»")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--budget-usd", type=float, default=None, help="бюджет всего прогона")
    parser.add_argument("--save", type=Path, help="куда записать результат прогона (JSON)")
    parser.add_argument("--before", type=Path, help="с чем сравнивать (по умолчанию опорный)")
    parser.add_argument("--accept", action="store_true", help="сделать этот прогон опорным")
    parser.add_argument("--rule", choices=["lower", "point"], default="lower")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--report", type=Path, help="записать отчёт в файл")
    parser.add_argument("--cassettes", type=Path, default=None, help="папка кассет (для опытов)")
    args = parser.parse_args(argv)

    suite = Suite.load(args.suite)
    print(f"набор {suite.name} v{suite.version}: {len(suite.items)} пунктов; слой {args.layer}; "
          f"коммит {commit()}", flush=True)  # fmt: skip
    result = run_suite(suite, args.layer, only=args.only, kinds=args.kind, seed=args.seed,
                       budget_usd=args.budget_usd, cassettes=args.cassettes)  # fmt: skip
    reference = REGRESS / "reference" / f"{args.layer}.json"
    before_path = args.before or (reference if reference.exists() else None)
    before = RunResult.load(before_path) if before_path else None
    report = compare(suite, before, result, args.today, args.rule)
    text = report.render()
    print()
    print(text)
    print(f"\nпрогон: {result.seconds:.1f} с, {len(result.items)} пунктов, "
          f"цена по счётчикам {result.cost_usd:.4f} $; сравнение с "
          f"{rel(before_path) if before_path else '— (опорного прогона нет)'}")  # fmt: skip
    if args.save:
        result.save(args.save)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if args.accept:
        result.save(reference)
        print(f"опорный прогон слоя {args.layer}: {rel(reference)}")
    return 1 if report.red else 0


if __name__ == "__main__":
    raise SystemExit(main())
