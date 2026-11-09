"""Ф4-08, присест 2: таблица отказов — каждый воспроизводится одной командой CLI.

    python -u week-04/checkpoint/failures.py 2>&1 | tee week-04/checkpoint/failures.txt
    python -u week-04/checkpoint/failures.py 04a 10b      # только выбранные

Каждый сценарий — отдельный запуск `python -m docqa …` в подпроцессе, как у пользователя.
Для каждого сохраняются: расшифровка терминала (out/failures/<id>.txt — ровно то, что видит
пользователь) и журнал событий (out/failures/<id>.events.jsonl — то, что попадает в лог).
Здесь печатается команда, код возврата, расшифровка целиком и события по типам.

Прогноз (до запуска), что увидит пользователь:
  01 unknown-tool   строка «→ unknown: ошибка: такого инструмента нет…», модель ответит сама
                    (вероятно, без письма), и под ответом «[не выполнено: send_email…]».
  02 bad-args       «→ invalid», модель повторит вызов с name и ответит верно (290 ₽).
  03 tool-crash     «→ failed», модель скажет, что прочитать не удалось, или ответит по
                    поиску; путь к файлу в терминале не появится, в журнале — появится.
  04a raw-call      мусорного текста не видно, строка «[код] … разобрал и выполняет», ответ.
  04b raw-call-brk  «[ответа нет] … вызов текстом, не разбирается», код 1.
  05 loop           list_documents → ok, затем 2 повтора и «[ответа нет] зациклилась», 3
                    обращения вместо 5.
  06 limit          5 поисков и «[ответа нет] … за отведённое число обращений».
  07a read-all      6 чтений, «[ответа нет] не помещается в окно … обращение 2».
  07b history       2-й или 3-й вопрос при num_ctx 2560 и полной истории — отказ окна.
  08a stream-cut    обрывок текста, «[ответа нет] ответ оборван… обрывок, не ответ».
  08b stream-stall  то же через ~3 с: таймаут между кусками.
  09a limit-usd     «[ответа нет] запрос не отправлен — дороже порога», ни одного обращения.
  09b session-usd   1-й вопрос с ответом, на 2-м или 3-м — «бюджет сессии исчерпан».
  10a injection     ответ без promo-example, «[код] … не передал модели 1 строк(и)».
  10b inj --unsafe  в ответе promo-example (модель выполнила; защит нет).
  11a busy-503      две строки «[код] сервер занят… попытка 2 из 4», затем ответ.
  11b busy-429      то же с 429.
  11c busy-forever  три паузы, «[ответа нет] сервер модели занят: 503 на 4 попыток».
  11d no-server     «[ответа нет] нет связи с сервером модели… запущен ли ollama».
  12 no-tools       ответ «Доставка у нас всегда бесплатная.» и под ним «[в этом ходе
                    документы не читались… — ответ не опирается на текст из папки]».
"""

import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from injection import attack_folder  # noqa: E402

OUT = ROOT / "week-04" / "checkpoint" / "out" / "failures"
Q = "Сколько стоит доставка курьером?"
LONG = "Перечисли подробно все способы доставки, сроки и цены по каждому."

SCENARIOS: list[tuple[str, str, list[str]]] = [
    ("01", "неизвестное имя инструмента", ["--fault", "unknown-tool", "-q", Q]),
    ("02", "невалидные аргументы", ["--fault", "bad-args", "-q", Q]),
    ("03", "инструмент упал", ["--fault", "tool-crash", "-q", Q]),
    ("04a", "вызов текстом в content — разобран", ["--fault", "raw-call", "-q", Q]),
    ("04b", "вызов текстом в content — оборван", ["--fault", "raw-call-broken", "-q", Q]),
    ("05", "цикл повторных вызовов", ["--fault", "loop", "-q", Q]),
    ("06", "лимит итераций", ["--fault", "limit", "-q", Q]),
    (
        "07a",
        "переполнение окна посреди хода",
        ["--fault", "read-all", "--num-ctx", "2048", "-q", "Расскажи всё, что есть в документах"],
    ),
    (
        "07b",
        "переполнение окна посреди сессии",
        [
            "--num-ctx",
            "2560",
            "--history",
            "full",
            "-q",
            Q,
            "-q",
            "А возврат электронной книги?",
            "-q",
            "А подарочным сертификатом можно оплатить доставку?",
        ],
    ),  # fmt: skip
    ("08a", "поток оборван", ["--fault", "stream-cut", "-q", LONG]),
    ("08b", "поток замолк", ["--fault", "stream-stall", "--timeout-between", "3", "-q", LONG]),
    ("09a", "порог цены обращения", ["--limit-usd", "0.005", "-q", Q]),
    (
        "09b",
        "бюджет сессии",
        [
            "--session-usd",
            "0.012",
            "-q",
            Q,
            "-q",
            "А возврат электронной книги?",
            "-q",
            "А подарочным сертификатом можно оплатить доставку?",
        ],
    ),  # fmt: skip
    ("10a", "инструкция в документе — защиты включены", ["--docs", "@A", "-q", "@PICKUP"]),
    ("10b", "инструкция в документе — --unsafe", ["--docs", "@A", "--unsafe", "-q", "@PICKUP"]),
    ("11a", "503 от сервера, потом успех", ["--fault", "busy-503", "-q", Q]),
    ("11b", "429 от сервера, потом успех", ["--fault", "busy-429", "-q", Q]),
    ("11c", "503 на все попытки", ["--fault", "busy-forever", "-q", Q]),
    ("11d", "сервер не запущен", ["--host", "http://127.0.0.1:9", "-q", Q]),
    ("12", "ответ без чтения документов", ["--fault", "no-tools", "-q", Q]),
]


def commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return head + (" + незакоммиченные правки" if dirty else "")


def resolve(args: list[str]) -> list[str]:
    from injection import QUESTION

    out = []
    for a in args:
        if a == "@A":
            out.append(str(attack_folder("A").relative_to(ROOT)))
        elif a == "@PICKUP":
            out.append(QUESTION)
        else:
            out.append(a)
    return out


def run(sid: str, title: str, args: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transcript, events = OUT / f"{sid}.txt", OUT / f"{sid}.events.jsonl"
    events.unlink(missing_ok=True)
    argv = [*resolve(args), "--log", str(transcript.relative_to(ROOT)),
            "--events", str(events.relative_to(ROOT))]  # fmt: skip
    print(f"\n{'=' * 100}\n[{sid}] {title}")
    print(f"$ python -m docqa {shlex.join(argv)}")
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-m", "docqa", *argv], cwd=ROOT, capture_output=True,
                          text=True, timeout=900, env={**os.environ, "PYTHONUNBUFFERED": "1"})  # fmt: skip
    seconds = time.monotonic() - started
    print(f"код возврата {done.returncode}, {seconds:.0f} с")
    print("--- терминал ---")
    print(done.stdout.rstrip())
    if done.stderr.strip():
        print("--- stderr ---")
        print(done.stderr.rstrip()[-1500:])
    print("--- журнал событий ---")
    records = (
        [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines()]
        if events.exists()
        else []
    )
    print(
        "   по типам: "
        + ", ".join(f"{k}×{v}" for k, v in Counter(r["event"] for r in records).items())
    )
    for r in records:
        if r["event"] in ("session", "request", "response", "turn_start"):
            continue
        fields = {k: v for k, v in r.items() if k not in ("ts", "event")}
        print(f"   {r['event']}: {json.dumps(fields, ensure_ascii=False)[:400]}")


def main() -> int:
    only = set(sys.argv[1:])
    print(f"коммит {commit()}; python -m docqa, модель по умолчанию; каждый сценарий — подпроцесс")
    print(
        f"   прогноз (до запуска):\n{(__doc__ or '').split('Прогноз (до запуска), что увидит пользователь:')[1].rstrip()}"
    )
    for sid, title, args in SCENARIOS:
        if not only or sid in only:
            run(sid, title, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
