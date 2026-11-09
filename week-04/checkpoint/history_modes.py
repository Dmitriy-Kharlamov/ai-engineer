"""Ф4-08, присест 1: что из хода хранить в истории — замер на четырёх вопросах.

    python -u week-04/checkpoint/history_modes.py 2>&1 | tee week-04/checkpoint/history_modes.txt

Один и тот же разговор из четырёх вопросов в трёх режимах истории docqa (full, stubs,
answers); temperature 0, seed 42 — каждый режим детерминирован, поэтому по одному прогону.
По каждому ходу: сколько вызовов инструментов, какие документы прочитаны (по журналу
исполнения), ответ и верен ли ключевой факт — по регулярному выражению против текста
документа. Ход без чтения документа с верным фактом — угадан, а не найден.

Прогноз (до запуска): answers — как в расшифровке присеста 1: вызовы только в ходе 1,
верных ответов 1–2 из 4; stubs — вызовы в каждом ходе, верных 4/4; full — тоже 4/4, но
промпт к ходу 4 в 2–3 раза больше, чем у stubs.
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path

import ollama

from docqa.documents import DocumentStore
from docqa.session import HistoryMode, Session, Settings
from ragkit.exceptions import RAGError
from ragkit.tools import ToolResult

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docqa" / "sample_docs"

# вопрос, правильный факт (регулярка по ответу), где он в документах
QUESTIONS = [
    ("Сколько стоит курьерская доставка по Казани?", r"290", "delivery.md: 290 рублей"),
    ("А если заказ на 3000 рублей?", r"бесплатн", "delivery.md: от 2500 — бесплатно"),
    (
        "Можно ли вернуть электронную книгу?",
        r"не подлеж|нельзя|не возвра|не принима",
        "returns.md: электронные книги возврату не подлежат",
    ),  # fmt: skip
    (
        "А подарочным сертификатом можно оплатить доставку?",
        r"нельзя|не может|не принима",
        "gift_cards.md: сертификатом нельзя оплатить доставку",
    ),  # fmt: skip
]
MODES: list[HistoryMode] = ["answers", "stubs", "full"]


class Quiet:
    def delta(self, text: str) -> None:
        pass

    def tool(self, result: ToolResult) -> None:
        pass

    def debug(self, line: str) -> None:
        pass


def commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return head + (" + незакоммиченные правки" if dirty else "")


async def run_mode(mode: HistoryMode) -> tuple[int, int, int]:
    session = Session(DocumentStore(DOCS), ollama.AsyncClient(), Quiet(),
                      Settings(history_mode=mode))  # fmt: skip
    right = found = prompt = 0
    for turn, (question, fact, where) in enumerate(QUESTIONS, start=1):
        try:
            report = await session.ask(question)
        except RAGError as error:  # отказ хода — тоже результат, а не падение замера
            print(f"   ход {turn}: ОТКАЗ — {type(error).__name__}: {error}")
            continue
        ok = bool(re.search(fact, report.answer.casefold()))
        read = report.documents_read
        right += ok
        found += ok and bool(read)
        prompt += report.prompt_tokens
        verdict = "верно" if ok else "НЕВЕРНО"
        basis = f"прочитано: {', '.join(read)}" if read else "документы не читались"
        print(
            f"   ход {turn}: вызовов {len(report.calls)}, {basis}; {verdict} (в документах — {where})"
        )
        print(
            f"      промпт за ход {report.prompt_tokens} ток.; ответ: {report.answer.strip()[:160]!r}"
        )
    return right, found, prompt


def main() -> int:
    print(f"коммит {commit()}; модель {Settings().model}; temperature 0, seed 42")
    print(f"   прогноз (до запуска): {(__doc__ or '').split('Прогноз (до запуска): ')[1].strip()}")
    summary = []
    for mode in MODES:
        print(f"\n[{mode}]")
        right, found, prompt = asyncio.run(run_mode(mode))
        summary.append((mode, right, found, prompt))
    print("\nитог: режим — верных из 4 / из них с прочитанным документом / промпт за 4 хода")
    for mode, right, found, prompt in summary:
        print(f"   {mode:<8} {right}/4  {found}/4  {prompt} ток.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
