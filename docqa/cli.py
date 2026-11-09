"""CLI: вопросы по папке с документами, ответ потоком, инструменты по ходу.

    python -m docqa --docs docqa/sample_docs
    python -m docqa --docs docqa/sample_docs --debug --log session.txt
    python -m docqa -q "Сколько стоит доставка?" -q "А если заказ больше 3000?"

Интерактивно: вопрос — строка, пустая строка или /exit — выход. С -q вопросы идут по
очереди одной сессией (многоходовой диалог), без ввода с клавиатуры.

Правду об отказе сообщает код: если ход не удался, CLI печатает, что именно случилось, и
что ответа нет. История при этом не меняется — следующий вопрос задаётся как будто этого
хода не было.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TextIO

import httpx
import ollama

from ragkit.conversation import BrokenHistoryError, BudgetExceededError
from ragkit.cost import CostLimitExceededError
from ragkit.exceptions import RAGError
from ragkit.llm import IncompleteResponseError
from ragkit.streaming import StreamTimeout, StreamTruncated
from ragkit.tools import IterationLimitError, ToolResult

from .documents import DocumentStore
from .session import DOCQA_PROMPT, Session, Settings

DEFAULT_DOCS = Path(__file__).parent / "sample_docs"


class Terminal:
    """Вывод в терминал и, если задан, в файл расшифровки — одно и то же."""

    def __init__(self, debug: bool, log: TextIO | None) -> None:
        self.show_debug = debug
        self.log = log
        self._line_open = False
        self._answering = False

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.log is not None:
            self.log.write(text)
            self.log.flush()

    def line(self, text: str) -> None:
        if self._line_open:
            self.write("\n")
            self._line_open = False
        self.write(text + "\n")

    def delta(self, text: str) -> None:
        if not self._answering:  # префикс — перед первым куском текста, а не до вызовов
            if self._line_open:
                self.write("\n")
            self.write("модель> ")
            self._answering = True
        self.write(text)
        self._line_open = True

    def tool(self, result: ToolResult) -> None:
        self._answering = False
        args = ", ".join(f"{k}={v!r}" for k, v in (result.requested_arguments or {}).items())
        name = result.tool_name if result.tool_name == result.requested else repr(result.requested)
        self.line(f"  · {name}({args}) → {result.kind}, {len(result.content)} симв. модели")

    def debug(self, text: str) -> None:
        if self.show_debug:
            self.line(f"  [debug] {text}")

    def end(self) -> None:
        """Закрыть строку ответа, если она открыта."""
        if self._line_open:
            self.write("\n")
            self._line_open = False

    def question(self, text: str) -> None:
        self.line(f"\nвы> {text}")
        self._answering = False


def failure(error: BaseException) -> str:
    """Что пользователь видит, когда ход не удался. Ни одна ветка не говорит «готово»."""
    if isinstance(error, CostLimitExceededError):
        return f"запрос не отправлен — дороже порога: {error}"
    if isinstance(error, BudgetExceededError):
        return f"запрос не отправлен — не помещается в окно модели: {error}"
    if isinstance(error, IterationLimitError):
        return f"ответа нет — модель не закончила за отведённое число обращений: {error}"
    if isinstance(error, StreamTimeout):
        return f"ответ оборван — {error}"
    if isinstance(error, StreamTruncated | IncompleteResponseError):
        return f"ответ оборван, в историю не записан: {error}"
    if isinstance(error, BrokenHistoryError):
        return f"ход не записан — неверная форма истории: {error}"
    if isinstance(error, ollama.ResponseError):
        return f"сервер модели ответил ошибкой {error.status_code}: {error.error}"
    if isinstance(error, httpx.HTTPError | ConnectionError):
        return f"нет связи с сервером модели: {type(error).__name__}: {error}"
    return f"{type(error).__name__}: {error}"


async def run(questions: list[str] | None, session: Session, term: Terminal) -> int:
    failures = 0

    async def one(question: str) -> None:
        nonlocal failures
        term.question(question)
        try:
            report = await session.ask(question)
        except (RAGError, ollama.ResponseError, httpx.HTTPError, ConnectionError) as error:
            failures += 1
            term.line(f"[ответа нет] {failure(error)}")
            return
        term.end()
        read = report.documents_read
        if read:
            term.line(f"  [прочитано в этом ходе: {', '.join(read)}]")
        else:
            calls = len(report.calls)
            term.line(
                "  [в этом ходе документы не читались"
                + (f" (вызовов инструментов: {calls})" if calls else ", инструменты не вызывались")
                + " — ответ не опирается на текст из папки]"
            )
        term.debug(
            f"ход {len(session.turns)}: обращений {len(report.iterations)}, промпт всего "
            f"{report.prompt_tokens} ток., ответ всего {report.output_tokens} ток.; в истории "
            f"{len(session.conversation.history)} сообщений"
        )

    if questions:
        for q in questions:
            await one(q)
        return 1 if failures else 0
    while True:
        try:
            question = (await asyncio.to_thread(input, "\nвопрос (пусто — выход)> ")).strip()
        except EOFError:
            break
        if not question or question == "/exit":
            break
        await one(question)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docqa", description="Вопросы по папке с документами")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="папка с .md и .txt")
    parser.add_argument("--model", default=Settings.model)
    parser.add_argument("--host", default=None, help="адрес ollama (по умолчанию OLLAMA_HOST)")
    parser.add_argument("--num-ctx", type=int, default=Settings.num_ctx)
    parser.add_argument("--num-predict", type=int, default=Settings.num_predict)
    parser.add_argument("--max-iterations", type=int, default=Settings.max_iterations)
    parser.add_argument(
        "--history",
        choices=["full", "stubs", "answers"],
        default=Settings.history_mode,
        help="что из хода хранить в истории: всё, вызовы с пометками вместо текста, ответы",
    )
    parser.add_argument("--limit-usd", type=float, default=Settings.limit_usd)
    parser.add_argument("-q", "--question", action="append", help="вопрос; можно несколько")
    parser.add_argument("--debug", action="store_true", help="версия промпта, бюджет, токены")
    parser.add_argument("--log", type=Path, help="записать расшифровку сессии в файл")
    args = parser.parse_args(argv)

    store = DocumentStore(args.docs)
    if not store.names():
        print(f"в папке {args.docs} нет документов .md или .txt", file=sys.stderr)
        return 2
    settings = Settings(
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        max_iterations=args.max_iterations,
        limit_usd=args.limit_usd,
        history_mode=args.history,
    )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("w", encoding="utf-8") if args.log else None
    try:
        term = Terminal(args.debug, log)
        client = ollama.AsyncClient(host=args.host)
        session = Session(store, client, term, settings)
        term.line(
            f"docqa: {len(store.names())} документов в {args.docs}; модель {settings.model}; "
            f"промпт {DOCQA_PROMPT.name} v{DOCQA_PROMPT.version}"
        )
        for line in session.describe():
            term.debug(line)
        return asyncio.run(run(args.question, session, term))
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
