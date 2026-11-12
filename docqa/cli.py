"""CLI: вопросы по папке с документами, ответ потоком, инструменты по ходу.

    python -m docqa --docs docqa/sample_docs
    python -m docqa --docs docqa/sample_docs --debug --log session.txt
    python -m docqa -q "Сколько стоит доставка?" -q "А если заказ больше 3000?"

Интерактивно: вопрос — строка, пустая строка или /exit — выход, /reset — забыть историю.
С -q вопросы идут по очереди одной сессией (многоходовой диалог), без ввода с клавиатуры.

Отказы воспроизводятся флагом --fault (docqa.faults), журнал событий — --events PATH.

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
from ragkit.streaming import (
    AsyncToolStreamingChatClient,
    StreamTimeout,
    StreamTimeouts,
    StreamTruncated,
)
from ragkit.tools import IterationLimitError, ToolResult, UnparsedToolCallError

from .documents import DocumentStore
from .events import EventLog
from .faults import FAULTS, CrashingStore, Fault, FaultyClient
from .session import (
    EmptyAnswerError,
    RepeatLoopError,
    ServerBusyError,
    Session,
    SessionBudgetError,
    Settings,
)

DEFAULT_DOCS = Path(__file__).parent / "sample_docs"
KIND_TEXT = {
    "unknown": "такого инструмента нет",
    "invalid": "аргументы не прошли проверку",
    "failed": "инструмент не смог выполнить",
}


class Terminal:
    """Вывод в терминал и, если задан, в файл расшифровки — одно и то же."""

    def __init__(self, debug: bool, log: TextIO | None) -> None:
        self.show_debug = debug
        self.log = log
        self._line_open = False
        self._answering = False
        self.shown = False  # видел ли пользователь в этом ходе хоть символ ответа

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
        self.shown = True
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
        line = f"  · {name}({args}) → {result.kind}, {len(result.content)} симв. модели"
        if not result.ok:  # текст ошибки — тот, что ушёл модели: он из белого списка
            line += f": {result.content[:160]}"
        self.line(line)

    def notice(self, text: str) -> None:
        """Строка от кода, не от модели."""
        self.line(f"  [код] {text}")

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


def failure(error: BaseException, shown: bool) -> str:
    """Что пользователь видит, когда ход не удался. Ни одна ветка не говорит «готово».

    shown — успел ли пользователь увидеть часть ответа: тогда о ней надо сказать прямо.
    """
    partial = " Текст выше — обрывок, не ответ; в историю он не записан." if shown else ""
    if isinstance(error, SessionBudgetError):
        return f"запрос не отправлен — бюджет сессии исчерпан: {error}"
    if isinstance(error, CostLimitExceededError):
        return f"дороже порога: {error}"
    if isinstance(error, BudgetExceededError):
        return (
            f"запрос не отправлен — не помещается в окно модели: {error}. Можно /reset "
            "(забыть историю), вопрос уже или больший --num-ctx"
        )
    if isinstance(error, RepeatLoopError):
        return f"модель зациклилась на одном вызове: {error}"
    if isinstance(error, IterationLimitError):
        return f"модель не закончила за отведённое число обращений: {error}"
    if isinstance(error, EmptyAnswerError):
        return f"модель вернула пустой ответ; в историю ничего не записано ({error})"
    if isinstance(error, UnparsedToolCallError):
        return (
            "модель вернула вызов инструмента текстом, и код не смог его разобрать; "
            f"ничего не выполнено. {error}"
        )
    if isinstance(error, StreamTimeout):
        return f"ответ оборван — {error}.{partial}"
    if isinstance(error, StreamTruncated | IncompleteResponseError):
        return f"ответ оборван: {error}.{partial}"
    if isinstance(error, ServerBusyError):
        return f"{error}. Попробуйте позже"
    if isinstance(error, BrokenHistoryError):
        return f"ход не записан — неверная форма истории: {error}"
    if isinstance(error, ollama.ResponseError):
        return f"сервер модели ответил ошибкой {error.status_code}: {error.error}.{partial}"
    if isinstance(error, httpx.HTTPError | ConnectionError):
        return (
            f"нет связи с сервером модели ({type(error).__name__}: {error}). Запущен ли "
            f"ollama (ollama serve) и верен ли --host?{partial}"
        )
    return f"{type(error).__name__}: {error}{partial}"


async def run(questions: list[str] | None, session: Session, term: Terminal) -> int:
    failures = 0

    async def one(question: str) -> None:
        nonlocal failures
        term.question(question)
        term.shown = False
        try:
            report = await session.ask(question)
        except (RAGError, ollama.ResponseError, httpx.HTTPError, ConnectionError) as error:
            failures += 1
            term.line(f"[ответа нет] {failure(error, term.shown)}")
            return
        term.end()
        if report.not_done:
            term.line(
                "  [не выполнено в этом ходе: "
                + "; ".join(f"{r.requested} — {KIND_TEXT[r.kind]}" for r in report.not_done)
                + "]"
            )
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
        if question == "/reset":
            session.reset()
            term.line("[код] история забыта; следующий вопрос — с чистого листа")
            continue
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
    parser.add_argument("--session-usd", type=float, default=None, help="порог на всю сессию")
    parser.add_argument("--temperature", type=float, default=Settings.temperature)
    parser.add_argument("--seed", type=int, default=Settings.seed)
    parser.add_argument("--timeout-first", type=float, default=Settings.timeouts.first)
    parser.add_argument("--timeout-between", type=float, default=Settings.timeouts.between)
    parser.add_argument("--timeout-total", type=float, default=Settings.timeouts.total)
    parser.add_argument("--prompt", choices=["base", "warn"], default=Settings.prompt)
    parser.add_argument(
        "--allow-domain", action="append", default=[], help="адрес, который можно показывать"
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="выключить защиты от инструкций в документах (для замера и демонстрации)",
    )
    parser.add_argument(
        "--fault",
        choices=sorted(FAULTS),
        help="воспроизвести отказ: " + "; ".join(f"{k} — {v}" for k, v in FAULTS.items()),
    )
    parser.add_argument("--events", type=Path, help="журнал событий, JSON по строке")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="показывать ответ без чтения документов сразу; код сам не ищет и не читает",
    )
    parser.add_argument("-q", "--question", action="append", help="вопрос; можно несколько")
    parser.add_argument("--debug", action="store_true", help="версия промпта, бюджет, токены")
    parser.add_argument("--log", type=Path, help="записать расшифровку сессии в файл")
    args = parser.parse_args(argv)

    store_type = CrashingStore if args.fault == "tool-crash" else DocumentStore
    store = store_type(args.docs, quarantine=not args.unsafe)
    if not store.names():
        print(f"в папке {args.docs} нет документов .md или .txt", file=sys.stderr)
        return 2
    settings = Settings(
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        max_iterations=args.max_iterations,
        limit_usd=args.limit_usd,
        session_limit_usd=args.session_usd,
        history_mode=args.history,
        temperature=args.temperature,
        seed=args.seed,
        timeouts=StreamTimeouts(args.timeout_first, args.timeout_between, args.timeout_total),
        prompt=args.prompt,
        quarantine=not args.unsafe,
        redact_addresses=not args.unsafe,
        allowed_domains=tuple(args.allow_domain),
        fallback=not args.no_fallback,
    )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("w", encoding="utf-8") if args.log else None
    events = EventLog.open(args.events) if args.events else None
    try:
        term = Terminal(args.debug, log)
        client: AsyncToolStreamingChatClient = ollama.AsyncClient(host=args.host)
        if args.fault and args.fault != "tool-crash":
            fault: Fault = args.fault
            client = FaultyClient(client, fault, store.names())
        session = Session(store, client, term, settings, events=events)
        term.line(
            f"docqa: {len(store.names())} документов в {args.docs}; модель {settings.model}; "
            f"промпт {session.prompt.name} v{session.prompt.version}"
            + (f"; ОТКАЗ {args.fault}: {FAULTS[args.fault]}" if args.fault else "")
            + ("; ЗАЩИТЫ ВЫКЛЮЧЕНЫ (--unsafe)" if args.unsafe else "")
        )
        if events is not None:
            events.emit("session", argv=sys.argv[1:], fault=args.fault, settings=session.describe())
        for line in session.describe():
            term.debug(line)
        return asyncio.run(run(args.question, session, term))
    finally:
        if log is not None:
            log.close()
        if events is not None:
            events.close()


if __name__ == "__main__":
    raise SystemExit(main())
