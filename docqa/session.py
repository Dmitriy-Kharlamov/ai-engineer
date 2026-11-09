"""Сессия: многоходовой диалог, в каждом ходе — потоковый цикл инструментов.

Ход устроен так:
1. Conversation.outgoing — сообщения для первого обращения: system из промпта с версией,
   история по политике обрезки, новый вопрос; чередование проверено, бюджет окна посчитан.
2. Обращение к модели (до max_iterations раз). Перед каждым — две проверки ДО отправки:
   окно (оценка входа вместе со схемами инструментов ≤ num_ctx − num_predict) и деньги
   (CostGuard по прайсу). num_predict задан явно: это и верхняя граница цены ответа.
3. Ответ читается потоком с тремя таймаутами (ragkit.streaming); текст сразу уходит в
   терминал, вызовы инструментов собираются из кусков. Поток закрывается явно — async with.
4. Есть вызовы — ToolLoop (ragkit.tools) выполняет их: повтор одинакового вызова не
   исполняется, ошибка становится результатом для модели, порядок результатов — порядок
   вызовов. Нет вызовов — это ответ: ход записывается в историю (commit_turn).
   Что из хода попадает в историю — HistoryMode:
     full    — цепочка целиком, с текстами документов;
     stubs   — вызовы остаются, а текст результата заменён пометкой (по умолчанию);
     answers — только вопрос и ответ.
   Почему не full: документ с инструкцией, оставшийся в истории, давал её на 10-м ходу в
   5/10, а вышедший из окна — 0/10 (Ф4-03, задание 6), и каждый ход переотправлял бы все
   прочитанные документы. Почему не answers: в присесте 1 с answers модель после первого
   хода перестала звать инструменты — 0 вызовов в ходах 2–4, ответ «электронные книги
   можно вернуть» против «возврату не подлежат» в returns.md и «поиск не дал результатов»
   без единого поиска. История — это пример: ходы без вызовов учат отвечать без вызовов.
5. Любой отказ посреди хода — исключение наружу, и история не меняется: в неё не попадает
   ни обрывок потока, ни половина цепочки вызовов (Ф4-04: PartialPolicy.DISCARD).

Что видит пользователь при каждом отказе — решает CLI, а не модель.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ragkit.conversation import BudgetExceededError, CharRatioEstimate, Conversation, LastTurns
from ragkit.cost import CostGuard, Price
from ragkit.prompts import Prompt
from ragkit.streaming import AsyncToolStreamingChatClient, StreamTimeouts, astream_tool_chat
from ragkit.tools import (
    IterationLimitError,
    Step,
    ToolLoop,
    ToolResult,
    assistant_message,
    validate_tool_results,
)

from .documents import DocumentStore, build_registry

Msg = Mapping[str, Any]
HistoryMode = Literal["full", "stubs", "answers"]

DOCQA_PROMPT = Prompt(
    name="docqa",
    version="1",
    system=(
        "Ты отвечаешь на вопросы по документам из папки пользователя. Документы доступны "
        "только через инструменты: list_documents — что есть в папке, search_documents — "
        "поиск по словам, read_document — текст документа по имени файла. Прежде чем "
        "отвечать, найди и прочитай нужный документ; отвечай только по прочитанному и "
        "называй документы, из которых взят ответ, по имени файла. Если в документах ответа "
        "нет — так и скажи. Отвечай по-русски, коротко."
    ),
    user_template="{question}",
)

# Цена — модель на прайсе Anthropic (Ф4-07): локальная модель бесплатна, но предохранитель
# проверяется на тех числах, которые были бы у платного API.
PRICE = Price(
    provider="Anthropic",
    model="Claude Sonnet 5.5",
    as_of="2026-11-08",
    source="https://platform.claude.com/docs/en/about-claude/pricing",
    input_per_mtok=2.0,
    output_per_mtok=10.0,
    cache_read_per_mtok=0.10,
)
# Худший замеренный курс 1,65 символа на токен (Ф4-02) × 0,9 и база 30 токенов на разметку
# шаблона: на 16 замерах из tests/fixtures/token_counts.json оценка не ниже факта (Ф4-07).
ESTIMATOR = CharRatioEstimate(chars_per_token=1.49, per_message=5, base=30)


@dataclass(frozen=True)
class Settings:
    model: str = "qwen2.5:7b"
    num_ctx: int = 8192
    num_predict: int = 512
    temperature: float = 0.0
    seed: int = 42
    max_iterations: int = 5
    history_turns: int = 4  # LastTurns: system с постановкой задачи остаётся всегда
    history_mode: HistoryMode = "stubs"  # что из хода попадает в историю, см. выше
    limit_usd: float = 0.02  # порог цены одного обращения
    timeouts: StreamTimeouts = StreamTimeouts(first=90.0, between=20.0, total=240.0)

    @property
    def options(self) -> dict[str, Any]:
        return {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    @property
    def window(self) -> int:
        """Сколько токенов может занять вход, чтобы ответу осталось num_predict."""
        return self.num_ctx - self.num_predict


class Output(Protocol):
    """Куда сессия пишет то, что видит пользователь. Реализация — в CLI, в тестах — фейк."""

    def delta(self, text: str) -> None: ...
    def tool(self, result: ToolResult) -> None: ...
    def debug(self, line: str) -> None: ...


@dataclass(frozen=True)
class Iteration:
    """Одно обращение к модели в ходе: что ушло (по оценке) и что пришло (по счётчикам)."""

    turn: int
    number: int
    sent_messages: int
    estimated_tokens: int
    estimated_usd: float
    prompt_eval_count: int
    eval_count: int
    prompt_ms: float
    eval_ms: float
    calls: tuple[ToolResult, ...] = ()


@dataclass
class TurnReport:
    answer: str
    iterations: list[Iteration] = field(default_factory=list)

    @property
    def calls(self) -> tuple[ToolResult, ...]:
        return tuple(r for i in self.iterations for r in i.calls)

    @property
    def documents_read(self) -> list[str]:
        """Какие документы прочитаны в этом ходе — по журналу исполнения, а не по ответу."""
        names: list[str] = []
        for r in self.calls:
            name = (r.arguments or {}).get("name")
            if r.ok and r.tool_name == "read_document" and name and name not in names:
                names.append(str(name))
        return names

    @property
    def prompt_tokens(self) -> int:
        return sum(i.prompt_eval_count for i in self.iterations)

    @property
    def output_tokens(self) -> int:
        return sum(i.eval_count for i in self.iterations)


class Session:
    def __init__(
        self,
        store: DocumentStore,
        client: AsyncToolStreamingChatClient,
        out: Output,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store
        self.client = client
        self.out = out
        self.registry = build_registry(store)
        self.schemas = self.registry.schemas()
        schema_text = json.dumps(self.schemas, ensure_ascii=False)
        self.tools_tokens = ESTIMATOR([{"role": "system", "content": schema_text}])
        self.guard = CostGuard(PRICE, self.settings.limit_usd, ESTIMATOR)
        self.conversation = Conversation(
            DOCQA_PROMPT,
            budget=self.settings.window - self.tools_tokens,
            count_tokens=ESTIMATOR,
            policy=LastTurns(self.settings.history_turns),
        )
        self.turns: list[TurnReport] = []

    def describe(self) -> list[str]:
        """Настройки сессии для --debug."""
        s, p = self.settings, DOCQA_PROMPT
        return [
            f"промпт {p.name} v{p.version}; модель {s.model}; temperature {s.temperature}, "
            f"seed {s.seed}",
            f"окно num_ctx={s.num_ctx}, ответ num_predict={s.num_predict} → вход ≤ {s.window} "
            f"токенов по оценке (из них схемы {len(self.schemas)} инструментов ≈ "
            f"{self.tools_tokens})",
            f"история: {self.conversation.policy!r}, ход в истории — {s.history_mode}; "
            f"оценщик {ESTIMATOR.chars_per_token} "
            f"симв/ток, база {ESTIMATOR.base}, {ESTIMATOR.per_message} на сообщение",
            f"цена: {PRICE.provider} {PRICE.model} на {PRICE.as_of}, порог {s.limit_usd} $ "
            "на обращение",
            f"таймауты потока: первый кусок {s.timeouts.first} с, между кусками "
            f"{s.timeouts.between} с, весь поток {s.timeouts.total} с; итераций ≤ "
            f"{s.max_iterations}",
        ]

    async def ask(self, question: str) -> TurnReport:
        s = self.settings
        turn = len(self.turns) + 1
        sent = self.conversation.outgoing(question=question)
        produced: list[dict[str, Any]] = []
        loop = ToolLoop(self.registry)
        report = TurnReport("")
        steps: list[Step] = []
        for number in range(1, s.max_iterations + 1):
            messages = [*sent, *produced]
            estimate = self.guard.check(
                messages, max_output_tokens=s.num_predict, tools=self.schemas
            )  # CostLimitExceededError — до отправки
            if estimate.input_tokens > s.window:
                raise BudgetExceededError(
                    f"ход {turn}, обращение {number}: вход по оценке {estimate.input_tokens} "
                    f"токенов при окне {s.window} (num_ctx {s.num_ctx} − num_predict "
                    f"{s.num_predict}); сообщений {len(messages)}"
                )
            self.out.debug(
                f"ход {turn}, обращение {number}: {len(messages)} сообщений, оценка входа "
                f"{estimate.input_tokens} ток. (окно ≤ {s.window}), цена ≤ "
                f"{estimate.cost:.5f} $ (порог {s.limit_usd})"
            )
            stream = await astream_tool_chat(
                self.client, s.model, messages, self.schemas, s.options, timeouts=s.timeouts
            )
            async with stream:  # поток закрывается явно и при ошибке тоже (Ф4-04, задание 6)
                async for delta in stream:
                    self.out.delta(delta)
            envelope = stream.envelope
            calls = [(c.name, c.arguments) for c in stream.tool_calls]
            results = loop.handle(number, calls) if calls else ()
            iteration = Iteration(
                turn,
                number,
                len(messages),
                estimate.input_tokens,
                estimate.cost,
                envelope.prompt_eval_count,
                envelope.eval_count,
                envelope.prompt_eval_ns / 1e6,
                envelope.eval_ns / 1e6,
                results,
            )
            report.iterations.append(iteration)
            steps.append(Step(number, envelope.prompt_eval_count, envelope.eval_count, results))
            over = estimate.input_tokens / max(envelope.prompt_eval_count, 1) - 1
            self.out.debug(
                f"ход {turn}, обращение {number}: prompt_eval_count={envelope.prompt_eval_count} "
                f"(оценка {estimate.input_tokens}, {over:+.0%}), "
                f"eval_count={envelope.eval_count}, prefill {iteration.prompt_ms:.0f} мс, "
                f"decode {iteration.eval_ms:.0f} мс, done_reason={envelope.done_reason!r}, "
                f"вызовов {len(calls)}"
            )
            if not calls:
                report.answer = stream.text
                produced.append({"role": "assistant", "content": stream.text})
                validate_tool_results([*sent, *produced])
                self.conversation.commit_turn(
                    sent,
                    for_history(produced, s.history_mode),
                    prompt_eval_count=report.prompt_tokens,
                    prompt_eval_ms=sum(i.prompt_ms for i in report.iterations),
                    eval_count=report.output_tokens,
                )
                self.turns.append(report)
                return report
            produced.append(assistant_message(stream.text, calls))
            for result in results:
                self.out.tool(result)
                produced.append(result.message())
        raise IterationLimitError(
            f"ход {turn}: модель просит инструменты после {s.max_iterations} обращений",
            steps=steps,
            messages=[*sent, *produced],
        )


def for_history(produced: Sequence[Mapping[str, Any]], mode: HistoryMode) -> list[Msg]:
    """Что из законченного хода записать в историю."""
    if mode == "full":
        return list(produced)
    if mode == "answers":
        return list(produced[-1:])
    out: list[Msg] = []
    for m in produced:
        if m.get("role") == "tool":
            size = len(str(m.get("content") or ""))
            out.append(
                {
                    **m,
                    "content": f"[результат {m.get('tool_name')} ({size} симв.) не хранится в "
                    "истории; чтобы опереться на него в новом ответе — вызови инструмент снова]",
                }
            )
        else:
            out.append(m)
    return out
