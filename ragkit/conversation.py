"""История диалога: только дописывается, проверяет чередование, знает свой бюджет (Ф4-03).

У модели нет памяти: каждый ход — новый вызов со всей стенограммой. Отсюда правила класса.

- История только дописывается. Правка или обрезка начала сбрасывает кеш префикса, и тогда
  весь prefill идёт заново (замер: week-04/lesson_03.py, задание 2). Обрезка — не правка
  истории: хранится всё, а политика решает, что из этого отправить.
- Обрезается только целыми ходами. Ход — сообщение user и всё, что идёт за ним до следующего
  user: ответ модели, а позже (Ф4-06) пара «вызов инструмента + результат». Ход режется
  целиком или не режется совсем, поэтому пара не рвётся ни при какой политике.
- Перед каждой отправкой проверяется чередование: шаблон отрендерит и сломанную форму, но
  модель ведёт себя на ней хуже (задание 4).
- Бюджет считается до отправки, по оценке. prompt_eval_count приходит слишком поздно: окно
  к этому моменту уже обрезано сервером. Выход за бюджет — исключение, а не тихая отправка.
- Каждый ход записывает имя и версию промпта (ragkit.prompts, Ф4-02).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from .exceptions import RAGError
from .llm import ChatClient, IncompleteResponseError
from .prompts import Prompt

Message = Mapping[str, Any]
Turn = tuple[Message, ...]
TokenCounter = Callable[[Sequence[Message]], int]


class BrokenHistoryError(RAGError):
    """Сообщения идут не в той форме, которую модель видела в обучении."""


class BudgetExceededError(RAGError):
    """Промпт по оценке не влезает в бюджет даже после обрезки."""


# ---------------------------------------------------------------------------
# Чередование
# ---------------------------------------------------------------------------
def _calls_tool(message: Message) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def validate_alternation(messages: Sequence[Message], *, outgoing: bool = True) -> None:
    """BrokenHistoryError, если форма истории не «system? (user assistant)* user».

    Допустимо: system только первым; дальше первым идёт user; за user — assistant; за
    assistant — user, а за assistant с tool_calls — tool; за tool — tool или assistant.
    outgoing=True — история уходит модели, и последним должен быть user (или tool).
    """
    previous: Message | None = None
    for i, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            if i != 0:
                raise BrokenHistoryError(f"сообщение {i}: system не первым")
            continue
        if previous is None:
            allowed = {"user"}
        elif previous["role"] == "user":
            allowed = {"assistant"}
        elif previous["role"] == "tool":
            allowed = {"tool", "assistant"}
        elif _calls_tool(previous):
            allowed = {"tool"}
        else:
            allowed = {"user"}
        if role not in allowed:
            after = "начала" if previous is None else str(previous["role"])
            raise BrokenHistoryError(
                f"сообщение {i}: {role} после {after}, ожидалось {' или '.join(sorted(allowed))}"
            )
        previous = message
    if outgoing and (previous is None or previous["role"] not in {"user", "tool"}):
        last = "пусто" if previous is None else str(previous["role"])
        raise BrokenHistoryError(f"последним уходит {last}, а должен user")


def split_turns(messages: Sequence[Message]) -> tuple[list[Message], list[Turn]]:
    """system-сообщения отдельно, остальное — ходами: user и всё до следующего user."""
    system = [m for m in messages if m.get("role") == "system"]
    turns: list[list[Message]] = []
    for message in messages:
        if message.get("role") == "system":
            continue
        if message.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    return system, [tuple(t) for t in turns]


# ---------------------------------------------------------------------------
# Политики обрезки: на вход ходы, на выход — те, что отправить. Режут только целыми ходами.
# ---------------------------------------------------------------------------
class TrimPolicy(Protocol):
    def __call__(self, turns: Sequence[Turn]) -> Sequence[Turn]: ...


@dataclass(frozen=True)
class KeepAll:
    """Вся история. Токены растут квадратично, зато кеш префикса работает."""

    def __call__(self, turns: Sequence[Turn]) -> Sequence[Turn]:
        return turns


@dataclass(frozen=True)
class LastTurns:
    """Последние k ходов. Начало — а там стояла задача — теряется."""

    k: int

    def __call__(self, turns: Sequence[Turn]) -> Sequence[Turn]:
        return turns[-self.k :] if self.k > 0 else ()


@dataclass(frozen=True)
class FirstAndLastTurns:
    """Первый ход (постановка задачи) и последние k. Теряется середина."""

    k: int

    def __call__(self, turns: Sequence[Turn]) -> Sequence[Turn]:
        if len(turns) <= self.k + 1:
            return turns
        return (turns[0], *LastTurns(self.k)(turns[1:]))


def apply_policy(messages: Sequence[Message], policy: TrimPolicy) -> list[Message]:
    """Сообщения после обрезки: все system на месте, ходы — по политике, целиком."""
    system, turns = split_turns(messages)
    return [*system, *(m for turn in policy(turns) for m in turn)]


# ---------------------------------------------------------------------------
# Оценка размера до отправки
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CharRatioEstimate:
    """Токены ≈ base + на каждое сообщение per_message + символы / chars_per_token.

    chars_per_token — не константа: 2,3 на русском тексте в Ф4-01, 1,65 на примерах в
    Ф4-02. Калибруется на своём тексте (week-04/lesson_03.py, задание 7).
    """

    chars_per_token: float
    per_message: int = 5
    base: int = 3

    def __call__(self, messages: Sequence[Message]) -> int:
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return (
            self.base + self.per_message * len(messages) + math.ceil(chars / self.chars_per_token)
        )


# ---------------------------------------------------------------------------
# Диалог
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TurnRecord:
    """Что ушло и что пришло на одном ходу — для замеров и для разбора жалоб."""

    number: int
    prompt: str
    prompt_version: str
    policy: str
    sent_messages: int
    estimated_tokens: int
    prompt_eval_count: int | None
    prompt_eval_ms: float | None
    eval_count: int | None
    answer: str


def _frozen(message: Message) -> Message:
    return MappingProxyType(dict(message))


class Conversation:
    """Диалог с моделью: хранит всё, отправляет то, что разрешает политика и бюджет.

    budget — предел оценки промпта в токенах (обычно num_ctx минус место под ответ);
    count_tokens — оценка до отправки, например CharRatioEstimate; policy — какие ходы
    отправлять. Системное сообщение берётся из prompt.system, сообщение пользователя —
    из prompt.user_template.
    """

    def __init__(
        self,
        prompt: Prompt,
        *,
        budget: int,
        count_tokens: TokenCounter,
        policy: TrimPolicy | None = None,
    ) -> None:
        self.prompt = prompt
        self.budget = budget
        self.count_tokens = count_tokens
        self.policy: TrimPolicy = policy if policy is not None else KeepAll()
        self._history: tuple[Message, ...] = ()
        self._records: tuple[TurnRecord, ...] = ()

    @property
    def history(self) -> tuple[Message, ...]:
        """Вся история без system — кортеж неизменяемых сообщений."""
        return self._history

    @property
    def records(self) -> tuple[TurnRecord, ...]:
        return self._records

    def _system(self) -> list[Message]:
        return [{"role": "system", "content": self.prompt.system}] if self.prompt.system else []

    def append(self, *messages: Message) -> None:
        """Дописать готовые сообщения в конец (например, историю из базы). Только в конец."""
        candidate = (*self._history, *(_frozen(m) for m in messages))
        validate_alternation([*self._system(), *candidate], outgoing=False)
        self._history = candidate

    def outgoing(self, **values: str) -> list[Message]:
        """Сообщения, которые уйдут модели на следующем ходу. Проверены и посчитаны."""
        user = self.prompt.messages(**values)[-1]
        messages = [*apply_policy([*self._system(), *self._history], self.policy), user]
        validate_alternation(messages)
        estimate = self.count_tokens(messages)
        if estimate > self.budget:
            raise BudgetExceededError(
                f"промпт {self.prompt.name} v{self.prompt.version}: по оценке {estimate} "
                f"токенов при бюджете {self.budget}, политика {self.policy!r}, "
                f"{len(messages)} сообщений"
            )
        return messages

    def ask(
        self,
        client: ChatClient,
        model: str,
        options: Mapping[str, Any] | None = None,
        **values: str,
    ) -> TurnRecord:
        """Один ход. Оборванный ответ — IncompleteResponseError, и история не меняется."""
        messages = self.outgoing(**values)
        reply = client.chat(model=model, messages=messages, options=options)
        content = reply.message.content or ""
        if reply.done_reason != "stop":
            raise IncompleteResponseError(
                f"ответ неполный: done_reason={reply.done_reason!r}, "
                f"промпт {self.prompt.name} v{self.prompt.version}, ход {len(self._records) + 1}"
            )
        self._history = (
            *self._history,
            _frozen(messages[-1]),
            _frozen({"role": "assistant", "content": content}),
        )
        duration = getattr(reply, "prompt_eval_duration", None)
        record = TurnRecord(
            number=len(self._records) + 1,
            prompt=self.prompt.name,
            prompt_version=self.prompt.version,
            policy=repr(self.policy),
            sent_messages=len(messages),
            estimated_tokens=self.count_tokens(messages),
            prompt_eval_count=getattr(reply, "prompt_eval_count", None),
            prompt_eval_ms=duration / 1e6 if duration is not None else None,
            eval_count=getattr(reply, "eval_count", None),
            answer=content,
        )
        self._records = (*self._records, record)
        return record
