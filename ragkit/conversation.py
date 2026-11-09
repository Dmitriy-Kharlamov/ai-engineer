"""История диалога: только дописывается, проверяет чередование, знает свой бюджет (Ф4-03).

У модели нет памяти: каждый ход — новый вызов со всей стенограммой. Отсюда правила класса.

- История только дописывается. Правка или обрезка начала сбрасывает кеш префикса, и тогда
  весь prefill идёт заново (замер: week-04/lesson_03.py, задание 2). Обрезка — не правка
  истории: хранится всё, а политика решает, что из этого отправить.
- Обрезается только целыми ходами. Ход — сообщение user и всё, что идёт за ним до следующего
  user: ответ модели, а позже (Ф4-06) пара «вызов инструмента + результат». Ход режется
  целиком или не режется совсем, поэтому пара не рвётся ни при какой политике.
- Перед каждой отправкой проверяется чередование: шаблон отрендерит и сломанную форму, но
  модель ведёт себя на ней хуже: два user подряд уронили соблюдение формата с 85 % до
  20–25 % (задание 4, p < 0,001).
- Бюджет считается до отправки, по оценке. prompt_eval_count приходит слишком поздно, и в
  диалоге переполнения по нему даже не видно: сервер ollama молча выбрасывает целые старые
  сообщения, оставляя system, и счётчик показывает «почти полное окно» (замер, задание 7:
  4083 из 4096, а первый ход потерян). Выход за бюджет — исключение, а не тихая отправка.
- Каждый ход записывает имя и версию промпта (ragkit.prompts, Ф4-02).
- Потоковый ход (ask_stream, Ф4-04): пользователь видит ответ раньше, чем известно,
  законченный ли он. Что записать в историю, если поток оборвался, решает PartialPolicy —
  параметром, а не по умолчанию «как получится».
- Ход с инструментами (commit_turn, Ф4-08) — это не пара, а цепочка: user → assistant с
  вызовами → результаты → … → ответ. Цикл живёт снаружи (ragkit.tools, CLI), а в историю
  ход попадает только законченным и целиком — тот же контракт, что у ask.
"""

import math
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from .exceptions import RAGError
from .llm import ChatClient, IncompleteResponseError
from .prompts import Prompt
from .streaming import Envelope, StreamingChatClient, stream_chat

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
    outcome: str = "stop"  # "stop" — законченный ответ; "partial:mark" и т. п. — обрывок


class PartialPolicy(Enum):
    """Что записать в историю, если поток оборвался после того, как его начали показывать.

    DISCARD — ничего: пара «вопрос + обрывок» пропадает целиком, история как будто хода не
      было. Пользователь видел текст, которого модель на следующем ходу «не помнит».
    MARK — обрывок с пометкой PARTIAL_MARK в конце: модель видит, что ответ не закончен.
    AS_COMPLETE — обрывок как полный ответ. Самый естественный вариант, если не думать, и
      самый опасный: обрывок становится «фактом» в истории, а свою историю модель защищает
      (Ф4-03, задание 5: ложь в собственном прошлом ответе — 20 повторов из 20, и инструкция
      «исправь ошибку» не изменила ни одного). Обрывок «не рекомендуется при…» на месте
      «не рекомендуется при беременности» — уже другой факт.

    По умолчанию в ask_stream — DISCARD: тот же контракт, что у ask (ход записан целиком или
    не записан вовсе), и в историю не попадает ничего непроверенного. Цена — модель не знает,
    что пользователь видел; это лечится интерфейсом («ответ оборван, повторите»), а ложь в
    истории — ничем.
    """

    DISCARD = "discard"
    MARK = "mark"
    AS_COMPLETE = "as_complete"


PARTIAL_MARK = "\n\n[ответ оборван: {reason}]"


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

    def _commit(
        self,
        messages: Sequence[Message],
        content: str,
        *,
        outcome: str,
        prompt_eval_count: int | None,
        prompt_eval_ms: float | None,
        eval_count: int | None,
    ) -> TurnRecord:
        """Записать ход целиком: вопрос и ответ одной парой, и строку в журнал."""
        return self._record(
            messages,
            [{"role": "assistant", "content": content}],
            content,
            outcome=outcome,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_ms=prompt_eval_ms,
            eval_count=eval_count,
        )

    def _record(
        self,
        messages: Sequence[Message],
        produced: Sequence[Message],
        content: str,
        *,
        outcome: str,
        prompt_eval_count: int | None,
        prompt_eval_ms: float | None,
        eval_count: int | None,
    ) -> TurnRecord:
        self._history = (
            *self._history,
            _frozen(messages[-1]),
            *(_frozen(m) for m in produced),
        )
        record = TurnRecord(
            number=len(self._records) + 1,
            prompt=self.prompt.name,
            prompt_version=self.prompt.version,
            policy=repr(self.policy),
            sent_messages=len(messages),
            estimated_tokens=self.count_tokens(messages),
            prompt_eval_count=prompt_eval_count,
            prompt_eval_ms=prompt_eval_ms,
            eval_count=eval_count,
            answer=content,
            outcome=outcome,
        )
        self._records = (*self._records, record)
        return record

    def commit_turn(
        self,
        sent: Sequence[Message],
        produced: Sequence[Message],
        *,
        prompt_eval_count: int | None = None,
        prompt_eval_ms: float | None = None,
        eval_count: int | None = None,
    ) -> TurnRecord:
        """Записать законченный ход из нескольких сообщений: вопрос — последнее из sent
        (как его вернул outgoing), produced — всё, что дал цикл, до ответа модели включительно.

        Счётчики — за весь ход (сумма по обращениям). Ход, который не кончился ответом модели,
        не записывается: BrokenHistoryError.
        """
        last = produced[-1] if produced else None
        if last is None or last.get("role") != "assistant" or last.get("tool_calls"):
            raise BrokenHistoryError("ход записывается только законченным: последним — ответ")
        validate_alternation([*self._system(), *self._history, sent[-1], *produced], outgoing=False)
        return self._record(
            sent,
            produced,
            str(last.get("content") or ""),
            outcome="stop",
            prompt_eval_count=prompt_eval_count,
            prompt_eval_ms=prompt_eval_ms,
            eval_count=eval_count,
        )

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
        duration = getattr(reply, "prompt_eval_duration", None)
        return self._commit(
            messages,
            content,
            outcome="stop",
            prompt_eval_count=getattr(reply, "prompt_eval_count", None),
            prompt_eval_ms=duration / 1e6 if duration is not None else None,
            eval_count=getattr(reply, "eval_count", None),
        )

    def _keep_partial(
        self, messages: Sequence[Message], text: str, reason: str, partial: PartialPolicy
    ) -> None:
        if partial is PartialPolicy.DISCARD:
            return
        content = (
            text + PARTIAL_MARK.format(reason=reason) if partial is PartialPolicy.MARK else text
        )
        self._commit(
            messages,
            content,
            outcome=f"partial:{partial.value}",
            prompt_eval_count=None,
            prompt_eval_ms=None,
            eval_count=None,
        )

    def ask_stream(
        self,
        client: StreamingChatClient,
        model: str,
        options: Mapping[str, Any] | None = None,
        *,
        partial: PartialPolicy = PartialPolicy.DISCARD,
        **values: str,
    ) -> Generator[str, None, None]:
        """Потоковый ход: отдаёт дельты; законченный ответ записывается после последней.

        Поток оборвался (StreamIncomplete, StreamTruncated, ошибка сервера посреди текста)
        или потребитель бросил чтение — в историю идёт то, что велит partial, а исключение
        летит дальше. Брошенный генератор закрывай явно (contextlib.closing или .close()):
        иначе политика применится, когда до него доберётся сборщик мусора.
        """
        messages = self.outgoing(**values)
        stream = stream_chat(client, model, messages, options)
        try:
            yield from stream
        except GeneratorExit:
            self._keep_partial(messages, stream.text, "чтение прервано", partial)
            raise
        except Exception as error:
            self._keep_partial(messages, stream.text, type(error).__name__, partial)
            raise
        finally:
            stream.close()
        envelope: Envelope = stream.envelope
        self._commit(
            messages,
            stream.text,
            outcome="stop",
            prompt_eval_count=envelope.prompt_eval_count,
            prompt_eval_ms=envelope.prompt_eval_ns / 1e6,
            eval_count=envelope.eval_count,
        )
