"""Conversation и политики обрезки: на фейке, без сети и без модели."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from ragkit.conversation import (
    BrokenHistoryError,
    BudgetExceededError,
    CharRatioEstimate,
    Conversation,
    FirstAndLastTurns,
    KeepAll,
    LastTurns,
    Message,
    apply_policy,
    validate_alternation,
)
from ragkit.llm import ChatReply, IncompleteResponseError
from ragkit.prompts import Prompt

CHAT = Prompt("test_chat", "3", "Отвечай коротко.", "{question}")


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeReply:
    done_reason: str | None
    message: FakeMessage
    prompt_eval_count: int | None = None


@dataclass
class ScriptedClient:
    """Отвечает «ответ 1», «ответ 2», … и запоминает, что ему отправили."""

    done_reason: str = "stop"
    calls: list[list[Mapping[str, Any]]] = field(default_factory=list)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> ChatReply:
        self.calls.append(list(messages))
        return FakeReply(self.done_reason, FakeMessage(f"ответ {len(self.calls)}"), 7)


def words(messages: Sequence[Message]) -> int:
    """Счётчик для тестов: одно слово — один «токен»."""
    return sum(len(str(m["content"]).split()) for m in messages)


def talk(conversation: Conversation, client: ScriptedClient, turns: int) -> None:
    for i in range(1, turns + 1):
        conversation.ask(client, "m", question=f"вопрос {i}")


def roles(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(m["role"]) for m in messages]


# ---------------------------------------------------------------------------
# Валидатор
# ---------------------------------------------------------------------------
S: Message = {"role": "system", "content": "s"}
U: Message = {"role": "user", "content": "u"}
A: Message = {"role": "assistant", "content": "a"}
CALL: Message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]}
RESULT: Message = {"role": "tool", "content": "42"}


@pytest.mark.parametrize(
    "messages",
    [
        [S, U, U, A, U],  # два user подряд
        [S, U, A, U, U],  # ответ на прошлый вопрос выброшен, вопрос остался
        [S, A, U, A, U],  # начинается с assistant
        [U, A, S, U],  # system не первым
        [S, U, A, A, U],  # два assistant подряд
        [S, U, CALL, U],  # вызов инструмента без результата
        [S, U, A],  # последним уходит ответ модели
    ],
    ids=[
        "two-users",
        "dropped-answer",
        "starts-with-assistant",
        "late-system",
        "two-assistants",
        "call-without-result",
        "ends-with-assistant",
    ],
)
def test_validator_rejects_broken_forms(messages: list[Message]) -> None:
    with pytest.raises(BrokenHistoryError):
        validate_alternation(messages)


@pytest.mark.parametrize(
    "messages",
    [[S, U], [U], [S, U, A, U], [S, U, CALL, RESULT, A, U], [S, U, CALL, RESULT]],
    ids=["first-turn", "no-system", "dialog", "tool-turn", "tool-result-last"],
)
def test_validator_accepts_correct_forms(messages: list[Message]) -> None:
    validate_alternation(messages)


# ---------------------------------------------------------------------------
# Обрезка
# ---------------------------------------------------------------------------
def test_trim_keeps_system_and_first_turn() -> None:
    client = ScriptedClient()
    conversation = Conversation(
        CHAT, budget=10_000, count_tokens=words, policy=FirstAndLastTurns(2)
    )
    talk(conversation, client, 6)
    sent = client.calls[-1]
    assert sent[0] == {"role": "system", "content": "Отвечай коротко."}
    assert [m["content"] for m in sent[1:3]] == ["вопрос 1", "ответ 1"]
    assert [m["content"] for m in sent[3:]] == [
        "вопрос 4",
        "ответ 4",
        "вопрос 5",
        "ответ 5",
        "вопрос 6",
    ]


def test_trim_never_splits_tool_call_from_result() -> None:
    history: list[Message] = [S, U, A, U, CALL, RESULT, A, U, A]
    for k in range(4):
        kept = apply_policy(history, LastTurns(k))
        validate_alternation([*kept, U])
        if CALL in kept:
            assert kept[kept.index(CALL) + 1] == RESULT
            assert kept[kept.index(CALL) - 1]["role"] == "user"


def test_last_turns_window_slides() -> None:
    client = ScriptedClient()
    conversation = Conversation(CHAT, budget=10_000, count_tokens=words, policy=LastTurns(2))
    talk(conversation, client, 5)
    assert roles(client.calls[-1]) == ["system", "user", "assistant", "user", "assistant", "user"]
    assert client.calls[-1][1]["content"] == "вопрос 3"


# ---------------------------------------------------------------------------
# Бюджет
# ---------------------------------------------------------------------------
def test_over_budget_raises_and_sends_nothing() -> None:
    client = ScriptedClient()
    conversation = Conversation(CHAT, budget=11, count_tokens=words)
    talk(conversation, client, 2)  # system 2 + ход 4 + вопрос 2 = 8 слов — влезает
    # третий ход: 2 + 4 + 4 + 2 = 12 слов при бюджете 11
    with pytest.raises(BudgetExceededError, match=r"test_chat v3: по оценке 12 .* бюджете 11"):
        conversation.ask(client, "m", question="вопрос 3")
    assert len(client.calls) == 2
    assert len(conversation.history) == 4


def test_trimming_policy_keeps_long_dialog_in_budget() -> None:
    client = ScriptedClient()
    conversation = Conversation(CHAT, budget=12, count_tokens=words, policy=LastTurns(2))
    talk(conversation, client, 10)
    assert len(client.calls) == 10
    assert len(conversation.history) == 20  # хранится всё, обрезается только отправка


def test_char_ratio_estimate() -> None:
    estimate = CharRatioEstimate(chars_per_token=2.0, per_message=5, base=3)
    assert estimate([{"role": "user", "content": "абвгд"}]) == 3 + 5 + 3


# ---------------------------------------------------------------------------
# История и записи
# ---------------------------------------------------------------------------
def test_history_only_grows_at_the_end() -> None:
    client = ScriptedClient()
    conversation = Conversation(CHAT, budget=10_000, count_tokens=words)
    talk(conversation, client, 2)
    before = conversation.history
    talk(conversation, client, 1)
    assert conversation.history[: len(before)] == before
    with pytest.raises(TypeError):
        conversation.history[0]["content"] = "правка задним числом"  # type: ignore[index]


def test_every_turn_records_prompt_version() -> None:
    client = ScriptedClient()
    conversation = Conversation(CHAT, budget=10_000, count_tokens=words, policy=LastTurns(1))
    talk(conversation, client, 3)
    assert [(r.number, r.prompt, r.prompt_version) for r in conversation.records] == [
        (1, "test_chat", "3"),
        (2, "test_chat", "3"),
        (3, "test_chat", "3"),
    ]
    assert conversation.records[-1].sent_messages == 4
    assert conversation.records[-1].prompt_eval_count == 7
    assert conversation.records[-1].policy == "LastTurns(k=1)"


def test_cut_answer_is_not_stored() -> None:
    client = ScriptedClient(done_reason="length")
    conversation = Conversation(CHAT, budget=10_000, count_tokens=words)
    with pytest.raises(IncompleteResponseError, match="test_chat v3, ход 1"):
        conversation.ask(client, "m", question="вопрос")
    assert conversation.history == ()


def test_append_rejects_broken_history() -> None:
    conversation = Conversation(CHAT, budget=10_000, count_tokens=words, policy=KeepAll())
    conversation.append(U, A)
    with pytest.raises(BrokenHistoryError):
        conversation.append(A)
    assert len(conversation.history) == 2
