"""complete_chat: оборванный ответ — исключение, законченный — текст. Без сети и без модели."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from ragkit.llm import ChatReply, IncompleteResponseError, complete_chat


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeReply:
    done_reason: str | None
    message: FakeMessage


@dataclass
class FakeClient:
    """Фейк ollama.Client: отдаёт заданный ответ и запоминает, с чем его позвали."""

    reply: FakeReply
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> ChatReply:
        self.calls.append({"model": model, "messages": messages, "options": options})
        return self.reply


MESSAGES = [{"role": "user", "content": "Расскажи историю."}]


def test_finished_answer_is_returned() -> None:
    client = FakeClient(FakeReply("stop", FakeMessage("Жили-были.")))
    assert complete_chat(client, "qwen2.5:7b", MESSAGES, {"num_predict": 50}) == "Жили-были."
    assert client.calls == [
        {"model": "qwen2.5:7b", "messages": MESSAGES, "options": {"num_predict": 50}}
    ]


def test_cut_by_length_raises() -> None:
    client = FakeClient(FakeReply("length", FakeMessage("Жили-были дед и ба")))
    with pytest.raises(IncompleteResponseError, match=r"done_reason='length', 18 символов"):
        complete_chat(client, "qwen2.5:7b", MESSAGES)


@pytest.mark.parametrize(
    "reason", [None, "", "load", "unload"], ids=["none", "empty", "load", "unload"]
)
def test_anything_but_stop_raises(reason: str | None) -> None:
    client = FakeClient(FakeReply(reason, FakeMessage("текст")))
    with pytest.raises(IncompleteResponseError, match="ответ неполный"):
        complete_chat(client, "qwen2.5:7b", MESSAGES)


def test_empty_finished_answer_is_empty_string() -> None:
    client = FakeClient(FakeReply("stop", FakeMessage(None)))
    assert complete_chat(client, "qwen2.5:7b", MESSAGES) == ""
