"""Промпты с версией: версия доезжает до ответа, шаблон не принимает чужих полей."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from ragkit.llm import ChatReply, IncompleteResponseError
from ragkit.prompts import DOCUMENT_QA, PROMPTS, SHORT_ANSWER, Prompt, ask_prompt


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeReply:
    done_reason: str | None
    message: FakeMessage


@dataclass
class FakeClient:
    reply: FakeReply
    calls: list[Sequence[Mapping[str, Any]]] = field(default_factory=list)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> ChatReply:
        self.calls.append(messages)
        return self.reply


def test_answer_carries_prompt_version() -> None:
    client = FakeClient(FakeReply("stop", FakeMessage("Растения делают сахар из света.")))
    answer = ask_prompt(client, "qwen2.5:7b", SHORT_ANSWER, question="Что такое фотосинтез?")
    assert (answer.prompt, answer.prompt_version) == ("short_answer", "1")
    assert answer.text == "Растения делают сахар из света."
    assert client.calls[0][0] == {"role": "system", "content": SHORT_ANSWER.system}


def test_new_version_is_visible_in_answer() -> None:
    v2 = Prompt("short_answer", "2", "Отвечай одним предложением.", "{question}")
    client = FakeClient(FakeReply("stop", FakeMessage("Сахар из света.")))
    assert ask_prompt(client, "m", v2, question="?").prompt_version == "2"


def test_cut_answer_names_the_prompt() -> None:
    client = FakeClient(FakeReply("length", FakeMessage("Растения дел")))
    with pytest.raises(IncompleteResponseError, match=r"document_qa v1"):
        ask_prompt(client, "m", DOCUMENT_QA, document="текст", question="о чём?")


@pytest.mark.parametrize(
    "values",
    [{"document": "текст"}, {"document": "т", "question": "в", "лишнее": "x"}],
    ids=["missing", "extra"],
)
def test_template_fields_must_match(values: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="document_qa v1"):
        DOCUMENT_QA.messages(**values)


def test_registry_names_are_unique_and_versioned() -> None:
    assert all(p.name == name and p.version for name, p in PROMPTS.items())
