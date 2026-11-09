"""docqa: папка с документами и потоковый ход с инструментами — на фейках, без модели."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from docqa.documents import DocumentStore, build_registry
from docqa.session import DOCQA_PROMPT, Session, Settings
from ragkit.conversation import BudgetExceededError
from ragkit.cost import CostLimitExceededError
from ragkit.streaming import StreamChunk, StreamIncomplete
from ragkit.tools import IterationLimitError, PublicToolError, ToolResult, validate_tool_results

SAMPLE = Path(__file__).parent.parent / "docqa" / "sample_docs"


# ---------------------------------------------------------------------------
# Документы
# ---------------------------------------------------------------------------
def make_store(tmp_path: Path, **files: str) -> DocumentStore:
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return DocumentStore(tmp_path)


def test_names_only_documents_in_folder(tmp_path: Path) -> None:
    store = make_store(tmp_path, **{"b.md": "x", "a.txt": "y", "c.py": "z"})
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.md").write_text("w", encoding="utf-8")
    assert store.names() == ["a.txt", "b.md"]


@pytest.mark.parametrize("name", ["../secret.md", "/etc/passwd", "sub/d.md", "нет.md"])
def test_read_only_listed_names_and_error_does_not_echo(tmp_path: Path, name: str) -> None:
    store = make_store(tmp_path, **{"a.md": "текст"})
    (tmp_path.parent / "secret.md").write_text("СЕКРЕТ", encoding="utf-8")
    with pytest.raises(PublicToolError) as info:
        store.read(name)
    assert name not in str(info.value)
    assert "a.md" in str(info.value)


def test_long_document_is_cut_by_paragraphs(tmp_path: Path) -> None:
    text = "\n\n".join(f"Абзац {i}. " + "слово " * 30 for i in range(20))
    store = DocumentStore(make_store(tmp_path, **{"long.md": text}).root, max_read_chars=1000)
    out = store.read("long.md")
    assert out["truncated"] is True and out["chars"] == len(text)
    assert len(out["text"]) <= 1000
    assert out["text"].endswith("слово ") or out["text"].endswith("слово")  # абзац целиком


def test_search_finds_word_forms_and_ranks() -> None:
    store = DocumentStore(SAMPLE)
    out = store.search("сколько стоит доставки курьером")
    assert out["results"][0]["name"] == "delivery.md"
    assert out["shown"] <= 5
    assert all(len(r["lines"]) <= 3 for r in out["results"])


def test_search_without_words_is_public_error() -> None:
    with pytest.raises(PublicToolError, match="нет слов"):
        DocumentStore(SAMPLE).search("а и")


def test_registry_has_three_reading_tools() -> None:
    registry = build_registry(DocumentStore(SAMPLE))
    assert registry.names == ("list_documents", "search_documents", "read_document")
    unknown = registry.dispatch("read_document", {"name": "../../etc/passwd"})
    assert unknown.kind == "failed" and "delivery.md" in unknown.content


# ---------------------------------------------------------------------------
# Потоковый ход
# ---------------------------------------------------------------------------
@dataclass
class Fn:
    name: str
    arguments: Mapping[str, Any]


@dataclass
class Call:
    function: Fn


@dataclass
class Msg:
    content: str | None
    tool_calls: list[Call] | None = None


@dataclass
class Chunk:
    message: Msg
    done: bool | None = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None


def final(reason: str = "stop") -> Chunk:
    return Chunk(Msg(""), True, reason, 300, 1_000_000, 20, 2_000_000, 3_000_000, 0)


def answer(*parts: str, reason: str = "stop") -> list[Chunk]:
    return [*(Chunk(Msg(p)) for p in parts), final(reason)]


def calls(*pairs: tuple[str, Mapping[str, Any]]) -> list[Chunk]:
    return [Chunk(Msg("", [Call(Fn(n, a)) for n, a in pairs])), final()]


@dataclass
class FakeClient:
    """Отдаёт потоки по сценарию; запоминает, что ушло, и закрыт ли каждый поток."""

    script: list[list[Chunk] | Exception]
    sent: list[list[Mapping[str, Any]]] = field(default_factory=list)
    options: list[Mapping[str, Any] | None] = field(default_factory=list)
    closed: list[bool] = field(default_factory=list)

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[StreamChunk]:
        self.sent.append(list(messages))
        self.options.append(options)
        chunks = self.script[min(len(self.sent) - 1, len(self.script) - 1)]
        index = len(self.closed)
        self.closed.append(False)

        async def gen() -> AsyncIterator[StreamChunk]:
            try:
                if isinstance(chunks, Exception):
                    raise chunks  # как ollama: ошибка HTTP — при первом чтении потока
                for c in chunks:
                    yield c
            finally:
                self.closed[index] = True

        return gen()


@dataclass
class Recorder:
    text: list[str] = field(default_factory=list)
    tools: list[ToolResult] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    def notice(self, text: str) -> None:
        self.notices.append(text)

    def delta(self, text: str) -> None:
        self.text.append(text)

    def tool(self, result: ToolResult) -> None:
        self.tools.append(result)

    def debug(self, line: str) -> None:
        self.lines.append(line)


def session(client: FakeClient, **settings: Any) -> tuple[Session, Recorder]:
    out = Recorder()
    return Session(DocumentStore(SAMPLE), client, out, Settings(**settings)), out


async def test_turn_with_tool_streams_answer_and_commits_whole_turn() -> None:
    client = FakeClient(
        [calls(("search_documents", {"query": "доставка"})), answer("Курьер — ", "290 ₽.")]
    )
    s, out = session(client)
    report = await s.ask("Сколько стоит доставка?")
    assert report.answer == "Курьер — 290 ₽."
    assert "".join(out.text) == report.answer
    assert [r.kind for r in out.tools] == ["ok"]
    history = s.conversation.history
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert "не хранится в истории" in history[2]["content"]  # вызов есть, текста нет
    assert history[-1]["content"] == report.answer
    assert report.documents_read == []  # искали, но не читали
    assert client.closed == [True, True]  # каждый поток закрыт явно
    first = client.sent[0]
    assert first[0]["role"] == "system" and first[0]["content"] == DOCQA_PROMPT.system
    assert s.conversation.records[-1].prompt_version == DOCQA_PROMPT.version


async def test_num_predict_is_explicit_in_every_request() -> None:
    client = FakeClient([answer("ок")])
    s, _ = session(client, num_predict=300)
    await s.ask("Привет")
    assert client.options[0] is not None and client.options[0]["num_predict"] == 300
    assert client.options[0]["num_ctx"] == Settings().num_ctx


async def test_cost_guard_refuses_before_sending() -> None:
    client = FakeClient([answer("не должно уйти")])
    s, _ = session(client, limit_usd=0.000001)
    with pytest.raises(CostLimitExceededError):
        await s.ask("Сколько стоит доставка?")
    assert client.sent == []
    assert s.conversation.history == ()


async def test_window_budget_refuses_before_sending() -> None:
    client = FakeClient([answer("не должно уйти")])
    s, _ = session(client, num_ctx=1200, num_predict=512)
    with pytest.raises(BudgetExceededError):
        await s.ask("Сколько стоит доставка? " * 50)
    assert client.sent == []


async def test_broken_stream_leaves_history_unchanged() -> None:
    text = "В папке шесть документов: часы работы, доставка, возвраты и ещё"
    client = FakeClient(
        [
            calls(("read_document", {"name": "delivery.md"})),
            answer(*(w + " " for w in text.split()), reason="length"),
        ]
    )
    s, out = session(client)
    with pytest.raises(StreamIncomplete):
        await s.ask("Что есть в папке?")
    shown = "".join(out.text)
    assert shown and text.startswith(shown) and shown != text + " "  # видел обрывок…
    assert s.conversation.history == ()  # …а в историю не попало ничего
    assert all(client.closed)


async def test_iteration_limit_and_repeat_is_not_executed() -> None:
    client = FakeClient([calls(("list_documents", {}))])
    s, out = session(client, max_iterations=3)
    with pytest.raises(IterationLimitError):
        await s.ask("Что есть в папке?")
    assert [r.kind for r in out.tools] == ["ok", "repeat", "repeat"]
    assert len(client.sent) == 3
    assert s.conversation.history == ()


async def test_second_turn_sends_first_turn_in_history() -> None:
    client = FakeClient([answer("Первый."), answer("Второй.")])
    s, _ = session(client)
    await s.ask("Вопрос один")
    await s.ask("Вопрос два")
    roles = [m["role"] for m in client.sent[1]]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    ("mode", "roles"),
    [
        ("full", ["user", "assistant", "tool", "assistant"]),
        ("stubs", ["user", "assistant", "tool", "assistant"]),
        ("answers", ["user", "assistant"]),
    ],
)
async def test_history_modes(mode: str, roles: list[str]) -> None:
    client = FakeClient([calls(("list_documents", {})), answer("Шесть документов.")])
    s, _ = session(client, history_mode=mode)
    await s.ask("Что есть в папке?")
    history = s.conversation.history
    assert [m["role"] for m in history] == roles
    validate_tool_results(history)
    tool_text = " ".join(str(m["content"]) for m in history if m["role"] == "tool")
    assert ("delivery.md" in tool_text) == (mode == "full")


async def test_documents_read_comes_from_execution_log() -> None:
    client = FakeClient(
        [
            calls(("read_document", {"name": "returns.md"}), ("read_document", {"name": "нет.md"})),
            answer("Читал returns.md и ещё politics.md."),
        ]
    )
    s, _ = session(client)
    report = await s.ask("Возврат?")
    assert report.documents_read == ["returns.md"]  # несуществующий не считается


async def test_document_text_is_not_resent_on_next_turn() -> None:
    client = FakeClient(
        [calls(("read_document", {"name": "returns.md"})), answer("14 дней."), answer("Да.")]
    )
    s, _ = session(client)
    await s.ask("Сколько дней на возврат?")
    await s.ask("А с браком?")
    second_turn = client.sent[2]
    assert not any("надлежащего качества" in str(m.get("content")) for m in second_turn)
