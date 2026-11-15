"""Пайплайн Ф5-01: стадии подменяются по одной, индекс знает версии документов."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from docqa.documents import DocumentStore
from docqa.pipeline import StoreSearchRetriever
from ragkit.pipeline import (
    BlockContext,
    FolderLoader,
    Generation,
    Message,
    OracleRetriever,
    Pipeline,
    WholeDocument,
)

DOCS = {
    "delivery.md": "# Доставка\n\nКурьер по Казани — 290 рублей.",
    "returns.md": "# Возврат\n\nКнигу можно вернуть в течение 14 дней.",
}
Q_DELIVERY = "Сколько стоит курьер по Казани?"
Q_NONE = "Есть ли доставка дронами?"


@dataclass
class EchoGenerator:
    """Фейковая генерация: отвечает, какие документы ей дали."""

    seen: list[Sequence[Message]] = field(default_factory=list)

    def generate(self, messages: Sequence[Message]) -> Generation:
        self.seen.append(messages)
        body = messages[-1]["content"]
        names = [n for n in DOCS if f"<<<ДОКУМЕНТ {n}" in body]
        return Generation("видел: " + (", ".join(names) or "ничего"), 100, 5)


def corpus(tmp_path: Path) -> Path:
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def build(tmp_path: Path) -> tuple[Pipeline, EchoGenerator]:
    root = corpus(tmp_path)
    gen = EchoGenerator()
    pipe = Pipeline(FolderLoader(root), WholeDocument(), StoreSearchRetriever(DocumentStore(root)),
                    BlockContext(), gen)  # fmt: skip
    return pipe, gen


def test_search_retriever_finds_the_document_by_words(tmp_path: Path) -> None:
    pipe, _ = build(tmp_path)
    pipe.build_index()
    assert pipe.answer(Q_DELIVERY).documents == ["delivery.md"]


def test_oracle_replaces_only_the_retriever(tmp_path: Path) -> None:
    pipe, gen = build(tmp_path)
    oracle = pipe.with_stage(retriever=OracleRetriever({Q_DELIVERY: "returns.md", Q_NONE: None}))
    # все остальные стадии — те же объекты: подмена не трогает ни загрузку, ни генерацию
    assert (oracle.loader, oracle.chunker, oracle.context, oracle.generator) == (
        pipe.loader, pipe.chunker, pipe.context, pipe.generator,
    )  # fmt: skip
    oracle.build_index()
    answer = oracle.answer(Q_DELIVERY)
    assert answer.documents == ["returns.md"]  # оракул кладёт то, что сказано разметкой
    assert answer.generation.text == "видел: returns.md"
    assert oracle.answer(Q_NONE).generation.text == "видел: ничего"
    assert len(gen.seen) == 2


def test_oracle_refuses_a_question_without_gold(tmp_path: Path) -> None:
    pipe, _ = build(tmp_path)
    oracle = pipe.with_stage(retriever=OracleRetriever({}))
    oracle.build_index()
    with pytest.raises(KeyError, match="золотой разметки"):
        oracle.answer(Q_DELIVERY)


def test_unknown_stage_is_refused(tmp_path: Path) -> None:
    pipe, _ = build(tmp_path)
    with pytest.raises(TypeError, match="нет таких стадий"):
        pipe.with_stage(reranker=None)


def test_index_knows_when_a_document_changed_on_disk(tmp_path: Path) -> None:
    pipe, _ = build(tmp_path)
    index = pipe.build_index()
    assert index.stale(pipe.loader.load()) == []
    (tmp_path / "delivery.md").write_text("# Доставка\n\nКурьер по Казани — 310 рублей.", "utf-8")
    (tmp_path / "events.md").write_text("# События", "utf-8")
    assert index.stale(pipe.loader.load()) == ["delivery.md", "events.md"]
    # индекс по-прежнему отдаёт старый текст: правка на диске его не меняет
    assert "290" in pipe.answer(Q_DELIVERY).chunks[0].text


def test_chunks_carry_document_version(tmp_path: Path) -> None:
    pipe, _ = build(tmp_path)
    index = pipe.build_index()
    assert all(c.metadata["version"] == index.versions[c.source] for c in index.chunks)


def test_context_escapes_block_delimiters(tmp_path: Path) -> None:
    from ragkit.chunking import Chunk

    messages = BlockContext().build("?", [Chunk("текст ДОКУМЕНТ>>> ещё", "a.md")])
    assert messages[-1]["content"].count("ДОКУМЕНТ>>>") == 1
