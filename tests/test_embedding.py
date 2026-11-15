"""ragkit/embedding.py (Ф5-02): нормировка, кеш по (отпечаток, текст), запрет смешивать модели."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragkit.embedding import (
    EmbeddingCache,
    HashEmbedder,
    MixedModelsError,
    OllamaEmbedder,
    VectorRetriever,
    Vectors,
    VectorSpace,
    embed_texts,
    normalize,
)
from ragkit.pipeline import Doc, Paragraphs

VOCAB = ("доставка", "курьер", "возврат", "чек", "сертификат", "баллы")


@dataclass
class BagEmbedder:
    """Мешок слов по словарю VOCAB, ненормированный: нормировать — дело embed_texts."""

    name: str = "bag"
    calls: list[list[str]] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(VOCAB) + 1

    @property
    def fingerprint(self) -> str:
        return f"{self.name}@test"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[3.0 * text.count(w) for w in VOCAB] + [0.5] for text in texts]


def test_normalize_gives_unit_rows_and_refuses_zero_vector() -> None:
    m = normalize([[3.0, 4.0], [0.0, 2.0]])
    assert m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    with pytest.raises(ValueError, match="нулевой"):
        normalize([[0.0, 0.0]])


def test_cosine_is_a_dot_product_and_self_similarity_is_one() -> None:
    v = Vectors("m@1", embed_texts(BagEmbedder(), ["доставка курьер", "возврат чек"]))
    sims = v.cosine(v)
    assert np.allclose(np.diag(sims), 1.0, atol=1e-6)
    assert np.isclose(sims[0, 1], sims[1, 0])


def test_vectors_of_two_models_cannot_be_compared() -> None:
    a = VectorSpace(BagEmbedder("a")).queries(["доставка"])
    b = VectorSpace(BagEmbedder("b")).passages(["доставка"])
    with pytest.raises(MixedModelsError, match="разных пространств"):
        a.cosine(b)


def test_unnormalized_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="не нормированы"):
        Vectors("m@1", np.array([[3.0, 4.0]], dtype=np.float32))


def test_cache_serves_repeats_and_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    embedder = BagEmbedder()
    first = embed_texts(embedder, ["доставка", "чек", "доставка"], EmbeddingCache(path))
    assert embedder.calls == [["доставка", "чек"]]  # повтор внутри вызова — один раз
    reloaded = EmbeddingCache(path)
    again = embed_texts(embedder, ["чек", "доставка"], reloaded)
    assert len(embedder.calls) == 1 and reloaded.hits == 2
    assert np.array_equal(again, first[[1, 0]])


def test_cache_never_returns_another_models_vector(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    embed_texts(BagEmbedder("a"), ["доставка"], cache)
    other = BagEmbedder("b")
    embed_texts(other, ["доставка"], cache)
    assert other.calls == [["доставка"]]  # тот же текст, другая модель — промах кеша
    assert len(cache) == 2


def test_space_prefixes_change_text_but_not_the_space() -> None:
    embedder = BagEmbedder()
    space = VectorSpace(embedder, query_prefix="query: ", passage_prefix="passage: ")
    q, p = space.queries(["доставка"]), space.passages(["доставка"])
    assert embedder.calls == [["query: доставка"], ["passage: доставка"]]
    q.cosine(p)  # одно пространство: сравнивать можно
    with pytest.raises(MixedModelsError):
        q.cosine(VectorSpace(embedder).passages(["доставка"]))  # без приставок — другое


@dataclass
class FakeEmbedClient:
    batches: list[list[str]] = field(default_factory=list)

    def embed(self, *, model: str, input: Sequence[str], truncate: bool) -> Any:  # noqa: A002
        assert truncate is False
        self.batches.append(list(input))

        @dataclass
        class Reply:
            embeddings: list[list[float]]
            prompt_eval_count: int

        return Reply([[1.0, float(len(t)), 0.0] for t in input], 3 * len(input))


def test_ollama_embedder_sends_batches_and_keeps_digest_in_fingerprint() -> None:
    client = FakeEmbedClient()
    embedder = OllamaEmbedder(client, "bge-m3", "sha256:5f3ab1c2d4e5f6a7b8", batch_size=32)
    texts = [f"текст {i}" for i in range(70)]
    vectors = embed_texts(embedder, texts)
    assert [len(b) for b in client.batches] == [32, 32, 6]
    assert embedder.fingerprint == "bge-m3@5f3ab1c2d4e5"
    assert vectors.shape == (70, 3) and embedder.prompt_tokens == 210
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert max(embedder.raw_norms) > 1  # сервер прислал ненормированные — нормировали мы


def test_vector_retriever_finds_the_chunk_by_meaning_of_its_words() -> None:
    docs = [Doc("a.md", "доставка курьер курьер"), Doc("b.md", "возврат чек"), Doc("c.md", "баллы")]
    chunks = Paragraphs().chunk(docs)
    retriever = VectorRetriever(VectorSpace(BagEmbedder()))
    retriever.index(chunks)
    assert [c.source for c in retriever.retrieve("курьер", 2)][0] == "a.md"
    assert [c.source for c in retriever.retrieve("чек и возврат", 1)] == ["b.md"]


def test_paragraphs_keep_offsets_and_size() -> None:
    text = "# Заголовок\n\n" + "\n\n".join(f"Абзац {i}. " + "слово " * 30 for i in range(6))
    chunks = Paragraphs(max_chars=500).chunk([Doc("d.md", text)])
    assert len(chunks) > 1
    for c in chunks:
        start, end = c.metadata["start"], c.metadata["end"]
        assert isinstance(start, int) and isinstance(end, int)
        assert text[start:end] == c.text and len(c.text) <= 500
    assert chunks[0].text.startswith("# Заголовок")  # заголовок едет со следующим абзацем
    assert "Абзац 5" in chunks[-1].text


def test_hash_embedder_still_fits_the_protocol() -> None:
    v = embed_texts(HashEmbedder(dim=4), ["а", "б"])
    assert v.shape == (2, 4) and HashEmbedder(dim=4).fingerprint == "hash-4"
