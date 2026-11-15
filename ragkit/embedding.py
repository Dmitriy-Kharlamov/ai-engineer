"""Эмбеддинги: протокол Embedder, ollama, нормировка, кеш, поиск умножением матриц (Ф5-02).

Протокол Embedder (Ф3-05): под него подходит любой объект с dim, fingerprint и embed() —
OpenAI, локальная модель, фейк в тестах; наследоваться не нужно, mypy сверяет структурно.

Правило отпечатка (Ф5-01, задание 5): вектор, сохранённый на диск, несёт отпечаток модели, которой
он посчитан. Векторы двух разных моделей — разные пространства: косинус между ними считается без
ошибок и ничего не значит. Поэтому Vectors.cosine() сверяет отпечатки и падает громко
(MixedModelsError), а ключ кеша — (отпечаток модели, отправленный текст).

Нормировка — один раз, при сохранении: в кеше и в Vectors строки единичной длины, косинус —
скалярное произведение, поиск по корпусу — одно умножение матрицы на вектор.

VectorSpace — модель плюс схема приставок (query: / passage:, если их требует карточка модели).
Вопросы и чанки одного пространства сравнивать можно, хотя тексты им отправлены разные; векторы
пространства с другой схемой приставок — уже нельзя.
"""

import base64
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .chunking import Chunk
from .exceptions import RAGError

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...  # всё, что меняет вектор: модель и файл весов

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class MixedModelsError(RAGError):
    """Сравнение векторов из разных пространств: косинус посчитался бы и ничего бы не значил."""


class HashEmbedder:
    """Детерминированный «эмбеддер» без сети — для тестов. Embedder не наследует."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"hash-{self._dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [digest[i % len(digest)] / 255 - 0.5 for i in range(self._dim)]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


def embed_chunks(embedder: Embedder, chunks: Sequence[Chunk]) -> list[list[float]]:
    vectors = embedder.embed([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks) or any(len(v) != embedder.dim for v in vectors):
        raise ValueError("эмбеддер вернул векторы не той формы")
    return vectors


# ---------------------------------------------------------------------------
# Векторы
# ---------------------------------------------------------------------------
def normalize(rows: Sequence[Sequence[float]] | Matrix) -> Matrix:
    """Строки единичной длины, float32. Нулевой вектор направления не имеет — это ошибка."""
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"нужна матрица строк-векторов, пришла форма {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("нулевой вектор: нормировать нечего")
    return (matrix / norms).astype(np.float32)


@dataclass(frozen=True)
class Vectors:
    """Нормированные векторы одного пространства: строка i — текст i."""

    fingerprint: str
    matrix: Matrix

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(f"Vectors: нужна матрица, пришла форма {self.matrix.shape}")
        norms = np.linalg.norm(self.matrix, axis=1)
        if len(norms) and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "Vectors: строки не нормированы — косинус не будет скалярным произведением"
            )

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def cosine(self, other: "Vectors") -> Matrix:
        """Матрица косинусов len(self) × len(other): одно умножение, без деления на нормы."""
        if other.fingerprint != self.fingerprint:
            raise MixedModelsError(
                f"векторы из разных пространств: {self.fingerprint} и {other.fingerprint}"
            )
        result: Matrix = self.matrix @ other.matrix.T
        return result

    def rows(self, indices: Sequence[int]) -> "Vectors":
        return Vectors(self.fingerprint, self.matrix[list(indices)])


# ---------------------------------------------------------------------------
# Кеш
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingCache:
    """Нормированные векторы по ключу (отпечаток модели, отправленный текст); файл — JSONL.

    Запись другой модели под тем же текстом — другой ключ: она никогда не вернётся вместо своей.
    path=None — только в памяти.
    """

    path: Path | None = None
    hits: int = 0
    misses: int = 0
    _vectors: dict[str, Matrix] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                raw = base64.b64decode(record["vector"])
                self._vectors[record["key"]] = np.frombuffer(raw, dtype=np.float32).copy()

    @staticmethod
    def key(fingerprint: str, text: str) -> str:
        return hashlib.sha256(f"{fingerprint}\x00{text}".encode()).hexdigest()

    def get(self, fingerprint: str, text: str) -> Matrix | None:
        vector = self._vectors.get(self.key(fingerprint, text))
        if vector is None:
            self.misses += 1
        else:
            self.hits += 1
        return vector

    def put(self, fingerprint: str, text: str, vector: Matrix) -> None:
        key = self.key(fingerprint, text)
        self._vectors[key] = vector
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "key": key,
                "fingerprint": fingerprint,
                "dim": int(vector.shape[0]),
                "vector": base64.b64encode(vector.astype(np.float32).tobytes()).decode(),
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def __len__(self) -> int:
        return len(self._vectors)

    @property
    def file_bytes(self) -> int:
        return self.path.stat().st_size if self.path is not None and self.path.exists() else 0


def embed_texts(
    embedder: Embedder, texts: Sequence[str], cache: EmbeddingCache | None = None
) -> Matrix:
    """Нормированные векторы текстов; каждый уникальный текст уходит в модель не больше раза."""
    fingerprint = embedder.fingerprint
    store = cache if cache is not None else EmbeddingCache()
    found: dict[str, Matrix] = {}
    missing: list[str] = []
    for text in dict.fromkeys(texts):
        vector = store.get(fingerprint, text)
        if vector is None:
            missing.append(text)
        else:
            found[text] = vector
    if missing:
        raw = embedder.embed(missing)
        if len(raw) != len(missing):
            raise ValueError(f"эмбеддер вернул {len(raw)} векторов на {len(missing)} текстов")
        for text, vector in zip(missing, normalize(raw), strict=True):
            store.put(fingerprint, text, vector)
            found[text] = vector
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.stack([found[t] for t in texts]).astype(np.float32)


@dataclass(frozen=True)
class VectorSpace:
    """Модель + схема приставок. Отпечаток пространства — то, что сверяет Vectors.cosine()."""

    embedder: Embedder
    cache: EmbeddingCache | None = None
    query_prefix: str = ""
    passage_prefix: str = ""

    @property
    def fingerprint(self) -> str:
        if not (self.query_prefix or self.passage_prefix):
            return self.embedder.fingerprint
        return f"{self.embedder.fingerprint}|{self.query_prefix!r}|{self.passage_prefix!r}"

    def queries(self, texts: Sequence[str]) -> Vectors:
        sent = [self.query_prefix + t for t in texts]
        return Vectors(self.fingerprint, embed_texts(self.embedder, sent, self.cache))

    def passages(self, texts: Sequence[str]) -> Vectors:
        sent = [self.passage_prefix + t for t in texts]
        return Vectors(self.fingerprint, embed_texts(self.embedder, sent, self.cache))


# ---------------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------------
class EmbedClient(Protocol):
    # input — имя параметра у ollama.Client.embed: протокол обязан совпасть с ним
    def embed(self, *, model: str, input: Sequence[str], truncate: bool) -> Any: ...  # noqa: A002


@dataclass
class OllamaEmbedder:
    """Пакетная отправка в /api/embed. digest — из ollama (probe_fingerprint): другой файл весов
    под тем же именем — другая модель, и кеш это видит."""

    client: EmbedClient
    model: str
    digest: str
    batch_size: int = 32
    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    raw_norms: list[float] = field(default_factory=list)  # нормы до нормировки: что прислал сервер
    _dim: int = 0

    @property
    def fingerprint(self) -> str:
        return f"{self.model}@{self.digest.removeprefix('sha256:')[:12]}"

    @property
    def dim(self) -> int:
        if not self._dim:
            self.embed(["проба размерности"])
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            began = time.monotonic()
            # truncate=False: текст длиннее окна модели — громкая ошибка, а не обрезанный вектор
            reply = self.client.embed(model=self.model, input=batch, truncate=False)
            self.seconds += time.monotonic() - began
            self.calls += 1
            self.prompt_tokens += int(getattr(reply, "prompt_eval_count", 0) or 0)
            vectors = [[float(x) for x in v] for v in reply.embeddings]
            if len(vectors) != len(batch):
                raise ValueError(f"ollama вернул {len(vectors)} векторов на {len(batch)} текстов")
            self._dim = len(vectors[0]) if vectors else self._dim
            self.raw_norms += [math.sqrt(sum(x * x for x in v)) for v in vectors]
            out += vectors
        return out


# ---------------------------------------------------------------------------
# Стадия Retriever для пайплайна (ragkit/pipeline.py)
# ---------------------------------------------------------------------------
@dataclass
class VectorRetriever:
    """Косинусный поиск по чанкам: index — векторы корпуса (офлайн), retrieve — вектор вопроса
    и одно умножение (онлайн). k ближайших, без порога: отсечка по порогу — Ф5-02, задание 3."""

    space: VectorSpace
    chunks: list[Chunk] = field(default_factory=list)
    vectors: Vectors | None = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.vectors = self.space.passages([c.text for c in self.chunks])

    def scores(self, question: str) -> NDArray[np.float32]:
        if self.vectors is None:
            raise RAGError("VectorRetriever: сначала index()")
        result: NDArray[np.float32] = self.space.queries([question]).cosine(self.vectors)[0]
        return result

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        order = np.argsort(-self.scores(question), kind="stable")[:k]
        return [self.chunks[int(i)] for i in order]
