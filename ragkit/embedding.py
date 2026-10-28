"""Протокол эмбеддера: под него подходит любой объект с методом embed().

OpenAI, локальная модель, фейк в тестах — им не нужно наследоваться от Embedder
и даже знать о нём. mypy проверяет совпадение сигнатуры структурно.
"""

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from .chunking import Chunk


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Детерминированный «эмбеддер» без сети — для тестов. Embedder не наследует."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

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
