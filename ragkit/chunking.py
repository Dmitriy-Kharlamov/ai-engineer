"""Чанки и документы."""

from collections.abc import Iterable, Iterator, Sequence
from functools import total_ordering
from typing import Literal, TypedDict, overload

from .exceptions import ChunkTooLargeError

# Только эти строки и никакие другие: опечатка "sentense" упадёт в mypy, а не в рантайме.
SplitMode = Literal["words", "sentences"]


class ChunkDict(TypedDict):
    """Чанк в виде словаря с известными ключами (для JSON и логов)."""

    text: str
    source: str
    tokens: int
    metadata: dict[str, object]


@total_ordering
class Chunk:
    def __init__(self, text: str, source: str, metadata: dict[str, object] | None = None) -> None:
        self.text = text
        self.source = source
        self.metadata: dict[str, object] = metadata if metadata is not None else {}

    @property
    def tokens(self) -> int:
        return len(self.text.split())

    def preview(self, n: int = 50) -> str:
        return self.text[:n] + "..." if len(self.text) > n else self.text

    def is_too_large(self, max_tokens: int = 500) -> bool:
        return self.tokens > max_tokens

    def to_dict(self) -> ChunkDict:
        return {
            "text": self.text,
            "source": self.source,
            "tokens": self.tokens,
            "metadata": dict(self.metadata),
        }

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.tokens < other.tokens

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.tokens == other.tokens

    def __hash__(self) -> int:
        # Согласован с __eq__: __eq__ сравнивает по числу токенов,
        # поэтому и хеш считаем по числу токенов — равные чанки дают равный хеш.
        # Следствие: все чанки с одинаковым числом слов для словаря — один ключ.
        return hash(self.tokens)

    def __repr__(self) -> str:
        return f"Chunk(source={self.source!r}, tokens={self.tokens})"


class PDFChunk(Chunk):
    def __init__(
        self, text: str, source: str, page: int, metadata: dict[str, object] | None = None
    ) -> None:
        super().__init__(text, source, metadata)
        self.page = page

    def __repr__(self) -> str:
        return f"PDFChunk(source={self.source!r}, tokens={self.tokens}, page={self.page})"


def split_text(text: str, source: str, max_tokens: int, mode: SplitMode = "words") -> list[Chunk]:
    """Нарезать текст на чанки не длиннее max_tokens слов.

    mode="words" — подряд по словам; mode="sentences" — по предложениям, не разрывая их
    (предложение длиннее лимита становится отдельным чанком).
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens должно быть ≥ 1, получено {max_tokens}")
    if mode == "words":
        words = text.split()
        return [
            Chunk(" ".join(words[i : i + max_tokens]), source)
            for i in range(0, len(words), max_tokens)
        ]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in (s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")):
        if not sentence:
            continue
        if current and len(" ".join(current + [sentence]).split()) > max_tokens:
            pieces.append(". ".join(current) + ".")
            current = []
        current.append(sentence)
    if current:
        pieces.append(". ".join(current) + ".")
    return [Chunk(piece, source) for piece in pieces]


def largest(chunks: Sequence[Chunk], k: int) -> list[Chunk]:
    """k самых больших чанков. На входе — любая последовательность, на выходе — список."""
    return sorted(chunks, key=lambda c: c.tokens, reverse=True)[:k]


class Document:
    def __init__(self, name: str, max_chunk_tokens: int = 50) -> None:
        self.name = name
        self.max_chunk_tokens = max_chunk_tokens
        self._chunks: list[Chunk] = []

    def add_chunk(self, chunk: Chunk) -> None:
        if chunk.is_too_large(self.max_chunk_tokens):
            raise ChunkTooLargeError(
                f"Чанк чрезмерно большой: {chunk.tokens} токенов при лимите {self.max_chunk_tokens}"
            )
        self._chunks.append(chunk)

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Только перебираем — значит, достаточно Iterable: подойдёт и генератор."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def total_tokens(self) -> int:
        return sum(chunk.tokens for chunk in self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return (
            f"Document(name={self.name!r}, chunks={self.chunk_count}, tokens={self.total_tokens()})"
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk: object) -> bool:
        return chunk in self._chunks

    @overload
    def __getitem__(self, index: int) -> Chunk: ...
    @overload
    def __getitem__(self, index: slice) -> list[Chunk]: ...
    def __getitem__(self, index: int | slice) -> Chunk | list[Chunk]:
        return self._chunks[index]

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._chunks)
