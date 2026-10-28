from functools import total_ordering

from .exceptions import ChunkTooLargeError


@total_ordering
class Chunk:
    def __init__(self, text, source, metadata=None):
        self.text = text
        self.source = source
        self.metadata = metadata if metadata is not None else {}

    @property
    def tokens(self):
        return len(self.text.split())

    def preview(self, n=50):
        return self.text[:n] + "..." if len(self.text) > n else self.text

    def is_too_large(self, max_tokens=500):
        return self.tokens > max_tokens

    def __lt__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.tokens < other.tokens

    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.tokens == other.tokens

    def __hash__(self):
        # Согласован с __eq__: __eq__ сравнивает по числу токенов,
        # поэтому и хеш считаем по числу токенов — равные чанки дают равный хеш.
        # Следствие: все чанки с одинаковым числом слов для словаря — один ключ.
        return hash(self.tokens)

    def __repr__(self):
        return f"Chunk(source={self.source!r}, tokens={self.tokens})"


class PDFChunk(Chunk):
    def __init__(self, text, source, page, metadata=None):
        super().__init__(text, source, metadata)
        self.page = page

    def __repr__(self):
        return f"PDFChunk(source={self.source!r}, tokens={self.tokens}, page={self.page})"


class Document:
    def __init__(self, name, max_chunk_tokens=50):
        self.name = name
        self.max_chunk_tokens = max_chunk_tokens
        self._chunks = []

    def add_chunk(self, chunk):
        if chunk.is_too_large(self.max_chunk_tokens):
            raise ChunkTooLargeError(
                f"Чанк чрезмерно большой: {chunk.tokens} токенов при лимите {self.max_chunk_tokens}"
            )
        self._chunks.append(chunk)

    def total_tokens(self):
        return sum(chunk.tokens for chunk in self._chunks)

    @property
    def chunk_count(self):
        return len(self._chunks)

    def __repr__(self):
        return (
            f"Document(name={self.name!r}, chunks={self.chunk_count}, tokens={self.total_tokens()})"
        )

    def __len__(self):
        return len(self._chunks)

    def __contains__(self, chunk):
        return chunk in self._chunks

    def __getitem__(self, index):
        return self._chunks[index]

    def __iter__(self):
        return iter(self._chunks)
