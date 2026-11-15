"""Поиск docqa как стадия Retriever пайплайна (Ф5-01).

docqa сам по себе — агент: модель решает, искать ли, что искать и что читать. Здесь — его
поиск без модели: DocumentStore.search по тексту вопроса (так же ищет подстраховка
session._fallback, session.py:868) и первые k найденных документов целиком. Это «как есть»
для пайплайна; агентный docqa меряется рядом, своим прогоном (week-05/lesson_01.py).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ragkit.chunking import Chunk
from ragkit.tools import PublicToolError

from .documents import DocumentStore
from .session import FALLBACK_QUERY_CHARS


@dataclass
class StoreSearchRetriever:
    store: DocumentStore
    _chunks: list[Chunk] = field(default_factory=list)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        query = " ".join(question.split())[:FALLBACK_QUERY_CHARS]
        try:
            names = [h["name"] for h in self.store.search(query)["results"]][:k]
        except PublicToolError:
            return []
        by_name = {c.source: c for c in self._chunks}
        return [by_name[n] for n in names if n in by_name]
