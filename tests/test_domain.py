"""Исключения предметного слоя (ragkit, services) — без HTTP.

match= сверяет текст сообщения регулярным выражением (re.search): тест краснеет, если
исключение того же класса вылетело не там и не по той причине.
"""

import pytest

from ragkit import Chunk, ChunkTooLargeError, Document, EmptyIndexError
from services.db import Database
from services.documents import DocumentLimitError, DocumentRepo


def test_chunk_over_limit_names_size_and_limit() -> None:
    document = Document("d", max_chunk_tokens=2)
    with pytest.raises(ChunkTooLargeError, match=r"3 токенов при лимите 2$"):
        document.add_chunks([Chunk("a b c", "s")])


def test_empty_document_names_itself() -> None:
    with pytest.raises(EmptyIndexError, match=r"в документе 'пустой' нет ни одного чанка"):
        Document("пустой").largest_chunk()


def test_document_limit_is_per_owner() -> None:
    repo = DocumentRepo(Database().session(), max_per_owner=1)
    repo.create("alice", "a.md", "x")
    repo.create("bob", "b.md", "y")  # у другого владельца свой лимит — исключения нет
    with pytest.raises(DocumentLimitError, match=r"^не больше 1 документов на владельца$"):
        repo.create("alice", "c.md", "z")
