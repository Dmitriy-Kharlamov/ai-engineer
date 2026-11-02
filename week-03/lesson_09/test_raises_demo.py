"""Задание 10: pytest.raises(Exception) зелёный на опечатке.

Вне testpaths, запуск: pytest week-03/lesson_09/test_raises_demo.py -p no:randomly -rA
Опечатка `documnet` намеренная; ruff (F821) и mypy (name-defined) её бы поймали —
поэтому здесь они заглушены построчно, а в tests/ такой код не прошёл бы pre-commit.
"""

import pytest

from ragkit import Chunk, ChunkTooLargeError, Document


def test_broad_raises_hides_typo() -> None:
    document = Document("d", max_chunk_tokens=2)  # noqa: F841
    with pytest.raises(Exception):  # noqa: B017 — ловит что угодно, в том числе NameError
        documnet.add_chunks([Chunk("a b c", "s")])  # type: ignore[name-defined]  # noqa: F821


def test_precise_raises_catches_typo() -> None:
    document = Document("d", max_chunk_tokens=2)  # noqa: F841
    with pytest.raises(ChunkTooLargeError, match=r"при лимите 2"):
        documnet.add_chunks([Chunk("a b c", "s")])  # type: ignore[name-defined]  # noqa: F821


def test_precise_raises_without_typo() -> None:
    document = Document("d", max_chunk_tokens=2)
    with pytest.raises(ChunkTooLargeError, match=r"при лимите 2"):
        document.add_chunks([Chunk("a b c", "s")])
