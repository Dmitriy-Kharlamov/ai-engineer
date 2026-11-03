"""Задания 1 и 3: тесты, зелёные на дублёре и красные на настоящем HashEmbedder.

Вне testpaths, запуск: pytest week-03/lesson_10 -p no:randomly -rA
В обоих случаях проверяемый код с ошибкой — и ловит её только настоящий объект.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ragkit import Embedder, HashEmbedder


# --- задание 1: Mock соглашается на неправильную подпись -------------------------
def embed_titles(embedder: Embedder, titles: list[str]) -> list[list[float]]:
    """Вектор на заголовок. ОШИБКА: embed ждёт список текстов, а получает строку.

    mypy её не видит: str — это Sequence[str], подпись формально соблюдена.
    """
    return [embedder.embed(title)[0] for title in titles]


def make_mock() -> Any:
    m = Mock()
    m.embed.return_value = [[0.6, 0.8]]
    return m


@pytest.mark.parametrize("embedder", [make_mock(), HashEmbedder(dim=2)], ids=["mock", "real"])
def test_embed_titles(embedder: Embedder) -> None:
    vectors = embed_titles(embedder, ["привет", "мир"])
    assert vectors[0] == embedder.embed(["привет"])[0]  # вектор заголовка — его собственный


# --- задание 3: MagicMock выполняет протоколы, не выполняя их смысла ------------
def embed_unique(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Эмбеддит каждый различный текст один раз. ОШИБКА: дубликаты не убраны."""
    return embedder.embed(texts)


@pytest.fixture(params=["patch-magicmock", "real"])
def embedder(request: pytest.FixtureRequest) -> Iterator[Embedder]:
    real = HashEmbedder(dim=2)
    if request.param == "real":
        yield real
        return
    with patch.object(HashEmbedder, "embed"):  # patch по умолчанию подставляет MagicMock
        yield real


def test_embed_unique_has_no_duplicates(embedder: Embedder) -> None:
    vectors = embed_unique(embedder, ["a", "a", "b"])
    assert len({tuple(v) for v in vectors}) == len(vectors)  # ни одного повтора
