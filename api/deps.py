"""Зависимости для Depends: настройки и эмбеддер из app.state."""

from fastapi import Request

from ragkit import Embedder, HashEmbedder

EMBEDDER_LOADS = 0  # сколько раз создавался эмбеддер — доказательство для задания 9


def load_embedder(dim: int = 128) -> Embedder:
    """«Тяжёлая» загрузка модели. Вызывается в lifespan ровно один раз."""
    global EMBEDDER_LOADS
    EMBEDDER_LOADS += 1
    return HashEmbedder(dim=dim)


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder
    return embedder
