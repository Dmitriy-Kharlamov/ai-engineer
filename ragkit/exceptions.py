"""Исключения ragkit: общий предок RAGError ловит всё, что бросает пакет."""


class RAGError(Exception):
    """Ошибка в RAG."""


class ChunkTooLargeError(RAGError):
    """Чанк превышает допустимый размер."""


class EmptyIndexError(RAGError):
    """Ошибка в индексе."""
