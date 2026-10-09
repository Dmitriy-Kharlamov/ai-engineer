class RAGError(Exception): "Ошибка в RAG"
class ChunkTooLargeError(RAGError): "Чанк превышает допустимый размер"
class EmptyIndexError(RAGError): "Ошибка в индексе"
