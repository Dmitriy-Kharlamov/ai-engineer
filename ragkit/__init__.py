"""ragkit — учебный пакет курса: чанки, документы, эмбеддеры, настройки."""

from .chunking import Chunk, ChunkDict, Document, PDFChunk, SplitMode, largest, split_text
from .embedding import Embedder, HashEmbedder, embed_chunks
from .exceptions import ChunkTooLargeError, EmptyIndexError, RAGError

__all__ = [
    "Chunk",
    "ChunkDict",
    "ChunkTooLargeError",
    "Document",
    "Embedder",
    "EmptyIndexError",
    "HashEmbedder",
    "PDFChunk",
    "RAGError",
    "SplitMode",
    "embed_chunks",
    "largest",
    "split_text",
]
