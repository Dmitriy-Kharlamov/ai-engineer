from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_embedder
from api.schemas import ChunkOut, DocumentIn, DocumentOut, EmbedIn, EmbedOut, SplitIn
from ragkit import Chunk, Document, Embedder, split_text

router = APIRouter(prefix="/chunks", tags=["chunks"])


def to_out(chunk: Chunk) -> ChunkOut:
    return ChunkOut(text=chunk.text, source=chunk.source, tokens=chunk.tokens)


@router.post("", status_code=201, summary="Нарезать текст на чанки")
async def split(body: SplitIn) -> list[ChunkOut]:
    # mode уже проверен: SplitIn.mode — Literal["words", "sentences"], опечатка — 422 до входа сюда
    return [to_out(c) for c in split_text(body.text, body.source, body.max_tokens, body.mode)]


@router.post("/documents", summary="Собрать документ и найти самый большой чанк")
async def build_document(body: DocumentIn) -> DocumentOut:
    doc = Document(body.name, max_chunk_tokens=body.max_chunk_tokens)
    doc.add_chunks(Chunk(text, source=body.name) for text in body.texts)  # ChunkTooLargeError
    largest = doc.largest_chunk()  # EmptyIndexError, если texts пуст
    return DocumentOut(
        name=doc.name, chunks=len(doc), total_tokens=doc.total_tokens(), largest=to_out(largest)
    )


@router.post("/embed", summary="Эмбеддинги текстов (модель загружена один раз)")
async def embed(body: EmbedIn, embedder: Annotated[Embedder, Depends(get_embedder)]) -> EmbedOut:
    return EmbedOut(dim=embedder.dim, vectors=embedder.embed(body.texts))
