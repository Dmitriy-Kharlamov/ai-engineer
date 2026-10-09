from ragkit import Chunk, Document

doc = Document("Демо", max_chunk_tokens=20)
doc.add_chunk(Chunk("проверка пакета", "demo.pdf"))
print(doc)