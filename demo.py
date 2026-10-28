from ragkit import Chunk, Document, PDFChunk, RAGError

doc = Document("Руководство", max_chunk_tokens=20)
c1 = Chunk("Python это язык программирования.", "manual.pdf")
doc.add_chunk(c1)
c2 = Chunk("Python это язык программирования без точки", "manual.pdf")
doc.add_chunk(c2)
c3 = Chunk("Исключения сигнализируют об ошибках.", "manual.pdf")
doc.add_chunk(c3)
c5 = Chunk("Исключения сигнализируют об оши", "manual.pdf")
print(doc)
print(len(doc))

b = Chunk("Много текстаааааа очень очент", "doc.pdf")
c = Chunk("Мало текста", "doc.pdf")
d = Chunk("Текст", "doc.pdf")
print(c <= b)
print(b > c)
print(sorted([b, d, c]))

long_chunk = PDFChunk("слово " * 600, "big.pdf", 1)
a = Chunk("первый", "doc.pdf")
print(long_chunk.preview())
print(long_chunk.is_too_large())

try:
    doc.add_chunk(long_chunk)
except RAGError as e:
    print("Не добавили:", e)
