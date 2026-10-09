# 1 и 5
# Создайте PDFChunk, наследник Chunk, с дополнительным полем page. Используйте super().__init__(). Переопределите __repr__, чтобы показывалась страница. Проверьте, что .tokens и .preview() работают без переписывания.
# Добавьте в Chunk метод __lt__ (сравнение по количеству токенов) и отсортируйте список чанков обычным sorted() без key=. Потом добавьте @total_ordering и проверьте, что > и <= тоже заработали.
from abc import ABC, abstractmethod
import time
from functools import total_ordering

@total_ordering
class Chunk:
    def __init__(self, text, source, metadata=None):
        self.text = text
        self.source = source
        self.metadata = metadata if metadata is not None else {}

    @property
    def tokens(self):
        return len(self.text.split())
    
    def preview(self, n=50):
        return self.text[:n] + "..." if len(self.text) > n else self.text
    
    def is_too_large(self, max_tokens=500):
        return self.tokens > max_tokens
    
    def __lt__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.tokens < other.tokens
    
    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.tokens == other.tokens
    
    def __repr__(self):
        return f"Chunk(source={self.source!r}, tokens={self.tokens})"


class PDFChunk(Chunk):
    def __init__(self, text, source, page, metadata=None):
        super().__init__(text, source, metadata)   
        self.page = page    
    def __repr__(self):
        return f"PDFChunk(source={self.source!r}, tokens={self.tokens}, page={self.page})" 

long_chunk = PDFChunk("слово " * 600, "big.pdf", 1)
a = Chunk("первый", "doc.pdf")
print(long_chunk.preview())
print(long_chunk.is_too_large())
print(long_chunk)
print(long_chunk.tokens)

b = Chunk("Много текстаааааа очень очент", "doc.pdf")
c = Chunk("Мало текста", "doc.pdf")
d = Chunk("Текст", "doc.pdf")
print(c <= b)
print(b > c)
print(sorted([b, a, d, c]))



# 2
# Проверьте isinstance и issubclass на своих классах. Убедитесь, что PDFChunk проходит проверку isinstance(x, Chunk), а обычный Chunk не проходит isinstance(x, PDFChunk). Объясните в комментарии, почему так.

print(isinstance(a, PDFChunk))    # False, потому что a относится к Chunk
print(isinstance(long_chunk, PDFChunk))  # True, потому что long_chunk относится к PDFChunk
print(isinstance(a, Chunk)) # True, потому что a относится к Chunk
print(isinstance(long_chunk, Chunk)) # True, потому что long_chunk относится к PDFChunk, а PDFChunk наследник Chunk 
print(issubclass(PDFChunk, Chunk))   # True, потому что PDFChunk наследник Chunk 
print(issubclass(Chunk, PDFChunk))  # False, потому что Chunk родитель PDFChunk 


# 3
# Иерархия исключений для RAG:

# python
# class RAGError(Exception): ...
# class ChunkTooLargeError(RAGError): ...
# class EmptyIndexError(RAGError): ...

# Напишите функцию, которая в зависимости от аргумента бросает одно из них. Покажите тремя примерами: перехват конкретного класса, перехват родителя RAGError (ловит оба), и что except ValueError их уже не поймает.


class RAGError(Exception): "Ошибка в RAG"
class ChunkTooLargeError(RAGError): "Чанк превышает допустимый размер"
class EmptyIndexError(RAGError): "Ошибка в индексе"

def check_index(problem):
    if problem == "large":
        raise ChunkTooLargeError("Чанк слишком большой для индекса")
    if problem == "empty":
        raise EmptyIndexError("Ошибка в индексе")
    else:
        return "Всё в порядке"

try:
    check_index("large")
except ChunkTooLargeError as e:
    print("Пример 1:", e)

for problem in ["large", "empty"]:
    try:
        check_index(problem)
    except RAGError as e:        
        print("Пример 2:", type(e).__name__, e)

print(check_index("ok"))

try:
    check_index("large")
except ValueError:
    print("сработал ValueError")
except ChunkTooLargeError as e:
    print("ValueError пропустил, поймал ChunkTooLargeError:", e)



# 4
class Document:
    def __init__(self, name, max_chunk_tokens=50):
        self.name = name
        self.max_chunk_tokens = max_chunk_tokens
        self._chunks = []

    def add_chunk(self, chunk):
        if chunk.is_too_large(self.max_chunk_tokens):
            raise ChunkTooLargeError(
                f"Чанк слишком большой: {chunk.tokens} токенов "
                f"при лимите {self.max_chunk_tokens}"
            )
        self._chunks.append(chunk)

    def total_tokens(self):
        return sum(chunk.tokens for chunk in self._chunks)

    @property
    def chunk_count(self):
        return len(self._chunks)

    def __repr__(self):
        return (f"Document(name={self.name!r}, chunks={self.chunk_count}, "
                f"tokens={self.total_tokens()})")

    def __len__(self):
        return len(self._chunks)
    
    def __contains__(self, chunk):
        return chunk in self._chunks
    
    def __getitem__(self, index):
        return self._chunks[index]
    
    def __iter__(self):
        return iter(self._chunks)
    

doc = Document("Руководство", max_chunk_tokens=20)

c1 = Chunk("Python это язык программирования.", "manual.pdf")
doc.add_chunk(c1)
c2 = Chunk("Python это язык программирования без точки", "manual.pdf")
doc.add_chunk(c2)
c3 = Chunk("Исключения сигнализируют об ошибках.", "manual.pdf")
doc.add_chunk(c3)
c5 = Chunk("Исключения сигнализируют об оши", "manual.pdf")


print(len(doc))
print(c1 in doc)
print(c5 in doc)
print(doc[0], doc[1:3])
for chunk in doc:
    print(chunk.text)
print(doc)



# 6
# Напишите контекстный менеджер Timer, который печатает время выполнения блока. Проверьте на чём-нибудь медленном, например на цикле в миллион итераций.

# Отдельно проверьте, что __exit__ срабатывает даже при исключении внутри блока.
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Заняло {time.time() - self.start:.2f} сек")

with Timer():
    total = 0
    for i in range(1_000_000):
        total += i

try:
    with Timer():
        raise ChunkTooLargeError("Чанк слишком большой для индекса")
    print("Эта строка не выполнится")
except ChunkTooLargeError as e:
    print("Пример 1:", e)


class Silencer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Проглотил: {exc_type.__name__}")
        return True          # ← вот оно


with Silencer():
    raise ValueError("ошибка!")

print("Мы сюда дошли, хотя выше было исключение")


# 7
# Базовый класс BaseRetriever с абстрактным search. Два наследника: DenseRetriever возвращает заглушку ["dense результат"], KeywordRetriever — ["keyword результат"]. Напишите функцию, которая принимает любой ретривер и печатает его результаты — одна функция работает с обоими.

# Попробуйте создать объект BaseRetriever напрямую и посмотрите, что произойдёт.

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query):
        pass

class DenseRetriever(BaseRetriever):
    def search(self, query):
        return ["dense результат"]


class KeywordRetriever(BaseRetriever):
    def search(self, query):
        return ["keyword результат"]  



def show_results(retriever, query):
    print(retriever.search(query))

show_results(DenseRetriever(), "что такое RAG?")
show_results(KeywordRetriever(), "что такое RAG?")


try:
    BaseRetriever()
except TypeError as e:
    print("BaseRetriever():", e)



# 8
class A:
    def hello(self):
        return "A"


class B(A):
    def hello(self):
        return "B + " + super().hello()


class C(B):
    pass


print(C().hello())          # B + A
print(isinstance(C(), A))   # True
print(issubclass(C, A))     # True
print(C.__mro__)            # порядок поиска методов: C -> B -> A -> object

# Почему так:
# 1. У C своего hello нет. Python ищет метод по цепочке C -> B -> A
#    и находит его в B.
# 2. B.hello возвращает "B + " и вызывает super().hello(). super() — это
#    «следующий класс в цепочке после B», то есть A. A.hello вернёт "A".
#    Итог: "B + A".
# 3. C наследует от B, а B от A, значит C(), хоть и через посредника,
#    является A. isinstance и issubclass учитывают всю цепочку,
#    не только прямого родителя. Поэтому оба раза True.