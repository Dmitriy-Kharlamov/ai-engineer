from dataclasses import dataclass, field


# 1 и 2
class Person:
    def __init__(self, name, age, city):
        self.name = name
        self.age = age
        self.city = city

    def greet(self):
        return f"Привет {self.name} из города {self.city}"

    def is_adult(self):
        return self.age >= 18

    def __repr__(self):
        return f"Person(name={self.name!r}, age={self.age}, city={self.city!r})"


p = Person("Дима", 55, "Москва")
p1 = Person("Дима", 27, "Москва")
p2 = Person("Диана", 28, "Москва")
print(p.city)
print(p.greet())
print(p.is_adult())
print(p)
print([p1, p2])


# 3
class BankAccount:
    def __init__(self, owner, balance=0):
        if balance < 0:
            raise ValueError("Начальный баланс не может быть отрицательным")
        self.owner = owner
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError(f"Сумма пополнения должна быть больше нуля, получено: {amount}")
        self._balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError(f"Сумма снятия должна быть больше нуля, получено: {amount}")
        if amount > self._balance:
            raise ValueError(f"Недостаточно средств: на счёте {self._balance}, запрошено {amount}")
        self._balance -= amount


# --- Успешные сценарии ---
acc = BankAccount("Dmitriy")
print(acc.owner, acc.balance)  # Dmitriy 0

acc.deposit(100)
print("После пополнения:", acc.balance)  # 100

acc.withdraw(30)
print("После снятия:", acc.balance)  # 70

acc.withdraw(70)
print("Сняли всё:", acc.balance)  # 0

# --- Ошибочные сценарии ---
try:
    acc.deposit(-50)
except ValueError as e:
    print("Ошибка:", e)

try:
    acc.deposit(0)
except ValueError as e:
    print("Ошибка:", e)

try:
    acc.withdraw(1000)
except ValueError as e:
    print("Ошибка:", e)

try:
    acc.withdraw(-10)
except ValueError as e:
    print("Ошибка:", e)

try:
    BankAccount("Ivan", -500)
except ValueError as e:
    print("Ошибка:", e)

try:
    acc.balance = 1_000_000  # property только для чтения
except AttributeError as e:
    print("Ошибка:", e)

print("Итоговый баланс:", acc.balance)  # 0 — ошибки баланс не испортили


# 4
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

    def __repr__(self):
        return f"Chunk(source={self.source!r}, tokens={self.tokens})"


c = Chunk("третий", "doc.pdf")
print(c.tokens)
print(c)

a = Chunk("первый", "doc.pdf")
b = Chunk("второй", "doc.pdf")
a.metadata["page"] = 3
print(a.metadata)  # {'page': 3}
print(b.metadata)  # {} — должно быть пусто

long_chunk = Chunk("слово " * 600, "big.pdf")
print(long_chunk.preview())
print(long_chunk.is_too_large())
print(c.is_too_large())


# 5
class ChunkTooLargeError(ValueError):
    """Чанк превышает допустимый размер."""


class Document:
    def __init__(self, name, max_chunk_tokens=50):
        self.name = name
        self.max_chunk_tokens = max_chunk_tokens
        self._chunks = []

    def add_chunk(self, chunk):
        if chunk.is_too_large(self.max_chunk_tokens):
            raise ChunkTooLargeError(
                f"Чанк слишком большой: {chunk.tokens} токенов при лимите {self.max_chunk_tokens}"
            )
        self._chunks.append(chunk)

    def total_tokens(self):
        return sum(chunk.tokens for chunk in self._chunks)

    @property
    def chunk_count(self):
        return len(self._chunks)

    def __repr__(self):
        return (
            f"Document(name={self.name!r}, chunks={self.chunk_count}, tokens={self.total_tokens()})"
        )


doc = Document("Руководство", max_chunk_tokens=20)

doc.add_chunk(Chunk("Python это язык программирования.", "manual.pdf"))
doc.add_chunk(Chunk("Классы объединяют данные и поведение.", "manual.pdf"))
doc.add_chunk(Chunk("Исключения сигнализируют об ошибках.", "manual.pdf"))

print(doc)
print("Чанков:", doc.chunk_count)
print("Токенов:", doc.total_tokens())

huge = Chunk("слово " * 1000, "big.pdf")
print(huge)

try:
    doc.add_chunk(huge)
except ChunkTooLargeError as e:
    print("Ошибка:", e)

print("После ошибки:", doc)


# 6


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def tokens(self):
        return len(self.text.split())

    def preview(self, n=50):
        return self.text[:n] + "..." if len(self.text) > n else self.text

    def is_too_large(self, max_tokens=500):
        return self.tokens > max_tokens

    def __repr__(self):
        return f"Chunk(source={self.source!r}, tokens={self.tokens})"


x = Chunk("привет", "doc.pdf")
y = Chunk("привет", "doc.pdf")

print(x == y)  # True  — поля одинаковые
print(x is y)  # False — но это два разных объекта

y.metadata["page"] = 1
print(x == y)  # False — metadata теперь отличается

print(Chunk("привет", "doc.pdf") == Chunk("пока", "doc.pdf"))  # False


# 7
class Counter:
    count = 0

    def __init__(self):
        self.count += 1


a = Counter()
b = Counter()
print(a.count, b.count, Counter.count)  # Будет выведено 1 1 и 0
