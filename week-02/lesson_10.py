import locale
import traceback
import random
import re
import time
import unicodedata
from collections import namedtuple
from math import log2
from operator import attrgetter, itemgetter


def best_of(func, *args, runs=7):
    """Минимальное время из runs прогонов."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return min(times)


# ===========================================================================
# ЧАСТЬ 1. Сортировки руками (задания A–E из раннего сообщения)
# ===========================================================================

def identity(x):
    return x


# ---------------------------------------------------------------------------
# A. Сортировка вставками
# ---------------------------------------------------------------------------
print("A. insertion_sort")


def insertion_sort(items, key=identity):
    result = list(items)                       # копия: вход не портим
    for i in range(1, len(result)):
        current = result[i]
        current_key = key(current)
        j = i - 1
        # сдвигаем вправо всё, что СТРОГО больше: равные не обгоняем → устойчиво
        while j >= 0 and key(result[j]) > current_key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = current
    return result


random.seed(42)
for _ in range(200):                           # проверка против эталона sorted
    data = [random.randint(-50, 50) for _ in range(random.randint(0, 30))]
    original = list(data)
    assert insertion_sort(data) == sorted(data)
    assert data == original                    # вход не изменился

random_5000 = [random.random() for _ in range(5000)]
sorted_5000 = sorted(random_5000)
t_random = best_of(insertion_sort, random_5000)
t_sorted = best_of(insertion_sort, sorted_5000)
print(f"5000 случайных:      {t_random:.4f} с")
print(f"5000 отсортированных: {t_sorted:.5f} с")
print(f"разница: в {t_random / t_sorted:.0f} раз")
# На отсортированных данных внутренний while не делает ни одного шага:
#   каждый новый элемент уже не меньше предыдущего. Остаётся один проход —
#   O(n), ~5000 сравнений. На случайных каждый элемент в среднем сдвигается
#   на половину уже отсортированной части — O(n²), ~n²/4 ≈ 6 млн сдвигов.
#   Поэтому вставками хороши для почти упорядоченных данных — и Timsort
#   использует их для коротких участков.


# ---------------------------------------------------------------------------
# B. Слияние
# ---------------------------------------------------------------------------
print("\nB. merge и merge_sort")


def merge(left, right, key=identity, strict=False):
    """Слить два отсортированных списка. strict=True — сравнение через < (ошибка)."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = key(left[i]), key(right[j])
        take_left = a < b if strict else a <= b
        if take_left:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])                     # хвост одного из списков
    result.extend(right[j:])
    return result


def merge_sort(items, key=identity, strict=False):
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    return merge(merge_sort(items[:mid], key, strict),
                 merge_sort(items[mid:], key, strict), key, strict)


assert merge([1, 4, 9], [2, 3, 10, 11]) == [1, 2, 3, 4, 9, 10, 11]
assert merge([], [1, 2]) == [1, 2] and merge([1], []) == [1]
for _ in range(200):
    data = [random.randint(-50, 50) for _ in range(random.randint(0, 40))]
    assert merge_sort(data) == sorted(data)

# Устойчивость: (ключ, номер) — сортируем только по ключу, номер показывает,
# сохранился ли исходный порядок равных.
tagged = [(random.randint(0, 3), i) for i in range(20)]
by_key = itemgetter(0)
stable = merge_sort(tagged, key=by_key)
unstable = merge_sort(tagged, key=by_key, strict=True)
print("с <= :", stable[:8])
print("с <  :", unstable[:8])
assert stable == sorted(tagged, key=by_key)        # sorted устойчив — эталон
assert unstable != sorted(tagged, key=by_key)
# По ключу оба результата отсортированы верно — ломается УСТОЙЧИВОСТЬ.
#   С <= при равных ключах берётся элемент из left — он стоял раньше, порядок
#   сохраняется. С < при равенстве срабатывает else и первым идёт элемент
#   из right — равные элементы переставляются. Проверка: сортировать пары
#   (ключ, номер) только по ключу и сравнить с sorted, который гарантированно
#   устойчив: при < номера внутри одного ключа идут не по возрастанию.


# ---------------------------------------------------------------------------
# C. Быстрая сортировка и её худший случай
# ---------------------------------------------------------------------------
print("\nC. quicksort")


def quicksort(items, key=identity):
    if len(items) <= 1:
        return list(items)
    pivot, rest = items[0], items[1:]                 # опорный — первый
    p = key(pivot)
    less = [x for x in rest if key(x) < p]
    greater = [x for x in rest if key(x) >= p]
    return quicksort(less, key) + [pivot] + quicksort(greater, key)


for _ in range(200):
    data = [random.randint(-50, 50) for _ in range(random.randint(0, 40))]
    assert quicksort(data) == sorted(data)

try:
    quicksort(list(range(1500)))
except RecursionError:
    print(traceback.format_exc())
# На отсортированном входе items[0] — минимум. less всегда пуст, greater —
#   все остальные n−1 элементов. Каждый вызов откусывает ровно один элемент,
#   глубина рекурсии — n = 1500, а предел Python — 1000 кадров
#   (sys.getrecursionlimit()). Время тоже O(n²), но до медленности дело не
#   доходит: программа падает с RecursionError — ошибкой, которая на первый
#   взгляд к сортировке и скорости отношения не имеет. Худший случай
#   алгоритма проявился как исчерпание стека.
# Причина — выбор опорного: первый элемент делит ровно только случайные
#   данные. На отсортированных (частый случай в жизни!) он худший из возможных.


def quicksort_random(items, key=identity):
    if len(items) <= 1:
        return list(items)
    p = key(random.choice(items))                      # случайный опорный
    less = [x for x in items if key(x) < p]
    equal = [x for x in items if key(x) == p]          # равные — сразу на место
    greater = [x for x in items if key(x) > p]
    return quicksort_random(less, key) + equal + quicksort_random(greater, key)


for _ in range(200):
    data = [random.randint(-50, 50) for _ in range(random.randint(0, 40))]
    assert quicksort_random(data) == sorted(data)
assert quicksort_random(list(range(1500))) == list(range(1500))
assert quicksort_random(list(range(100_000))) == list(range(100_000))
assert quicksort_random([7] * 5000) == [7] * 5000
print("quicksort_random на list(range(1500)) и list(range(100_000)): OK")
# Случайный опорный в среднем делит список примерно пополам независимо от
#   того, как упорядочен вход: глубина ~ log2(n) (для 100 000 — порядка 20–40),
#   до предела 1000 далеко. Худший случай не исчез, но стал маловероятным:
#   нужно, чтобы случай раз за разом выбирал минимум.
# Отдельный список equal нужен для повторов: без него на [7] * 5000 все
#   элементы шли бы в одну сторону, и снова получилась бы глубина n.


# ---------------------------------------------------------------------------
# D. Многоключевая сортировка и quicksort
# ---------------------------------------------------------------------------
print("\nD. оценка ↓, при равной — имя ↑")
results = [("doc_b", 0.9), ("doc_a", 0.9), ("doc_c", 0.7), ("doc_a", 0.7), ("doc_b", 0.7)]

double = sorted(results, key=itemgetter(0))                   # младший: имя ↑
double = sorted(double, key=itemgetter(1), reverse=True)      # старший: оценка ↓
one_key = sorted(results, key=lambda r: (-r[1], r[0]))
print("двойная сортировка:", double)
print("один кортеж:       ", one_key)
assert double == one_key == [("doc_a", 0.9), ("doc_b", 0.9),
                             ("doc_a", 0.7), ("doc_b", 0.7), ("doc_c", 0.7)]

by_score = quicksort(results, key=lambda r: -r[1])
print("quicksort по оценке:", by_score)


def quicksort_inplace(items, key=identity):
    """Классическая быстрая сортировка на месте (разбиение Ломуто)."""
    a = list(items)

    def sort(lo, hi):
        if lo >= hi:
            return
        p = key(a[hi])
        i = lo
        for j in range(lo, hi):
            if key(a[j]) < p:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        sort(lo, i - 1)
        sort(i + 1, hi)

    sort(0, len(a) - 1)
    return a


print("quicksort на месте: ", quicksort_inplace(results, key=lambda r: -r[1]))
# Почему на quicksort здесь полагаться нельзя:
#   1. Мы сортируем только по оценке, а порядок равных (doc_b и doc_a с 0.9)
#      этим ключом не задан. Что получится среди равных, решает алгоритм.
#   2. Наша quicksort на фильтрах случайно оказалась устойчивой: генераторы
#      списков сохраняют порядок, поэтому равные вышли в порядке ВХОДА
#      (doc_b раньше doc_a) — это не порядок по имени.
#   3. Устойчивость у быстрой сортировки — не гарантия, а деталь реализации.
#      Классическая версия на месте (Ломуто) переставляет равные обменами, и
#      порядок doc_* среди 0.7 у неё уже другой. Поэтому двойной сортировкой
#      через quicksort пользоваться нельзя: она держится только на
#      устойчивости, которую quicksort не обещает. sorted (Timsort) обещает.


# ---------------------------------------------------------------------------
# E. Прикидка без кода
# ---------------------------------------------------------------------------
# O(n²): n выросло в 50 000 / 1 000 = 50 раз → время в 50² = 2 500 раз.
#   2 с × 2 500 = 5 000 с ≈ 83 минуты ≈ 1 ч 23 мин.
# O(n log n): время растёт в (50 000 · log2 50 000) / (1 000 · log2 1 000)
#   = 50 × (15,61 / 9,97) ≈ 50 × 1,57 ≈ 78 раз.
#   2 с × 78 ≈ 157 с ≈ 2,6 минуты.
# Разница между алгоритмами на 50 000 элементов — примерно в 32 раза
#   (5 000 / 157), и с ростом n она только увеличивается.
print("\nE. проверка прикидки:",
      f"O(n²): {2 * 50 ** 2} с;",
      f"O(n log n): {2 * 50_000 * log2(50_000) / (1_000 * log2(1_000)):.0f} с")


# ===========================================================================
# ЧАСТЬ 2. sorted(key=) — задания урока Ф2-10
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Естественная сортировка
# ---------------------------------------------------------------------------
print("\n1. natural_key")


def natural_key(s):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", s)]


files = ["файл10.txt", "файл9.txt", "файл1.txt", "Файл2.txt", "отчёт.txt"]
plain = sorted(files)
natural = sorted(files, key=natural_key)
print("обычный:  ", plain)
print("natural:  ", natural)
print("ключ для 'файл10.txt':", natural_key("файл10.txt"))
assert natural == ["отчёт.txt", "файл1.txt", "Файл2.txt", "файл9.txt", "файл10.txt"]


def natural_key_no_fold(s):
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", s)]


no_fold = sorted(files, key=natural_key_no_fold)
print("без casefold:", no_fold)
assert no_fold[0] == "Файл2.txt"
# Обычный sorted сравнивает посимвольно по кодам: '1' < '9', поэтому
#   «файл10» раньше «файл9»; а заглавная «Ф» (U+0424) меньше любой строчной
#   (о — U+043E, ф — U+0444), поэтому «Файл2» оказался первым.
# natural_key: re.split(r"(\d+)") режет строку на куски и, благодаря скобкам,
#   оставляет числа: 'файл10.txt' → ['файл', 10, '.txt']. Числа сравниваются
#   как числа (9 < 10), текст — как текст.
# casefold() внутри ключа делает сравнение текстовых кусков нечувствительным
#   к регистру: 'Файл' и 'файл' дают одинаковый ключ, и решает только номер.
#   Без него «Файл2» снова уезжает в начало, раньше даже «отчёт» — заглавные
#   буквы в юникоде идут до всех строчных.
# Почему int и str не сталкиваются: re.split со скобками ВСЕГДА начинает с
#   текстового куска (может быть пустым ''), дальше строго чередует текст и
#   число. Поэтому на одной позиции у двух ключей всегда один тип.


# ---------------------------------------------------------------------------
# 2. Кириллица и «ё»
# ---------------------------------------------------------------------------
print("\n2. Кириллица")
words = ["ёж", "яблоко", "ель", "Ёлка", "абажур"]
DICTIONARY_ORDER = ["абажур", "ёж", "Ёлка", "ель", "яблоко"]   # как в словаре

plain = sorted(words)
by_key = sorted(words, key=lambda s: s.casefold().replace("ё", "е"))
print("обычный:          ", plain)
print("casefold + ё→е:   ", by_key)
assert by_key == DICTIONARY_ORDER

try:
    locale.setlocale(locale.LC_COLLATE, "ru_RU.UTF-8")
    by_locale = sorted(words, key=locale.strxfrm)
    print("locale.strxfrm:   ", by_locale)
    print("совпало со словарём:", by_locale == DICTIONARY_ORDER)
except locale.Error as e:
    print("locale.strxfrm:    ошибка:", repr(e))
finally:
    locale.setlocale(locale.LC_COLLATE, "C")      # вернуть как было
# Обычный sorted: «Ёлка» первая (заглавная Ё — U+0401, раньше всего алфавита
#   А–я, U+0410–U+044F), «ёж» последний (строчная ё — U+0451, после я U+044F).
# casefold().replace("ё", "е"): регистр и ё убраны из ключа, остаётся алфавит
#   а…я подряд → словарный порядок. В словаре е и ё на первом уровне равны,
#   поэтому «ёж» (ж) раньше «Ёлка» (л), а «Ёлка» (к) раньше «ель» (ь).
# strxfrm зависит от того, какие локали установлены в ОС и как ОС их
#   реализует. Если ru_RU.UTF-8 нет — setlocale падает с ошибкой; если есть,
#   но реализация слабая (так бывает на macOS) — ошибки нет, а порядок
#   неправильный. Поведение кода меняется от машины к машине, тихо, без
#   изменений в самом коде — это главный аргумент против strxfrm в проде.
#   К тому же setlocale меняет глобальное состояние всего процесса.


# ---------------------------------------------------------------------------
# 3. None в данных
# ---------------------------------------------------------------------------
print("\n3. None в данных")
scores = [0.9, None, 0.3, None, 0.7]
try:
    sorted(scores)
except TypeError as e:
    print("sorted без ключа:", e)

none_last = sorted(scores, key=lambda x: (x is None, x))
none_first = sorted(scores, key=lambda x: (x is not None, x))
print("None в конце: ", none_last)
print("None в начале:", none_first)
assert none_last == [0.3, 0.7, 0.9, None, None]
assert none_first == [None, None, 0.3, 0.7, 0.9]
# Кортежи сравниваются по элементам слева направо, и до второго элемента дело
#   доходит, только если первые равны. Первый элемент — bool: False < True.
#   (False, 0.3) и (True, None) решаются уже на первом элементе — None с числом
#   никогда не сравнивается. Два None сравниваются как (True, None) против
#   (True, None): первые равны, вторые None == None — равны, сравнения «<»
#   между ними не нужно, и ошибки нет.
# Без ключа Timsort сравнивает None с float через «<», а у NoneType такого
#   сравнения нет → TypeError.


# ---------------------------------------------------------------------------
# 4. Смешанные направления: source ↓, внутри tokens ↑
# ---------------------------------------------------------------------------
print("\n4. Смешанные направления")
Chunk = namedtuple("Chunk", "source tokens text")
chunks = [
    Chunk("arxiv", 300, "a1"),
    Chunk("github", 50, "g1"),
    Chunk("confluence", 700, "c1"),
    Chunk("arxiv", 100, "a2"),
    Chunk("github", 900, "g2"),
    Chunk("confluence", 200, "c2"),
    Chunk("arxiv", 200, "a3"),
]

# От младшего ключа к старшему:
step1 = sorted(chunks, key=attrgetter("tokens"))                  # младший: tokens ↑
final = sorted(step1, key=attrgetter("source"), reverse=True)     # старший: source ↓
for c in final:
    print("   ", c.source, c.tokens, c.text)

assert [c.text for c in final] == ["g1", "g2", "c2", "c1", "a2", "a3", "a1"]
# проверка свойством: источники по убыванию, внутри — токены по возрастанию
for prev, cur in zip(final, final[1:]):
    assert prev.source > cur.source or (
        prev.source == cur.source and prev.tokens <= cur.tokens)
# Почему работает: Timsort УСТОЙЧИВ — элементы с равным ключом сохраняют
#   взаимный порядок. Вторая сортировка (по source) переставляет группы, но
#   внутри группы с одинаковым source оставляет порядок от первой сортировки,
#   то есть по tokens. reverse=True тоже сохраняет устойчивость: Python не
#   переворачивает готовый список, а сортирует «наоборот» с сохранением
#   порядка равных.
# Почему не кортежем в одном key: кортеж (source, tokens) сортирует оба поля
#   в ОДНУ сторону, а reverse=True переворачивает весь кортеж целиком —
#   source ↓, но и tokens ↓. Развернуть одно поле можно только минусом, а
#   минус есть только у чисел: «-source» для строки не существует.
# Оговорка честности: здесь младший ключ tokens — число, поэтому есть обход
#   (source, -tokens) с reverse=True: reverse перевернёт и минус, и выйдет
#   tokens ↑. Это совпадает с двойной сортировкой (assert ниже), но только
#   из-за того, что tokens числовой. Если оба поля строки (например, source ↓,
#   а внутри text ↑), одним key это не выражается — только двойной сортировкой.
assert sorted(chunks, key=lambda c: (c.source, -c.tokens), reverse=True) == final


# ---------------------------------------------------------------------------
# 5. Цена ключа: сколько раз вызывается key
# ---------------------------------------------------------------------------
print("\n5. Цена ключа")
calls = 0


def expensive_key(s):
    global calls
    calls += 1
    return unicodedata.normalize("NFKD", s).casefold()


random.seed(42)
ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFﬁﬂ"
items = ["".join(random.choices(ALPHABET, k=200)) for _ in range(1000)]
n = len(items)

sorted(items, key=expensive_key)
print(f"вызовов key: {calls}")
print(f"n = {n},  n·log2(n) ≈ {n * log2(n):.0f}")
assert calls == n


class Counted:
    """Обёртка, которая считает сравнения «<» — их и делает Timsort."""
    comparisons = 0

    def __init__(self, key):
        self.key = key

    def __lt__(self, other):
        Counted.comparisons += 1
        return self.key < other.key


sorted(Counted(expensive_key(s)) for s in items)
print(f"сравнений при сортировке: {Counted.comparisons}")
# key вызвался ровно n = 1000 раз, а сравнений Timsort сделал порядка n·log n
#   (~8–9 тысяч). Python сначала один раз вычисляет ключ для каждого элемента,
#   запоминает пары (ключ, элемент) и дальше сравнивает уже готовые ключи.
#   Если бы та же нормализация жила в __lt__, она выполнялась бы при каждом
#   сравнении — дважды на сравнение, то есть ~17 тысяч раз вместо 1000.


# ---------------------------------------------------------------------------
# 6. itemgetter против lambda
# ---------------------------------------------------------------------------
print("\n6. itemgetter против lambda")
random.seed(42)
rows = [(i, random.random(), "x") for i in range(1_000_000)]
assert sorted(rows, key=itemgetter(1)) == sorted(rows, key=lambda r: r[1])

t_item = best_of(lambda: sorted(rows, key=itemgetter(1)))
t_lambda = best_of(lambda: sorted(rows, key=lambda r: r[1]))
print(f"itemgetter(1):     {t_item:.3f} с")
print(f"lambda r: r[1]:    {t_lambda:.3f} с")
print(f"lambda / itemgetter: {t_lambda / t_item:.2f}")
# Ключ вызывается миллион раз. lambda — это вызов Python-функции: создать
#   кадр, выполнить байткод r[1], вернуть результат. itemgetter написан на C
#   и делает только индексацию — поэтому itemgetter быстрее.
# Но разница небольшая (у меня 5–12% от запуска к запуску): ключ вызывается всего n раз (задание 5),
#   а основное время уходит на ~n·log n ≈ 20 млн сравнений случайных float —
#   и они одинаковы в обоих вариантах. Выигрыш itemgetter заметнее, когда
#   сортировка дешёвая (почти упорядоченные данные) и доля вызовов key больше.


# ---------------------------------------------------------------------------
# 7. Устно, потом письменно
# ---------------------------------------------------------------------------
# Почему key вызывается n раз, а не n·log n: sorted сначала проходит по списку
#   и вычисляет ключ для каждого элемента ровно один раз, сохраняя его рядом
#   с элементом. Сортировка потом сравнивает только готовые ключи — а
#   сравнений действительно порядка n·log n (задание 5: 1000 вызовов key
#   против ~8–9 тысяч сравнений).
# Почему работает двойная сортировка: Timsort устойчив — равные по ключу
#   элементы сохраняют порядок, который был до сортировки. Поэтому сначала
#   сортируют по младшему ключу, потом по старшему: старший расставляет группы,
#   а внутри группы остаётся порядок младшего (задание 4).
# Почему «ёж» после «яблоко»: строки сравниваются по кодам символов. Алфавит
#   а–я занимает подряд U+0430–U+044F, а «ё» вынесена отдельно — U+0451, после
#   «я». Значит, для sorted «ё» больше любой буквы алфавита (задание 2).