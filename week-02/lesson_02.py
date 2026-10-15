import random


def check(my_func, reference_func, trials=500, low=-50, high=50, max_size=30):
    for _ in range(trials):
        size = random.randint(0, max_size)
        data = [random.randint(low, high) for _ in range(size)]
        expected = reference_func(data[:])      # копия: функция может менять вход
        actual = my_func(data[:])
        assert actual == expected, f"вход {data}: получил {actual}, ждал {expected}"
    print(f"{my_func.__name__}: {trials} проверок пройдено")


# 1 Если выбрать None, то функция не вернет ошибку. Плюс - вызывающий код может использовать if result is None:
def max_and_min(numbers):
    if not numbers:
        return None
    biggest = smallest = numbers[0]
    for n in numbers[1:]:
        if n > biggest:
            biggest = n
        if n < smallest:
            smallest = n
    return (biggest, smallest)


def max_and_min_reference(numbers):
    if not numbers:
        return None
    return (max(numbers), min(numbers))


# 2. Память: O(n) — в prefix хранится n + 1 чисел.
#    Не окупается: (1) если запрос один — построение O(n) + запрос O(1) = O(n),
#    а один sum тоже O(n), стоимость одинаковая, но префиксы ещё занимают O(n) памяти;
#    (2) если список часто меняется — при каждом изменении префиксы надо
#    пересчитывать за O(n), и запрос перестаёт быть быстрым.
class RangeSum:
    def __init__(self, items):
        self.prefix = [0]
        for x in items:
            self.prefix.append(self.prefix[-1] + x)

    def query(self, i, j):
        return self.prefix[j] - self.prefix[i]


def check_range_sum(trials=500):
    for _ in range(trials):
        size = random.randint(0, 30)
        items = [random.randint(-50, 50) for _ in range(size)]
        rs = RangeSum(items)
        i = random.randint(0, size)
        j = random.randint(i, size)
        expected = sum(items[i:j])
        actual = rs.query(i, j)
        assert actual == expected, f"вход {items}, i={i}, j={j}: получил {actual}, ждал {expected}"
    print(f"RangeSum: {trials} проверок пройдено")


# 3
# Инвариант: перед каждым шагом в items[:write] лежат все ненулевые элементы из items[:read]
# в изначальном порядке, а в items[write:read] — только 0.
def move_zeroes(items):
    write = 0                          # куда положить следующее ненулевое
    for read in range(len(items)):     # read идёт по всем элементам
        if items[read] != 0:
            items[write], items[read] = items[read], items[write]   # поменять местами
            write += 1
    return items


def move_zeroes_reference(items):
    return [x for x in items if x != 0] + [0] * items.count(0)


# 4
def remove_duplicates(sorted_items):
    if not sorted_items:
        return 0
    write = 1
    for read in range(1, len(sorted_items)):
        if sorted_items[read] != sorted_items[write - 1]:
            sorted_items[write] = sorted_items[read]
            write += 1
    return write


def check_remove_duplicates(trials=500):
    for _ in range(trials):
        size = random.randint(0, 30)
        items = sorted(random.randint(-5, 5) for _ in range(size))
        original = items[:]
        k = remove_duplicates(items)
        assert k == len(set(original)), f"вход {original}: k={k}"
        assert items[:k] == sorted(set(original)), f"вход {original}: получил {items[:k]}"
    print(f"remove_duplicates: {trials} проверок пройдено")


if __name__ == "__main__":
    check(max_and_min, max_and_min_reference)
    check_range_sum()
    check(move_zeroes, move_zeroes_reference, low=-2, high=2)
    check_remove_duplicates()