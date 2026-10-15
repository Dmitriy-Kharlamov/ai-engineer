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


# 5. Деление ломается из-за нулей:
#    один ноль — total = 0, на месте нуля получается 0 / 0 → ZeroDivisionError,
#    а правильный ответ там — произведение остальных (для [1, 2, 0, 4] это 8).
#    Два нуля — total = 0, ответ везде 0 верный, но на местах нулей
#    всё равно делим 0 на 0 → ZeroDivisionError. Информация о ненулевых числах
#    теряется в total, и делением её не восстановить.
def product_except_self(numbers):
    n = len(numbers)
    result = [1] * n
    left = 1
    for i in range(n):              # проход слева направо
        result[i] = left             # произведение всех левее i
        left *= numbers[i]
    right = 1
    for i in range(n - 1, -1, -1):  # проход справа налево
        result[i] *= right             # домножаем на произведение всех правее i
        right *= numbers[i]
    return result


def product_except_self_reference(numbers):
    result = []
    for i in range(len(numbers)):
        product = 1
        for j in range(len(numbers)):
            if j != i:
                product *= numbers[j]
        result.append(product)
    return result


# 6. -0 в Python равно 0, поэтому items[-0:] — это items[0:], то есть весь список,
#    а items[:-0] — это items[:0], то есть пустой список.
def rotate_reference(items, k):
    n = len(items)
    if n == 0:
        return items
    k = k % n
    return items[-k:] + items[:-k]


def reverse_part(items, lo, hi):
    while lo < hi:
        items[lo], items[hi] = items[hi], items[lo]
        lo += 1
        hi -= 1


def rotate(items, k):
    n = len(items)
    if n == 0:
        return items
    k = k % n
    reverse_part(items, 0, n - 1)
    reverse_part(items, 0, k - 1)
    reverse_part(items, k, n - 1)
    return items


def check_rotate(trials=500):
    for _ in range(trials):
        size = random.randint(0, 30)
        items = [random.randint(-50, 50) for _ in range(size)]
        for k in range(2 * size + 1):
            expected = rotate_reference(items[:], k)
            actual = rotate(items[:], k)
            assert actual == expected, f"вход {items}, k={k}: получил {actual}, ждал {expected}"
    print(f"rotate: {trials} проверок пройдено")


# 7. Если все числа отрицательные, функция вернёт отрицательное — самое большое число списка
# (для [-3, -1, -2] это -1, отрезок [-1] из одного элемента).
# Почему так: любой отрезок длиннее одного элемента складывает отрицательные
# числа, и сумма от этого только уменьшается. Значит, лучший отрезок — один элемент.
# Это правильно для «непустого» подмассива: пустой отрезок дал бы сумму 0,
# и она была бы больше любого отрицательного числа, но условие запрещает
# брать пустой отрезок, поэтому 0 вернуть нельзя.
def max_subarray(numbers):
    if not numbers:
        return None
    current = best = numbers[0]
    for x in numbers[1:]:
        current = max(current + x, x)
        best = max(best, current)
    return best


def max_subarray_reference(numbers):
    if not numbers:
        return None
    n = len(numbers)
    return max(sum(numbers[i:j]) for i in range(n) for j in range(i + 1, n + 1))


if __name__ == "__main__":
    check(max_and_min, max_and_min_reference)
    check_range_sum()
    check(move_zeroes, move_zeroes_reference, low=-2, high=2)
    check_remove_duplicates()
    check(product_except_self, product_except_self_reference, low=-3, high=3, max_size=10)
    assert product_except_self([1, 2, 0, 4]) == [0, 0, 8, 0]
    assert product_except_self([0, 2, 0, 4]) == [0, 0, 0, 0]
    print("product_except_self: проверки с нулями пройдены")
    check_rotate()
    check(max_subarray, max_subarray_reference)