import random
from collections import Counter


def two_sum(numbers, target):
    seen = {}
    for index, value in enumerate(numbers):
        complement = target - value
        if complement in seen:
            return (seen[complement], index)
        seen[value] = index
    return None


# 1. Сломанная версия кладёт элемент в seen до проверки. Если элемент равен
#    половине target (4 при target = 8), его дополнение — он сам, и проверка
#    находит в словаре тот же элемент: [4, 1], 8 → (0, 0) вместо None.
#    Правильный порядок — сначала искать, потом класть: проверка идёт только
#    среди элементов до текущего, поэтому элемент не может стать парой себе.
def two_sum_broken(numbers, target):
    seen = {}
    for index, value in enumerate(numbers):
        complement = target - value
        seen[value] = index               # СЛОМАНО: сначала кладём
        if complement in seen:            # потом ищем
            return (seen[complement], index)
    return None


def two_sum_reference(numbers, target):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return (i, j)
    return None


def check_two_sum(trials=500):
    for _ in range(trials):
        numbers = [random.randint(-10, 10) for _ in range(random.randint(0, 10))]
        target = random.randint(-20, 20)
        result = two_sum(numbers, target)
        if two_sum_reference(numbers, target) is None:
            assert result is None, f"вход {numbers}, target={target}: получил {result}"
        else:
            i, j = result
            assert i < j and numbers[i] + numbers[j] == target, \
                f"вход {numbers}, target={target}: неверная пара {result}"
    print(f"two_sum: {trials} проверок пройдено")


# 2. Определение: число уникальных пар значений (a, a + k).
#    k < 0 → 0, разница по определению неотрицательная;
#    k = 0 → число значений, встречающихся хотя бы дважды;
#    дубликаты на входе не размножают пары.
def count_pairs_with_diff(numbers, k):
    if k < 0:
        return 0
    counts = Counter(numbers)
    if k == 0:
        return sum(1 for c in counts.values() if c >= 2)
    return sum(1 for v in counts if v + k in counts)


def count_pairs_with_diff_reference(numbers, k):
    if k < 0:
        return 0
    pairs = set()
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j and numbers[j] - numbers[i] == k:
                pairs.add((numbers[i], numbers[j]))
    return len(pairs)


def check_count_pairs(trials=500):
    assert count_pairs_with_diff([1, 1, 3], 2) == 1          # дубликаты
    assert count_pairs_with_diff([1, 1, 2, 2, 3], 0) == 2    # k = 0: (1,1) и (2,2)
    assert count_pairs_with_diff([1, 3], -2) == 0            # k < 0
    for _ in range(trials):
        numbers = [random.randint(-5, 5) for _ in range(random.randint(0, 10))]
        k = random.randint(-3, 3)
        assert count_pairs_with_diff(numbers, k) == count_pairs_with_diff_reference(numbers, k), \
            f"вход {numbers}, k={k}"
    print(f"count_pairs_with_diff: {trials} проверок пройдено")


# 3. Без counts = {0: 1} теряются все отрезки, которые начинаются с индекса 0
#    (то есть целые префиксы списка). Для такого отрезка total - k = 0 —
#    «префикс до начала списка», а в пустом словаре его нет.
#    [3], k = 3 → 1 и 0 (потерян [3]); [1, 2], k = 3 → 1 и 0 (потерян [1, 2]).
#    {0: 1} — тот же фиктивный элемент, что 0 в начале prefix в RangeSum (Ф2-02).
def count_subarrays_with_sum(numbers, k):
    counts = {0: 1}                  # префикс 0 «до начала списка» встречался 1 раз
    total = 0
    result = 0
    for value in numbers:
        total += value
        result += counts.get(total - k, 0)
        counts[total] = counts.get(total, 0) + 1
    return result


def count_subarrays_broken(numbers, k):
    counts = {}                      # СЛОМАНО: нет фиктивного префикса 0
    total = 0
    result = 0
    for value in numbers:
        total += value
        result += counts.get(total - k, 0)
        counts[total] = counts.get(total, 0) + 1
    return result


def count_subarrays_with_sum_reference(numbers, k):
    count = 0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers) + 1):   # отрезок numbers[i:j], непустой
            if sum(numbers[i:j]) == k:
                count += 1
    return count


def check_subarrays(trials=500):
    for _ in range(trials):
        numbers = [random.randint(-3, 3) for _ in range(random.randint(0, 10))]
        k = random.randint(-3, 3)                    # включая 0
        assert count_subarrays_with_sum(numbers, k) == \
            count_subarrays_with_sum_reference(numbers, k), f"вход {numbers}, k={k}"
    print(f"count_subarrays_with_sum: {trials} проверок пройдено")


if __name__ == "__main__":
    check_two_sum()
    print("правильная:", two_sum([4, 1], 8))
    print("сломанная: ", two_sum_broken([4, 1], 8))
    check_count_pairs()
    for numbers, k in [([3], 3), ([1, 2], 3)]:
        print(numbers, k, "→ правильная:", count_subarrays_with_sum(numbers, k),
            "| сломанная:", count_subarrays_broken(numbers, k))
    check_subarrays()