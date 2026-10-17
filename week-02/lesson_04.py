import random
from collections import Counter, defaultdict
from itertools import groupby


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


# 4. groupby группирует только подряд идущие элементы с одинаковым ключом:
#    он режет последовательность на участки и начинает новую группу каждый раз,
#    когда ключ меняется. В WORDS длины идут 3, 4, 3, 4, 3, 5, 3 — поэтому ключ 3
#    встречается 4 раза отдельными группами. После сортировки по len
#    одинаковые длины стоят рядом, и groupby даёт настоящие группы (O(n log n)).
#    defaultdict собирает группы по всему списку за один проход, O(n), без сортировки.
WORDS = ["кот", "слон", "дом", "мышь", "ток", "жираф", "лев"]


def group_by_length(words):
    groups = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return dict(groups)


def group_by_length_groupby(words):
    return [(key, list(group)) for key, group in groupby(words, key=len)]


def group_by_length_groupby_sorted(words):
    ordered = sorted(words, key=len)
    return [(key, list(group)) for key, group in groupby(ordered, key=len)]


# 5. Контрпример для одного словаря: то и аа. forward проверяет только, что каждая
#    буква первой строки всегда переходит в одну и ту же букву второй, но не
#    запрещает двум разным буквам перейти в одну букву: {т→а, о→а} он принимает.
#    Второй словарь backward проверяет обратное направление и ловит это.
def is_isomorphic(a, b):
    if len(a) != len(b):
        return False
    forward, backward = {}, {}
    for x, y in zip(a, b):
        if forward.setdefault(x, y) != y:
            return False
        if backward.setdefault(y, x) != x:
            return False
    return True


def pattern(s):
    first = {}
    return [first.setdefault(ch, len(first)) for ch in s]


def is_isomorphic_reference(a, b):
    return pattern(a) == pattern(b)


def is_isomorphic_one_dict(a, b):
    if len(a) != len(b):
        return False
    forward = {}
    for x, y in zip(a, b):
        if forward.setdefault(x, y) != y:
            return False
    return True


def check_isomorphic(trials=500):
    for _ in range(trials):
        a = "".join(random.choice("аб") for _ in range(random.randint(0, 4)))
        b = "".join(random.choice("вг") for _ in range(random.randint(0, 4)))
        assert is_isomorphic(a, b) == is_isomorphic_reference(a, b), f"{a!r}, {b!r}"
    print(f"is_isomorphic: {trials} проверок пройдено")


# 6. Counter & — пересечение с кратностью: по каждому слову берётся минимум из двух
#    счётчиков («кот» 3 и 2 → 2, «пёс» 1 и 3 → 1). set & — только факт наличия,
#    кратность теряется.
#    Counter нужен, когда важно, СКОЛЬКО раз: например, можно ли составить слово
#    «мама» из букв «мамонт» (нужны две «м» и две «а», а не просто наличие букв),
#    или насколько сильно пересекаются два документа по словам.
#    set нужен, когда важно только, ЕСТЬ ЛИ: общие ключевые слова или темы
#    двух документов, пересечение тегов.
def common_words(text_a, text_b):
    return Counter(text_a.lower().split()) & Counter(text_b.lower().split())


def common_words_set(text_a, text_b):
    return set(text_a.lower().split()) & set(text_b.lower().split())


if __name__ == "__main__":
    check_two_sum()
    print("правильная:", two_sum([4, 1], 8))
    print("сломанная: ", two_sum_broken([4, 1], 8))
    check_count_pairs()
    for numbers, k in [([3], 3), ([1, 2], 3)]:
        print(numbers, k, "→ правильная:", count_subarrays_with_sum(numbers, k),
            "| сломанная:", count_subarrays_broken(numbers, k))
    check_subarrays()
    print("defaultdict:     ", group_by_length(WORDS))
    print("groupby без sort:", group_by_length_groupby(WORDS))
    print("groupby с sort:  ", group_by_length_groupby_sorted(WORDS))
    assert group_by_length(WORDS) == dict(group_by_length_groupby_sorted(WORDS))
    a, b = "то", "аа"
    print("один словарь:", is_isomorphic_one_dict(a, b), "| правильно:", is_isomorphic(a, b))
    check_isomorphic()
    text_a = "кот кот кот пёс рыба"
    text_b = "кот кот пёс пёс пёс"
    print("Counter &:", common_words(text_a, text_b))
    print("set &:    ", common_words_set(text_a, text_b))
    assert common_words(text_a, text_b) == Counter({"кот": 2, "пёс": 1})
    assert common_words_set(text_a, text_b) == {"кот", "пёс"}