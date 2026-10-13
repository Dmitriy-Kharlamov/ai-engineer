# а - O(n²): вложенный цикл, для каждого элемента сравниваем со всеми следующими
def first_duplicate(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return items[i]
    return None

# б - O(n), потому что надо пройтись по каждому w в списке
def count_long(words):
    return sum(1 for w in words if len(w) > 5)

# в - O(n), потому что seen — множество, а проверка in у множества стоит O(1)
def has_pair_summing(numbers, target):
    seen = set()
    for n in numbers:
        if target - n in seen:
            return True
        seen.add(n)
    return False

# г - O(n log n): sorted сортирует весь список, срез [:3] дешёвый, берём самый дорогой шаг
def top_three(scores):
    return sorted(scores, reverse=True)[:3]

# д - O(n²) — каждое сложение копирует всю строку заново
def build_report(rows):
    report = ""
    for row in rows:
        report += f"{row}\n"
    return report

# е O(log n): на каждом шаге n уменьшается вдвое, поэтому шагов log₂(n)
def halve_until_one(n):
    steps = 0
    while n > 1:
        n = n // 2
        steps += 1
    return steps

