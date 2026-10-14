import time
from collections import deque


def measure(func, *args):
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


# а - O(n²): вложенный цикл, для каждого элемента сравниваем со всеми следующими
def first_duplicate(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return items[i]
    return None


# Задача 2: O(n) — один проход по списку, а проверка "in seen" у множества стоит O(1).
# Плата за скорость: дополнительная память O(n) под seen (если повторов нет, туда попадут все элементы),
# и элементы должны быть хэшируемыми — список [[1], [1]] вызовет TypeError.
# Разница в поведении: на [1, 2, 2, 1] старая версия вернёт 1, новая 2, потому что
# старая ищет пару для каждого элемента по порядку, а новая останавливается на первом элементе, который встретился второй раз.
def first_duplicate_fast(items):
    seen = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
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


# Задача 3: замеры на моём компьютере:
#   +=   : 20 000 → 0.0011 с, 200 000 → 0.0124 с, рост ≈ 10×
#   join : 20 000 → 0.0011 с, 200 000 → 0.0112 с, рост ≈ 10×
# Прогноз: += это O(n²), должно было вырасти в 100 раз; join это O(n) — в 10 раз.
# join совпал с прогнозом, += нет. Причина: CPython оптимизирует +=, если у строки только одна ссылка,
# и дописывает в неё на месте, без копирования. Это не гарантия языка, поэтому join надёжнее.
def build_report_join(rows):
    return "".join(f"{row}\n" for row in rows)


# е O(log n): на каждом шаге n уменьшается вдвое, поэтому шагов log₂(n)
def halve_until_one(n):
    steps = 0
    while n > 1:
        n = n // 2
        steps += 1
    return steps


# O(n·m), если stop_words — список (in перебирает его целиком); O(n), если множество (in за O(1))
# Задача 4: список 1.8151 с, множество 0.0016 с, множество быстрее в 1120 раз.
# Со списком O(n·m): для каждого из n слов "in" перебирает до m стоп-слов.
# С множеством O(n): каждая проверка "in" стоит O(1).
# Разница порядка m = 5 000; вышло около тысячи, потому что поиск в списке часто
# останавливается раньше конца, а проверка в множестве тоже не бесплатна (вычисляется хеш).  
def remove_stop_words(words, stop_words):
    return [w for w in words if w not in stop_words]


# Задача 5: list.pop(0) 2.2579 с, deque.popleft() 0.0025 с, deque быстрее в 913 раз.
# Список хранит элементы подряд: pop(0) сдвигает все оставшиеся на одно место, это O(n) за шаг,
# значит, обработка всей очереди — O(n²).
# deque берёт элемент с начала без сдвига: popleft() — O(1) за шаг, вся очередь — O(n).
def process_with_list(queue):
    while queue:
        queue.pop(0)


def process_with_deque(queue):
    while queue:
        queue.popleft()


if __name__ == "__main__":
    tests = [[3, 1, 4, 1, 5], [1, 2, 3], [], [1, 2, 2, 1]]
    for t in tests:
        print(t, first_duplicate(t), first_duplicate_fast(t))

    for size in (20_000, 200_000):
        rows = [f"строка номер {i}" for i in range(size)]
        plus_time = measure(build_report, rows)
        join_time = measure(build_report_join, rows)
        print(f"n={size}: += {plus_time:.4f} с, join {join_time:.4f} с")

    words = [f"слово{i % 10_000}" for i in range(100_000)]
    stop_list = [f"слово{i}" for i in range(5_000)]
    stop_set = set(stop_list)

    list_time = measure(remove_stop_words, words, stop_list)
    set_time = measure(remove_stop_words, words, stop_set)
    print(f"список: {list_time:.4f} с, множество: {set_time:.4f} с")
    print(f"множество быстрее в {list_time / set_time:.0f} раз")

    queue_list = list(range(200_000))
    queue_deque = deque(range(200_000))
    list_queue_time = measure(process_with_list, queue_list)
    deque_time = measure(process_with_deque, queue_deque)
    print(f"list.pop(0): {list_queue_time:.4f} с, deque.popleft(): {deque_time:.4f} с")
    print(f"deque быстрее в {list_queue_time / deque_time:.0f} раз")