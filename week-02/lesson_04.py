import random


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


if __name__ == "__main__":
    check_two_sum()
    print("правильная:", two_sum([4, 1], 8))
    print("сломанная: ", two_sum_broken([4, 1], 8))