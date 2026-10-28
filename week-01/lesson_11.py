# 1
def greet(name):
    return f"Привет, {name}!"


print(greet("Дмитрий"))


# 2
def calculate(a, b, operation="+"):
    if operation == "+":
        return a + b
    if operation == "-":
        return a - b
    if operation == "*":
        return a * b
    if operation == "/":
        if b != 0:
            return a / b
        print("На ноль делить нельзя")
        return None
    print("Неизвестная операция")
    return None


print(f"Результат: {calculate(10, 0, '.')}")


# 3
def get_stats(numbers):
    if not numbers:
        return None
    minimum = min(numbers)
    maximum = max(numbers)
    sum_numbers = sum(numbers)
    average = sum(numbers) / len(numbers)
    return minimum, maximum, sum_numbers, average


print(get_stats([1, 2, 3, 4, 5]))


# 4
def clean_text(text, stop_words=None):
    if stop_words is None:
        stop_words = set()
    words = text.lower().split()
    stripped_words = [word.strip('.,!?;:"()[]') for word in words]
    return [w for w in stripped_words if w not in stop_words]


text = "Это пример текста с Некоторыми стоп-словами"

print(clean_text(text, stop_words={"пример", "текста"}))


# 5
def word_frequency(text, top_n=3):
    words = clean_text(text)
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    sorted_words = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]


print(word_frequency("Это пример текста с некоторыми словами. Это текст с повторяющимися словами."))


# 6
def jaccard(text_a, text_b):
    set_a = set(clean_text(text_a))
    set_b = set(clean_text(text_b))
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


print(jaccard("Это пример текста", "Это другой текст с примером"))


# 7
def add_tag(tag, tags=[]):
    tags.append(tag)
    return tags


print(add_tag("a"))
print(add_tag("b"))
print(add_tag("c", []))
print(add_tag("d"))
# В последнем вызове будет выведен список ['a', 'b', 'd'], потому что список tags по умолчанию сохраняет свое состояние между вызовами функции. 'c' добавляется в новый список, переданный в качестве аргумента, поэтому он не влияет на список по умолчанию.
