# 1
numbers = [1, 2, 3, 4, 5]
print(numbers[0])
print(numbers[-1])
print(numbers[0:3])
print(numbers[-2:])
print(numbers[::-1])
print(len(numbers))
print(sum(numbers))


# 2
models = ["BERT", "GPT", "T5"]
models.append("Llama")
models.insert(0, "Word2Vec")
models.pop(2)
print(models)
print(len(models))


# 3
raw = input("Введите числа через пробел: ")
numbers = [int(x) for x in raw.split()]
new_numbers_1 = sorted(numbers)
new_numbers_2 = sorted(numbers, reverse=True)
print(numbers)
print(new_numbers_1)
print(new_numbers_2)


# 4
tokens = ["кот", "пёс", "кот", "рыба", "пёс", "кот"]
print(tokens.count("кот"))
new_tokens = []
for token in tokens:
    if token not in new_tokens:
        new_tokens.append(token)
print(f"Список без дубликатов: {new_tokens}")

counts = {}
for token in tokens:
    if token in counts:
        counts[token] += 1
    else:
        counts[token] = 1

most_frequent = None
max_count = 0

for token, count in counts.items():
    if count > max_count:
        max_count = count
        most_frequent = token

print(f"Самый частый элемент: {most_frequent} (встречается {max_count} раз)")


# 5
numbers = [12, 5, 8, 130, 44, 3, 99]
new_numbers_1 = [x for x in numbers if x % 2 == 0]
new_numbers_2 = [x for x in numbers if x > 10]
new_numbers_3 = [x**2 for x in numbers]

print(new_numbers_1)
print(new_numbers_2)
print(new_numbers_3)


# 6
# [1, 2, 3, 4] - потому что a и b теперь один объект
# [1, 2, 3] - потому что c - это копия a до добавления 4
