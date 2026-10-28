# 1
coords = (55.75, 37.61, 48.02)
x, y, z = coords
print(x)
print(y)
print(z)
# coords[0] = 56.75  # TypeError: 'tuple' object does not support item assignment

# 2
a = 10
b = 20
a, b = b, a
print(a, b)

# 3
numbers = [10, 20, 30, 40, 50]
first, *rest = numbers
print(first, rest)
*rest, last = numbers
print(rest, last)
first, *rest, last = numbers
print(first, rest, last)


# 4
results = [("BERT", 0.87), ("GPT-4", 0.94), ("T5", 0.79), ("Llama", 0.88)]
for name, score in results:
    print(f"{name}: {score * 100:.0f}%")
best_name, best_score = results[0]

for name, score in results:
    if score > best_score:
        best_name = name
        best_score = score

print(f"Лучший результат у модели {best_name}: {best_score:.0%}")


# 5
names = []
scores = []
for name, score in results:
    names.append(name)
    scores.append(score)
print(names)
print(scores)

zipped_names, zipped_scores = zip(*results)
print(zipped_names)
print(zipped_scores)

# 6
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
print(sorted_results[0:2])

# 7
# Список не может быть использован в качестве ключа словаря, а кортеж может
