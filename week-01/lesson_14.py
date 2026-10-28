import sys

# 1
nums = [-3, 4, -1, 6, 7, -8, 2]
nums_quad = [x**2 for x in nums]
print(nums_quad)
nums_even = [x for x in nums if x % 2 == 0]
print(nums_even)
nums_even_quad = [x**2 for x in nums if x % 2 == 0]
print(nums_even_quad)
nums_clipped = [x if x > 0 else 0 for x in nums]
print(nums_clipped)


# 2
matrix = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flat = [x for row in matrix for x in row]
print(flat)
big = [x for x in flat if x > 3]
print(big)


# 3
grid = [[0 for _ in range(4)] for _ in range(4)]
print(grid)


# 4
names = ["bert", "gpt-4", "t5", "llama"]
scores = [0.87, 0.94, 0.79, 0.88]
model_dict = dict(zip(names, scores, strict=True))
print(model_dict)
model_dict_more_than_0_85 = {x: y for x, y in model_dict.items() if y > 0.85}
print(model_dict_more_than_0_85)
model_dict_x_upper = {x.upper(): y for x, y in model_dict.items()}
print(model_dict_x_upper)
model_dict_turn_over = {y: x for x, y in model_dict.items()}
print(model_dict_turn_over)

# 5
text = "машинное обучение это подраздел искусственного интеллекта"
unique_words = set(text.split())
print(unique_words)

len_words = {x: len(x) for x in text.split()}
print(len_words)

long_upper = [w.upper() for w in text.split() if len(w) > 5]
print(long_upper)


# 6
listcomp = [x**2 for x in range(1_000_000)]
gen = (x**2 for x in range(1_000_000))

print(f"Список: {sys.getsizeof(listcomp):,} байт")
print(f"Генератор: {sys.getsizeof(gen):,} байт")


# 7
pairs = [("кот", 5), ("пёс", 1), ("рыба", 3), ("кит", 2), ("краб", 7)]
frequent = {k: v for k, v in sorted(pairs, key=lambda x: x[1], reverse=True) if v > 2}
print(frequent)

# 8
nums = [1, 2, 3, 4]

print(
    [x for x in nums if x > 2]
)  # [3, 4], потому что здесь используется фильтр, который оставляет только те элементы, которые больше 2
print(
    [x if x > 2 else 0 for x in nums]
)  # [0, 0, 3, 4], потому что здесь используется тернарный оператор, который заменяет элементы, которые не удовлетворяют условию (x > 2), на 0. Таким образом, первые два элемента (1 и 2) заменяются на 0, а оставшиеся элементы (3 и 4) остаются без изменений.
print(
    len([x for x in nums if x > 2])
)  # 2, потому что здесь используется фильтр, который оставляет только те элементы, которые больше 2, и затем вычисляется длина полученного списка. В данном случае, только два элемента (3 и 4) удовлетворяют условию (x > 2), поэтому длина списка равна 2.
print(
    len([x if x > 2 else 0 for x in nums])
)  # 4, потому что здесь используется тернарный оператор, который заменяет элементы, которые не удовлетворяют условию (x > 2), на 0. Таким образом, все элементы (1, 2, 3, 4) заменяются на 0 или остаются без изменений, и затем вычисляется длина полученного списка. В данном случае, длина списка равна 4.
