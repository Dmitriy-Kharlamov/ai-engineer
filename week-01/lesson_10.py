# 1
tokens = ["кот", "пёс", "кот", "рыба", "пёс", "кот", "кит"]
unique = set(tokens) #  в множестве выводятся уникальные элементы, порядок не сохраняется
print(unique)
unique = list(dict.fromkeys(tokens)) #  в словаре сохраняется порядок добавления элементов, а ключи словаря уникальны, поэтому при преобразовании обратно в список мы получаем уникальные элементы в порядке их появления
print(unique)


# 2
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}
print(a | b)
print(a & b)
print(a - b)
print(a ^ b)


# 3
text_a = "машинное обучение это подраздел искусственного интеллекта"
text_b = "глубокое обучение это подраздел машинного обучения"
set_a = set(text_a.split())
set_b = set(text_b.split())
print(set_a & set_b)
print(set_a - set_b)
print(set_b - set_a)
similarity = len(set_a & set_b) / len(set_a | set_b)
print(f"Схожесть: {similarity:.2f}")

# 4
text = "кот сидит на окне и смотрит в сад с интересом"
STOP_WORDS = {"и", "в", "на", "с", "это"}
words = text.split()
filtered = [w for w in words if w not in STOP_WORDS]
print(filtered)
print(f"Удалено слов: {len(words) - len(filtered)}")


# 5
dense_results = [101, 205, 310, 415, 520]
sparse_results = [205, 310, 601, 702, 415]
set_dense = set(dense_results)
set_sparse = set(sparse_results)
print(set_dense & set_sparse)
print(set_dense - set_sparse)
print(set_sparse - set_dense)
print(len(set_dense | set_sparse))






# 6
s = {1, 2, 3}
s.add(2)
s.add("2")
print(s)
print(len(s))
# 2 добавится только один раз, потому что множество не может содержать дубликаты. "2" добавится, потому что это строка, а не число 2. В итоге в множестве будет 4 элемента: 1, 2, 3 и "2".