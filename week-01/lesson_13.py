# 1
raw = ["10", "25", "3", "47"]
raw_number_1 = []
for x in raw:
    raw_number_1.append(int(x))
raw_number_2 = list(map(int, raw))
raw_number_3 = [int(x) for x in raw]

print(raw_number_1)
print(raw_number_2)
print(raw_number_3)


# 2 
words = ["трансформер", "токен", "эмбеддинг", "слой", "attention"]
words_1 = [word for word in words if len(word) > 5]
print(words_1)
len_words = [len(word) for word in words]
print(len_words)
upper_words = [word.upper() for word in words]
print(upper_words)


# 3
print(sum(1 for w in words if len(w) > 5))

# 4
scores = {"precision": 0.87, "recall": 0.72, "f1": 0.79}

print(all(score > 0.5 for score in scores.values()))
print(any(score > 0.9 for score in scores.values()))
print(all(score < 1.0 for score in scores.values()))


# 5
people = [
    {"name": "Иван", "city": "Москва", "age": 30},
    {"name": "Анна", "city": "Казань", "age": 25},
    {"name": "Пётр", "city": "Москва", "age": 25},
    {"name": "Мария", "city": "Казань", "age": 35},
]
print(sorted(people, key=lambda p: p["age"]))
print(sorted(people, key=lambda p: p["city"]))
print(sorted(people, key=lambda p: (p["city"], p["age"])))
print(sorted(people, key=lambda p: (p["city"], -p["age"])))


# 6
min_age_person = min(people, key=lambda p: p["age"])
print(min_age_person["name"], min_age_person["age"])
max_age_person = max(people, key=lambda p: p["age"])
print(max_age_person["name"], max_age_person["age"])


# 7
text = "машинное обучение и нейронные сети"
keywords = ["трансформер", "нейронные", "токенизация"]
print(any(keyword in text for keyword in keywords))

# 8
nums = map(int, ["1", "2", "3"])
print(sum(nums))
print(sum(nums))
# Первый вызов даёт результат ["1", "2", "3"], второй вызов даёт пустоту, потому что итератор одноразовый.
