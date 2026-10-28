from collections import Counter, defaultdict

# 1
model = {
    "name": "Дмитрий",
    "age": 29,
    "city": "Москва",
    "year_start_programming": 2026,
    "subject": "Python",
}
model["sex"] = "мужской"
model["age"] = 30
del model["year_start_programming"]
print(model)


# 2
text = "кот пёс кот рыба пёс кот попугай"
words = text.split()
counts = {}
for word in words:
    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1
print(counts)

counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
print(counts)

counts = Counter(words)
print(counts)


# 3
high = {k: v for k, v in counts.items() if v > 1}
print(high)


counts = Counter(words)
word, count = counts.most_common(1)[0]
print(f"Самое частое: {word} ({count} раз)")

# 4
scores = {"BERT": 0.87, "GPT-4": 0.94, "T5": 0.79, "Llama": 0.88, "Mistral": 0.91}
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print(sorted_scores)

print(f"Топ-3: {sorted_scores[:3]}")

high = {k: v for k, v in scores.items() if v > 0.9}
print(high)

average = sum(scores.values()) / len(scores)

print(f"Средний балл: {average:.2f}")


# 5
words = ["кот", "пёс", "кит", "попугай", "краб", "рыба"]
groups = {}
for word in words:
    letter = word[0]
    groups[letter] = groups.get(letter, []) + [word]
print(groups)

groups = defaultdict(list)
for word in words:
    groups[word[0]].append(word)
print(groups)


# 6
response = {
    "model": "claude-opus-5",
    "usage": {"input_tokens": 1250, "output_tokens": 340},
    "choices": [
        {"text": "Первый вариант", "score": 0.92},
        {"text": "Второй вариант", "score": 0.85},
    ],
}
print(f"Название модели: {response['model']}")
total_tokens = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
print(f"Общее число токенов: {total_tokens}")
best_choice = max(response["choices"], key=lambda x: x["score"])
print(f"Текст варианта с лучшим score: {best_choice['text']}")
print(response.get("usage", {}).get("output_tokens", 0))


# 7
inverted = {value: key for key, value in scores.items()}
print(inverted)
