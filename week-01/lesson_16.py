import json
import csv
from pathlib import Path

# 1
with open("notes.txt", "w", encoding="utf-8") as f:
    f.write("Первая строка\n")
    f.write("Вторая строка\n")
    f.write("Третья строка\n")
    f.write("Четвертая строка\n")
    f.write("Пятая строка\n")

# Чтение файла целиком
with open("notes.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print("Чтение целиком:")
    print(content)

# Чтение файла как список строк
with open("notes.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print("Чтение как список строк:")
    for line in lines:
        print(line.strip())

# Чтение файла построчно
with open("notes.txt", "r", encoding="utf-8") as f:
    print("Чтение построчно:")
    for line in f:
        print(line.strip())



# 2
with open("notes.txt", "a", encoding="utf-8") as f:
    f.write("Шестая строка\n")
    f.write("Седьмая строка\n")

with open("notes.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print("Чтение целиком:")
    print(content)


# 3
def count_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return len(f.readlines())
    except FileNotFoundError:
        print(f"Файл {path} не найден")
        return 0
    
print(count_lines("notes.txt"))    # существующий
print(count_lines("нет.txt"))      # отсутствующий


# 4
config = {
    "model": "claude-opus-5",
    "temperature": 0.7,
    "описание": "конфигурация для разбора документов",
    "stop_words": ["и", "в", "на"],
}

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

with open("config.json", encoding="utf-8") as f:
    loaded = json.load(f)

print(loaded)
print(loaded == config)


# 5
# Функция load_config(path) читает JSON. Если файла нет — возвращает дефолтный словарь. Если файл повреждён — печатает ошибку и тоже возвращает дефолты. Проверьте все три случая: создайте нормальный файл, битый файл (запишите туда мусор вручную) и укажите несуществующий путь.

DEFAULT_CONFIG = {"debug": False, "language": "ru", "max_items": 10}


def load_config(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_CONFIG.copy()
    except json.JSONDecodeError as e:
        print(f"Ошибка: файл {path.name} повреждён: {e}")
        return DEFAULT_CONFIG.copy()


base = Path(__file__).parent

# 1. Нормальный файл
good = base / "config_good.json"
good.write_text(json.dumps({"debug": True, "language": "en", "max_items": 50}), encoding="utf-8")

# 2. Битый файл
bad = base / "config_bad.json"
bad.write_text("{это не json, ]]", encoding="utf-8")

# 3. Несуществующий путь
missing = base / "config_missing.json"

print("1. Нормальный:", load_config(good))
print("2. Битый:", load_config(bad))
print("3. Нет файла:", load_config(missing))
    

# 6
# Сохраните список словарей в формате JSON Lines, затем прочитайте обратно и выведите. Это основа работы с датасетами.
chunks = [
    {"id": 1, "text": "первый чанк", "tokens": 12},
    {"id": 2, "text": "второй чанк", "tokens": 15},
    {"id": 3, "text": "третий чанк", "tokens": 9},
]
with open("chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

with open("chunks.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]
print (items)


# 7
with open("chunks.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "tokens"])
    writer.writeheader()
    writer.writerows(chunks)

with open("chunks.csv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    total = sum(int(row["tokens"]) for row in reader)
print(total)


# 8
folder = Path(__file__).parent  # папка, где лежит сам скрипт, то есть week-01

for path in sorted(folder.glob("*.py")):
    size = path.stat().st_size
    print(f"{path.name:<25} {path.stem:<20} {size} байт")

# 9
data = {"items": (1, 2, 3), "when": {1, 2}}
json.dumps(data)
# Результатом выполнения будет ошибка TypeError: Object of type set is not JSON serializable, так как множество не может быть сериализовано в JSON. Если убрать множество, то кортеж молча превратится в список, и при обратном чтении вернётся список, а не кортеж.
