import json

# 1
while True:
    raw = input("Введите число: ")
    try:
        number = float(raw)
        break
    except ValueError:
        print("Это не число, попробуйте ещё раз")

print(f"Вы ввели {number}")


# 2
result = None
try:
    a = float(input("Первое число: "))
    operation = input("Операция (+, -, *, /): ")
    b = float(input("Второе число: "))
    if operation == "+":
        result = a + b
    elif operation == "-":
        result = a - b
    elif operation == "*":
        result = a * b
    elif operation == "/":
        result = a / b
    else:
        print(f"Неизвестная операция: {operation}")

except ZeroDivisionError:
    print("Деление на ноль")

except (TypeError, ValueError):
    print("Проблема с типом или значением")

if result is not None:
    print(f"Результат: {result}")


# 3
def safe_get_if(data, key):
    if key in data:
        return data[key]
    return "не найдено"


print(safe_get_if({"a": 1, "b": 2}, "a"))  # 1
print(safe_get_if({"a": 1, "b": 2}, "c"))  # не найдено


def safe_get_try(data, key):
    try:
        return data[key]
    except KeyError:
        return "не найдено"


print(safe_get_try({"a": 1, "b": 2}, "a"))  # 1
print(safe_get_try({"a": 1, "b": 2}, "c"))  # не найдено


def safe_get_get(data, key):
    return data.get(key, "не найдено")


print(safe_get_get({"a": 1, "b": 2}, "a"))  # 1
print(safe_get_get({"a": 1, "b": 2}, "c"))  # не найдено

# 4
raw = "0.87 абв 0.94 0.79 ??? 0.88"


def parse_scores(raw):
    scores = []
    for item in raw.split():
        try:
            scores.append(float(item))
        except ValueError:
            print(f"Предупреждение: '{item}' не является числом")
    return scores


# 5
def validate_config(config):
    if "model" not in config:
        raise KeyError("Отсутствует ключ 'model'")

    if "temperature" not in config:
        raise KeyError("Отсутствует ключ 'temperature'")
    temperature = config["temperature"]
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature должен быть числом")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature должен быть в диапазоне 0–2")

    if "max_tokens" not in config:
        raise KeyError("Отсутствует ключ 'max_tokens'")
    if not isinstance(config["max_tokens"], int):
        raise TypeError("max_tokens должен быть целым числом")


configs = [
    {"model": "claude-opus-5", "temperature": 0.7, "max_tokens": 1000},
    {"temperature": 0.7, "max_tokens": 1000},
    {"model": "gpt-4", "temperature": 5, "max_tokens": 1000},
    {"model": "gpt-4", "temperature": 0.7, "max_tokens": "много"},
]

for config in configs:
    try:
        validate_config(config)
        print("OK")
    except (KeyError, ValueError, TypeError) as e:
        print(f"{type(e).__name__}: {e}")


# 6
class ChunkTooLargeError(ValueError):
    pass


def add_chunk(text, max_size=500):
    if len(text) > max_size:
        raise ChunkTooLargeError(f"Чанк {len(text)} символов, лимит {max_size}")


try:
    add_chunk("x" * 600)
except ChunkTooLargeError as e:
    print(f"Не добавили: {e}")


# 7
responses = [
    '{"text": "ответ", "score": 0.9}',
    "не json вовсе",
    '{"text": "другой", "score": 0.8}',
    '{"text": "сломан", "score":}',
]

successful = []
failed = []

for resp in responses:
    try:
        data = json.loads(resp)
        successful.append(data)
    except json.JSONDecodeError:
        failed.append(resp)

print(f"Успешно: {successful}")
print(f"Не удалось: {failed}")

# 8
try:
    print("A")
    raise ValueError("ошибка")
except ValueError:
    print("B")
    raise
except Exception:
    print("C")
finally:
    print("D")
print("E")

# Этот код выведет ошибку и завершится с исключением ValueError, так как после блока except ValueError вызывается raise без аргументов, что повторно возбуждает то же исключение.
