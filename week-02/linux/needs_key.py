import os
import sys

try:
    key = os.environ["API_KEY"]
except KeyError:
    sys.exit(
        "ошибка: не задана переменная окружения API_KEY.\n"
        "Задай её: export API_KEY=... или API_KEY=... python3 needs_key.py"
    )

# сам ключ никогда не печатаем — только факт, что он есть
print(f"API_KEY получен, длина {len(key)} символов")
