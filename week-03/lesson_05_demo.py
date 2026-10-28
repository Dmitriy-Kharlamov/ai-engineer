"""Ф3-05, задание 1: семь спрятанных ошибок. Номер ошибки — в комментарии.

Проверка:
    mypy --config-file "" week-03/lesson_05_demo.py                       # без настроек
    mypy --config-file "" --strict week-03/lesson_05_demo.py
    mypy --config-file "" --strict --warn-unreachable week-03/lesson_05_demo.py
    ruff check week-03/lesson_05_demo.py
"""

import json  # 7. неиспользуемый импорт


def shout(text):  # 1. функция без аннотаций, ошибка внутри: строка + число
    return text.upper() + 5


def parse_port(raw: str):  # 2. возвращает то int, то str (тип возврата не указан)
    if raw.isdigit():
        return int(raw)
    return "default"


def greet_length(name: str | None) -> int:  # 3. str | None без проверки на None
    return len(name)


def total(xs: list[int]) -> int:  # 4. код после return
    return sum(xs)
    print("посчитано")


def add_item(item: str, bucket: list[str] = []) -> list[str]:  # 5. изменяемое умолчание
    bucket.append(item)
    return bucket


def describe(list: list[int]) -> str:  # 6. параметр перекрывает встроенное имя list
    return f"{len(list)} элементов"


if __name__ == "__main__":
    port = parse_port("abc")
    print(port + 1)  # упадёт: "default" + 1
