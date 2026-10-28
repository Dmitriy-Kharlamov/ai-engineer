"""Ф3-05, задание 2: Any заражает.

Проверка (вывод mypy и есть результат задания; ошибки здесь — намеренные):
    mypy --config-file "" week-03/lesson_05_any.py             # без настроек
    mypy --config-file "" --strict week-03/lesson_05_any.py    # строго
reveal_type — инструкция для mypy; при обычном запуске Python печатает тип в stderr.
"""

import json
from typing import Any, reveal_type


# --- 2.1 reveal_type: неаннотированный параметр против аннотированного ---------
def untyped(x):  # без аннотаций — намеренно
    reveal_type(x)  # Any
    reveal_type(x.upper())  # Any
    return x.upper() + 5  # строка + число: mypy молчит


def typed(s: str) -> None:
    reveal_type(s)  # str
    reveal_type(s.upper())  # str
    print(s.upper() + 5)  # та же ошибка — и mypy её видит


# --- 2.2 цепочка из трёх функций: Any из первой гасит ошибку в третьей ------------
def load_config(raw):  # 1) не аннотирована → вызов возвращает Any
    return json.loads(raw)


def get_port(config: dict[str, Any]) -> Any:  # 2) честно пропускает Any дальше
    return config["port"]


def build_url(host: str, raw_config: str) -> str:  # 3) здесь настоящая ошибка
    port = get_port(load_config(raw_config))
    reveal_type(port)  # Any
    return "http://" + host + ":" + port  # в рантайме port = 8080 (int) → TypeError


# --- 2.3 то же через object: mypy заставляет проверить тип ------------------------
def get_port_obj(config: dict[str, object]) -> object:
    return config["port"]


def build_url_obj(host: str, config: dict[str, object]) -> str:
    port = get_port_obj(config)
    return "http://" + host + ":" + port  # mypy: str + object — ошибка


def build_url_checked(host: str, config: dict[str, object]) -> str:
    port = get_port_obj(config)
    if not isinstance(port, int):  # object требует проверку — после неё тип int
        raise TypeError(f"port должен быть int, пришло {type(port).__name__}")
    reveal_type(port)  # int
    return f"http://{host}:{port}"


if __name__ == "__main__":
    print(build_url_checked("localhost", {"port": 8080}))
    try:
        build_url("localhost", '{"port": 8080}')
    except TypeError as e:
        print("build_url упал в рантайме:", e)
