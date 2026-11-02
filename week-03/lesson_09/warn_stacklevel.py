"""Задание 2: какой модуль попадает в поле module фильтра — решает stacklevel в чужом коде.

Цепочка вызовов как у starlette 1.6 и anyio:
    __main__  ->  starlette_like.use()  ->  anyio_like.deprecated()  ->  warnings.warn(...)
Для каждого stacklevel проверяем, какой фильтр "ignore::DeprecationWarning:<модуль>"
глушит предупреждение. Запуск: python week-03/lesson_09/warn_stacklevel.py
"""

import warnings
from collections.abc import Callable
from typing import Any

ANYIO_SRC = """
import warnings
def deprecated(level):
    warnings.warn("alias is deprecated", DeprecationWarning, stacklevel=level)
"""
STARLETTE_SRC = """
def use(level):
    deprecated(level)
"""


def module(name: str, src: str, **extra: Any) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": name, **extra}
    exec(compile(src, f"<{name}>", "exec"), namespace)
    return namespace


anyio_like = module("anyio_like", ANYIO_SRC)
starlette_like = module("starlette_like", STARLETTE_SRC, deprecated=anyio_like["deprecated"])
use: Callable[[int], None] = starlette_like["use"]


def swallowed_by(target: str, level: int) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=target)
        try:
            use(level)
        except DeprecationWarning:
            return False
        return True


if __name__ == "__main__":
    candidates = ["anyio_like", "starlette_like", "__main__"]
    print("stacklevel | какой фильтр module=... глушит")
    for level in (1, 2, 3):
        hits = [name for name in candidates if swallowed_by(name, level)]
        print(f"{level:>10} | {', '.join(hits)}")
