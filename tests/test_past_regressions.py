"""Ф5.5-02, задание 1: проверки на прошлые поломки — в блокирующем потоке на каждый push.

Сами проверки — week-05.5/past_checks.py: тот же файл lesson_02.py запускает на старых
коммитах и показывает, что там они красные.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PATH = Path(__file__).resolve().parent.parent / "week-05.5" / "past_checks.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("past_checks", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", list(load().CHECKS))
def test_past_regression_stays_fixed(name: str) -> None:
    ok, detail = load().CHECKS[name]()
    assert ok, detail
