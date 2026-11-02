"""Направление зависимостей: ragkit ↛ services/api, services ↛ api."""

from pathlib import Path

from tools.check_layers import find_violations

ROOT = Path(__file__).resolve().parent.parent


def test_repository_layers_are_clean() -> None:
    assert find_violations(ROOT) == []


def test_checker_catches_violations(tmp_path: Path) -> None:
    (tmp_path / "ragkit").mkdir()
    (tmp_path / "services").mkdir()
    (tmp_path / "ragkit" / "bad.py").write_text("from api.main import create_app\n")
    (tmp_path / "services" / "bad.py").write_text("import api.deps\n")
    (tmp_path / "services" / "ok.py").write_text("from ragkit import Chunk\nfrom . import db\n")
    assert find_violations(tmp_path) == [
        "ragkit/bad.py:1: ragkit не может импортировать api",
        "services/bad.py:1: services не может импортировать api",
    ]
