"""Проверка направления зависимостей: api → services → ragkit, никогда обратно.

Запуск: python -m tools.check_layers [корень]   (код возврата 1, если есть нарушения)
"""

import ast
import sys
from pathlib import Path

# пакет → какие пакеты ему импортировать ЗАПРЕЩЕНО
FORBIDDEN: dict[str, frozenset[str]] = {
    "ragkit": frozenset({"api", "services"}),
    "services": frozenset({"api"}),
}


def imported_top_packages(source: str) -> list[tuple[int, str]]:
    """(строка, верхний пакет) для каждого абсолютного импорта в файле."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name.split(".")[0]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module.split(".")[0]))
    return found


def find_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for package, forbidden in FORBIDDEN.items():
        for path in sorted((root / package).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, top in imported_top_packages(text):
                if top in forbidden:
                    rel = path.relative_to(root)
                    violations.append(f"{rel}:{lineno}: {package} не может импортировать {top}")
    return violations


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    violations = find_violations(root)
    for line in violations:
        print(line)
    print("слои в порядке" if not violations else f"нарушений: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
