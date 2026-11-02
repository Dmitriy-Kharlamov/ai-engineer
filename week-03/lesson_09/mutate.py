"""Задание 8: может ли тест упасть. Одна точечная поломка за раз, полный прогон tests/.

Каждая поломка — замена ровно одного фрагмента в одном файле; после прогона файл
возвращается как был (finally), даже если прогон прервали.
Запуск из корня репозитория: python week-03/lesson_09/mutate.py
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mutation:
    test: str  # какой тест должен покраснеть
    path: str
    old: str
    new: str
    what: str  # поломка человеческими словами


MUTATIONS = [
    Mutation(
        "test_unknown_ctx_key_is_dropped",
        "api/errors.py",
        " if k in SAFE_CTX_KEYS}",
        "}",
        "safe_error пропускает любой ключ ctx",
    ),
    Mutation(
        "test_foreign_document_is_404",
        "services/documents.py",
        "if doc is None or doc.owner != owner:",
        "if doc is None:",
        "get_owned не сверяет владельца",
    ),
    Mutation(
        "test_rate_limit_429",
        "services/ratelimit.py",
        "if len(hits) >= self.limit:",
        "if len(hits) > self.limit:",
        "лимитер пропускает limit+1 запрос (>= -> >)",
    ),
    Mutation(
        "test_admin_router_protects_new_endpoint",
        "api/routers/admin.py",
        "dependencies=[Depends(require_admin)]",
        "dependencies=[]",
        "у роутера /admin нет защиты",
    ),
    Mutation(
        "test_jwt_expired_and_tampered",
        "api/security.py",
        "jwt.decode(token, secret, algorithms=[ALGORITHM])",
        'jwt.decode(token, secret, algorithms=[ALGORITHM], options={"verify_signature": False})',
        "подпись JWT не проверяется",
    ),
    Mutation(
        "test_health",
        "api/routers/health.py",
        'return version("ragkit")',
        "return UNKNOWN_VERSION",
        '/health всегда отдаёт "unknown"',
    ),
    Mutation(
        "test_embedder_loaded_once",
        "api/deps.py",
        "embedder: Embedder = request.app.state.embedder",
        "embedder: Embedder = load_embedder()",
        "эмбеддер грузится на каждый запрос",
    ),
]


def failed_tests() -> list[str]:
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rf",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("FAILED ")]
    return [line.split()[1].split("::")[1].split(" ")[0] for line in lines]


def main() -> None:
    assert failed_tests() == [], "до поломок набор должен быть зелёным"
    print("| тест | поломка | результат | покраснели |")
    print("|---|---|---|---|")
    for m in MUTATIONS:
        path = Path(m.path)
        original = path.read_text()
        assert original.count(m.old) == 1, f"{m.path}: фрагмент не найден или не уникален"
        try:
            path.write_text(original.replace(m.old, m.new))
            failed = failed_tests()
        finally:
            path.write_text(original)
        red = any(name.startswith(m.test) for name in failed)
        verdict = "КРАСНЫЙ" if red else "ЗЕЛЁНЫЙ — тест не ловит"
        others = ", ".join(sorted(set(failed))) or "—"
        print(f"| {m.test} | {m.what} | {verdict} | {others} |")


if __name__ == "__main__":
    main()
