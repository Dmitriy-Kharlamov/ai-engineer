"""Задание 8: может ли тест упасть. Одна точечная поломка за раз, полный прогон tests/.

Каждая поломка — замена ровно одного фрагмента в одном файле; после прогона файл
возвращается как был (finally), даже если прогон прервали.
Запуск из корня репозитория:
    python week-03/lesson_09/mutate.py       # поломки Ф3-09
    python week-03/lesson_09/mutate.py 10    # поломки Ф3-10: код, проверенный моком, и часы
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
    Mutation(  # Ф3-10: тест разделён на tampered и expires, decode получил options
        "test_jwt_tampered_is_invalid",
        "api/security.py",
        '"verify_exp": False, "verify_iat": False}',
        '"verify_exp": False, "verify_iat": False, "verify_signature": False}',
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
    Mutation(  # задание 11: ветка, найденная покрытием
        "test_login_failure_is_401_and_indistinguishable",
        "services/users.py",
        "return record.user if check_password(password, record.password_hash) else None",
        "return record.user",
        "authenticate не проверяет пароль",
    ),
]


MUTATIONS_10 = [
    Mutation(  # задание 2: настоящая реализация, которую проверяет только мок
        "test_embed_route_with_mock",
        "ragkit/embedding.py",
        'digest = hashlib.sha256(text.encode("utf-8")).digest()',
        'digest = hashlib.sha256(b"").digest()',
        "HashEmbedder игнорирует текст: у всех текстов один вектор",
    ),
    Mutation(
        "test_embed_route_with_mock",
        "ragkit/embedding.py",
        "vectors.append([x / norm for x in raw])",
        "vectors.append(raw)",
        "HashEmbedder не нормирует векторы",
    ),
    Mutation(  # задание 7: граница срока токена по внедрённым часам
        "test_jwt_expires_exactly_at_ttl",
        "api/security.py",
        "if exp <= moment:",
        "if exp < moment:",
        "токен живёт на секунду дольше ttl (<= -> <)",
    ),
    Mutation(
        "test_rate_limit_429",
        "services/ratelimit.py",
        "while hits and hits[0] <= now - self.window:",
        "while hits and hits[0] < now - self.window:",
        "запрос не выходит из окна ровно через 60 с (<= -> <)",
    ),
    Mutation(  # задание 6: retry
        "test_retry_after_seconds_is_respected",
        "services/http_retry.py",
        "if p.respect_retry_after and response is not None:",
        "if False:",
        "Retry-After игнорируется",
    ),
    Mutation(
        "test_404_is_final_not_retried",
        "services/http_retry.py",
        "RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})",
        "RETRY_STATUSES = frozenset({404, 408, 425, 429, 500, 502, 503, 504})",
        "404 повторяется",
    ),
]

SETS = {"9": MUTATIONS, "10": MUTATIONS_10}


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
    mutations = SETS[sys.argv[1] if len(sys.argv) > 1 else "9"]
    assert failed_tests() == [], "до поломок набор должен быть зелёным"
    print("| тест | поломка | результат | покраснели |")
    print("|---|---|---|---|")
    for m in mutations:
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
