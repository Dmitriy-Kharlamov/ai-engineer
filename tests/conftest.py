import os
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

import ragkit
from api.main import create_app
from ragkit.config import LLMSettings, Settings
from services.clock import FakeClock

START = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)  # время в тестах стоит, пока его не сдвинут


# --- диагностика окружения (Ф3-09, задание 1) ------------------------------------
def environment_report() -> tuple[bool, str]:
    """Честная диагностика: установлен ли пакет и откуда на самом деле импортирован код."""
    code = Path(ragkit.__file__).parent
    try:
        dist = distribution("ragkit")
    except PackageNotFoundError:
        return False, (
            f"ragkit: метаданных пакета нет, код импортирован из {code}. "
            "/health отдаст ragkit_version='unknown'. Чтобы было как в проде: pip install -e ."
        )
    return True, f"ragkit {dist.version}: пакет установлен, код импортирован из {code}"


def pytest_report_header() -> str:
    return environment_report()[1]  # шапка; при -q не печатается


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    installed, text = environment_report()
    if not installed and terminalreporter.verbosity < 0:  # при -q шапки нет — пишем в итог
        terminalreporter.write_sep("-", "окружение")
        terminalreporter.write_line(text)


# --- фикстуры ------------------------------------------------------------------------
@pytest.fixture
def settings_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Тестовые Settings; любое поле меняется аргументом: settings_factory(debug=False).

    _env_file=None отключает только файл .env; переменные APP_* из окружения процесса
    Settings читает всё равно (замерено в Ф3-10: APP_JWT_TTL_MINUTES=60 ронял тест срока).
    Поэтому на время теста они убираются; monkeypatch вернёт их после.
    """
    for name in [n for n in os.environ if n.upper().startswith("APP_")]:
        monkeypatch.delenv(name)

    def make(**overrides: object) -> Settings:
        base: dict[str, object] = {
            "api_key": SecretStr("test-key"),
            "debug": True,
            "max_retries": 7,
            "llm": LLMSettings(model="test-model"),
            "jwt_secret": SecretStr("test-jwt-secret-at-least-32-bytes-long!!"),
            "bcrypt_rounds": 4,  # тестам не нужна стоимость прода: 4 раунда ≈ 1 мс
            "rate_limit_per_minute": 3,
            "_env_file": None,  # не читать настоящий .env
        }
        base.update(overrides)
        return Settings(**base)  # type: ignore[arg-type]

    return make


@pytest.fixture
def settings(settings_factory: Callable[..., Settings]) -> Settings:
    """Настройки по умолчанию; папка с тестами может переопределить их своим conftest.py."""
    return settings_factory()


@pytest.fixture
def clock() -> FakeClock:
    """Часы приложения: clock.advance(60) вместо time.sleep(60)."""
    return FakeClock(START)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> FastAPI:
    """Свежее приложение на каждый тест: своё app.state, свой лимитер, свои overrides."""
    return create_app(settings, clock=clock)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """TestClient внутри with: lifespan отрабатывает до первого запроса и после последнего."""
    with TestClient(app) as c:  # with — чтобы отработал lifespan
        yield c


@pytest.fixture
async def aclient(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Асинхронный клиент без сети и без сервера: запросы идут прямо в ASGI-приложение.

    ASGITransport lifespan не запускает (замерено: без этого app.state пуст и /health
    отвечает 500), поэтому входим в него сами.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
