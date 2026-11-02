from collections.abc import Callable, Iterator
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

import ragkit
from api.main import create_app
from ragkit.config import LLMSettings, Settings


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
def settings_factory() -> Callable[..., Settings]:
    """Тестовые Settings; любое поле меняется аргументом: settings_factory(debug=False)."""

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
def app(settings: Settings) -> FastAPI:
    """Свежее приложение на каждый тест: своё app.state, свой лимитер, свои overrides."""
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """TestClient внутри with: lifespan отрабатывает до первого запроса и после последнего."""
    with TestClient(app) as c:  # with — чтобы отработал lifespan
        yield c
