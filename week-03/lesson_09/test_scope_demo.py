"""Задание 6: session-фикстура — одна на весь прогон, и поэтому общее состояние.

Лежит вне testpaths: обычный `pytest` её не собирает, запускать явно:
    pytest week-03/lesson_09/test_scope_demo.py -p randomly --randomly-seed=1 -s
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.main import create_app
from ragkit.config import Settings

CREATED = 0  # сколько раз фикстура строила приложение за процесс


@pytest.fixture(scope="session")
def shared_client() -> Iterator[TestClient]:
    """Одно приложение на весь прогон: «дорогая» подготовка делается один раз."""
    global CREATED
    CREATED += 1
    settings = Settings(
        api_key=SecretStr("k"),
        jwt_secret=SecretStr("demo-secret-at-least-32-bytes-long!!"),
        bcrypt_rounds=4,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as c:
        yield c
    print(f"\n[shared_client] за прогон создана {CREATED} раз")


# --- счётчик: три теста, и каждый видит одну и ту же фикстуру ---------------------
@pytest.mark.parametrize("n", [1, 2, 3])
def test_fixture_created_once(shared_client: TestClient, n: int) -> None:
    assert shared_client.get("/health").status_code == 200
    assert CREATED == 1  # со scope="function" второй и третий случаи покраснеют


# --- поломка: два теста меняют общее состояние -----------------------------------
USER = {"username": "alice", "password": "password-123"}


def test_register_alice(shared_client: TestClient) -> None:
    assert shared_client.post("/auth/register", json=USER).status_code == 201


def test_register_alice_again_conflicts(shared_client: TestClient) -> None:
    # Молча рассчитывает, что test_register_alice уже отработал на ЭТОМ ЖЕ приложении.
    assert shared_client.post("/auth/register", json=USER).status_code == 409
