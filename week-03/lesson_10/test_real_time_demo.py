"""Задание 7, «до»: окно лимитера на настоящих часах — тест обязан ждать 60 секунд.

Вне testpaths, запуск: pytest week-03/lesson_10/test_real_time_demo.py -p no:randomly --durations=1
В tests/test_security.py::test_rate_limit_429 то же самое делает clock.advance(60).
"""

import time

from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.main import create_app
from ragkit.config import Settings
from services.apikeys import ApiKeyStore


def test_window_slides_on_real_clock() -> None:
    settings = Settings(
        api_key=SecretStr("k"),
        rate_limit_per_minute=3,
        jwt_secret=SecretStr("lesson-jwt-secret-at-least-32-bytes!!"),
        _env_file=None,
    )
    app = create_app(settings)  # clock не передан — SystemClock, настоящее время
    with TestClient(app) as c:
        keys: ApiKeyStore = app.state.api_keys
        headers = {"X-API-Key": keys.issue("dave", frozenset())}
        codes = [c.get("/me/key", headers=headers).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        time.sleep(60)  # ждать, пока первый запрос выйдет из окна
        assert c.get("/me/key", headers=headers).status_code == 200
