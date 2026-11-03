"""Контрактные тесты против живого PyPI: правда ли ещё то, что записано в tests/fixtures/.

Ходят в сеть, поэтому помечены slow и в обычный прогон не входят (addopts: -m 'not slow').
Запуск: pytest -m slow
Стабы в test_http_retry.py — догадка о провайдере; эти тесты сверяют догадку с ним самим.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from services.pypi import PYPI_URL, PackageNotFoundOnPyPIError, fetch_package

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.slow


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


async def test_live_200_still_matches_recording() -> None:
    rec = fixture("pypi_respx_200.json")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(rec["url"])
    assert r.status_code == 200
    assert r.headers["content-type"] == rec["headers"]["content-type"]
    body = r.json()
    assert set(rec["top_level_keys"]) <= set(body)  # новые ключи можно, пропажа — нет
    for key, value in rec["json"]["info"].items():
        assert isinstance(body["info"][key], type(value)), key  # тип поля не поменялся


async def test_live_404_still_matches_recording() -> None:
    rec = fixture("pypi_missing_404.json")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(rec["url"])
    assert r.status_code == 404
    assert r.json() == rec["json"]


async def test_live_client_end_to_end() -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        info = await fetch_package(client, "httpx")
        assert info.name.lower() == "httpx" and info.version[0].isdigit()
        with pytest.raises(PackageNotFoundOnPyPIError):
            await fetch_package(client, "no-such-package-xyz-123")
    assert PYPI_URL.startswith("https://")
