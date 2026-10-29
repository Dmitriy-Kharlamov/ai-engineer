"""Ф3-06. FastAPI: эндпоинты и роутеры — прогон всех заданий.

Запуск: python week-03/lesson_06.py
Задание 3 поднимает настоящий uvicorn в фоновом потоке (TestClient последователен
и конкурентность не покажет); остальное идёт через TestClient.
Сервис целиком: uvicorn api.main:app --reload  →  http://127.0.0.1:8000/docs
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
import uvicorn
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api import deps
from api.main import app
from ragkit.config import LLMSettings, Settings, get_settings


class ResponseLike(Protocol):
    """И httpx.Response, и ответ TestClient (в starlette 1.x это отдельный класс)."""

    @property
    def status_code(self) -> int: ...

    def json(self) -> Any: ...


def show(title: str, r: ResponseLike) -> Any:
    body = r.json()
    print(f"   {title}: {r.status_code} {json.dumps(body, ensure_ascii=False)}")
    return body


def test_settings() -> Settings:
    return Settings(
        api_key=SecretStr("test-key"),
        debug=True,
        max_retries=7,
        llm=LLMSettings(model="подменённая-модель"),
        _env_file=None,
    )


# ===========================================================================
# 1. Схема: куда попала аннотация возврата
# ===========================================================================
def task1(c: TestClient) -> None:
    print("\n1. /health и /openapi.json")
    show("GET /health", c.get("/health"))
    schema = c.get("/openapi.json").json()
    ok = schema["paths"]["/health"]["get"]["responses"]["200"]
    print("   paths./health.get.responses.200 =", json.dumps(ok, ensure_ascii=False))
    health_out = schema["components"]["schemas"]["HealthOut"]
    print("   components.schemas.HealthOut.required =", health_out["required"])
    print(
        "   components.schemas.HealthOut.properties.started_at =",
        json.dumps(health_out["properties"]["started_at"], ensure_ascii=False),
    )


# Аннотация `-> HealthOut` попала в ДВА места схемы: в ответ 200 эндпоинта —
#   ссылкой {"$ref": "#/components/schemas/HealthOut"} — и в components.schemas, где
#   лежит сама модель: поля, типы (datetime → "format": "date-time"), обязательность и
#   description из Field(...). /docs рисует страницу ровно из этого JSON.


# ===========================================================================
# 2. Откуда берётся параметр
# ===========================================================================
def task2(c: TestClient) -> None:
    print("\n2. path / query / header / body")
    # Прогноз (записан до запуска): doc_id — path (есть в {}), limit — query (скаляр),
    # x_request_id — header (Header()), body — тело (pydantic-модель).
    show(
        "POST /lab/params/7?limit=5 + X-Request-Id + тело",
        c.post("/lab/params/7?limit=5", headers={"X-Request-Id": "abc-1"}, json={"note": "hi"}),
    )
    show("то же без limit и заголовка", c.post("/lab/params/7", json={"note": "hi"}))
    r = show("doc_id=0 (Path(ge=1))", c.post("/lab/params/0", json={"note": "hi"}))
    assert r["detail"][0]["loc"] == ["path", "doc_id"]
    show("скаляр с именем как в пути: /same-name/5?item_id=9", c.get("/lab/same-name/5?item_id=9"))
    try:
        from fastapi import FastAPI

        from api.schemas import ParamsBody

        probe = FastAPI()

        @probe.get("/b/{item}")
        def b(item: ParamsBody) -> ParamsBody:
            return item
    except AssertionError as e:
        print(
            f"   pydantic-модель с именем как в пути: при объявлении маршрута {type(e).__name__}: {e}"
        )


# Прогноз совпал: doc_id из пути, limit из строки запроса (без него — умолчание 10),
#   x_request_id из заголовка X-Request-Id (подчёркивание → дефис), body из JSON-тела.
#   Ошибка у doc_id=0 указывает источник: loc = ["path", "doc_id"].
# Скаляр, названный как {item_id} в пути, берётся ТОЛЬКО из пути: ?item_id=9 проигнорирован.
# Pydantic-модель, названная как часть пути, — ошибка уже при объявлении маршрута, сервис
#   даже не стартует: путь — это строка, а модель — объект; FastAPI это запрещает заранее.
#   Модель без такого имени в пути (как body выше) — всегда тело.


# ===========================================================================
# 3. def против async def — настоящий сервер
# ===========================================================================
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


async def five_at_once(base: str, path: str) -> tuple[float, int]:
    async with httpx.AsyncClient(base_url=base, timeout=30, trust_env=False) as client:
        start = time.perf_counter()
        responses = await asyncio.gather(*[client.get(path) for _ in range(5)])
        elapsed = time.perf_counter() - start
    threads = {r.json()["thread"] for r in responses}
    return elapsed, len(threads)


def task3() -> None:
    print("\n3. def против async def: 5 одновременных запросов по 1 с (uvicorn)")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    results = {}
    for name, path in [
        ("def + time.sleep", "/lab/sleep/sync-def"),
        ("async def + time.sleep", "/lab/sleep/async-blocking"),
        ("починка 1: asyncio.sleep", "/lab/sleep/async-sleep"),
        ("починка 2: asyncio.to_thread", "/lab/sleep/async-to-thread"),
        ("починка 3: просто def", "/lab/sleep/fixed-def"),
    ]:
        elapsed, threads = asyncio.run(five_at_once(base, path))
        results[name] = elapsed
        print(f"   {name:30}: {elapsed:5.2f} с | разных потоков: {threads}")
    server.should_exit = True
    thread.join()
    assert results["async def + time.sleep"] > 4.5
    assert all(t < 1.5 for n, t in results.items() if n != "async def + time.sleep")


# def + time.sleep: ~1 с и 5 потоков — FastAPI запускает обычный def в пуле потоков, пять
#   запросов спят одновременно в пяти разных потоках.
# async def + time.sleep: ~5 с и 1 поток — async def исполняется прямо в цикле событий,
#   time.sleep замораживает его целиком, запросы обрабатываются по очереди.
# asyncio.sleep: ~1 с, 1 поток — сон отдаёт управление циклу, все пять ждут вместе
#   в ОДНОМ потоке; лучший вариант, когда неблокирующий аналог существует.
# asyncio.to_thread: ~1 с, 1 поток в ответе — обработчик живёт в потоке цикла, а сон уехал
#   в рабочие потоки (их ident в ответ не попадает).
# просто def: ~1 с, 5 потоков — то же, что первый вариант.
# Для синхронного драйвера базы выберу обычный def: весь обработчик целиком уходит в поток,
#   и невозможно забыть обернуть ОДИН из нескольких вызовов в to_thread — а один забытый
#   вызов в async def снова замораживает весь сервис. to_thread — для случая, когда в
#   async-обработчике один блокирующий вызов среди неблокирующих.


# ===========================================================================
# 4. Утечка в 422
# ===========================================================================
def task4() -> None:
    print("\n4. Пароль в ответе 422")
    from fastapi import FastAPI
    from pydantic import BaseModel, ConfigDict, Field

    class LoginSafe(BaseModel):
        model_config = ConfigDict(hide_input_in_errors=True)
        password: SecretStr = Field(min_length=12)

    raw = FastAPI()  # без нашего обработчика — как FastAPI ведёт себя по умолчанию

    @raw.post("/login")
    def login(body: LoginSafe) -> dict[str, bool]:
        return {"ok": True}

    r = TestClient(raw).post("/login", json={"password": "secret123"})
    print(
        f"   без обработчика, hide_input_in_errors=True: пароль в ответе: {'secret123' in r.text}"
    )
    print(f"   {r.text}")
    assert "secret123" in r.text

    with TestClient(app) as c:
        r = c.post("/lab/login", json={"password": "secret123"})
    print(f"   с обработчиком RequestValidationError: пароль в ответе: {'secret123' in r.text}")
    print(f"   {r.text}")
    assert "secret123" not in r.text and r.json()["detail"][0]["loc"] == ["body", "password"]


# hide_input_in_errors не помог: эта настройка меняет только str(e) — текст ошибки. FastAPI
#   строит ответ 422 не из текста, а из структурного exc.errors(), а там поле input с
#   паролем остаётся. Это ровно то, что я написал в Ф3-04 про e.errors(): маскирование
#   стояло не на том выходе.
# Обработчик вырезает только input. loc ["body", "password"], type too_short, msg и ctx
#   с min_length=12 и actual_length=9 остались — клиент точно знает, какое поле и почему
#   не прошло, но своего пароля обратно не получает.


# ===========================================================================
# 5. Три модели
# ===========================================================================
def task5() -> None:
    print("\n5. Три модели: UserIn → UserDB → UserOut")
    user = {"name": "dmitriy", "password": "my-password"}
    with TestClient(app, raise_server_exceptions=False) as c:
        show("-> UserDB (аннотация = схема ответа)", c.post("/lab/users-leak", json=user))
        show("response_model=UserOut", c.post("/lab/users", json=user))
        r = c.post("/lab/users-missing", json=user)
        print(f"   response_model с полем email, которого нет: {r.status_code} {r.text}")
    assert r.status_code == 500


# -> UserDB отдаёт наружу всю модель базы: хеш пароля и внутренний id. response_model=UserOut
#   пропускает объект через UserOut, и лишние поля отбрасываются — это фильтр.
# Если в модели ответа есть обязательное поле, которого в объекте нет, FastAPI бросает
#   ResponseValidationError и клиент получает 500 Internal Server Error: это ошибка сервера
#   (мы нарушили собственный контракт), а не клиента. Если у поля есть умолчание —
#   подставится умолчание.


# ===========================================================================
# 6. POST /chunks и опечатка в Literal
# ===========================================================================
def task6(c: TestClient) -> None:
    print("\n6. POST /chunks")
    body = {
        "text": "Первое предложение. Второе предложение тут.",
        "source": "a.pdf",
        "max_tokens": 3,
        "mode": "sentences",
    }
    show("mode=sentences", c.post("/chunks", json=body))
    r = show("mode=sentense (опечатка)", c.post("/chunks", json=body | {"mode": "sentense"}))
    assert r["detail"][0]["type"] == "literal_error"


# Опечатка дала 422 literal_error с текстом «Input should be 'words' or 'sentences'»,
#   loc ["body", "mode"], — и в коде эндпоинта нет ни одной проверки mode. FastAPI взял
#   SplitMode = Literal[...] из ragkit (тот же тип, что mypy проверял в Ф3-05) и построил
#   по нему валидацию на границе сервиса. Аннотация окупилась второй раз.


# ===========================================================================
# 7. Роутеры
# ===========================================================================
def task7(c: TestClient) -> None:
    print("\n7. Роутеры: группы в /docs")
    schema = c.get("/openapi.json").json()
    groups: dict[str, list[str]] = {}
    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            groups.setdefault(op["tags"][0], []).append(f"{method.upper()} {path}")
    for tag, ops in groups.items():
        print(f"   [{tag}] " + ", ".join(ops))


# Каждый роутер объявляет prefix и tags один раз; /docs группирует эндпоинты по tags —
#   три раздела health, chunks, lab, как в выводе.


# ===========================================================================
# 8. Depends и подмена
# ===========================================================================
def task8() -> None:
    print("\n8. Depends(get_settings) и dependency_overrides")
    app.dependency_overrides[get_settings] = test_settings
    try:
        with TestClient(app) as c:
            show("GET /health/config с подменой", c.get("/health/config"))
            assert c.get("/health/config").json()["llm_model"] == "подменённая-модель"
    finally:
        app.dependency_overrides.clear()


# Обработчик ничего не знает о подмене, но вернул «подменённая-модель» и max_retries=7 —
#   значит, получил именно тестовые настройки. Подмена касается одного объекта app и
#   снимается одной строкой; cache_clear() плюс переменные окружения меняют глобальное
#   состояние всего процесса и протекают в соседние тесты, если их забыть вернуть.


# ===========================================================================
# 9. Lifespan: эмбеддер создаётся один раз
# ===========================================================================
def task9() -> None:
    print("\n9. Lifespan: 10 запросов к /chunks/embed")
    before = deps.EMBEDDER_LOADS
    with TestClient(app) as c:
        for _ in range(10):
            c.post("/chunks/embed", json={"texts": ["привет", "мир"]})
        loads = deps.EMBEDDER_LOADS - before
        print(f"   запросов: 10, создано эмбеддеров за этот запуск сервиса: {loads}")
        health = show("GET /health после 10 запросов", c.get("/health"))
    assert loads == 1 and health["embedder_loads"] == 1


# load_embedder вызывается в lifespan, до первого запроса, и кладёт объект в app.state.
#   Эндпоинты берут его через Depends(get_embedder) — 10 запросов, одно создание; /health
#   показывает то же самое изнутри сервиса. Счётчик считается «за этот запуск»: в скрипте
#   сервис стартует несколько раз (каждый `with TestClient(app)` и uvicorn в задании 3 —
#   это отдельный запуск lifespan), и в каждом запуске эмбеддер создаётся ровно один раз.


# ===========================================================================
# 10. Исключения ragkit → коды HTTP
# ===========================================================================
def task10(c: TestClient) -> None:
    print("\n10. Исключения ragkit → HTTP")
    show(
        "чанк больше лимита → ChunkTooLargeError",
        c.post(
            "/chunks/documents", json={"name": "doc", "max_chunk_tokens": 2, "texts": ["a b c d"]}
        ),
    )
    show(
        "пустой документ → EmptyIndexError",
        c.post("/chunks/documents", json={"name": "doc", "texts": []}),
    )
    show(
        "нормальный документ",
        c.post("/chunks/documents", json={"name": "doc", "texts": ["a b", "c d e"]}),
    )


# ChunkTooLargeError → 413 Content Too Large (часть запроса больше допустимого),
#   EmptyIndexError → 422 (запрос корректен по форме, но обработать нечего), общий RAGError →
#   400. Перевод живёт в api/errors.py: ragkit поднимает свои исключения и не знает, что
#   такое HTTP, — его можно вызвать из CLI, из ноутбука, из воркера очереди, где кодов
#   ответа нет вовсе. Иначе библиотека стала бы пригодна только внутри веб-сервиса.


# ===========================================================================
# 11. Устно, потом письменно
# ===========================================================================
# Почему async def с блокирующим вызовом хуже обычного def: обычный def FastAPI сам
#   запускает в пуле потоков, и блокировка занимает только свой поток. async def
#   исполняется прямо в цикле событий, и блокирующий вызов останавливает весь цикл — все
#   запросы встают в очередь (задание 3: 5 с вместо 1).
# Откуда FastAPI знает, что проверять: из аннотаций функции. Имя есть в пути → path,
#   pydantic-модель → тело, остальной скаляр → query; Annotated[..., Query(ge=1)] и
#   Field(...) добавляют ограничения, Literal — допустимые значения. Те же аннотации
#   становятся схемой /openapi.json.
# Почему аннотация возврата — фильтр, а не описание: FastAPI пропускает возвращаемый объект
#   через модель ответа и отдаёт только её поля. Написал -> UserDB — отдал всю модель базы
#   с хешем пароля; response_model=UserOut — только имя (задание 5).
# Почему hide_input_in_errors не закрыл 422: он меняет только текстовое представление
#   str(e). FastAPI строит ответ из структурного exc.errors(), где input остаётся.
#   Закрывает свой обработчик RequestValidationError (задание 4).
# Зачем Depends, если можно вызвать функцию: зависимость объявлена, а не зашита в код, —
#   поэтому её можно подменить через dependency_overrides без изменения обработчика
#   (задание 8), FastAPI кеширует её в пределах запроса и показывает в схеме. Прямой
#   вызов get_settings() внутри обработчика подменить можно только глобально.


def main() -> None:
    steps: list[Callable[[TestClient], None]] = [task1, task2]
    app.dependency_overrides[get_settings] = test_settings
    with TestClient(app) as c:
        for step in steps:
            step(c)
    app.dependency_overrides.clear()
    task3()
    task4()
    task5()
    with TestClient(app) as c:
        task6(c)
        task7(c)
    task8()
    task9()
    with TestClient(app) as c:
        task10(c)


if __name__ == "__main__":
    main()
