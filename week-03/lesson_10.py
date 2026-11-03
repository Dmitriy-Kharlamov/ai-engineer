"""Ф3-10. pytest: моки и тестирование API.

Запуск из корня репозитория: python week-03/lesson_10.py
Тесты урока — в tests/ (обычный прогон: pytest). Демонстрации, которые должны падать,
— в week-03/lesson_10/ (вне testpaths). Задания 2, 6, 7, 9, 10 — результаты прогонов
pytest и mutate.py в комментариях; команда для повтора указана рядом.
"""

import asyncio
import inspect
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, create_autospec, patch

import httpx

from ragkit import HashEmbedder
from services.pypi import fetch_package

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "lesson_10"))  # пакет clockdemo


def attempt(label: str, action: Callable[[], object]) -> None:
    try:
        result = action()
    except Exception as e:
        print(f"   {label:<38}: {type(e).__name__}: {e}")
    else:
        print(f"   {label:<38}: прошло -> {result!r}"[:110])


# ===========================================================================
# 1. Мок соглашается на всё
# ===========================================================================
def task1() -> None:
    print("1. Mock() вместо HashEmbedder")
    m: Any = Mock()
    attempt("m.embedd(['a'])  — опечатка", lambda: m.embedd(["a"]))
    attempt("m.embed()  — без texts", lambda: m.embed())
    attempt(
        "m.embed('строка', 1, 2, 3, лишний=True)", lambda: m.embed("строка", 1, 2, 3, лишний=True)
    )
    attempt("m.совершенно_новый_метод().и_ещё()", lambda: m.совершенно_новый_метод().и_ещё())
    attempt(
        "assert_any_call с мусорной подписью",
        lambda: m.embed.assert_any_call("строка", 1, 2, 3, лишний=True),
    )
    real = HashEmbedder(dim=2)
    attempt("тот же вызов на HashEmbedder", lambda: real.embed("строка", 1, 2, 3, лишний=True))  # type: ignore[call-arg]
    print(
        "   тест, зелёный на моке и красный на настоящем (week-03/lesson_10/test_mock_vs_real.py):"
    )
    run_pytest("week-03/lesson_10/test_mock_vs_real.py", "-k", "embed_titles")


def run_pytest(*args: str) -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *args,
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-rA",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    ).stdout
    for line in out.splitlines():
        if line.startswith(("PASSED", "FAILED", "E       assert")) or " passed" in line:
            print("      " + line[:100])


# Все пять строк «прошло»: Mock создаёт атрибут на любое имя и принимает любой вызов, а
#   assert_any_call проверяет только то, что такой вызов был записан. Настоящий embed на
#   той же подписи — TypeError (unexpected keyword argument 'лишний').
#
# Тест test_embed_titles[mock] зелёный, [real] красный. В embed_titles ошибка: embed
#   получает строку вместо списка. Мок зелёный, потому что возвращает один и тот же
#   заранее заданный ответ на любой аргумент, и сравнение «вектор заголовка == embed(
#   ['привет'])[0]» сравнивает этот ответ сам с собой. Настоящий HashEmbedder перебирает
#   строку по буквам и отдаёт вектор буквы «п», поэтому сравнение падает — TypeError тут
#   не возникает вовсе, ошибка тихая, и поймать её может только настоящий объект. mypy
#   тоже молчит: str — это Sequence[str], подпись формально соблюдена.


# ===========================================================================
# 2. create_autospec
# ===========================================================================
def task2() -> None:
    print("\n2. create_autospec(HashEmbedder, instance=True)")
    a: Any = create_autospec(HashEmbedder, instance=True)
    attempt("опечатка в имени", lambda: a.embedd(["a"]))
    attempt("без аргументов", lambda: a.embed())
    attempt("лишние аргументы", lambda: a.embed(["a"], 1, 2))
    attempt("лишний именованный", lambda: a.embed(["a"], лишний=True))
    attempt("правильный вызов", lambda: a.embed(["a"]))


# Опечатка — AttributeError, пропуск texts — TypeError: missing a required argument,
#   лишние позиционные — TypeError: too many positional arguments, лишний именованный —
#   unexpected keyword argument. Подпись закрыта. Но ответ autospec не знает: правильный
#   вызов вернул MagicMock, а не векторы. Отсюда главный результат урока:
#
# python week-03/lesson_09/mutate.py 10, тест test_embed_route_with_mock (create_autospec,
#   коммит 55c111c — до задания 10):
#   | test_embed_route_with_mock | HashEmbedder игнорирует текст: у всех текстов один вектор | ЗЕЛЁНЫЙ — тест не ловит | — |
#   | test_embed_route_with_mock | HashEmbedder не нормирует векторы                         | ЗЕЛЁНЫЙ — тест не ловит | — |
#   | test_embed_route_with_mock | маршрут эмбеддит только первый текст                      | ЗЕЛЁНЫЙ — тест не ловит | — |
#   Все 3 поломки из 3 зелёные, и не только этот тест — во всём наборе из 41 теста не
#   покраснел ни один. Связь с Ф3-09: там test_health и test_embedder_loaded_once были
#   зелёными из-за слабых утверждений, и лечилось это утверждением. Здесь утверждения
#   точные (тело ответа целиком, assert_called_once_with), но настоящей реализации в
#   тесте нет — проверять нечего. Autospec закрывает подпись, но не поведение: мок
#   проверяет, что вызов был, а не то, что из него вышло.


# ===========================================================================
# 3. MagicMock против Mock
# ===========================================================================
def task3() -> None:
    print("\n3. Ответ embed([...]) у Mock и MagicMock")

    def probe(v: Any) -> None:
        attempt("len(v)", lambda: len(v))
        attempt("v[0]", lambda: v[0])
        attempt("'x' in v", lambda: "x" in v)
        attempt("list(v)", lambda: list(v))
        attempt("v == [[1.0]]", lambda: v == [[1.0]])

    for name, m in (("Mock", Mock()), ("MagicMock", MagicMock())):
        print(f"   {name}:")
        probe(m.embed(["a"]))
    with patch.object(HashEmbedder, "embed") as patched:
        print(f"   patch.object(HashEmbedder, 'embed') подставил: {type(patched).__name__}")
    print("   тест, зелёный на MagicMock и красный на настоящем:")
    run_pytest("week-03/lesson_10/test_mock_vs_real.py", "-k", "unique")


# Mock: len, индексация, in и перебор — TypeError (протоколов у Mock нет); == даёт False
#   (сравнение по идентичности). MagicMock: len 0, v[0] — снова MagicMock, 'x' in v —
#   False, list(v) — [], == — False. Каждая строка «прошла» и вернула правдоподобное
#   «пусто». Тест test_embed_unique_has_no_duplicates[patch-magicmock] зелёный: множество
#   из пустого перебора длины 0 равно len 0. С настоящим HashEmbedder он красный: 3
#   вектора, уникальных 2 — ошибка «дубликаты не убраны» найдена.
#
# patch по умолчанию создаёт MagicMock, потому что подменяет что угодно — функцию, класс,
#   файл в with, итерируемый объект, — и должен «подходить» на месте любого из них:
#   with, for, len, [] не должны падать на самой подмене. Опасность в том же: подмена
#   удовлетворяет протоколам, не выполняя их смысла, и проверки вида «пусто», «нет
#   повторов», all(...) по пустому перебору проходят вхолостую. Лечится autospec=True:
#   patch.object(HashEmbedder, "embed", autospec=True) знает подпись настоящего метода.


# ===========================================================================
# 4. patch там, где имя используется
# ===========================================================================
def task4() -> None:
    import clockdemo.service as early
    import clockdemo.service_late as late

    print("\n4. patch и момент привязки имени")
    with patch("clockdemo.clock.now", return_value="ПОДМЕНА"):
        print(f"   ранняя привязка, patch('clockdemo.clock.now')         : {early.stamp()}")
    with patch("clockdemo.service.now", return_value="ПОДМЕНА"):
        print(f"   ранняя привязка, patch('clockdemo.service.now')       : {early.stamp()}")
    with patch("clockdemo.clock.now", return_value="ПОДМЕНА"):
        print(f"   поздняя привязка, patch('clockdemo.clock.now')        : {late.stamp()}")
    with patch("clockdemo.service_late.clock.now", return_value="ПОДМЕНА"):
        print(f"   поздняя привязка, patch('clockdemo.service_late.clock.now'): {late.stamp()}")
    try:
        with patch("clockdemo.service_late.now", return_value="ПОДМЕНА"):
            pass
    except AttributeError as e:
        print(f"   поздняя привязка, patch('clockdemo.service_late.now') : AttributeError: {e}")


# `from clockdemo.clock import now` выполняется один раз, при импорте service, и кладёт в
#   словарь модуля service ссылку на функцию. patch("clockdemo.clock.now") заменяет атрибут
#   модуля clock, а service продолжает держать старую ссылку — первая строка показывает
#   настоящее время. patch по месту использования ("clockdemo.service.now") меняет именно
#   ту ссылку, которую читает stamp, — «ПОДМЕНА».
#
# service_late импортирует модуль: `clock.now()` ищет атрибут now в модуле clock в момент
#   вызова. Поэтому подмена у источника работает, а "clockdemo.service_late.clock.now" —
#   это тот же самый атрибут того же объекта-модуля, второй путь к нему. Третий вариант,
#   "clockdemo.service_late.now", теперь падает AttributeError: такого имени в модуле нет,
#   и patch говорит об этом сразу, а не подменяет молча мимо.
#
# Заранее так стоит импортировать то, что подменять будут наверняка: часы, случайность,
#   клиенты к чужим API, отправку писем. В нашем коде я пошёл дальше и для часов patch
#   не нужен вовсе: services/clock.py, Clock внедряется в create_app (задание 7).


# ===========================================================================
# 5. Три стороны дрейфа
# ===========================================================================
class RealEmbedder:
    """Настоящий объект после изменения: появился параметр normalize."""

    def embed(self, texts: list[str], *, normalize: bool = True) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class StaleFake:
    """Фейк, написанный до изменения."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def task5() -> None:
    print("\n5. Код уже вызывает embed(texts, normalize=True)")

    def call(double: Any) -> Callable[[], object]:
        return lambda: double.embed(["a"], normalize=True)

    for name, double in (("настоящий", RealEmbedder()), ("фейк", StaleFake()), ("мок", Mock())):
        attempt(name, call(double))
    print("   дрейф протокола Embedder в копии репозитория, mypy --strict:")
    drift_mypy()


DRIFT = [
    (
        "ragkit/embedding.py",
        "    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...",
        "    def embed(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]: ...",
    ),
    (
        "ragkit/embedding.py",
        "    vectors = embedder.embed([chunk.text for chunk in chunks])",
        "    vectors = embedder.embed([chunk.text for chunk in chunks], normalize=True)",
    ),
]


def drift_mypy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", "HEAD"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        for path, old, new in DRIFT:
            file = Path(tmp) / path
            text = file.read_text()
            assert text.count(old) == 1, old
            file.write_text(text.replace(old, new))
        start = time.perf_counter()
        out = subprocess.run(
            [sys.executable, "-m", "mypy"], cwd=tmp, capture_output=True, text=True, check=False
        ).stdout
        took = time.perf_counter() - start
        shutil.rmtree(Path(tmp) / ".mypy_cache", ignore_errors=True)
    for line in out.splitlines():
        if ": error:" in line or line.startswith("Found"):
            print("      " + line[:150])
    print(f"      mypy отработал за {took:.1f} с, ни один тест не запускался")


# Таблица на одном вызове embed(["a"], normalize=True):
#   настоящий — [[1.0, 0.0, 0.0]]; фейк — TypeError: unexpected keyword argument
#   'normalize'; мок — прошло, вернул <Mock name='mock.embed()'>.
#
# Дрейф в настоящем коде: протоколу Embedder добавлен normalize, embed_chunks начал им
#   пользоваться, HashEmbedder не тронут. mypy --strict нашёл несовпадение во всех
#   местах, где HashEmbedder подставляется вместо Embedder (вывод выше: load_embedder в
#   api/deps.py, tests/test_api.py, week-03/lesson_05.py и демо урока — 5 ошибок в 4
#   файлах) до запуска тестов, за время работы mypy (8–9 с, почти всё — холодный кеш). Фейк
#   связан с протоколом структурно: mypy сверяет подписи каждый раз.
#
# Мок такой защиты не имеет в принципе: для mypy Mock() — это Any, а с Any совместимо
#   всё, сверять нечего. create_autospec(Embedder) читает подпись в момент создания, во
#   время теста — значит, повторяет протокол уже после изменения и падать не будет: его
#   «эталон» — сам протокол, а не реализация. Он защищает от опечатки в вызове, но не от
#   расхождения протокола и настоящего класса.


# ===========================================================================
# 8. AsyncMock против Mock
# ===========================================================================
def task8() -> None:
    print("\n8. Подмена корутины fetch_package")

    async def use(fake: Any) -> object:
        return await fake(None, "respx")

    attempt("Mock()", lambda: asyncio.run(use(Mock(return_value="0.23.1"))))
    attempt("AsyncMock()", lambda: asyncio.run(use(AsyncMock(return_value="0.23.1"))))
    spec: Any = create_autospec(fetch_package, return_value="0.23.1")
    print(
        f"   create_autospec(fetch_package): iscoroutinefunction={inspect.iscoroutinefunction(spec)}"
    )
    attempt("create_autospec(fetch_package)", lambda: asyncio.run(use(spec)))
    attempt("он же с неверной подписью", lambda: asyncio.run(spec(None)))
    print("   ASGITransport без входа в lifespan:")

    async def no_lifespan() -> int:
        from pydantic import SecretStr

        from api.main import create_app
        from ragkit.config import Settings

        app = create_app(Settings(api_key=SecretStr("k"), _env_file=None))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return (await c.get("/health")).status_code

    attempt("GET /health", lambda: asyncio.run(no_lifespan()))


# Mock: TypeError: object str can't be used in 'await' expression — вызов вернул
#   готовое значение, не корутину. AsyncMock: вызов возвращает корутину, await отдаёт
#   '0.23.1'. create_autospec async-функции сам делает AsyncMock (iscoroutinefunction=True)
#   и вдобавок проверяет подпись: вызов без name — TypeError: missing a required argument.
#
# ASGITransport не запускает lifespan: app.state пуст, и /health падает AttributeError
#   ('State' object has no attribute 'started_at'). Поэтому фикстура aclient в
#   tests/conftest.py входит в app.router.lifespan_context сама.
#
# Что дали async-тесты сверх TestClient (tests/test_security.py): несколько запросов в
#   полёте одновременно через asyncio.gather. test_concurrent_registration_creates_one_user
#   шлёт 10 одновременных POST /auth/register одного имени и сразу нашёл гонку:
#       E  assert Counter({409: 8, 201: 2}) == {201: 1, 409: 9}
#   register — обычный def, FastAPI выполняет его в пуле потоков, а между проверкой
#   «имя свободно» и записью стоит bcrypt. Два пользователя получали 201, второй затирал
#   пароль первого. Починено блокировкой в services/users.py; тест зелёный 5 прогонов из 5,
#   поломка из mutate.py 10 («под блокировкой нет повторной проверки») — красная.
#   TestClient шлёт запросы по одному из одного потока — эту гонку он не покажет.
#
# Чего не получили: настоящего сервера и сети. Нет разбора HTTP байтами, таймаутов
#   сокетов, keep-alive, нескольких процессов воркеров; lifespan приходится запускать
#   вручную. Одновременность только в одном процессе и одном цикле событий — для
#   проверок под нагрузкой по-прежнему нужен uvicorn, как в Ф3-06.


# ===========================================================================
# 6. HTTP без сети: respx  (tests/test_http_retry.py, tests/test_contract_pypi.py)
# ===========================================================================
# Retry из Ф3-03 перенесён из урока в services/http_retry.py и типизирован; поверх него —
#   services/pypi.py, клиент к настоящему чужому API (PyPI JSON API: публичный, без ключа).
#   Пауза (sleep=), «сейчас» (now=) и случайность внедряются, поэтому тест знает каждую
#   паузу и ничего не ждёт:
#     test_retry_on_503_then_200            503 -> 200, route.call_count == 2, паузы [0.5]
#     test_retry_after_seconds_is_respected 429 + Retry-After: 7 -> паузы [7.0], а не 0.5
#     test_retry_after_http_date_is_respected Retry-After датой через 30 с -> паузы [30.0]
#     test_404_is_final_not_retried         404 -> 1 запрос, пауз 0, PackageNotFoundOnPyPIError
#     test_gives_up_after_last_attempt      503 x3 -> 3 запроса, паузы [0.5, 1.0], HTTPStatusError
#     test_connect_error_is_retried         ConnectError -> 200, паузы [0.5]
#     test_retry_after_parsing (5 случаев)  секунды, дата, дата в прошлом -> 0, мусор и
#                                           пустой заголовок -> None (пауза по формуле)
#   Ответы 200 и 404 в стабах — не выдуманные: tests/fixtures/*.json однажды записаны с
#   живого PyPI (curl, 2026-11-03); тело 200 урезано до полей, которые мы читаем.
#
# Чего мой стаб не знает о настоящем провайдере:
#   1. Формат Retry-After. RFC разрешает и секунды, и HTTP-дату; урок 3 понимал только
#      секунды — float("Wed, 21 Oct ...") упал бы ValueError. Теперь оба, но какой шлёт
#      PyPI (и шлёт ли вообще), стаб не знает: 429 от PyPI я ни разу не видел.
#   2. Как выглядит отказ по лимиту. Fastly перед PyPI может вернуть 429 или 503 с HTML,
#      а не JSON; другие API (GitHub) отвечают 403 и x-ratelimit-reset (секунды эпохи), а
#      не Retry-After, — такой ответ мой клиент сочтёт окончательным и не повторит.
#   3. Кеш. Настоящий ответ несёт cache-control: max-age=900 и ETag (записано в фикстуре);
#      клиент их игнорирует, стаб не проверяет, что повторный запрос мог бы быть дешевле.
#   4. Размер и время. Настоящее тело 200 для respx — 80 121 байт (стаб — урезанные 3 поля),
#      ответ идёт через CDN с задержкой; таймауты чтения стаб не воспроизводит.
#   5. Обрыв посреди тела (ReadError после заголовков) и редиректы: на имена вроде PyJWT
#      PyPI сейчас отвечает 200 сразу, но httpx редиректы по умолчанию не проходит —
#      если PyPI начнёт отвечать 301, стабы этого не покажут.
#
# Контрактные тесты: tests/test_contract_pypi.py, pytestmark = slow, в обычный прогон не
#   входят (addopts: -m 'not slow'; обычный прогон — «42 passed, 3 deselected»). Явно:
#     $ pytest -m slow -rA
#     PASSED tests/test_contract_pypi.py::test_live_200_still_matches_recording
#     PASSED tests/test_contract_pypi.py::test_live_404_still_matches_recording
#     PASSED tests/test_contract_pypi.py::test_live_client_end_to_end
#     3 passed, 42 deselected in 0.43s
#   Они проверяют, что запись ещё правда: тот же content-type, все ключи верхнего уровня
#   на месте, типы полей info те же, тело 404 то же — и что клиент работает вживую.


# ===========================================================================
# 7. Внедрённые часы  (services/clock.py)
# ===========================================================================
# Где тесты зависели от настоящего времени:
#   - test_jwt_expired_and_tampered собирал просроченный токен руками:
#     jwt.encode({"sub", "exp": datetime.now(UTC) - 1 мин}) — в обход create_access_token,
#     то есть проверял не тот токен, который выдаёт сервис (без iat, без roles);
#   - test_rate_limit_429 проверял Retry-After диапазоном 1..60: точное значение зависело
#     от того, сколько прошло по настоящим часам;
#   - выход запроса из окна лимитера (services/ratelimit.py:23) не проверял никто: на
#     настоящих часах это 60 секунд ожидания.
# Clock — протокол с now() и monotonic(); SystemClock — настоящий, FakeClock — фейк,
#   оба проверяет mypy. create_app(settings, clock=...) кладёт часы в app.state; от них
#   берут время lifespan (started_at), лимитер, выдача токена (/auth/login) и проверка
#   (current_user). exp и iat сверяет наш код по этим часам: PyJWT вызывает
#   datetime.now() внутри и внедрённые часы не видит.
# Новые тесты: test_jwt_expires_exactly_at_ttl (через 14:59 — 200, через 15:00 — 401
#   «токен просрочен»), test_jwt_from_the_future_is_invalid, test_rate_limit_429 с
#   Retry-After ровно "60", потом "1" через 59 с, и 200 ещё через 1 с.
#
# Время прогона (3 прогона подряд, -p no:randomly):
#   до:    27 passed in 0.68s / 0.64s / 0.68s   (до урока; pytest-asyncio отключён -p
#          no:asyncio — без настройки loop scope он ронял весь прогон INTERNALERROR)
#   после: 42 passed, 3 deselected in 1.09s / 1.08s / 1.25s
#   Набор стал дольше на 0,4–0,6 с, потому что тестов на 15 больше; ни один старый тест
#   настоящего времени не ждал. Разница видна на тесте, которого раньше не было:
#   выход из окна лимитера на настоящих часах (week-03/lesson_10/test_real_time_demo.py)
#     60.03s call  test_window_slides_on_real_clock      1 passed in 60.50s
#   и то же через clock.advance(60) в tests/test_security.py::test_rate_limit_429 —
#     0.03s call.  Сам тест в 2000 раз быстрее, и граница проверена с точностью до секунды.


# ===========================================================================
# 9. Инвентаризация подмен — все подмены в tests/
# ===========================================================================
# | # | что подменено            | чем                               | где                          | чьё     | вид          |
# |---|--------------------------|-----------------------------------|------------------------------|---------|--------------|
# | 1 | importlib.metadata.version| monkeypatch: {"ragkit":"9.9.9"}, | test_health,                 | чужое   | стаб         |
# |   |                          | функция с PackageNotFoundError    | test_health_survives_missing_metadata |  |              |
# | 2 | load_embedder            | counting_loader через             | test_embedder_loaded_once    | своё    | фейк-шпион   |
# |   |                          | dependency_overrides              |                              |         |              |
# | 3 | get_embedder             | create_autospec(HashEmbedder)     | test_embed_route_with_mock   | своё    | мок          |
# |   |                          | (до задания 10, коммит 55c111c)   |                              |         | -> фейк      |
# | 4 | системное время          | FakeClock (services/clock.py)     | conftest: clock -> app; все  | чужое за| фейк         |
# |   |                          |                                   | тесты через client/aclient   | своим протоколом |     |
# | 5 | asyncio.sleep            | SleepRecorder                     | tests/test_http_retry.py     | чужое   | фейк         |
# | 6 | «сейчас» для Retry-After | now=lambda: NOW                   | tests/test_http_retry.py     | чужое   | стаб         |
# | 7 | случайность (джиттер)    | jitter=0.0 в RetryPolicy          | tests/test_http_retry.py     | чужое   | отключена    |
# | 8 | HTTP к PyPI              | respx_mock (транспорт httpx)      | tests/test_http_retry.py     | чужое   | стаб         |
# | 9 | тела ответов PyPI        | записи tests/fixtures/*.json      | tests/test_http_retry.py     | чужое   | запись       |
# |10 | файл .env и APP_* в env  | _env_file=None + monkeypatch.delenv| settings_factory            | чужое   | изоляция     |
# |11 | цена bcrypt              | bcrypt_rounds=4 вместо 12         | settings_factory             | чужое   | настройка    |
# |12 | сеть и сервер            | TestClient / ASGITransport        | conftest: client, aclient    | чужое   | транспорт    |
# Не подмены: Database в памяти (это и есть реализация хранилища), JWT-секрет из
#   настроек, OTHER_SECRET в tamper (настоящее шифрование чужим ключом), tmp_path в
#   test_layers (настоящая файловая система), test_domain и test_layers целиком —
#   настоящие объекты; test_contract_pypi — настоящая сеть.
#
# Чем подмена чужого может разойтись с реальностью:
#   1. version: настоящая читает метаданные установленного пакета; стаб не знает о
#      нормализации имени (ragkit / Ragkit), о версиях вида 0.1.0.dev0 и о том, что
#      метаданные могут быть от другой копии пакета (egg-info из Ф3-09).
#   4. FakeClock двигает now() и monotonic() вместе и только по команде. Настоящие часы
#      идут сами во время запроса (между выдачей и проверкой токена), а стенное время
#      может прыгнуть назад при синхронизации NTP — monotonic нет. Тест этого не видит.
#   5. Настоящий sleep отдаёт управление циклу событий и может быть отменён (таймаут
#      сверху, CancelledError посреди паузы); фейк возвращается сразу.
#   6. Стаб «сейчас» не расходится с часами сервера; в жизни дата из Retry-After
#      сравнивается с нашими часами, а они могут отставать от часов провайдера.
#   7. Распределение джиттера не проверено вовсе: тесты идут с jitter=0.
#   8–9. Весь список задания 6: формат лимита, кеш, размер, обрывы, редиректы; записи
#      стареют — поэтому контрактные тесты.
#   10. Найдено именно здесь: _env_file=None отключал только файл, а переменные APP_*
#      окружения Settings читал всё равно. Замер: APP_JWT_TTL_MINUTES=60
#      APP_MAX_DOCUMENTS_PER_OWNER=0 pytest -> 2 failed (test_foreign_document_is_404,
#      test_jwt_expires_exactly_at_ttl). Починено: settings_factory убирает APP_* через
#      monkeypatch; с теми же переменными — 42 passed.
#   11. bcrypt с 4 раундами в ~250 раз дешевле; всё, что зависит от длительности (пул
#      потоков под нагрузкой, выравнивание времени входа из Ф3-08), в тестах выглядит
#      иначе, чем в проде.
#   12. Нет разбора HTTP, сокетов, таймаутов; ASGITransport не запускает lifespan.
#
# Моки своего и можно ли заменить фейком:
#   2. counting_loader — уже фейк: возвращает настоящий HashEmbedder и считает вызовы;
#      метка dim=3 (Ф3-09) проверяет, что запросы получают именно его. Оставляю.
#   3. create_autospec(HashEmbedder) — единственный мок своего кода и самое слабое место:
#      проверял вызов, а не результат, и пропустил 3 поломки из 3 (задание 2). Фейк есть
#      — сам HashEmbedder. Заменён в задании 10.


# ===========================================================================
# 10. Один мок на фейк  (tests/test_api.py::test_embed_route_with_fake)
# ===========================================================================
# Было: create_autospec(HashEmbedder) с return_value=[[0.6, 0.8, 0.0]] — утверждал тело
#   ответа и assert_called_once_with(["привет"]).
# Стало: настоящий HashEmbedder из lifespan; три текста ["привет", "мир", "привет"];
#   утверждения о результате: по вектору на текст в том же порядке, тот же текст — тот же
#   вектор, разные тексты — разные, размерность 128, длина каждого вектора 1.
# Таблица поломок, python week-03/lesson_09/mutate.py 10, обе строки на каждую поломку:
#   | тест                       | поломка                                 | результат               |
#   | test_embed_route_with_mock | HashEmbedder игнорирует текст           | ЗЕЛЁНЫЙ — тест не ловит |
#   | test_embed_route_with_fake | HashEmbedder игнорирует текст           | КРАСНЫЙ                 |
#   | test_embed_route_with_mock | HashEmbedder не нормирует векторы       | ЗЕЛЁНЫЙ — тест не ловит |
#   | test_embed_route_with_fake | HashEmbedder не нормирует векторы       | КРАСНЫЙ                 |
#   | test_embed_route_with_mock | маршрут эмбеддит только первый текст    | ЗЕЛЁНЫЙ — тест не ловит |
#   | test_embed_route_with_fake | маршрут эмбеддит только первый текст    | КРАСНЫЙ                 |
# Остальные строки того же прогона, все КРАСНЫЙ: граница срока токена (<= -> <), выход
#   из окна лимитера (<= -> <), гонка регистрации без повторной проверки, Retry-After
#   игнорируется (краснеют оба теста Retry-After), 404 повторяется. Итог: 8 из 8.
# Третья поломка — в маршруте, не в эмбеддере: мок с одним текстом её тоже пропускал,
#   потому что return_value одинаков при любом числе текстов.


# ===========================================================================
# 11. Устно, потом письменно
# ===========================================================================
# Почему мок проверяет вызов, а не результат. Мок ничего не вычисляет: его ответ задал
#   сам тест. Всё, что тест может утверждать о моке, — что и с чем позвали; «результат»
#   — это эхо return_value. Если ошибка в том, что вызов неверен по смыслу (строка вместо
#   списка, задание 1) или в самой реализации (задание 2), мок о ней не узнает.
#
# Почему MagicMock опаснее Mock. MagicMock реализует дандеры: len 0, перебор пуст, in —
#   False, индексация — ещё один MagicMock. Проверки «пусто», «нет повторов», all(...)
#   по пустому перебору на нём проходят вхолостую (задание 3). Mock на тех же операциях
#   падает TypeError — то есть хотя бы сообщает, что настоящего объекта нет. А patch по
#   умолчанию даёт именно MagicMock.
#
# Почему patch ставят по месту использования. `from x import y` при импорте копирует
#   ссылку в пространство имён модуля-потребителя; patch меняет атрибут одного модуля.
#   Подменишь источник — потребитель продолжит смотреть на свою копию (задание 4). Если
#   потребитель импортирует модуль (`import x; x.y()`), имя ищется при каждом вызове, и
#   подмена у источника работает.
#
# Чем фейк лучше мока при изменении настоящего объекта. Фейк — работающая реализация с
#   той же подписью, и mypy сверяет её с протоколом: изменили протокол — mypy указал все
#   места за секунды до тестов (задание 5), а на рантайме фейк падает TypeError. Мок
#   примет новый вызов молча и будет зелёным, пока ошибку не найдёт прод. Фейк гниёт
#   заметно, мок — незаметно.
#
# Что можно подменять и что нельзя. Можно — чужое, медленное или недетерминированное на
#   границе системы: сеть к чужим API, часы, случайность, sleep, почту и платежи. Нельзя
#   — своё: свои сервисы, репозитории, ragkit. Мок своего класса проверяет, что код
#   вызывает то, что вызывает, а не что система работает. Если без мока своего не
#   обойтись, это знак сильной связности: нужен протокол и фейк.
#
# Чем стаб HTTP-ответа отличается от знания о настоящем API. Стаб — моя догадка,
#   записанная в тест: он отвечает так, как я думаю, что отвечает провайдер, и тест
#   проверяет мой код против моей догадки. Знание о провайдере — это то, что он делает
#   на самом деле: формат Retry-After, тело 429, кеш, редиректы. Получить его можно
#   только у провайдера: записать настоящий ответ в фикстуру и регулярно сверять запись
#   с живым API контрактным тестом (задание 6).


def main() -> None:
    task1()
    task2()
    task3()
    task4()
    task5()
    task8()


if __name__ == "__main__":
    main()
