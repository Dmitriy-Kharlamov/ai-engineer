# Ф3-09. pytest: основы и фикстуры — отчёт

Код урока — в коммитах ветки; демонстрации, которые не должны попадать в обычный прогон,
лежат в `week-03/lesson_09/` (вне `testpaths`). Все выводы ниже — настоящие прогоны.

Окружения:
- **чистое** — свежий `git clone`, `pip install -r requirements.txt`, без `pip install -e .`;
- **установленное** — то же плюс `pip install -e ".[dev]"`.

Версии из `requirements.txt`: starlette 1.10.0, anyio 4.15.1, PyJWT 2.15.1, pytest 9.2.0.

Итог набора: было 17 тестов, стало 27 тестов. Все 27 зелёные:
- в обоих окружениях;
- на 10 сидах `pytest-randomly`;
- под `pytest -n 4`.

---

## 1. Два падения на свежем клоне

**До починки**, чистое окружение, коммит 2618397. Каталога `ragkit.egg-info` нет:

```
FAILED tests/test_api.py::test_health - importlib.metadata.PackageNotFoundErr...
FAILED tests/test_security.py::test_health_is_open - importlib.metadata.Packa...
E           importlib.metadata.PackageNotFoundError: No package metadata was found for ragkit
========================= 2 failed, 15 passed in 0.80s =========================
```

Тот же коммит в установленном окружении: `17 passed in 0.37s`, то есть все 17 тестов зелёные.

Сначала я не смог воспроизвести падение: в моём рабочем клоне оставался `ragkit.egg-info` от старого `pip install -e .`. Каталог в `.gitignore`, git его не показывает, но `importlib.metadata` находит его через `sys.path`, потому что корень репозитория стоит в `sys.path` первым. Значит, «чистое окружение» — это чистый клон, а не только чистый venv.

**Починка, слой 1 — запас в сервисе** (`api/routers/health.py`):

```python
def ragkit_version() -> str:
    try:
        return version("ragkit")
    except PackageNotFoundError:
        return UNKNOWN_VERSION  # "unknown"
```

Новый тест `test_health_survives_missing_metadata` подменяет `version` на функцию, которая бросает `PackageNotFoundError`. Он проверяет, что `/health` отвечает 200 и `"unknown"`.

**Слой 2 — честная диагностика окружения** (`tests/conftest.py`):
- `pytest_report_header` пишет в шапку прогона, установлен ли пакет и откуда на самом деле импортирован код.
- При `-q` шапка не печатается. Поэтому если пакет не установлен, то же сообщение уходит ещё и в итог прогона через `pytest_terminal_summary`.

**После**, чистое окружение (свежий клон, `ls *.egg-info` → `No such file or directory`):

```
...........................                                              [100%]
---------------------------------- окружение -----------------------------------
ragkit: метаданных пакета нет, код импортирован из /tmp/claude-0/fresh9/ragkit. /health отдаст ragkit_version='unknown'. Чтобы было как в проде: pip install -e .
27 passed in 0.68s
```

**После**, установленное окружение:

```
...........................                                              [100%]
27 passed in 0.67s
```

Шапка того же прогона без `-q`:

```
ragkit 0.1.0: пакет установлен, код импортирован из /tmp/claude-0/inst9/ragkit
```

Зелёные 27 из 27 в обоих окружениях.

Диагностика не делает набор красным: тест не должен падать из-за способа установки. Но человек с чистым клоном видит, что у него не так и чем это лечится.

---

## 2. Предупреждения как ошибки

Поставил `filterwarnings = ["error"]` и прогнал в двух окружениях.

| окружение | starlette | результат с `["error"]` |
|---|---|---|
| версии из `requirements.txt` | 1.10.0 | `18 passed`, не всплыло ни одного предупреждения |
| системный Python в моей песочнице | 1.6.0 | падение ещё на загрузке conftest, вывод ниже |

Вывод для starlette 1.6.0:

```
ImportError while loading conftest '/tmp/claude-0/fresh9/tests/conftest.py'.
/usr/local/lib/python3.13/dist-packages/starlette/testclient.py:53: in <module>
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]
/usr/local/lib/python3.13/dist-packages/anyio/_lazyimport.py:84: in emit_deprecation_warning
E   DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
```

Это ровно твоё предупреждение. Значит, у тебя starlette старше 1.10: в 1.10.0 `testclient.py` уже обращается к `anyio.from_thread.BlockingPortal`, и предупреждения нет.

Итоговая настройка (`pyproject.toml`):

```toml
addopts = "-q --strict-markers --strict-config"
filterwarnings = [
    "error",
    "ignore:The anyio.abc.BlockingPortal alias is deprecated:DeprecationWarning",
]
markers = ["slow: тест дольше секунды"]
```

С этой настройкой набор зелёный на обеих версиях starlette. Игнор нужен только старым окружениям, комментарий в `pyproject.toml` это говорит.

**Срабатывал ли прежний фильтр?** Замер на starlette 1.6.0: в фильтр подставлял каждый вариант по очереди, прогонял набор.

| `filterwarnings` | результат |
|---|---|
| `["error", "ignore::DeprecationWarning:starlette.*"]` (прежний модуль) | `18 passed` — **срабатывал** |
| `["error", "ignore::DeprecationWarning:anyio.*"]` | падение — модуль `anyio` в фильтр **не** попадает |
| `["error", "ignore:The anyio.abc.BlockingPortal alias is deprecated:DeprecationWarning"]` | `18 passed` |

Прямой замер места, на которое Python записывает предупреждение:

```
DeprecationWarning | starlette/testclient.py 53 | The anyio.abc.BlockingPortal alias is deprecated, ...
```

Почему так: anyio выпускает предупреждение с `stacklevel=3` (`anyio/_lazyimport.py:84`). Отсчёт кадров такой:

1. `emit_deprecation_warning`;
2. `__getattr__` модуля `anyio.abc`;
3. строка в `starlette.testclient`, которая написала `anyio.abc.BlockingPortal`.

В поле `module` фильтра подставляется `__name__` третьего кадра, то есть `starlette.testclient`. Поэтому прежний фильтр по шаблону `starlette.*` это предупреждение глушил. Плохо в нём другое: он глушил **любые** DeprecationWarning, приписанные коду starlette. Это ровно тот класс сигналов «starlette скоро сломается на новой зависимости», который нужно слышать.

Зависимость от `stacklevel` отдельно, на модели той же цепочки `__main__ → starlette_like → anyio_like → warn` (`week-03/lesson_09/warn_stacklevel.py`):

```
stacklevel | какой фильтр module=... глушит
         1 | anyio_like
         2 | starlette_like
         3 | __main__
```

Тот же вызов `warn` при `stacklevel` 1, 2, 3 приписывается трём разным модулям. Какой из них — решает автор чужой библиотеки. Фильтр по модулю ломается, когда они поменяют `stacklevel`. Фильтр по тексту сообщения от этого не зависит.

Попутная находка для задания 3: `jwt.encode(..., "другой-секрет")` под `error` падает с `InsecureKeyLengthWarning: The HMAC key is 25 bytes long, which is below the minimum recommended length of 32 bytes`. Строка «другой-секрет» занимает 25 байт в UTF-8, а для HS256 рекомендовано не меньше 32 байт. Поэтому чужой ключ в тестах — 63 байта.

---

## 3. Испорченный токен

Скрипт: `week-03/lesson_09/jwt_tamper.py`. Ниже прогон на PyJWT 2.15.1 из `requirements.txt`:

```
--- 1. Откуда 43 символа
HMAC-SHA256: 32 байта = 256 бит
base64url: 6 бит на символ, 256 / 6 = 42.67 -> 43 символа = 258 бит
в последнем символе 4 бита данных и 2 бита-заполнителя
--- 2. Сколько вариантов последнего символа дают ту же подпись
base64: последний символ '4', те же 32 байта дают 4 из 64 -> ['4', '5', '6', '7']
base64: два последних символа, те же байты дают 4 из 4096
PyJWT 2.15.1: принимает 1 из 64 -> ['4']
--- 3. Старая порча (замена двух последних на AA/BB) на 100 000 токенах
подпись по байтам не изменилась: 0 из 100000 = 0.000%; наивная оценка для случайной пары: 3/4096 = 0.073%
PyJWT 2.15.1 принял испорченный токен: 0 из 100000
последний символ настоящих подписей: 16 разных -> AEIMQUYcgkosw048
кодировщик пишет 2 лишних бита нулями: последний символ всегда A/E/I/...,
поэтому пары 'A'+B/C/D не бывает, а 'AA' меняется на 'BB' — старая порча портит всегда
--- 4. Новая порча (пересборка с другим секретом) на 1000 токенах
подпись испорчена в 1000 из 1000 токенов
```

Тот же скрипт с PyJWT 2.13.0, разделы 2 и 3:

```
base64: последний символ 'E', те же 32 байта дают 4 из 64 -> ['E', 'F', 'G', 'H']
PyJWT 2.13.0: принимает 4 из 64 -> ['E', 'F', 'G', 'H']
PyJWT 2.13.0 принял испорченный токен: 0 из 100000
```

Что подтвердилось и что нет:

- **43 символа и 4 варианта из 64 — подтверждаю.** 256 бит не делятся на 6. Последний символ несёт 4 бита подписи и 2 бита, которые декодер отбрасывает. Четыре символа с одинаковыми старшими 4 битами дают те же 32 байта. Для двух последних символов совпадают 4 комбинации из 4096: предпоследний символ должен остаться прежним, последний — один из четырёх.
- **Зависит от версии PyJWT** — проверил 2.10.1, 2.12.1, 2.13.0 и 2.14.0:
  - 2.10.1, 2.12.1 и 2.13.0 принимают все 4 варианта;
  - с 2.14.0 PyJWT перекодирует подпись и сравнивает с присланной (`base64url_encode(decoded) != segment` → `DecodeError`), поэтому принимает только 1 канонический вариант.

  У нас 2.15.1. В `pyproject.toml` стоит `pyjwt>=2.10`, так что у коллеги может оказаться и снисходительная версия.
- **Твоя оценка 4/4096 для именно этой порчи не подтвердилась.** Получилось 0 из 100 000. Причина: кодировщик всегда пишет 2 лишних бита нулями. Поэтому последний символ настоящей подписи — только один из 16 (A, E, I, M, …, 8), а B, C, D там не бывает никогда. Чтобы замена на «AA» не изменила байты, подпись должна была кончаться на «A» + B/C/D — такого не бывает. Если же она кончается на «AA», код ставит «BB» и меняет предпоследний символ. Старая порча работала всегда, но держалась на неочевидном свойстве кодировщика. Ревьюер этого не видит, и любая «похожая» правка (например, замена только последнего символа) уже ломается в 3 случаях из 4 на PyJWT ≤ 2.13.

**Новая порча** (`tests/test_security.py`):

```python
OTHER_SECRET = "другой-секрет-той-же-длины-32-байта!"  # >= 32 байт: иначе InsecureKeyLengthWarning


def tamper(token: str) -> str:
    payload = jwt.decode(token, options={"verify_signature": False})
    return jwt.encode(payload, OTHER_SECRET, algorithm="HS256")
```

Заголовок и payload остаются прежними, меняется только подпись. Подпись чужим ключом совпадает с настоящей с вероятностью 2⁻²⁵⁶, то есть никогда. Это не зависит ни от кодировщика, ни от версии PyJWT.

Проверка на 1000 токенах стала тестом `test_tamper_always_breaks_signature`, он идёт 0,22 с. На каждом токене новый случайный секрет. Тест проверяет:
- у испорченного токена та же часть до подписи;
- `decode_access_token` бросает `InvalidSignatureError` с `match="Signature verification failed"`.

Результат: 1000 из 1000 токенов испорчены.

---

## 4. parametrize

`test_ragkit_errors_mapped` и `test_response_model_filters` теперь по одному случаю на тест.

Две находки по ходу:

- Названия случаев (`ids`) написал латиницей. Кириллические pytest экранирует, в выводе было бы `test_ragkit_errors_mapped[чанк ...]`. Включать это можно только опцией `disable_test_id_escaping_and_forfeit_all_rights_to_community_support` — название говорит само за себя.
- `pytest -v` при `addopts = "-q"` даёт обычную подробность: `-q` и `-v` взаимно гасятся. Имена случаев видны при `-vv`.

```
$ pytest -vv -p no:randomly -k "errors_mapped or response_model"
tests/test_api.py::test_response_model_filters[leak-without-response-model] PASSED [ 25%]
tests/test_api.py::test_response_model_filters[response-model-filters] PASSED [ 50%]
tests/test_api.py::test_ragkit_errors_mapped[chunk-too-large] PASSED     [ 75%]
tests/test_api.py::test_ragkit_errors_mapped[empty-document] PASSED      [100%]
======================= 4 passed, 18 deselected in 0.13s =======================
```

Из 22 тестов набора на тот момент выбраны 4 случая, оба теста разошлись на два.

`test_response_model_filters` теперь сверяет полный набор ключей:
- в варианте с утечкой: `{"name", "password_hash", "internal_id"}`;
- в отфильтрованном: `{"name"}`.

Раньше проверялось только наличие `password_hash`, и утечка `internal_id` проходила незамеченной.

**Сломал один случай**: в `api/errors.py` код `"empty_index"` заменил на `"empty_idx"`.

```
tests/test_api.py::test_ragkit_errors_mapped[chunk-too-large] PASSED     [ 50%]
tests/test_api.py::test_ragkit_errors_mapped[empty-document] FAILED      [100%]
E       AssertionError: assert 'empty_idx' == 'empty_index'
FAILED tests/test_api.py::test_ragkit_errors_mapped[empty-document] - Asserti...
================== 1 failed, 1 passed, 20 deselected in 0.07s ==================
```

Красный 1 случай из 2. Второй отработал и прошёл, падение первого его не скрыло. Поломку откатил.

---

## 5. Фикстура вместо импорта из conftest

`from tests.conftest import make_settings` убран. В `conftest.py` теперь цепочка фикстур `settings_factory → settings → app → client`.

```
$ pytest --fixtures
--------------------- fixtures defined from tests.conftest ---------------------
settings_factory -- tests/conftest.py:41
    Тестовые Settings; любое поле меняется аргументом: settings_factory(debug=False).

settings -- tests/conftest.py:62
    Настройки по умолчанию; папка с тестами может переопределить их своим conftest.py.

app -- tests/conftest.py:68
    Свежее приложение на каждый тест: своё app.state, свой лимитер, свои overrides.

client -- tests/conftest.py:74
    TestClient внутри with: lifespan отрабатывает до первого запроса и после последнего.
```

Пример использования: `test_lab_exists_only_in_debug(settings_factory)` собирает приложение с `debug=False` и проверяет, что `/lab` отвечает 404.

Что получил сверх прежнего, в двух предложениях. Настройки стали узлом в графе фикстур: `app` зависит от `settings`, поэтому папка с тестами может подменить настройки своим `conftest.py`, не трогая ни одного теста, а фикстуры видны в `pytest --fixtures` с описанием. Импортированная функция была обычным вызовом: её не переопределить на уровне папки, она не попадает в список фикстур, и тесты зависели от расположения файла `tests/conftest.py` как от модуля.

---

## 6. Области видимости

Файл `week-03/lesson_09/test_scope_demo.py`, вне `testpaths`. В нём одна фикстура `shared_client` со `scope="session"` и счётчик `CREATED`.

Тесты по порядку в файле, с `-s`:

```
$ pytest week-03/lesson_09/test_scope_demo.py -p no:randomly -s
.....
[shared_client] за прогон создана 1 раз
5 passed
```

Прошло 5 тестов: три случая `test_fixture_created_once` и два теста регистрации. Фикстура создана 1 раз на все 5 тестов.

Контрольный опыт: та же фикстура со `scope="function"`.

```
test_fixture_created_once[1] PASSED
test_fixture_created_once[2] FAILED
test_fixture_created_once[3] FAILED
test_register_alice PASSED
test_register_alice_again_conflicts FAILED
3 failed, 2 passed
```

Счётчик дошёл до 2 и 3, поэтому второй и третий случаи покраснели. Значит, тест действительно ловит число созданий.

**Как ломается.** Два теста меняют общее состояние, оба регистрируют alice:
- `test_register_alice` ждёт ответ 201;
- `test_register_alice_again_conflicts` ждёт 409 и молча рассчитывает, что первый тест уже отработал на том же приложении.

Прогон на 10 сидах:

```
seed=1   2 failed, 3 passed
seed=2   5 passed
seed=3   5 passed
seed=4   5 passed
seed=5   5 passed
seed=6   2 failed, 3 passed
seed=7   5 passed
seed=8   2 failed, 3 passed
seed=9   5 passed
seed=10  2 failed, 3 passed
```

Красные сиды — 4 из 10: 1, 6, 8 и 10. Порядок на сиде 1:

```
Using --randomly-seed=1
test_register_alice_again_conflicts FAILED     E  assert 201 == 409
test_fixture_created_once[3] PASSED
test_register_alice FAILED                     E  assert 409 == 201
...
```

Тест «повторной» регистрации пошёл первым и получил 201, потому что alice ещё нет. Затем «первая» регистрация получила 409, потому что alice уже есть. Красными оказались оба теста, хотя код приложения ни при чём.

Починка — не делить состояние:
- вернуть `scope="function"`, как у `app` в `tests/conftest.py`;
- или оставить session-фикстуру только для того, что тесты не меняют (неизменяемые данные, «тяжёлую» модель);
- или каждому тесту давать своё имя пользователя.

---

## 7. Оснастка ушла из прода

Изменения в коде:
- `EMBEDDER_LOADS` удалён из `api/deps.py`. `load_embedder` теперь просто создаёт эмбеддер.
- Поле `embedder_loads` убрано из `HealthOut`: оно существовало только для теста.
- Lifespan берёт загрузчик через тот же механизм подмены, что и зависимости:

```python
load: Callable[..., Embedder] = app.dependency_overrides.get(deps.load_embedder, deps.load_embedder)
app.state.embedder = load(dim=128)
```

Тест:

```python
def test_embedder_loaded_once(app: FastAPI) -> None:
    loads: list[int] = []  # счётчик живёт в тесте

    def counting_loader(dim: int = 128) -> Embedder:
        loads.append(dim)
        return HashEmbedder(dim=3)  # метка: запросы обязаны получить именно этот объект

    app.dependency_overrides[deps.load_embedder] = counting_loader
    with TestClient(app) as c:
        for _ in range(10):
            r = c.post("/chunks/embed", json={"texts": ["привет"]})
            assert r.status_code == 200
            assert r.json()["dim"] == 3
    assert loads == [128]
```

Тест проверяет то же, что раньше: 10 запросов, ровно 1 загрузка. Счётчик — локальный список, другие тесты его не видят и не двигают, а `app` свой на каждый тест.

Метку `dim=3` добавил в задании 8: без неё тест не замечал эмбеддер, созданный в обход загрузчика.

`week-03/lesson_06.py`, задание 9, переведено на такой же подменённый загрузчик. Вывод: `запросов: 10, создано эмбеддеров за этот запуск сервиса: 1`.

---

## 8. Может ли тест упасть

Скрипт: `week-03/lesson_09/mutate.py`. Он вносит одну поломку, гоняет весь `tests/` и возвращает файл как был. Каждая поломка — замена одного фрагмента в одном файле.

**Первый прогон**, до усиления тестов:

| тест | поломка | результат |
|---|---|---|
| test_unknown_ctx_key_is_dropped | `safe_error` пропускает любой ключ ctx | красный |
| test_foreign_document_is_404 | `get_owned` не сверяет владельца | красный |
| test_rate_limit_429 | лимитер пропускает limit+1 запрос (`>=` → `>`) | красный |
| test_admin_router_protects_new_endpoint | у роутера `/admin` нет `dependencies` | красный |
| test_jwt_expired_and_tampered | подпись JWT не проверяется (`verify_signature: False`) | красный |
| test_health | `/health` всегда отдаёт `"unknown"` вместо версии | **зелёный — тест не ловит** |
| test_embedder_loaded_once | эмбеддер создаётся на каждый запрос в обход загрузчика | **зелёный — тест не ловит** |

Разбор двух выживших:

- **test_health** проверял только `status == "ok"`. Версия, `started_at` и вообще то, что эндпоинт отдаёт что-то осмысленное, не проверялись. Версию я не мог сравнить с `version("ragkit")`: в чистом окружении её нет (задание 1). Поэтому тест теперь сам задаёт её через `monkeypatch.setattr(health, "version", {"ragkit": "9.9.9"}.__getitem__)` и проверяет `"9.9.9"`. Так ловится и `"unknown"` вместо версии, и опечатка в имени пакета: `KeyError` даёт 500. Ещё тест проверяет, что `started_at` не позже начала запроса.
- **test_embedder_loaded_once** считал вызовы загрузчика, но не проверял, что запросы получают именно загруженный объект. Поломка `get_embedder → load_embedder()` грузила модель на каждый запрос мимо подмены, а счётчик честно показывал 1. Починка — метка `dim=3`, см. задание 7.

**Второй прогон**, после усиления. Сюда же добавлена поломка из задания 11:

| тест | поломка | результат | покраснели |
|---|---|---|---|
| test_unknown_ctx_key_is_dropped | safe_error пропускает любой ключ ctx | КРАСНЫЙ | test_unknown_ctx_key_is_dropped |
| test_foreign_document_is_404 | get_owned не сверяет владельца | КРАСНЫЙ | test_foreign_document_is_404 |
| test_rate_limit_429 | лимитер пропускает limit+1 запрос (>= -> >) | КРАСНЫЙ | test_rate_limit_429 |
| test_admin_router_protects_new_endpoint | у роутера /admin нет защиты | КРАСНЫЙ | test_admin_router_protects_new_endpoint |
| test_jwt_expired_and_tampered | подпись JWT не проверяется | КРАСНЫЙ | test_jwt_expired_and_tampered, test_tamper_always_breaks_signature |
| test_health | /health всегда отдаёт "unknown" | КРАСНЫЙ | test_health |
| test_embedder_loaded_once | эмбеддер грузится на каждый запрос | КРАСНЫЙ | test_embedder_loaded_once |
| test_login_failure_is_401_and_indistinguishable | authenticate не проверяет пароль | КРАСНЫЙ | test_login_failure_is_401_and_indistinguishable[wrong-password] |

Красные 8 из 8. Почти везде краснеет ровно целевой тест. Исключение — проверка подписи: её ловят 2 теста, и это правильно, оба про подпись.

---

## 9. Порядок и параллельность

Чистое окружение, свежий клон, итоговый коммит, 10 сидов:

```
seed=1   27 passed in 0.60s
seed=2   27 passed in 0.60s
seed=3   27 passed in 0.70s
seed=4   27 passed in 0.62s
seed=5   27 passed in 0.60s
seed=6   27 passed in 0.66s
seed=7   27 passed in 0.64s
seed=8   27 passed in 0.85s
seed=9   27 passed in 0.70s
seed=10  27 passed in 0.63s
$ pytest -n 4
27 passed in 2.12s
```

Зелёные все 10 прогонов по 27 тестов и параллельный прогон в 4 процесса. Дополнительно проверил сиды с 1 по 50: красных прогонов 0. Прежний набор, 17 тестов на коммите 2618397, тоже зелёный на 10 сидах и под `-n 4`.

**Почему подсказка про «3 запроса в минуту на все тесты» у нас не срабатывает.** Фикстура `app` — function-scope, и каждый тест получает новое приложение. В его lifespan создаётся новый `RateLimiter`, новый `UserStore`, новый секрет. Лимит 3 действует внутри одного теста, а не на весь прогон. Ключ лимитера — имя пользователя (`limiter.hit(principal.name)`), и `/me/key` дёргает один тест.

Отдельно проверил худший случай: временно сделал `settings_factory`, `settings`, `app` и `client` session-scope. Набор и тогда прошёл на 10 сидах из 10, потому что тесты берут разные имена: alice, bob, carol, dave, erin. Это везение, а не свойство: второй тест про `/me/key` от имени dave сделал бы набор зависимым от порядка.

Удалённый `EMBEDDER_LOADS` порядок тоже не ломал: тест считал разницу «до/после» внутри себя. Но под `-n 4` счётчик жил бы в каждом процессе свой, и на будущее это ловушка, поэтому он удалён.

**Сид, на котором падало: `--randomly-seed=1`**, также 6, 8 и 10. Это демонстрация из задания 6 (`week-03/lesson_09/test_scope_demo.py`). В основном наборе зависимостей от порядка не нашлось, и я не стал их выдумывать.

---

## 10. pytest.raises

Три проверки с `match=`, файл `tests/test_domain.py`, предметный слой без HTTP:

```python
with pytest.raises(ChunkTooLargeError, match=r"3 токенов при лимите 2$"):
    document.add_chunks([Chunk("a b c", "s")])

with pytest.raises(EmptyIndexError, match=r"в документе 'пустой' нет ни одного чанка"):
    Document("пустой").largest_chunk()

with pytest.raises(DocumentLimitError, match=r"^не больше 1 документов на владельца$"):
    repo.create("alice", "c.md", "z")  # у bob до этого свой документ — исключения не было
```

Четвёртая проверка — в задании 3: `pytest.raises(jwt.InvalidSignatureError, match="Signature verification failed")`.

**raises(Exception) на опечатке**, файл `week-03/lesson_09/test_raises_demo.py`:

```python
def test_broad_raises_hides_typo() -> None:
    document = Document("d", max_chunk_tokens=2)
    with pytest.raises(Exception):
        documnet.add_chunks([Chunk("a b c", "s")])  # опечатка: documnet
```

```
E           NameError: name 'documnet' is not defined
PASSED week-03/lesson_09/test_raises_demo.py::test_broad_raises_hides_typo
PASSED week-03/lesson_09/test_raises_demo.py::test_precise_raises_without_typo
FAILED week-03/lesson_09/test_raises_demo.py::test_precise_raises_catches_typo
1 failed, 2 passed
```

Результаты:
- `raises(Exception)` с опечаткой зелёный: `NameError` — тоже `Exception`, а `add_chunks` даже не вызывался.
- Та же опечатка под `raises(ChunkTooLargeError, match=...)` красная, с `NameError` в выводе.
- Без опечатки точная проверка зелёная.

В нашем проекте такой код до коммита не дошёл бы. `ruff check --ignore-noqa` на этом файле находит 5 ошибок, в том числе `B017 Do not assert blind exception: Exception` и `F821 Undefined name documnet`. Правило B017 включено в `select`. В демо обе проверки заглушены построчно и намеренно.

---

## 11. Покрытие

`pytest --cov=ragkit --cov=api --cov=services --cov-branch --cov-report=term-missing`, итог:

| пакет | строк | не выполнено | веток | частично | покрытие |
|---|---|---|---|---|---|
| api | 425 | 28 | 28 | 2 | 93% |
| services | 154 | 4 | 22 | 5 | 95% |
| ragkit | 153 | 49 | 26 | 2 | 61% |
| всего | 732 | 81 | 76 | 9 | 86% |

Худшие места по модулям:
- `ragkit/chunking.py` — 50% (40 строк не выполнено);
- `api/routers/lab.py` — 66% (19 строк);
- `ragkit/embedding.py` — 79%;
- `ragkit/__main__.py` — 0%;
- `services/users.py` — 91%;
- `services/ratelimit.py` — 91%, не выполнена строка 23 (`popleft` — выход запроса из окна).

Полная таблица — в выводе команды.

**Стоило покрыть: неудачный вход.** В первом прогоне `api/routers/auth.py:29` (`raise unauthorized("Bearer", "неверный логин или пароль")`) не выполнялась ни разу. В `services/users.py` не выполнялась ветка «пользователя нет» (строки 41–45). То есть ни один тест не проверял, что вход с неверным паролем отклоняется. Поломка «`authenticate` не проверяет пароль» до нового теста оставалась зелёной:

```
| test_login_failure_is_401_and_indistinguishable | authenticate не проверяет пароль | ЗЕЛЁНЫЙ — тест не ловит | — |
```

Покрытие нашло проблему не прямо. Проверка пароля — тернарный оператор в одной строке (`return record.user if check_password(...) else None`), а `--cov-branch` считает ветки между строками и показал её «выполненной». Выдала только невыполненная строка 29 в роутере.

Новый тест параметризован на 2 случая: `wrong-password` и `unknown-user`. Оба дают 401, одинаковое тело `{"detail": "неверный логин или пароль"}` и `WWW-Authenticate: Bearer`. Одинаковый ответ — часть требования: по нему нельзя узнать, какие логины существуют. После теста `auth.py` 100%, а поломка ловится.

**Не стоит покрывать: `api/routers/lab.py`, 19 строк.** Это учебные эндпоинты уроков 6–7: прогноз источника параметров, одноимённые параметры, `/lab/login-naive` без выравнивания времени. Роутер подключается только при `debug=True`, в проде его нет. Их поведение уже показано выводами `week-03/lesson_06.py` и `lesson_07.py`. Тесты на них прибили бы учебные опыты к набору и подняли бы процент, не защитив ни одной строки прода. Туда же `ragkit/__main__.py` (0%, 4 строки демо-запуска).

Что я не трогал и почему: `ragkit/chunking.py` (50%) — честный долг. Разбиение текста тестируется только через HTTP-слой, юнит-тестов у `split_text` нет. Покрывать его стоит отдельным уроком с осмысленными утверждениями, а не ради процента.

---

## 12. Устные вопросы — письменно

**Почему assert в pytest показывает значения и где это не работает.** pytest при импорте тестового модуля переписывает его AST: каждый `assert` превращается в код, который сохраняет промежуточные значения (`r.status_code`, обе стороны `==`) и при провале печатает их, например `assert 'empty_idx' == 'empty_index'` с построчным diff. Переписываются только модули, которые pytest импортирует как тесты: `test_*.py`, `*_test.py` и `conftest.py`, плюс зарегистрированные через `pytest.register_assert_rewrite`. `assert` в обычном модуле-помощнике, импортированном из теста, остаётся голым `AssertionError` без значений. Ещё он исчезает целиком при `python -O`. Поэтому утверждения пишут в тестах, а помощники только готовят данные, как `tamper()` в задании 3.

**Чем фикстура лучше функции, импортированной из conftest.** Фикстура — узел графа: она получает другие фикстуры по имени, у неё есть область видимости и teardown через `yield`. Её можно переопределить в `conftest.py` подпапки, не трогая тесты, и она видна в `pytest --fixtures` с описанием. Импортированная функция — просто вызов: её не переопределить для папки, у неё нет teardown и области видимости. К тому же тест начинает зависеть от `conftest.py` как от модуля по пути, который pytest на самом деле грузит по-своему.

**Почему широкая область видимости опасна.** Объект живёт дольше теста, значит его состояние — общее. Тест, который его меняет, влияет на все следующие, и результат начинает зависеть от порядка. Задание 6: на 4 сидах из 10 красными стали 2 теста, хотя код приложения верен. Такие падения не воспроизводятся при запуске одного теста. Широкую область берут только для дорогой и неизменяемой подготовки.

**Зачем match= в pytest.raises.** Класс исключения говорит «что-то такого рода случилось где-то внутри блока», а `match` — «случилось то, что я проверяю». Без `match` тест зелёный, если `ChunkTooLargeError` вылетел не из той проверки или по другой причине. `pytest.raises(Exception)` зелёный даже на `NameError` от опечатки (задание 10).

**Почему 100 % покрытия — плохая цель.** Покрытие измеряет «строка выполнилась», а не «результат строки проверен». До задания 8 `/health` был покрыт на 100%, а тест не замечал `"unknown"` вместо версии. Проверка пароля была «выполнена», а неверный пароль не проверял никто (задание 11). Погоня за процентом рождает тесты без утверждений: число растёт, защита нет. Полезно обратное: смотреть, что **не** покрыто, и решать по каждой ветке — это забытый случай или ненужный код.

**Почему тест, который нельзя заставить упасть, хуже отсутствия теста.** Отсутствие теста видно: дыру можно найти и закрыть. Тест, который не падает, создаёт ложную уверенность: в отчёте зелёная галочка с именем `test_health`, и все считают поведение проверенным. Его не перепишут, потому что он «есть». Он стоит времени на каждый прогон и поддержку, ничего не давая взамен. Проверка одна — сломать код и посмотреть, покраснеет ли тест.
