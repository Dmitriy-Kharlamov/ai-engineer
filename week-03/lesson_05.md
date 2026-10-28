# Ф3-05. Type hints, ruff, mypy — отчёт

Версии те же, что в уроке: **mypy 2.4.1, ruff 0.17.3**, Python 3.14.
Где мои числа расходятся с уроком, я пишу свои и объясняю, откуда разница.

Файлы урока:
- `week-03/lesson_05_demo.py` — задание 1, семь намеренных ошибок;
- `week-03/lesson_05_any.py` — задание 2, заражение через `Any`;
- `week-03/lesson_05.py` — задание 8, `should_retry` с типами, плюс проверка `Protocol` к заданию 7;
- `ragkit/` — задание 7;
- `pyproject.toml` и `.pre-commit-config.yaml` — задания 3 и 9.

---

## 1. Семь ошибок × инструмент × режим

```
mypy (без настроек)                  → 1
mypy --strict                        → 3
mypy --strict --warn-unreachable     → 4
ruff (настройки проекта)             → 4
ruff --isolated (умолчания ruff)     → 2
```

Команды: `mypy --config-file "" [флаги] week-03/lesson_05_demo.py` и `ruff check week-03/lesson_05_demo.py`.
Флаг `--config-file ""` отключает `pyproject.toml`. Без него «без настроек» превратилось бы в strict из проекта.

| # | Ошибка | mypy | mypy --strict | + warn-unreachable | ruff (проект) | ruff (умолч.) |
|---|---|---|---|---|---|---|
| 1 | `text.upper() + 5` в функции без аннотаций | — | ~ «missing a type annotation» | ~ то же | — | — |
| 2 | возвращает то `int`, то `str` | — | ~ «missing a return type annotation» | ~ то же | — | — |
| 3 | `len(name)` при `name: str \| None` | **да** `arg-type` | **да** | **да** | — | — |
| 4 | `print` после `return` | — | — | **да** `unreachable` | ~ `RET503` | — |
| 5 | `bucket: list[str] = []` | — | — | — | **да** `B006` | **да** `B006` |
| 6 | параметр `list` перекрывает встроенное имя | — | — | — | **да** `A002` | — |
| 7 | `import json` не используется | — | — | — | **да** `F401` | **да** `F401` |

«да» — инструмент нашёл саму ошибку. «~» — нашёл симптом рядом, но не её.

**Расхождение с уроком.** У тебя strict нашёл 5, у меня 3. Разница в том, что считать находкой. Ошибки 1 и 2 strict «ловит» только тем, что у функции нет аннотаций, а саму ошибку внутри (строка + число, два типа возврата) не видит. Стоит дописать `-> int` к `parse_port`, и mypy сразу скажет `Incompatible return value type (got "str", expected "int")`. Ошибку 4 ruff тоже видит косвенно: после `print` функция неявно возвращает `None`, и это `RET503`, а не «недостижимый код».

**Вывод.** mypy без настроек находит одну ошибку из семи. Строгий mypy с `warn_unreachable` находит две напрямую и две косвенно. ruff находит три, которые mypy не видит в принципе. Инструменты почти не пересекаются.

---

## 2. Any заражает

`mypy --config-file "" --strict week-03/lesson_05_any.py`:

```
lesson_05_any.py:14: error: Function is missing a type annotation  [no-untyped-def]
lesson_05_any.py:15: note: Revealed type is "Any"          ← reveal_type(x)
lesson_05_any.py:16: note: Revealed type is "Any"          ← reveal_type(x.upper())
lesson_05_any.py:21: note: Revealed type is "str"          ← reveal_type(s)
lesson_05_any.py:22: note: Revealed type is "str"          ← reveal_type(s.upper())
lesson_05_any.py:23: error: Unsupported operand types for + ("str" and "int")  [operator]
lesson_05_any.py:27: error: Function is missing a type annotation  [no-untyped-def]
lesson_05_any.py:36: error: Call to untyped function "load_config" in typed context  [no-untyped-call]
lesson_05_any.py:37: note: Revealed type is "Any"          ← port в build_url
lesson_05_any.py:38: error: Returning Any from function declared to return "str"  [no-any-return]
lesson_05_any.py:48: error: Unsupported operand types for + ("str" and "object")  [operator]
lesson_05_any.py:55: note: Revealed type is "int"          ← после isinstance
Found 6 errors in 1 file
```

- **`untyped(x)`.** `x` имеет тип `Any`, `x.upper()` тоже `Any`, и `x.upper() + 5` проходит молча. Та же строка в `typed(s: str)` даёт `operator` (строка 23). Ошибка не исчезла, она стала невидимой.
- **Цепочка из трёх функций.** `load_config(raw)` без аннотаций, поэтому её результат — `Any`. `get_port(config) -> Any` передаёт `Any` дальше. В `build_url` выражение `"http://" + host + ":" + port` при `port = 8080` падает в рантайме с `TypeError: can only concatenate str (not "int") to str`, это видно при запуске файла. При этом mypy **не видит саму ошибку сложения**, потому что `port` имеет тип `Any`.
  - Без настроек mypy молчит полностью.
  - `--strict` даёт лишь два косвенных сигнала: «вызов нетипизированной функции» (строка 36) и «возвращаешь Any» (строка 38). Если `get_port` вернёт `Any` из стороннего JSON без нетипизированного вызова, не останется и их.
- **Почему `object` безопаснее.** С `Any` разрешено всё, с `object` — только то, что есть у любого объекта. `get_port_obj(...) -> object` даёт ошибку `str + object` (строка 48) и заставляет проверить тип. После `isinstance(port, int)` тип сужается до `int` (строка 55), и дальше всё честно. `Any` говорит «не проверяй», `object` — «проверь, прежде чем трогать».

---

## 3. Настройки (`pyproject.toml`)

```toml
[tool.mypy]
python_version = "3.14"
files = ["ragkit", "week-03/lesson_05.py"]
strict = true
warn_unreachable = true
show_error_codes = true
warn_unused_ignores = true
plugins = ["pydantic.mypy"]

[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = [".venv", "week-03/lesson_05_demo.py"]

[tool.ruff.lint]
select = ["E", "W", "F", "B", "A", "SIM", "UP", "RET", "C4", "PTH", "I"]

[tool.ruff.lint.per-file-ignores]
"week-*/**" = ["E501", "PTH", "RET504"]
"todo/**" = ["PTH"]
```

- **`files`.** Строго проверяется только то, что аннотировано: пакет и `lesson_05.py`. Учебные `week-*` не типизированы. strict на них дал бы 893 строки, из которых 797 — «нет аннотаций». Проверка, которая всегда красная, перестаёт что-либо значить.
- **`plugins = ["pydantic.mypy"]`.** Без плагина mypy не понимает, что `Settings()` читает `api_key` из окружения, и выдаёт ложное `Missing named argument "api_key"`. То же с `_env_file=` (задание 6).
- **`line-length = 100`, а не 88.** На 88 ruff находит 136 длинных строк. На 100 после форматирования остаются 36, и все они — комментарии в учебных файлах. 100 — частый компромисс: на экран по-прежнему помещаются два файла рядом.
- **Наборы правил.** Каждый нашёл в репозитории хотя бы одну настоящую проблему (задание 4):
  - `E`/`W` — pycodestyle;
  - `F` — pyflakes: неиспользуемое и неопределённое;
  - `B` — bugbear: изменяемые умолчания, `zip` без `strict`, `raise` без `from`;
  - `A` — перекрытые встроенные имена;
  - `SIM`/`C4` — упрощения;
  - `UP` — устаревший синтаксис;
  - `RET` — возвраты, ловит «то значение, то `None`»;
  - `PTH` — `pathlib`;
  - `I` — сортировка импортов.
- **`per-file-ignores`** — только для `week-*` и `todo`. В `ragkit` не игнорируется ничего. Причины по каждому правилу — в части (в) задания 4.
- **`target-version = "py311"`** совпадает с `requires-python`. Без этого `UP` предложил бы синтаксис 3.14, и пакет перестал бы ставиться на 3.11.
- **Почему `warn_unreachable` руками.** `--strict` — фиксированный набор флагов, которые почти никогда не дают ложных срабатываний на нормальном коде. `warn_unreachable` в него не входит: он срабатывает и на защитный код, который «не может» выполниться по типам, но нужен на случай кривых данных извне (например, `else: raise` после исчерпывающего `isinstance`). Поэтому его включают сознательно. Ошибка «код после `return`» в моём журнале была трижды (Ф1-04, Ф1-18, Ф2-04), и здесь дисциплину заменяет настройка.

---

## 4. Разбор находок ruff

С правилами и `line-length` из урока (88) на текущем `main` ruff выдаёт **312** находок, а не 300. На коммите `5695755` (до урока 4) я насчитал **297**. Разница — это `lesson_04.py` и `config.py`, добавленные после твоего замера.

| Этап | Находок |
|---|---|
| правила урока, `line-length` 88 | 312 |
| мой конфиг: 100 + `per-file-ignores` | 155 |
| после `ruff format .` | 53 |
| после `ruff check --fix` | 28 |
| после ручных исправлений | 6 (только `ragkit/__init__.py`) |
| после аннотирования `ragkit` | **0** |

### (а) Форматирование и автоматические правки — 127 находок

`ruff format` закрыл 102:
- `W292` ×36 — нет перевода строки в конце файла;
- `W291` ×31 — пробелы в конце строки;
- `W293` ×29 — пробелы в пустой строке;
- `E701` ×6 — две инструкции в одной строке.

`ruff check --fix` закрыл ещё 25 безопасными автоисправлениями:
- `I001` ×12 — порядок импортов;
- `UP015` ×5 — лишнее `"r"` в `open`;
- `RET505` ×3 — `else` после `return`;
- `UP033` ×2 — `lru_cache(maxsize=None)` → `cache`;
- `RET506` ×1;
- `F401` ×2 — неиспользуемые импорты в уроках.

Ещё 136 длинных строк `E501` формат не закрывает, потому что это комментарии. Они ушли в часть (в).

### (б) Настоящие ошибки, исправленные руками — 24 находки

Ещё 4 из 28 — осознанные `noqa`, они в части (в).

| Правило | Где | Что сделал |
|---|---|---|
| `A004` ×1 | `week-03/lesson_01.py:7` | см. задание 5 |
| `RET503` ×2 | `week-03/lesson_03.py` | см. задание 5 |
| `B905` ×6 | `lesson_08`, `lesson_14`, `week-02/lesson_04` ×2, `lesson_10`, `week-03/lesson_03` | где длины обязаны совпадать — `zip(..., strict=True)`; в двух местах — **`itertools.pairwise`** (см. ниже) |
| `B904` ×1 | `todo/storage.py:23` | `raise CorruptedFileError(...) from err`: исходная `JSONDecodeError` теперь сохраняется в `__cause__`, а не теряется (проверено) |
| `B007` ×3 | `week-02/sql_block_02.py` | неиспользуемые переменные цикла → `_doc_id`, `_cid`, `_text` |
| `E741` ×2 | `lesson_03`, `lesson_04` | `l` → `line` / `ln`: `l` неотличима от `1` и `I` |
| `E402` ×1 | `week-01/lesson_17.py` | импорт `dataclasses` перенесён наверх |
| `C416` ×1 | `week-01/lesson_14.py` | `{k: v for k, v in zip(...)}` → `dict(zip(..., strict=True))` |
| `F401` ×1 | `week-03/lesson_04.py` | убран неиспользуемый импорт `Settings` |
| `F401` ×6 | `ragkit/__init__.py` | добавлен `__all__`: это не мусор, а публичный API пакета, теперь объявленный явно |

Две находки `B905` оказались ловушкой. `zip(final, final[1:])` и `zip(times, times[1:])` — это пары соседей, и длины здесь **по замыслу** отличаются на единицу. Если бездумно добавить `strict=True`, как просит правило, код упадёт с `ValueError`. Правильное исправление — `pairwise()`. Подсказку линтера нужно понимать, а не исполнять механически.

### (в) Осознанный игнор

| Правило | Где | Скрыто | Причина |
|---|---|---|---|
| `E501` | `week-*/**` | 36 | все 36 — комментарии-объяснения рядом с кодом; разрыв ухудшил бы чтение, а этих комментариев требует отчёт |
| `PTH` | `week-*/**`, `todo/**` | 19 (`PTH123` ×16, `PTH110` ×2, `PTH107` ×1) | уроки специально учат `open()` и `os.path`, это материал курса; в новом коде `ragkit` правило действует |
| `RET504` | `week-*/**` | 2 | `text = ...; return text` — именованный шаг конвейера (`clean_pdf_text`) и пример области видимости (`lesson_12`); присваивание и есть смысл |
| `B006` | `week-01/lesson_11.py:85` (`noqa`) | 1 | это **демонстрация** ловушки, а не регрессия (задание 5) |
| `SIM110` | `week-02/lesson_04.py` (`noqa`) | 1 | `all(forward.setdefault(x, y) == y ...)` прячет побочный эффект в генераторе; явный цикл честнее |
| `F841` ×2 | `week-03/lesson_04.py` (`noqa`) | 2 | переменные `key` и `secret` нужны, чтобы их показала трассировка: в этом суть задания 5.3 урока 4 |
| весь файл | `week-03/lesson_05_demo.py` (`extend-exclude`) | 4 | семь намеренных ошибок задания 1; явный запуск `ruff check week-03/lesson_05_demo.py` его проверяет |

---

## 5. Три слабых места

### A004 — `week-03/lesson_01.py:7`

```python
from requests.exceptions import ConnectionError, HTTPError, ...   # было
```

**Чем грозит.** Имя `ConnectionError` в модуле теперь означает класс из requests, а не встроенный. Класс из requests **не** наследник встроенного: я проверил, `issubclass(...)` даёт `False`, его MRO — `RequestException → OSError`. Любой `except ConnectionError:` в этом файле, написанный в расчёте на встроенное исключение (`ConnectionResetError`, `BrokenPipeError` из сокетов или asyncio), перестаёт его ловить. Программа падает там, где по коду «всё обработано».

**Починка:** `from requests.exceptions import ConnectionError as RequestsConnectionError`.

### B006 — `week-01/lesson_11.py:85` (было `:78`)

```python
def add_tag(tag, tags=[]):
```

**Честный ответ: это не регрессия.** Это задание 7 урока Ф1-11, где функция написана **специально**, чтобы показать ловушку. Под ней идут вызовы и комментарий, почему третий вывод — `['a', 'b', 'd']`. Если её исправить, урок перестанет демонстрировать то, ради чего написан.

**Чем грозит в настоящем коде.** Список создаётся один раз при определении функции и общий для всех вызовов. Теги одного пользователя утекают в вызов другого.

**Как чинится:** `tags: list[str] | None = None` и `if tags is None: tags = []`. Здесь — `# noqa: B006` с причиной. Правило продолжает действовать во всём остальном коде.

### RET503 — `week-03/lesson_03.py`

1. **`retry_request`, строка 230 (было 212).** При `attempts=0` цикл не выполняется ни разу, и функция молча возвращает `None` вместо `Response`. Вызывающий упадёт дальше, на `AttributeError: 'NoneType' object has no attribute 'json'`, далеко от места причины. **Починка:** после цикла `raise ValueError(f"attempts должно быть ≥ 1, ...")`.
2. **`place_order_buggy`, строка 544 (было 475).** Если все три попытки получили 503, функция возвращает `None`. Код, который вызвал её ради заказа, получит `None` вместо ошибки и решит, что заказ оформлен. Это ровно «ждал вещь, пришло ничего». **Починка:** после цикла `raise RuntimeError("заказ не подтверждён за N попыток")`.

---

## 6. Настоящие ошибки типов

В `week-*` повторяются имена файлов (`lesson_01.py` в трёх неделях), а в именах каталогов есть дефисы. Поэтому mypy на всём репозитории одной командой не запускается. Я прогонял его группами: `ragkit todo demo.py`, `week-01`, `week-02`, `week-02/linux`, `week-03`.

| | до урока (`cca207a`) | после |
|---|---|---|
| mypy без настроек | 64 | 66: +3 намеренные в demo/any, −1 исправлено |
| `mypy --strict` | 984 | 893 |
| из них не `no-untyped-*` | **103** | 96 (92 без двух демо-файлов урока) |

Ошибка, исчезнувшая без настроек: `demo.py: Unsupported left operand type for <= ("Chunk")`. После аннотирования `ragkit` mypy понял `@total_ordering`, потому что `__lt__` теперь типизирован.

Настоящих ошибок до урока 103:

| Код | Сколько |
|---|---|
| `var-annotated` | 25 |
| `attr-defined` | 22 |
| `arg-type` | 14 |
| `assignment` | 10 |
| `call-arg` | 6 |
| `operator` | 5 |
| `union-attr` | 4 |
| `return-value` | 4 |
| прочее | 13 |

Это больше, чем 32 в уроке, потому что с тех пор добавились мои файлы уроков 2–4 недели 3.

| Категория | Где | Падающий вход |
|---|---|---|
| **union-attr** | `week-02/lesson_07.py:179` — `m.groupdict()`, где `m = LOG.fullmatch(...)` имеет тип `Match[str] \| None` | строка лога без времени, `"2026-10-19 ERROR ragkit: без времени"`: `fullmatch` вернёт `None` → `AttributeError: 'NoneType' object has no attribute 'groupdict'` (проверено). Сейчас не падает только потому, что на вход подаётся одна заведомо правильная строка. **mypy прав.** |
| **arg-type** | `week-02/lesson_04.py:361` — `total([1, 2, 3])` для функции под `@cache`: `expected "Hashable"` | `[1, 2, 3]` → `TypeError: unhashable type: 'list'` (проверено). В уроке это намеренная демонстрация в `try/except`; без `try` упало бы. **mypy прав.** |
| **attr-defined** | `week-01/lesson_09.py:89` — `response.get("usage", {}).get(...)`: `"Collection[Collection[str]]" has no attribute "get"` | **упасть не может**: `response["usage"]` — словарь, и `.get` у него есть. **mypy ошибается**: словарь-литерал со значениями разных типов (`str`, `dict`, `list[dict]`) он сводит к общему предку `Collection[...]`, и знание «это `dict`» теряется. Лечится `TypedDict` для ответа API. Похожий случай — `week-01/lesson_17.py:237` (`"Chunk" has no attribute "count"`): имя `a` сначала было `Chunk`, потом `Counter`. В рантайме всё работает, но одна переменная с двумя типами — то, что mypy запрещает, и по делу. |
| **call-arg** | `week-03/lesson_04.py:130` — `Settings(_env_file=env3)`: `Unexpected keyword argument "_env_file"` | **упасть не может**: `_env_file` — документированный аргумент `BaseSettings`. **mypy ошибается**: без плагина он не знает о динамическом `__init__` pydantic. С `plugins = ["pydantic.mypy"]` (задание 3) эта ошибка и ложные `arg-type` «`str` вместо `SecretStr`» исчезают (проверено). |

---

## 7. ragkit

`mypy --strict --warn-unreachable ragkit` до урока выдавал **70** ошибок. После — **`Success: no issues found in 6 source files`**, и `ruff check ragkit` тоже проходит чисто.

- **`Protocol`.** В `ragkit/embedding.py` — `Embedder` со свойством `dim` и методом `embed(texts: Sequence[str]) -> list[list[float]]`. `HashEmbedder`, детерминированный фейк для тестов, **не наследует** `Embedder`, но подходит структурно. Функция `embed_chunks(embedder: Embedder, ...)` примет и его, и будущий клиент OpenAI или локальной модели. А `WrongEmbedder`, у которого `embed` возвращает `list[float]`, mypy отвергает с ошибкой `arg-type`. Эта проверка лежит в `lesson_05.py` под `TYPE_CHECKING`.
- **`Literal`.** `SplitMode = Literal["words", "sentences"]` в новой функции `split_text(text, source, max_tokens, mode)`. Опечатка `"sentense"` упадёт в mypy, а не в рантайме.
- **`Sequence` на входе, `list` на выходе.** `largest(chunks: Sequence[Chunk], k) -> list[Chunk]`. В `Document.add_chunks(chunks: Iterable[Chunk])` хватает `Iterable`: функция только перебирает, поэтому подойдёт и генератор.
- **`TypedDict`.** `ChunkDict` и `Chunk.to_dict() -> ChunkDict`.
- **Прочее:**
  - `__eq__(self, other: object) -> bool` с `NotImplemented`;
  - `@overload` для `Document.__getitem__`: `int` → `Chunk`, `slice` → `list[Chunk]`;
  - `__all__` в `__init__.py`;
  - `Chunk.metadata: dict[str, object]`.

Поведение не изменилось: `demo.py`, `python -m ragkit`, `week-02/lesson_01.py` и `week-02/lesson_03.py` (оба импортируют `Chunk`) выводят то же, что раньше.

---

## 8. `should_retry` с типами (`week-03/lesson_05.py`)

Честная аннотация исходной функции:

```python
Outcome: TypeAlias = int | type[httpx.TransportError]

def should_retry(outcome: Outcome, method: str, idempotency_key: str | None = None) -> bool:
```

mypy теперь ловит:
- `"503"` вместо `503`;
- **экземпляр** `ReadTimeout("x")` вместо класса (в рантайме это `TypeError: issubclass() arg 1 must be a class`);
- `ValueError` вместо исключения httpx;
- опечатку в имени аргумента.

В файле эти вызовы стоят под `if TYPE_CHECKING:` с `# type: ignore[код]`. Благодаря `warn_unused_ignores` это доказывает, что mypy их ловит: если бы не ловил, подавление стало бы лишним, и проверка упала бы. Я проверил и обратным путём: убрал одно подавление, и mypy выдал `arg-type`.

**Стало ли понятнее? Отчасти.** Тип `int | type[TransportError]` честно говорит, что функция смешивает два разных мира: «сервер ответил» и «ответа не было». Это и есть сигнал о проектировании. К тому же `method: str` пропускает `"GTE"`, а исключение приходится передавать **классом**, хотя у вызывающего на руках экземпляр. Разделённый вариант:

```python
Method = Literal["GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE", "POST", "PATCH"]

def is_replay_safe(method: Method, idempotency_key: str | None) -> bool
def retry_on_status(status: int, method: Method, idempotency_key: str | None = None) -> bool
def retry_on_error(error: httpx.TransportError, method: Method, idempotency_key: str | None = None) -> bool
```

У каждой функции подпись в одну строку без объединения типов. `Literal` ловит `"GTE"`, а `retry_on_error` принимает само исключение. Ключ идемпотентности используется в одном месте, в `is_replay_safe`. С ключом повтор POST после `ReadTimeout` и после 500 разрешён, без ключа — нет.

---

## 9. pre-commit

`.pre-commit-config.yaml` подключает хуки `ruff-check --fix` и `ruff-format` из `astral-sh/ruff-pre-commit` версии `v0.17.3`. Установка: `pre-commit install`.

```
$ git add bad_example.py && git commit -m "проверка: заведомо плохой файл"
ruff check...............................................................Failed
- hook id: ruff-check
- exit code: 1
- files were modified by this hook

B006 Do not use mutable data structures for argument defaults
 --> bad_example.py:1:11
  |
1 | def f(x,y=[]):
  |           ^^
2 |   return x
  |
help: Replace with `None`; initialize within function

Found 3 errors (2 fixed, 1 remaining).
1 unsafe fix available with `--fix --unsafe-fixes`.

ruff format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook

1 file reformatted

git commit exit code: 1
```

Здесь `bad_example.py` — это `import os`, `import json` и `def f(x,y=[]): return x`.

- Хук сам убрал два неиспользуемых импорта («2 fixed»).
- Третью находку, `B006`, автоматически чинить небезопасно. Она осталась, и коммит **отклонён**: код возврата 1.
- `ruff-format` переформатировал файл.
- Правки хуков остались в рабочей копии и в коммит не попали. Их нужно посмотреть, сделать `git add` и повторить коммит.
- После демонстрации файл удалён.

**Почему mypy не в хуке.** Хук получает только изменённые файлы, а mypy нужен весь проект: ошибка `call-arg` может возникнуть в файле, который ты не трогал, если ты поменял подпись в другом. Кроме того, mypy работает секундами и требует установленных зависимостей, а хук должен быть мгновенным. Поэтому mypy держат в CI, где он один раз проверяет весь проект в чистом окружении.

---

## 10. Устно

- **Почему аннотация может врать.** Python аннотации не проверяет, он только сохраняет их в `__annotations__`. `def tokens(text: str) -> int` спокойно принимает `42` и падает уже на `.split()`. Без запуска mypy аннотация — это комментарий, который никто не сверяет с кодом. Код меняется, комментарий нет, и аннотация начинает сообщать ложное знание.
- **Почему `Any` опаснее отсутствия аннотации.** Отсутствие аннотации хотя бы видно, и `--strict` о нём скажет. `Any` выглядит как тип, но выключает проверку и **распространяется**: `Any.upper()` — тоже `Any`, `Any + 5` законно. Одна неаннотированная функция гасит ошибки в коде, который её вызывает (задание 2: `build_url` падает в рантайме, а mypy молчит). Если тип правда неизвестен, нужен `object`: он требует проверки перед использованием.
- **Почему `Sequence` на входе, а `list` на выходе.** `list` инвариантен: `list[int]` не подходит туда, где ждут `list[float]`, потому что функция могла бы положить в него `float`. `Sequence` доступна только для чтения и поэтому ковариантна: `Sequence[float]` принимает и `list[int]`, и `tuple`. На входе пишут самое общее, чего достаточно, — тогда функцию можно вызвать с чем угодно подходящим. На выходе пишут конкретное, чтобы вызывающий мог пользоваться всем, что умеет `list`.
- **Что находит ruff и не находит mypy, и наоборот.** ruff видит то, что заметно в одном файле без знания типов: неиспользуемые импорты и переменные, перекрытые встроенные имена, изменяемые умолчания, `zip` без `strict`, `raise` без `from`, форматирование. mypy строит модель типов всего проекта и находит `None` там, где ждут значение, не тот тип аргумента, не то число аргументов, несуществующий атрибут, недостижимый код. Они почти не пересекаются (задание 1), поэтому нужны оба.
- **Почему `# type: ignore` пишут с кодом.** Голый `# type: ignore` гасит **все** ошибки в строке, включая те, что появятся позже и будут настоящими. С кодом в скобках гасится ровно одна категория. А `warn_unused_ignores` сообщит, когда подавление стало лишним, иначе такие комментарии копятся годами.
