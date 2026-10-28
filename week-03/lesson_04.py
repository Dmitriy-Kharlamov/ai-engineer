import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import dotenv
import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

print(
    f"pydantic {version('pydantic')}, pydantic-settings {version('pydantic-settings')}, "
    f"python-dotenv {version('python-dotenv')}"
)

# Все опыты — во временной папке и с префиксом L4_, чтобы не задеть настоящий
# .env репозитория и переменные твоей оболочки.
TMP = Path(tempfile.mkdtemp(prefix="lesson04_"))


def write_env(name, text):
    path = TMP / name
    path.write_text(text, encoding="utf-8")
    return path


def clean_env(prefix="L4_"):
    for k in [k for k in os.environ if k.startswith(prefix)]:
        del os.environ[k]


# ---------------------------------------------------------------------------
# 1. Иерархия источников
# ---------------------------------------------------------------------------
print("\n1. Иерархия: конструктор > окружение > .env > умолчание")
env1 = write_env(
    ".env.task1", "L4_ONLY_FILE=из_файла\nL4_FILE_AND_ENV=из_файла\nL4_INIT_ARG=из_файла\n"
)


class Task1(BaseSettings):
    model_config = SettingsConfigDict(env_file=env1, env_prefix="L4_")
    only_default: str = "умолчание"
    only_file: str = "умолчание"
    file_and_env: str = "умолчание"
    init_arg: str = "умолчание"


clean_env()
os.environ["L4_FILE_AND_ENV"] = "из_окружения"
os.environ["L4_INIT_ARG"] = "из_окружения"

prediction = {  # прогноз — записан ДО запуска
    "only_default": "умолчание",  # нигде больше не задана
    "only_file": "из_файла",  # есть только в .env
    "file_and_env": "из_окружения",  # окружение старше файла
    "init_arg": "из_конструктора",  # аргумент кода старше всего
}
s = Task1(init_arg="из_конструктора")
print(f"   {'поле':14} | {'где задана':44} | {'прогноз':16} | результат")
where_set = {
    "only_default": "умолчание",
    "only_file": "умолчание + .env",
    "file_and_env": "умолчание + .env + окружение",
    "init_arg": "умолчание + .env + окружение + конструктор",
}
for name, expected in prediction.items():
    got = getattr(s, name)
    print(f"   {name:14} | {where_set[name]:44} | {expected:16} | {got}")
    assert got == expected
clean_env()
# Каждое значение берётся из самого старшего источника, где оно задано.
#   init_arg задан во всех четырёх — победил конструктор. file_and_env — в
#   файле и окружении, победило окружение. Прогноз совпал во всех четырёх.


# ---------------------------------------------------------------------------
# 2. override у load_dotenv
# ---------------------------------------------------------------------------
print("\n2. load_dotenv и override")
env2 = write_env(".env.task2", "L4_API_KEY=from_file\n")
os.environ["L4_API_KEY"] = "from_environ"
print("   в окружении:                    ", os.environ["L4_API_KEY"])
dotenv.load_dotenv(env2)
print("   после load_dotenv():            ", os.environ["L4_API_KEY"])
assert os.environ["L4_API_KEY"] == "from_environ"
dotenv.load_dotenv(env2, override=True)
print("   после load_dotenv(override=True):", os.environ["L4_API_KEY"])
assert os.environ["L4_API_KEY"] == "from_file"
clean_env()
# В проде настоящие значения приходят переменными окружения от Docker,
#   Kubernetes или CI, а .env там либо нет, либо он случайно попал в образ со
#   значениями разработчика. Поэтому файл по умолчанию только дополняет
#   окружение и никогда его не перекрывает — иначе забытый .env молча заменил
#   бы прод-ключ на тестовый.


# ---------------------------------------------------------------------------
# 3. Класс настроек
# ---------------------------------------------------------------------------
print("\n3. Settings для проекта")


class LLMSettings(BaseModel):
    model: str = "claude-sonnet-5-5"
    timeout: float = 30.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="L4_APP_", env_nested_delimiter="__", extra="ignore"
    )
    api_key: SecretStr
    database_url: str = "sqlite:///dev.db"
    max_retries: int = 3
    debug: bool = False
    llm: LLMSettings = LLMSettings()


env3 = write_env(
    ".env.task3",
    "L4_APP_API_KEY=sk-test-1234567890\n"
    "L4_APP_MAX_RETRIES=5\n"
    "L4_APP_DEBUG=yes\n"
    "L4_APP_LLM__TIMEOUT=60\n",
)
s = Settings(_env_file=env3)
print(f"   max_retries = {s.max_retries!r} ({type(s.max_retries).__name__})  ← из строки '5'")
print(f"   debug       = {s.debug!r} ({type(s.debug).__name__})  ← из строки 'yes'")
print(f"   llm         = {s.llm!r}  ← L4_APP_LLM__TIMEOUT=60, model по умолчанию")
print(f"   database_url = {s.database_url!r}  ← не задана, умолчание")
print(f"   api_key     = {s.api_key!r}")
assert s.max_retries == 5 and s.debug is True and s.llm.timeout == 60.0

try:
    Settings(_env_file=None)  # ни файла, ни переменной
except ValidationError as e:
    print("   без L4_APP_API_KEY приложение не стартует:")
    print("   " + str(e).replace("\n", "\n   "))
# Строки из .env превратились в типы поля при создании объекта: "5" → int 5,
#   "yes" → True (pydantic принимает true/1/yes/on). Вложенная группа llm
#   собрана из L4_APP_LLM__TIMEOUT: двойное подчёркивание — разделитель уровня.
#   Без обязательного api_key объект не создаётся — ошибка сразу при старте, а
#   не в момент первого запроса к API через час работы.


# ---------------------------------------------------------------------------
# 4. Утечка через ValidationError
# ---------------------------------------------------------------------------
print("\n4. Утечка ключа через ValidationError")
LEAKY_KEY = "sk-ОЧЕНЬ-СЕКРЕТНЫЙ-КЛЮЧ"


class Strict(BaseSettings):
    api_key: SecretStr = Field(min_length=40)
    port: int = 0


try:
    Strict(api_key=LEAKY_KEY, port="не число")
except ValidationError as e:
    text = str(e)
    print("   " + text.replace("\n", "\n   "))
    print(f"   → ключ в тексте ошибки: {LEAKY_KEY in text}")
    assert LEAKY_KEY in text


class StrictSafe(BaseSettings):
    model_config = SettingsConfigDict(hide_input_in_errors=True)
    api_key: SecretStr = Field(min_length=40)
    port: int = 0


try:
    StrictSafe(api_key=LEAKY_KEY, port="не число")
except ValidationError as e:
    text = str(e)
    print("   с hide_input_in_errors=True:")
    print("   " + text.replace("\n", "\n   "))
    print(f"   → ключ в тексте ошибки: {LEAKY_KEY in text}")
    print(f"   → но в e.errors()[0]['input']: {e.errors()[0]['input']!r}")
    print(f"   → e.errors(include_input=False): {e.errors(include_input=False)[0].keys()}")
    assert LEAKY_KEY not in text
# Почему утекает: валидация идёт ДО того, как значение станет SecretStr, и
#   pydantic показывает вход как есть (input_value=...). Маскирует только
#   готовый объект, а его ещё нет.
# Выбранный способ — hide_input_in_errors=True в конфигурации модели.
#   Что даёт: из текста ошибки (str(e) — того, что попадает в трассировку и в
#   лог) пропадает input_value у ВСЕХ полей. Ключ больше не печатается.
#   Чего не даёт:
#   1) структурированный e.errors() по-прежнему содержит 'input' с ключом
#      (видно в выводе). Если код логирует или отдаёт клиенту e.errors() —
#      а FastAPI так делает в ответе 422, — ключ утечёт. Там нужен
#      e.errors(include_input=False).
#   2) скрывается вход и у безобидных полей: про port тоже не видно, что
#      пришло «не число», — отладка становится труднее.
#   3) другие пути утечки (локальные переменные в трассировке, URL, логи)
#      не закрывает — задание 5.


# ---------------------------------------------------------------------------
# 5. Пути утечки
# ---------------------------------------------------------------------------
print("\n5. Пути утечки")
KEY = "sk-live-0123456789abcdef"

# 5.1 Печать объекта целиком
print("   5.1 печать объекта")


@dataclass
class PlainConfig:
    api_key: str
    debug: bool = False


@dataclass
class FixedConfig:
    api_key: str = field(repr=False)  # не участвует в repr
    debug: bool = False


print(f"      dataclass с str:      {PlainConfig(KEY)}")
print(f"      field(repr=False):    {FixedConfig(KEY)}")
print(f"      Settings + SecretStr: {Settings(api_key=KEY, _env_file=None)}")
assert KEY in repr(PlainConfig(KEY)) and KEY not in repr(FixedConfig(KEY))
# logger.info("config: %s", cfg) или print(cfg) в отладке печатают repr. Обычный
#   str печатается целиком. Закрыть: SecretStr (как в Settings) или
#   field(repr=False) в dataclass.

# 5.2 Ключ в строке запроса
print("   5.2 ключ в URL")
fake_api = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
r = fake_api.get("https://api.example.com/v1/models", params={"api_key": KEY})
print(f"      в params:  r.request.url = {r.request.url}")
r = fake_api.get("https://api.example.com/v1/models", headers={"Authorization": f"Bearer {KEY}"})
print(f"      в заголовке: r.request.url = {r.request.url}")
assert KEY not in str(r.request.url)
# MockTransport отвечает 200 без сети — запрос строится по-настоящему. URL с
#   ключом пишется в лог доступа сервера, в логи прокси, в историю браузера, в
#   Referer, да и в наш собственный лог, если мы логируем r.request.url или
#   текст исключения (httpx пишет URL в сообщение HTTPStatusError). Закрыть:
#   ключ только в заголовке Authorization.

# 5.3 Трассировка с локальными переменными
print("   5.3 трассировка с локальными переменными")


def call_api_leaky(settings):
    key = settings.api_key.get_secret_value()  # noqa: F841 — распакован; нужен трассировке
    raise ConnectionError("API недоступен")


def call_api_safe(settings):
    secret = settings.api_key  # noqa: F841 — остаётся SecretStr; нужен трассировке
    raise ConnectionError("API недоступен")


cfg = Settings(api_key=KEY, _env_file=None)
for fn in (call_api_leaky, call_api_safe):
    try:
        fn(cfg)
    except ConnectionError as e:
        tb = "".join(traceback.TracebackException.from_exception(e, capture_locals=True).format())
        line = next(
            ln.strip() for ln in tb.splitlines() if ln.strip().startswith(("key =", "secret ="))
        )
        print(f"      {fn.__name__:15} локальная переменная в отчёте: {line}")
        assert (KEY in line) == (fn is call_api_leaky)
# capture_locals=True делает то же, что Sentry и rich.traceback(show_locals=True):
#   печатает локальные переменные КАЖДОГО кадра (включая кадр модуля, где
#   локальные = глобальные: поэтому и константа KEY на уровне модуля попала бы в
#   отчёт — секреты не держат в глобальных переменных). Распакованный ключ виден
#   целиком. Закрыть: держать SecretStr до последнего момента и распаковывать
#   прямо в выражении, которое его отправляет; в Sentry — включить очистку
#   данных (send_default_pii=False, before_send).

# 5.4 и 5.5 — без кода:
#   5.4 ValidationError с input_value — задание 4.
#   5.5 git и история оболочки — задание 6 и Ф3-01: коммит с .env навсегда
#       остаётся в истории; `export API_KEY=...` или `curl ...?token=...`
#       оседают в ~/.zsh_history. Закрыть: .gitignore + .env.example; ключи
#       не набирать в команде, а читать из файла или менеджера секретов.
#   Ещё один путь из урока: ENV API_KEY=... в Dockerfile остаётся в слое образа.


# ---------------------------------------------------------------------------
# 6. git помнит
# ---------------------------------------------------------------------------
print("\n6. git помнит")
repo = TMP / "leaky_repo"
repo.mkdir()


def git(*args):
    out = subprocess.run(
        ["git", "-c", "user.name=demo", "-c", "user.email=demo@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


git("init", "-q")
(repo / ".env").write_text("API_KEY=sk-это-секрет-из-коммита\n", encoding="utf-8")
(repo / "app.py").write_text("print('hi')\n", encoding="utf-8")
git("add", ".")
git("commit", "-q", "-m", "first: app + .env (ошибка)")
git("rm", "-q", "--cached", ".env")
(repo / ".gitignore").write_text(".env\n", encoding="utf-8")
git("add", ".gitignore")
git("commit", "-q", "-m", "remove .env from index")

print("   $ git log --oneline\n      " + git("log", "--oneline").replace("\n", "\n      "))
print(
    "   $ git ls-files   (что в последнем коммите)\n      "
    + git("ls-files").replace("\n", "\n      ")
)
first = git("rev-list", "--max-parents=0", "HEAD")
leaked = git("show", f"{first}:.env")
print(f"   $ git show {first[:7]}:.env\n      {leaked}")
print(
    "   $ git log --all --full-history --oneline -- .env\n      "
    + git("log", "--all", "--full-history", "--oneline", "--", ".env").replace("\n", "\n      ")
)
assert "sk-это-секрет-из-коммита" in leaked
# git rm --cached убрал .env только из СЛЕДУЮЩИХ коммитов: в последнем его нет
#   (ls-files), но первый коммит неизменен, и git show <коммит>:.env достаёт
#   секрет одной командой. Так же он лежит в каждом клоне, форке и кэше GitHub.
# Порядок в реальности: 1) сразу отозвать ключ у провайдера и выпустить новый;
#   2) положить новый туда, где ему место (окружение, менеджер секретов);
#   3) только потом, если нужно, чистить историю (git filter-repo / BFG) и
#   просить всех переклонировать. Почему так: пока ключ действует, его может
#   использовать любой, у кого есть копия истории, — а чистка не дотянется до
#   чужих клонов. Отозванный ключ безвреден, где бы он ни лежал.


# ---------------------------------------------------------------------------
# 7. Репозиторий: ragkit/config.py
# ---------------------------------------------------------------------------
print("\n7. ragkit.config")
try:
    from ragkit.config import get_settings

    print("   get_settings() is get_settings():", get_settings() is get_settings())
    print("   ", get_settings())
except ImportError as e:
    print("   ragkit.config не найден — сначала pip install -e . (см. задание 7):", e)
except ValidationError as e:
    print("   .env не заполнен — приложение честно не стартует:")
    print("   " + str(e).replace("\n", "\n   "))
# get_settings() обёрнута в lru_cache: объект создаётся один раз при первом
#   вызове, дальше возвращается тот же (is → True). Тесты подменяют настройки,
#   вызывая get_settings.cache_clear() и задавая переменные окружения.


# ---------------------------------------------------------------------------
# 8. Устно, потом письменно
# ---------------------------------------------------------------------------
# Почему окружение старше .env: в проде значения задаёт оркестратор через
#   переменные окружения, а .env — удобство разработчика. Если бы файл был
#   старше, случайно попавший в образ .env перекрыл бы настоящие прод-значения.
# Что защищает SecretStr: только от случайной ПЕЧАТИ готового объекта — repr,
#   str, print(settings), model_dump_json, показ в трассировке. От чего не
#   защищает: это не шифрование, значение лежит в памяти обычной строкой;
#   ValidationError показывает вход до превращения в SecretStr (задание 4);
#   распакованный get_secret_value() в локальной переменной виден в
#   трассировке с locals (задание 5.3); URL и git он не трогает вовсе.
# Почему chmod 600 не защищает секрет в репозитории: git хранит из прав только
#   бит исполняемости. 600 защищает файл на моём диске, но в коммит уходит
#   содержимое, и у каждого клона файл будет с обычными правами. Защищает
#   только то, что секрета в репозитории нет.
# Почему ключ отзывают раньше чистки истории: чистка не дотягивается до уже
#   сделанных клонов, форков и кэшей и занимает время — всё это время ключ
#   работает. Отозванный ключ безвреден, поэтому сначала отзыв, потом уборка.
# Конфигурация против константы: конфигурация — то, что различается между
#   окружениями (ключи, адрес базы, debug). Если значение одинаково на ноутбуке,
#   в тестах и в проде — это константа в коде. Каждая лишняя настройка — ещё
#   одна вещь, которую могут забыть задать в проде.
