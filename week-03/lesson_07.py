"""Ф3-07. Pydantic: валидация и схемы.

Запуск: python week-03/lesson_07.py
Файл входит в [tool.mypy] files: assert_type в задании 5 проверяет mypy.
Прогнозы в заданиях 1 и 2 записаны в коде ДО первого запуска; скрипт сверяет их с
реальностью и печатает, где я ошибся.
"""

import dataclasses
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self, assert_type

from fastapi.testclient import TestClient
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from api.main import create_app
from api.schemas import DocumentIn, SourceName, SplitIn
from ragkit import Chunk, ChunkDict
from ragkit.config import Settings

# Ф3-11: при debug=False без jwt_secret Settings отказывается стартовать
app = create_app(
    Settings(
        api_key=SecretStr("lesson-key"),
        jwt_secret=SecretStr("lesson-jwt-secret-at-least-32-bytes!!"),
        _env_file=None,
    )
)


def first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    return f"ОШИБКА {err['type']} loc={err['loc']}"


def attempt(make: Any) -> str:
    try:
        return repr(make())
    except ValidationError as e:
        return first_error(e)


# ===========================================================================
# 1. Три дефолта
# ===========================================================================
def task1() -> None:
    print("\n1. Три дефолта")

    class Lax(BaseModel):
        max_tokens: int = 100

    class Forbid(BaseModel):
        model_config = ConfigDict(extra="forbid")
        max_tokens: int = 100

    print("   а) опечатка max_token=5")
    print("      прогноз: Lax — max_tokens=100 молча; Forbid — extra_forbidden")
    print("      Lax:   ", attempt(lambda: Lax.model_validate({"max_token": 5})))
    print("      Forbid:", attempt(lambda: Forbid.model_validate({"max_token": 5})))
    # Больно: клиент API шлёт max_token=5, получает 200 и результат с лимитом 100 — и
    # месяц думает, что лимит работает.

    class MutLax(BaseModel):
        n: int = 0

    class Mut(BaseModel):
        model_config = ConfigDict(validate_assignment=True)
        n: int = 0

    print("   б) присваивание после создания: obj.n = 'не число'")
    print("      прогноз: MutLax — присвоится молча; Mut — int_parsing")
    ml = MutLax()
    ml.n = "не число"  # type: ignore[assignment]  # mypy видит, pydantic — нет
    print(f"      MutLax: n={ml.n!r}")
    m = Mut()
    try:
        m.n = "не число"  # type: ignore[assignment]
    except ValidationError as e:
        print("      Mut:   ", first_error(e))
    # Больно: долгоживущий объект (настройки, состояние сессии) испорчен присваиванием в
    # одном месте, а падает через час в другом — где этот n впервые используют как число.

    class Strict(BaseModel):
        model_config = ConfigDict(strict=True)
        n: int

    class LaxN(BaseModel):
        n: int

    print("   в) strict: n='5'")
    print("      прогноз: lax — 5; strict — int_type")
    print("      lax:   ", attempt(lambda: LaxN.model_validate({"n": "5"})))
    print("      strict:", attempt(lambda: Strict.model_validate({"n": "5"})))
    # Больно (в обратную сторону): включить strict на модели JSON-API — и все клиенты,
    # которые шлют числа из HTML-формы строками, внезапно получают 422.


# ===========================================================================
# 2. Таблица приведений
# ===========================================================================
def task2() -> None:
    print("\n2. Таблица приведений (lax)")
    cases: list[tuple[str, Any, Any, str]] = [
        # (тип, вход, тип-адаптер, ПРОГНОЗ — записан до запуска)
        ("int", "5", int, "5"),
        ("int", 5.0, int, "5"),
        ("int", 5.5, int, "ОШИБКА int_from_float"),
        ("int", True, int, "1"),
        ("str", 5, str, "ОШИБКА string_type"),
        ("str", True, str, "ОШИБКА string_type"),
        ("bool", "yes", bool, "True"),
        ("bool", "on", bool, "True"),
        ("bool", 1, bool, "True"),
        ("bool", "maybe", bool, "ОШИБКА bool_parsing"),
        # три своих
        ("int", "  7  ", int, "7"),
        ("int", "1_000", int, "ОШИБКА int_parsing"),
        ("bool", "Да", bool, "ОШИБКА bool_parsing"),
    ]
    misses = 0
    for type_name, value, typ, predicted in cases:
        try:
            got = repr(TypeAdapter(typ).validate_python(value))
        except ValidationError as e:
            got = f"ОШИБКА {e.errors()[0]['type']}"
        mark = "✓" if got == predicted else "✗ ПРОМАХ"
        misses += got != predicted
        print(f"   {type_name:4} <- {value!r:8} прогноз {predicted:22} факт {got:22} {mark}")
    print(f"   промахов: {misses}")


# int <- True → 1: в Python bool — подкласс int (True == 1, isinstance(True, int) — True).
#   Pydantic в lax принимает подкласс как есть. Флаг, пришедший вместо числа, тихо
#   становится 1 — это стоит помнить для полей вроде limit или max_tokens.
# str <- 5 — ошибка: в pydantic v2 lax-приведение идёт только «из строки» (строку
#   разбирают в число, дату, bool), но не «в строку». Иначе любой объект превращался бы
#   в строку через str() — так было в v1 и маскировало ошибки клиента.
# Мой промах — '1_000' → 1000. Я ждал ошибку, а pydantic разбирает целые как Python
#   int(): подчёркивания-разделители разрядов разрешены (int('1_000') == 1000). '  7  ' →
#   7: пробелы по краям обрезаются. 'Да' — ошибка: bool принимает только английские
#   слова (true/yes/on/1 и наоборот).


# ===========================================================================
# 3. Валидатор, который не течёт
# ===========================================================================
SECRET = "sk-test-СЕКРЕТНЫЙ-123"


class KeyLeakyMsg(BaseModel):
    key: SecretStr

    @field_validator("key")
    @classmethod
    def check(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if not raw.startswith("sk-proj-"):
            raise ValueError(f"ключ {raw} должен начинаться с sk-proj-")  # течёт через msg
        return v


class KeyLeakyCtx(BaseModel):
    key: SecretStr

    @field_validator("key")
    @classmethod
    def check(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if not raw.startswith("sk-proj-"):
            # msg без значения, но значение подложено в ctx — тоже утечка
            raise PydanticCustomError("bad_key_prefix", "неверный префикс ключа", {"got": raw})
        return v


class KeySafe(BaseModel):
    key: SecretStr

    @field_validator("key")
    @classmethod
    def check(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().startswith("sk-proj-"):
            raise PydanticCustomError("bad_key_prefix", "ключ должен начинаться с sk-proj-")
        return v


def leaks(err: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "msg": SECRET in str(err.get("msg", "")),
        "input": SECRET in str(err.get("input", "")),
        "ctx": SECRET in str(err.get("ctx", "")),
    }


def task3() -> None:
    from api.errors import safe_error

    print("\n3. Валидатор, который не течёт (сырые e.errors() и после обработчика 422)")
    for model in (KeyLeakyMsg, KeyLeakyCtx, KeySafe):
        try:
            model.model_validate({"key": SECRET})
        except ValidationError as e:
            raw = e.errors()[0]
            served = safe_error(raw)
            print(f"   {model.__name__:12} сырые: {leaks(raw)}")
            print(f"   {'':12} в ответе: {leaks(served)} → {served}")
    with TestClient(app) as c:
        r = c.post("/chunks", json={"text": "a b", "source": "../etc/passwd"})
        print(f"   настоящий API, плохой source: {r.status_code} {r.json()}")


# Три пути утечки и чем каждый закрыт:
#   msg   — ValueError с f-строкой кладёт ключ в текст. Обработчик тут бессилен: msg нужен
#           клиенту, его не выбросить. Закрывает только валидатор — PydanticCustomError с
#           текстом БЕЗ значения (KeySafe: msg чист уже в сырых ошибках).
#   input — у ВСЕХ трёх моделей сырые ошибки содержат ключ в input: валидатор на это не
#           влияет, настройкой модели не отключить. Закрывает обработчик: input нет в
#           белом списке (в ответе input: False).
#   ctx   — KeyLeakyCtx подложил значение в context. Да и обычный ValueError течёт сюда же:
#           pydantic кладёт в ctx сам объект исключения ({"error": ValueError(...)}), а его
#           текст содержит ключ (у KeyLeakyMsg сырой ctx: True). Обработчик из Ф3-06
#           вырезал только input, и ctx ушёл бы клиенту. Теперь ctx тоже по белому списку:
#           проходят лишь числовые ограничения (min_length, ge, …), остальное — нет.
# Итог для KeySafe: в ответе ни msg, ни input, ни ctx не содержат ключа, а type
#   bad_key_prefix, loc ('key',) и понятный msg остались — клиент знает, что исправить.
# Для KeyLeakyMsg утечка через msg остаётся даже после обработчика — поэтому слоёв два.


# ===========================================================================
# 4. Пустой loc у model_validator
# ===========================================================================
class RangeModel(BaseModel):
    lo: int
    hi: int

    @model_validator(mode="after")
    def order(self) -> Self:
        if self.lo > self.hi:
            raise ValueError("lo не может быть больше hi")
        return self


class RangeField(BaseModel):
    lo: int
    hi: int

    @field_validator("hi")
    @classmethod
    def hi_not_below_lo(cls, hi: int, info: Any) -> int:
        lo = info.data.get("lo")  # поля проверяются по порядку: lo уже проверен
        if lo is not None and hi < lo:
            raise PydanticCustomError("range_order", "hi должно быть не меньше lo")
        return hi


def task4() -> None:
    print("\n4. loc у межполевой ошибки")
    for model in (RangeModel, RangeField):
        try:
            model(lo=10, hi=1)
        except ValidationError as e:
            err = e.errors()[0]
            print(f"   {model.__name__:10}: type={err['type']} loc={err['loc']} msg={err['msg']!r}")


# model_validator(mode="after") проверяет модель целиком, и ошибка относится к ней, а не
#   к полю: loc=(). В 422 это loc ["body"] — форма не может подсветить ни lo, ни hi,
#   клиенту остаётся гадать. field_validator на hi видит уже проверенный lo через
#   info.data (поля валидируются в порядке объявления) и даёт loc=('hi',) — подсвечивается
#   именно поле, которое надо поправить.


# ===========================================================================
# 5. Валидатор как тип
# ===========================================================================
def task5() -> None:
    print("\n5. SourceName = Annotated[str, AfterValidator(check_source_name)]")
    split = SplitIn(text="текст", source="  manual.pdf  ")
    doc = DocumentIn(name="report.md", texts=[])
    assert_type(split.source, str)  # mypy проверяет: тип поля — str
    assert_type(doc.name, str)
    print(f"   SplitIn.source: {split.source!r} (пробелы обрезаны валидатором)")
    print("   DocumentIn.name='../x':", attempt(lambda: DocumentIn(name="../x", texts=[])))
    print("   SplitIn.source='a/b':  ", attempt(lambda: SplitIn(text="t", source="a/b")))

    class Another(BaseModel):  # третья модель — то же правило без копирования метода
        file: SourceName

    print("   Another.file='':       ", attempt(lambda: Another(file="   ")))


# Одно правило, три модели. Для mypy SourceName — это str: assert_type проходит, а
#   присваивание int в такое поле mypy отвергнет как обычный str. Для pydantic это str
#   плюс валидатор. Метод @field_validator пришлось бы копировать в каждый класс.


# ===========================================================================
# 6. extra="forbid" на границе
# ===========================================================================
def task6() -> None:
    print("\n6. extra='forbid' на моделях входа API")
    with TestClient(app) as c:
        r = c.post("/chunks", json={"text": "a b c", "source": "s", "max_token": 1})
        print(f"   POST /chunks с max_token: {r.status_code} {r.json()}")
    from fastapi import FastAPI

    class Out(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str

    class Inner(BaseModel):
        name: str
        password_hash: str

    probe = FastAPI()

    @probe.get("/obj", response_model=Out)
    def obj() -> Inner:
        return Inner(name="d", password_hash="h")

    @probe.get("/dict", response_model=Out)
    def as_dict() -> dict[str, str]:
        return {"name": "d", "password_hash": "h"}

    with TestClient(probe, raise_server_exceptions=False) as c:
        for path in ("/obj", "/dict"):
            r = c.get(path)
            print(f"   forbid на модели ОТВЕТА, GET {path}: {r.status_code} {r.text}")


# Опечатка max_token теперь даёт 422 extra_forbidden с loc ["body", "max_token"] — клиент
#   видит ровно своё неверное поле.
# На модель ответа forbid ставить не стоит, и это видно на опыте. Для объекта FastAPI
#   берёт из него только поля модели ответа: лишнее поле просто не прочитано, forbid ничего
#   не проверяет (200). Для словаря forbid срабатывает и превращает «лишний ключ в моём же
#   ответе» в 500 вместо фильтрации. То есть он либо бесполезен, либо ломает ту самую
#   фильтрацию из Ф3-06. Строгость нужна там, где данные приходят снаружи. Наш ответ —
#   данные, которые мы контролируем; а вот КЛИЕНТЫ нашего API, наоборот, должны терпеть
#   новые поля в ответе, иначе любое добавление поля сломает их.


# ===========================================================================
# 7. frozen вместо двух дандеров
# ===========================================================================
class FrozenChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    source: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens(self) -> int:
        return len(self.text.split())


def task7() -> None:
    print("\n7. frozen=True против ручных __eq__/__hash__ из ragkit (случай из Ф2-03)")
    a, b, c = ("привет мир", "a.pdf"), ("привет мир", "a.pdf"), ("пока всем", "b.pdf")
    manual = [Chunk(t, s) for t, s in (a, b, c)]
    frozen = [FrozenChunk(text=t, source=s) for t, s in (a, b, c)]
    for name, (x, y, z) in (("ragkit.Chunk", manual), ("FrozenChunk", frozen)):
        print(
            f"   {name:12}: a==b {x == y}, a==c {x == z}, hash(a)==hash(c) "
            f"{hash(x) == hash(z)}, len({{a, b, c}}) = {len({x, y, z})}"
        )
    try:
        frozen[0].text = "другой"  # type: ignore[misc]  # mypy тоже запрещает
    except ValidationError as e:
        print("   присваивание:", first_error(e))

    class FrozenWithDict(BaseModel):
        model_config = ConfigDict(frozen=True)
        text: str
        metadata: dict[str, str] = {}

    try:
        hash(FrozenWithDict(text="t"))
    except TypeError as e:
        print("   frozen + поле dict, hash():", e)


# Ручные __eq__/__hash__ в ragkit ведут себя НЕ так же, и это главная находка задания.
#   После Ф3-05 у ragkit.Chunk поменялась подпись (`other: object`, NotImplemented), но не
#   смысл: равенство и хеш по-прежнему по числу токенов. «привет мир» из a.pdf и «пока
#   всем» из b.pdf равны, и set склеивает три чанка в один — ровно авария из Ф2-03,
#   она в пакете до сих пор (её же фиксирует assert в week-02/lesson_03.py).
# FrozenChunk сравнивает по всем полям: a == b (тот же текст и источник), a != c, в set
#   два элемента. __eq__ и __hash__ построены из одних и тех же полей и разойтись не могут.
#   Присваивание — frozen_instance.
# Оговорка: frozen хеширует ВСЕ поля, и поле dict (metadata у ragkit.Chunk) делает объект
#   нехешируемым — TypeError. Выбрать поля для хеша pydantic не даёт. Это аргумент в
#   задании 10.


# ===========================================================================
# 8. Размеченное объединение
# ===========================================================================
class WordsCfg(BaseModel):
    mode: Literal["words"]
    max_tokens: int = Field(ge=1)


class SentCfg(BaseModel):
    mode: Literal["sentences"]
    max_sentences: int = Field(ge=1)


Cfg = Annotated[WordsCfg | SentCfg, Field(discriminator="mode")]
Plain: TypeAdapter[WordsCfg | SentCfg] = TypeAdapter(WordsCfg | SentCfg)
Tagged: TypeAdapter[WordsCfg | SentCfg] = TypeAdapter(Cfg)


def task8() -> None:
    print("\n8. Объединение с дискриминатором и без")
    bad = {"mode": "words", "max_tokens": 0}
    for name, ta in (("без дискриминатора", Plain), ("с дискриминатором", Tagged)):
        try:
            ta.validate_python(bad)
        except ValidationError as e:
            print(f"   {name}: {e.error_count()} ош., loc: {[err['loc'] for err in e.errors()]}")
    try:
        Tagged.validate_python({"mode": "paragraphs", "max_tokens": 5})
    except ValidationError as e:
        err = e.errors()[0]
        print(f"   неизвестный режим: {err['type']} — {err['msg']}")
    print("   правильный вход:", Tagged.validate_python({"mode": "sentences", "max_sentences": 3}))


# Без разметки pydantic пробует обе ветки и возвращает ошибки обеих — у меня 3, а не 2,
#   как в уроке: «max_tokens ≥ 1» из WordsCfg, а из SentCfg и «mode должен быть
#   'sentences'», и «нет max_sentences». Клиент получает шум про поля, которые к его
#   запросу не относятся. С дискриминатором pydantic сначала читает mode, выбирает ветку
#   words и сообщает одну ошибку с loc ('words', 'max_tokens'). Неизвестный тег — сразу
#   union_tag_invalid со списком допустимых.


# ===========================================================================
# 9. TypeAdapter на пачке
# ===========================================================================
class ChunkRecord(BaseModel):
    text: str
    source: str
    tokens: int = Field(ge=0)


def best_of(func: Any, runs: int = 7) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def task9() -> None:
    print("\n9. TypeAdapter(list[ChunkRecord]) против цикла, 10 000 записей")
    rows = [{"text": f"чанк {i}", "source": "a.pdf", "tokens": i % 50} for i in range(10_000)]
    adapter = TypeAdapter(list[ChunkRecord])
    t_adapter = best_of(lambda: adapter.validate_python(rows))
    t_loop = best_of(lambda: [ChunkRecord.model_validate(r) for r in rows])
    print(f"   адаптер: {t_adapter * 1000:.1f} мс | цикл: {t_loop * 1000:.1f} мс")

    broken = list(rows)
    broken[5000] = {"text": "битая", "source": "a.pdf", "tokens": "много"}
    broken[7000] = {"text": "битая", "source": "a.pdf", "tokens": -1}
    try:
        adapter.validate_python(broken)
    except ValidationError as e:
        print(
            f"   адаптер: одна ValidationError, ошибок {e.error_count()}, "
            f"loc: {[err['loc'] for err in e.errors()]}"
        )
    good = []
    try:
        for r in broken:
            good.append(ChunkRecord.model_validate(r))
    except ValidationError as e:
        print(
            f"   цикл: упал на записи №{len(good)}, loc: {e.errors()[0]['loc']}, "
            f"до 7000 не дошёл; успели {len(good)} записей"
        )


# Время: разница невелика (pydantic-core на Rust, цикл в Python не катастрофа).
# Поведение — главное: адаптер проверил всю пачку и вернул ОДНУ ошибку со всеми проблемами,
#   loc начинается с индекса записи — (5000, 'tokens') и (7000, 'tokens'). Цикл упал на
#   первой битой записи: про вторую мы не узнали, а 5000 хороших остались в недоделанном
#   списке. Чтобы собрать все ошибки циклом, нужен ручной try/except вокруг каждой
#   записи и свой формат отчёта.


# ===========================================================================
# 10. Проектное решение: ragkit.Chunk — pydantic-модель?
# ===========================================================================
class PydChunk(BaseModel):
    text: str
    source: str
    metadata: dict[str, object] = Field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DataChunk:
    text: str
    source: str
    metadata: dict[str, object] = dataclasses.field(default_factory=dict, compare=False, hash=False)


def task10() -> None:
    print("\n10. Цена создания 100 000 чанков")
    n = 100_000
    t_class = best_of(lambda: [Chunk(f"слово {i}", "a.pdf") for i in range(n)], runs=5)
    t_data = best_of(lambda: [DataChunk(f"слово {i}", "a.pdf") for i in range(n)], runs=5)
    t_pyd = best_of(lambda: [PydChunk(text=f"слово {i}", source="a.pdf") for i in range(n)], runs=5)
    print(
        f"   ragkit.Chunk: {t_class * 1e6 / n:.2f} мкс/шт | frozen dataclass: "
        f"{t_data * 1e6 / n:.2f} мкс/шт | pydantic: {t_pyd * 1e6 / n:.2f} мкс/шт "
        f"(pydantic / класс = {t_pyd / t_class:.1f}×)"
    )
    print("   hash(DataChunk) с metadata-словарём:", hash(DataChunk("t", "s", {"page": 1})) != 0)
    out = TypeAdapter(list[ChunkDict]).dump_json([Chunk("a b", "x.pdf").to_dict()])
    print("   to_dict() → JSON через TypeAdapter:", out.decode())


# Решение: НЕТ, ragkit.Chunk не становится pydantic-моделью. Обоснование — через то, где
#   ragkit будет жить в Ф5:
#   API — у него свои модели границы (SplitIn, ChunkOut) с extra="forbid", алиасами и
#     примерами. Chunk наружу и так не отдаётся: ChunkOut.model_validate(chunk) читает
#     атрибуты (from_attributes, сделано в api/schemas.py). Pydantic внутри Chunk API
#     ничего не добавит.
#   Воркер — сотни тысяч чанков на документ, и сообщения в очереди. Сообщение из очереди —
#     это ГРАНИЦА, как запрос в API: его валидирует своя модель или TypeAdapter (задание 9),
#     один раз на входе. Дальше чанки создаёт наш же код из уже проверенных строк —
#     валидировать их второй раз незачем, а платить пришлось бы на каждом чанке: по замеру
#     pydantic-чанк создаётся в несколько раз дольше нынешнего класса (отношение — в
#     выводе; абсолютные мкс плавают от запуска к запуску). На 100 000 чанков это десятые
#     доли секунды — немного рядом с эмбеддингами, поэтому скорость здесь довод
#     второстепенный, а не решающий.
#   CLI и ноутбук — Chunk("текст", "a.pdf") позиционными аргументами: так он создаётся в
#     demo.py, python -m ragkit и уроках недели 2. BaseModel позиционные аргументы не
#     принимает — пришлось бы писать свой __init__ поверх pydantic или переписывать всех
#     вызывающих.
#   Довод «за» про frozen и сагу __eq__/__hash__ — сильный, но его закрывает и
#     стандартный frozen dataclass (в моих прогонах от 1,5 до 2,6 раза дороже обычного
#     класса, потому что пишет поля через object.__setattr__), причём лучше: он позволяет
#     исключить metadata из сравнения и хеша (compare=False, hash=False), а frozen
#     pydantic с полем dict нехешируем вовсе (задание 7). Зависимость от pydantic, наоборот, не довод «против»:
#     ragkit уже тянет её через config.py.
# Что делать с to_dict() и сериализацией: оставить to_dict() -> ChunkDict (TypedDict из
#   Ф3-05) как единственный формат «чанк как данные». JSON — через TypeAdapter(ChunkDict)
#   или json.dumps; на границе API — ChunkOut.model_validate(chunk); в ноутбуке —
#   pd.DataFrame([c.to_dict() for c in chunks]).
# Следующий шаг, который я предлагаю отдельно: перевести ragkit.Chunk на frozen dataclass
#   с равенством по (text, source) и metadata вне сравнения. Это и есть окончательная
#   починка Ф2-03, но она меняет поведение: assert в week-02/lesson_03.py, фиксирующий
#   склейку set, перестанет выполняться. Поэтому это отдельный коммит с явным решением,
#   а не побочный эффект урока про pydantic.


# ===========================================================================
# Алиасы, computed_field, два model_dump — для отчёта
# ===========================================================================
class AliasDoc(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    text: str = Field(
        validation_alias=AliasChoices("text", "content", "body"),
        serialization_alias="text",
        description="Текст документа",
    )
    tokens: int
    when: datetime
    key: SecretStr | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def density(self) -> float:
        return round(self.tokens / max(len(self.text), 1), 3)


def extras() -> None:
    print("\nДоп.: алиасы и два model_dump")
    d = AliasDoc.model_validate(
        {
            "content": "привет мир",
            "tokens": 2,
            "when": datetime(2026, 10, 30, tzinfo=UTC),
            "key": "sk-x",
        }
    )
    print("   принял 'content':", d.text, "| density:", d.density)
    py = d.model_dump()
    js = d.model_dump(mode="json")
    print(f"   mode=python: when {type(py['when']).__name__}, key {py['key']!r}")
    print(f"   mode=json:   when {type(js['when']).__name__}, key {js['key']!r}")


# ===========================================================================
# 11. Устно, потом письменно
# ===========================================================================
# Почему валидация на границе короче проверок по коду: неизвестный вход один раз
#   превращается в объект с гарантированными типами и ограничениями (SplitIn). Дальше
#   каждая функция может на них опираться — проверки «а если пусто, а если не int» не
#   нужно повторять в каждом месте, где значение используется.
# Почему extra="ignore" опасен именно для API: опечатка клиента в имени поля не вызывает
#   ошибки — поле молча отбрасывается, и применяется умолчание. Клиент получает 200 и
#   уверен, что его параметр сработал (задание 1а). В библиотеке такая терпимость
#   удобна, в API — это ложное подтверждение.
# model_dump() против model_dump(mode="json"): первый отдаёт объекты Python (datetime,
#   SecretStr, Decimal) — для работы внутри программы. Второй приводит всё к
#   JSON-совместимому (строки, числа) — для сети и файлов: json.dumps на результате
#   первого падает на datetime.
# Что даёт frozen=True: __eq__ и __hash__ строятся из одних и тех же полей и не могут
#   разойтись, плюс присваивание запрещено — значит, хеш объекта в set не изменится
#   после того, как его туда положили. Ручные дандеры можно рассогласовать — и в ragkit
#   они до сих пор сравнивают по числу токенов (задание 7).
# Зачем дискриминатор: без него pydantic пробует все ветки и возвращает ошибки каждой —
#   шум про чужие поля. С ним выбирается одна ветка по тегу, ошибка одна и с понятным
#   loc, неизвестный тег — union_tag_invalid. Плюс это быстрее: проверяется одна ветка.


def main() -> None:
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()
    task10()
    extras()


if __name__ == "__main__":
    main()
