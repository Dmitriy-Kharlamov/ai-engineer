"""Структурированный ответ модели: форму задаёт pydantic, три отказа — три исключения (Ф4-05).

Модель pydantic — источник истины о форме. Её схема (model_json_schema) уходит серверу в
format=, и сервер на каждом шаге вычёркивает токены, которые схему нарушили бы: невалидный
JSON не маловероятен, а невозможен. Но грамматика гарантирует форму, а не правду, и не всё
выражает: согласованность полей, ограничения из валидаторов — это уже model_validate после
разбора.

Три отказа различаются, потому что лечатся по-разному:
- TruncatedOutputError — done_reason != "stop": вывод оборван по лимиту. Битый JSON не
  чиним и даже не пробуем разобрать: дописать скобку — значит догадаться о данных, а догадка
  в контексте становится фактом (Ф4-03, задание 5). Лечится запасом num_predict;
- NotJSONError — не разобралось как JSON (бывает без схемы: просьба в промпте);
- SchemaMismatchError — JSON есть, но не прошёл модель pydantic.

Ошибка — поверхность атаки. ValidationError несёт input_value — вывод модели, а в Ф5 вывод
строится по чужим документам. Поэтому в сообщениях здесь — белый список (как api/errors.py,
Ф3-06): тип ошибки, путь и текст pydantic, и только известные схеме имена полей. Исходные
исключения не прицепляются (from None): их текст содержит вывод модели. Сам вывод доступен
явно — .raw, — для того, кто решит, куда его можно класть.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from types import UnionType
from typing import Any, Literal, Protocol, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from .exceptions import RAGError
from .llm import ChatReply

T = TypeVar("T", bound=BaseModel)


class StructuredClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
        format: dict[str, Any] | Literal["json"],  # noqa: A002 — имя параметра у ollama
    ) -> ChatReply: ...


class StructuredOutputError(RAGError):
    """Предок трёх отказов. В тексте — ни строки вывода модели."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self._raw = raw

    @property
    def raw(self) -> str:
        """Вывод модели как есть. Недоверенный: в логи и ответы API — только осознанно."""
        return self._raw


class TruncatedOutputError(StructuredOutputError):
    """Генерация остановилась не сама (done_reason != "stop"): объект оборван."""

    def __init__(self, done_reason: str | None, *, raw: str) -> None:
        super().__init__(
            f"вывод оборван: done_reason={done_reason!r}, {len(raw)} символов; "
            "битый JSON не чиним — нужен запас num_predict",
            raw=raw,
        )
        self.done_reason = done_reason


class NotJSONError(StructuredOutputError):
    """Вывод не разбирается как JSON."""


class SchemaMismatchError(StructuredOutputError):
    """JSON разобрался, но не прошёл модель pydantic. errors — очищенный список ошибок."""

    def __init__(self, message: str, *, raw: str, errors: list[dict[str, Any]]) -> None:
        super().__init__(message, raw=raw)
        self.errors = errors


# ---------------------------------------------------------------------------
# Белый список для ошибок валидации
# ---------------------------------------------------------------------------
# Типы ошибок pydantic, чей msg собран из схемы, а не из присланного значения.
SAFE_MSG_TYPES = frozenset(
    {
        "missing",
        "extra_forbidden",
        "int_type",
        "int_parsing",
        "int_from_float",
        "float_type",
        "float_parsing",
        "string_type",
        "bool_type",
        "bool_parsing",
        "list_type",
        "dict_type",
        "model_type",
        "model_attributes_type",
        "none_required",
        "literal_error",
        "enum",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "too_short",
        "too_long",
        "string_too_short",
        "string_too_long",
    }
)
SAFE_CTX_KEYS = frozenset(
    {"min_length", "max_length", "actual_length", "ge", "gt", "le", "lt", "expected"}
)
UNKNOWN_FIELD = "<поле не из схемы>"


def _models_in(annotation: Any) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    if get_origin(annotation) in (Union, UnionType, list, tuple, set, dict) or get_args(annotation):
        for arg in get_args(annotation):
            yield from _models_in(arg)


def known_fields(schema: type[BaseModel]) -> frozenset[str]:
    """Все имена полей схемы и вложенных моделей — и по атрибуту, и по псевдониму."""
    names: set[str] = set()
    seen: set[type[BaseModel]] = set()
    todo = [schema]
    while todo:
        model = todo.pop()
        if model in seen:
            continue
        seen.add(model)
        for name, field in model.model_fields.items():
            names.add(name)
            if field.alias:
                names.add(field.alias)
            todo.extend(_models_in(field.annotation))
    return frozenset(names)


def safe_errors(error: ValidationError, schema: type[BaseModel]) -> list[dict[str, Any]]:
    """Ошибки валидации без вывода модели: ни input, ни незнакомых имён полей, ни чужих msg.

    Имя лишнего поля — тоже вывод модели (его можно заполнить инъекцией), поэтому в loc
    остаются только номера элементов и имена, которые есть в схеме.
    """
    known = known_fields(schema)
    out: list[dict[str, Any]] = []
    for err in error.errors(include_url=False, include_input=False, include_context=True):
        loc = [p if isinstance(p, int) or p in known else UNKNOWN_FIELD for p in err["loc"]]
        safe: dict[str, Any] = {"type": err["type"], "loc": loc}
        safe["msg"] = err["msg"] if err["type"] in SAFE_MSG_TYPES else "значение не прошло проверку"
        ctx = {k: v for k, v in (err.get("ctx") or {}).items() if k in SAFE_CTX_KEYS}
        if ctx:
            safe["ctx"] = ctx
        out.append(safe)
    return out


# ---------------------------------------------------------------------------
# Разбор
# ---------------------------------------------------------------------------
def parse_structured(content: str, done_reason: str | None, schema: type[T]) -> T:
    """Вывод модели → объект schema, или одно из трёх исключений. Порядок проверок важен.

    Сначала done_reason: оборванный вывод отвергается, даже если случайно разобрался бы.
    """
    if done_reason != "stop":
        raise TruncatedOutputError(done_reason, raw=content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        # error.doc — весь вывод модели: исключение не цепляем
        raise NotJSONError(
            f"не JSON: {error.msg}, позиция {error.pos} из {len(content)}", raw=content
        ) from None
    try:
        return schema.model_validate(data)
    except ValidationError as error:
        errors = safe_errors(error, schema)
        where = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or '<корень>'}: {e['type']}" for e in errors
        )
        raise SchemaMismatchError(
            f"не прошло схему {schema.__name__}: {len(errors)} ошиб.: {where}",
            raw=content,
            errors=errors,
        ) from None


def ask_structured(
    client: StructuredClient,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    schema: type[T],
    options: Mapping[str, Any] | None = None,
    *,
    mode: Literal["schema", "json"] = "schema",
) -> T:
    """Запрос со схемой (или с format='json') и разбор ответа в объект schema."""
    fmt: dict[str, Any] | Literal["json"] = (
        schema.model_json_schema() if mode == "schema" else "json"
    )
    reply = client.chat(model=model, messages=messages, options=options, format=fmt)
    return parse_structured(reply.message.content or "", reply.done_reason, schema)


# ---------------------------------------------------------------------------
# Стриминг: объекты из массива — по мере закрытия
# ---------------------------------------------------------------------------
class ArrayItemParser:
    """Достаёт из потока JSON объекты — элементы массива, как только объект закрылся.

    Префикс валидного JSON сам по себе невалиден, но массив объектов можно отдавать по
    частям: каждый закрытый объект внутри массива — уже законченный JSON. Берутся элементы
    массива верхнего уровня или массива, лежащего прямо в корневом объекте
    ({"books": [{...}, {...}]}). Скобки и кавычки внутри строк не считаются.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._pos = 0
        self._stack: list[str] = []
        self._in_string = False
        self._escape = False
        self._start: int | None = None

    @property
    def text(self) -> str:
        return "".join(self._text)

    def _item_level(self) -> bool:
        return self._stack in (["["], ["{", "["])

    def feed(self, delta: str) -> list[Any]:
        """Дописать кусок; вернуть объекты, которые закрылись в этом куске."""
        self._text.append(delta)
        done: list[Any] = []
        whole: str | None = None
        for ch in delta:
            pos = self._pos
            self._pos += 1
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                continue
            if ch == '"':
                self._in_string = True
            elif ch in "{[":
                if ch == "{" and self._item_level():
                    self._start = pos
                self._stack.append(ch)
            elif ch in "}]":
                if self._stack:
                    self._stack.pop()
                if ch == "}" and self._start is not None and self._item_level():
                    whole = whole if whole is not None else self.text
                    done.append(json.loads(whole[self._start : pos + 1]))
                    self._start = None
        return done
