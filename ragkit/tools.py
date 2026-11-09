"""Вызов инструментов: реестр, диспетчер с тремя отказами, цикл с предохранителями (Ф4-06).

Модель ничего не запускает: она пишет просьбу «позови name(arguments)», а выполняет её
этот код. Поэтому всё, что модель может попросить, проходит здесь через одни ворота:

- реестр знает имена, описания и схемы аргументов. Схема строится из модели pydantic, как
  в Ф4-05. Значения по умолчанию в ней честно не обязательны: схема, которую ollama 0.6.3
  собирает из подписи функции, делает обязательными все аргументы
  (week-04/lesson_06.py, задание 2). Описания из схемы уходят модели в промпт:
  tools= рендерится шаблоном в текст, в отличие от format= (задание 3);
- диспетчер различает три отказа, потому что лечатся они по-разному:
  UnknownToolError — такого имени нет (имя модель пишет сама, грамматика его не держит);
  ToolArgumentsError — аргументы не прошли модель pydantic;
  ToolFailedError — функция упала;
- ошибку диспетчер отдаёт модели результатом, текстом: тогда модель может исправиться, а
  исключение наружу значило бы «цикл умер, ответа не будет». Текст ошибки — белый список:
  ни имени, которое придумала модель, ни значений её аргументов, ни текста исключения
  (там бывают пути и фрагменты запроса — задание 8). Свой текст для модели инструмент
  передаёт только явно, через PublicToolError;
- цикл ограничен числом обращений к модели. Лимит токенов этого не заменяет: num_predict
  ограничивает один ответ, а вызов инструмента — два десятка токенов (задание 6);
- одинаковый вызов (то же имя, те же аргументы) второй раз не выполняется: модель
  получает отказ «повтор», а функция — ни одного лишнего запуска. Для пишущих
  инструментов это ещё и защита от дубликата;
- tool_call_id в протоколе ollama нет: результат связан с вызовом только порядком
  (задание 4). Поэтому результаты дописываются строго в порядке вызовов, а успешный
  результат по умолчанию повторяет свои проверенные аргументы — так модель может
  сопоставить его с вызовом, даже если имя одно и то же.

Клиент — любой объект с методом chat(...), чей ответ похож на ollama.ChatResponse:
message.content, message.tool_calls[i].function.name и .arguments, done_reason.
От библиотеки ollama ragkit не зависит.
"""

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .conversation import BrokenHistoryError
from .exceptions import RAGError
from .llm import IncompleteResponseError
from .structured import safe_errors

A = TypeVar("A", bound=BaseModel)
Message = Mapping[str, Any]
Kind = Literal["ok", "unknown", "invalid", "failed", "repeat"]

UNKNOWN_TOOL = "unknown"  # tool_name для результата вызова несуществующего инструмента


# ---------------------------------------------------------------------------
# Отказы
# ---------------------------------------------------------------------------
class ToolError(RAGError):
    """Предок трёх отказов диспетчера. В тексте — ни строки вывода модели."""

    kind: ClassVar[Kind]


class UnknownToolError(ToolError):
    """Модель попросила инструмент, которого нет в реестре."""

    kind = "unknown"


class ToolArgumentsError(ToolError):
    """Аргументы не прошли модель pydantic. errors — очищенный список (safe_errors)."""

    kind = "invalid"

    def __init__(self, message: str, *, errors: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.errors = errors


class ToolFailedError(ToolError):
    """Функция инструмента упала. Исходное исключение — в .original, не в тексте."""

    kind = "failed"

    def __init__(self, message: str, *, original: BaseException) -> None:
        super().__init__(message)
        self._original = original

    @property
    def original(self) -> BaseException:
        """Исходное исключение. Его текст недоверенный: в контекст модели — никогда."""
        return self._original


class PublicToolError(Exception):
    """Инструмент сообщает модели причину отказа — этот текст автор инструмента разрешил.

    Например «город не найден»: модель может переспросить. Любое другое исключение
    модель видит только как «инструмент упал».
    """


class IterationLimitError(RAGError):
    """Модель всё ещё просит инструменты после max_iterations обращений."""

    def __init__(
        self, message: str, *, steps: Sequence["Step"], messages: Sequence[Message]
    ) -> None:
        super().__init__(message)
        self.steps = tuple(steps)
        self.messages = tuple(messages)


# ---------------------------------------------------------------------------
# Реестр
# ---------------------------------------------------------------------------
def _strip_titles(node: Any) -> Any:
    """Схема без "title": pydantic ставит их везде, а схема в tools= — это токены промпта."""
    if isinstance(node, dict):
        return {
            k: _strip_titles(v)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _render(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class Tool:
    """Инструмент: имя, описание для модели, модель аргументов и функция."""

    name: str
    description: str
    args: type[BaseModel]
    func: Callable[[Any], object]

    def schema(self) -> dict[str, Any]:
        """Описание в формате tools= (ollama, OpenAI): его модель читает как промпт."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.args.model_json_schema()),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """Чем кончился один вызов. content уходит модели; остальное — для кода и журнала."""

    kind: Kind
    requested: str  # имя, которое попросила модель: недоверенное, в content не попадает
    tool_name: str  # имя для сообщения role=tool: из реестра или UNKNOWN_TOOL
    content: str
    arguments: Mapping[str, Any] | None = None  # проверенные аргументы, если дошло до них
    requested_arguments: Mapping[str, Any] | None = None  # как прислала модель: недоверенные

    @property
    def ok(self) -> bool:
        return self.kind == "ok"

    def message(self) -> dict[str, Any]:
        return {"role": "tool", "tool_name": self.tool_name, "content": self.content}


class ToolRegistry:
    """Имя → инструмент. Диспетчер различает три отказа и не пропускает чужой текст."""

    def __init__(self, *, echo_arguments: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self.echo_arguments = echo_arguments

    def add(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"инструмент {tool.name!r} уже есть")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self, args: type[A], *, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[[A], object]], Callable[[A], object]]:
        """Декоратор: функция от модели аргументов → инструмент в реестре.

        description по умолчанию — docstring функции целиком: модель его читает.
        """

        def register(func: Callable[[A], object]) -> Callable[[A], object]:
            text = description if description is not None else inspect.getdoc(func) or ""
            self.add(Tool(name or func.__name__, text, args, func))
            return func

        return register

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[object, Mapping[str, Any]]:
        """Выполнить вызов: (результат функции, проверенные аргументы) или один из отказов."""
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(
                f"инструмента нет (запрошено имя длиной {len(name)}); "
                f"доступны: {', '.join(self._tools) or 'ничего'}"
            )
        try:
            args = tool.args.model_validate(dict(arguments))
        except ValidationError as error:
            errors = safe_errors(error, tool.args)
            where = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '<корень>'}: {e['msg']}" for e in errors
            )
            raise ToolArgumentsError(
                f"{tool.name}: аргументы не прошли проверку — {where}", errors=errors
            ) from None
        checked = args.model_dump(mode="json")
        try:
            result = tool.func(args)
        except PublicToolError as error:
            raise ToolFailedError(f"{tool.name}: {error}", original=error) from None
        except Exception as error:
            raise ToolFailedError(
                f"{tool.name}: инструмент завершился с ошибкой ({type(error).__name__} — "
                "подробности в журнале сервера, не в контексте)",
                original=error,
            ) from None
        return result, checked

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Вызов, который не бросает отказы: ошибка становится результатом для модели."""
        known = name in self._tools
        tool_name = name if known else UNKNOWN_TOOL
        try:
            result, checked = self.call(name, arguments)
        except UnknownToolError:
            return ToolResult(
                "unknown",
                name,
                tool_name,
                "ошибка: такого инструмента нет. Доступны только: "
                f"{', '.join(self._tools) or 'ничего'}. Если нужного нет — скажи "
                "пользователю, что этого сделать не можешь.",
            )
        except ToolArgumentsError as error:
            where = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '<корень>'}: {e['msg']}" for e in error.errors
            )
            return ToolResult(
                "invalid", name, tool_name, f"ошибка: аргументы не прошли проверку — {where}"
            )
        except ToolFailedError as error:
            public = isinstance(error.original, PublicToolError)
            content = (
                f"ошибка: {error.original}"
                if public
                else "ошибка: инструмент не смог выполнить запрос. Повтор с теми же "
                "аргументами не поможет."
            )
            return ToolResult("failed", name, tool_name, content)
        content = (
            _render({"аргументы": checked, "результат": result})
            if self.echo_arguments
            else _render(result)
        )
        return ToolResult("ok", name, tool_name, content, checked)


# ---------------------------------------------------------------------------
# История: результатов ровно столько, сколько вызовов, и в том же порядке
# ---------------------------------------------------------------------------
def validate_tool_results(messages: Sequence[Message]) -> None:
    """BrokenHistoryError, если за assistant с k вызовами идёт не k результатов подряд.

    validate_alternation (Ф4-03) проверяет только роли: tool после вызова — да, но сколько
    их и к какому вызову какой — нет. Без tool_call_id связь держится лишь на порядке,
    значит порядок и число надо проверять отдельно.
    """
    i = 0
    while i < len(messages):
        message = messages[i]
        calls = message.get("tool_calls") or ()
        if message.get("role") != "assistant" or not calls:
            if message.get("role") == "tool":
                raise BrokenHistoryError(f"сообщение {i}: результат без вызова")
            i += 1
            continue
        results = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            results.append(messages[j])
            j += 1
        if len(results) != len(calls):
            raise BrokenHistoryError(
                f"сообщение {i}: вызовов {len(calls)}, результатов {len(results)}"
            )
        for k, (call, result) in enumerate(zip(calls, results, strict=True)):
            name = _call_name(call)
            if result.get("tool_name") not in (name, UNKNOWN_TOOL):
                raise BrokenHistoryError(f"сообщение {i + 1 + k}: результат не к вызову {k + 1}")
        i = j


def _call_name(call: Any) -> str:
    function = call["function"] if isinstance(call, Mapping) else call.function
    name = function["name"] if isinstance(function, Mapping) else function.name
    return str(name)


# ---------------------------------------------------------------------------
# Цикл
# ---------------------------------------------------------------------------
class _Function(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> Mapping[str, Any]: ...


class _ToolCall(Protocol):
    @property
    def function(self) -> _Function: ...


class _ToolMessage(Protocol):
    @property
    def content(self) -> str | None: ...

    @property
    def tool_calls(self) -> Sequence[_ToolCall] | None: ...


class ToolReply(Protocol):
    @property
    def done_reason(self) -> str | None: ...

    @property
    def message(self) -> _ToolMessage: ...

    @property
    def prompt_eval_count(self) -> int | None: ...

    @property
    def eval_count(self) -> int | None: ...


class ToolChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> ToolReply: ...


@dataclass(frozen=True)
class Step:
    """Одно обращение к модели и вызовы, которые она в нём попросила."""

    iteration: int
    prompt_eval_count: int | None
    eval_count: int | None
    results: tuple[ToolResult, ...] = ()
    prompt_eval_ns: int | None = None  # время prefill и decode на сервере, если ответ их несёт
    eval_ns: int | None = None


@dataclass(frozen=True)
class LoopResult:
    answer: str
    messages: tuple[Message, ...]
    steps: tuple[Step, ...] = field(default=())

    @property
    def prompt_tokens(self) -> int:
        """Сумма prompt_eval_count по всем обращениям: столько промпта прочитала модель."""
        return sum(s.prompt_eval_count or 0 for s in self.steps)


def call_key(name: str, arguments: Mapping[str, Any]) -> str:
    """Одинаковый вызов — то же имя и те же аргументы, порядок ключей не важен."""
    return json.dumps([name, arguments], ensure_ascii=False, sort_keys=True, default=str)


def assistant_message(
    content: str, calls: Sequence[tuple[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Сообщение модели с вызовами — в том виде, в каком оно уходит обратно в историю."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"function": {"name": n, "arguments": dict(a)}} for n, a in calls],
    }


class ToolLoop:
    """Одна итерация цикла без ввода-вывода: запрет повтора, вызов, порядок результатов.

    Её используют run_tools (без потока) и потоковые циклы (Ф4-08): как получить вызовы от
    модели — дело цикла, а что с ними делать — одно на всех.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._seen: dict[str, int] = {}

    def handle(
        self, iteration: int, calls: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> tuple[ToolResult, ...]:
        """Результаты строго в порядке вызовов: другой связи между ними нет (Ф4-06)."""
        results: list[ToolResult] = []
        for name, arguments in calls:
            key = call_key(name, arguments)
            if key in self._seen:
                result = ToolResult(
                    "repeat",
                    name,
                    name if name in self.registry.names else UNKNOWN_TOOL,
                    f"повтор: такой же вызов уже выполнен на итерации {self._seen[key]}; "
                    "результат выше и не изменится. Не повторяй его — ответь пользователю.",
                )
            else:
                self._seen[key] = iteration
                result = self.registry.dispatch(name, arguments)
            results.append(replace(result, requested_arguments=dict(arguments)))
        return tuple(results)


def run_tools(
    client: ToolChatClient,
    model: str,
    messages: Sequence[Message],
    registry: ToolRegistry,
    *,
    max_iterations: int = 5,
    options: Mapping[str, Any] | None = None,
) -> LoopResult:
    """Цикл «модель просит — код выполняет», пока модель не ответит текстом.

    max_iterations — сколько раз можно обратиться к модели. Если и последнее обращение
    вернуло вызовы — IterationLimitError (в нём шаги и история: можно, например, спросить
    модель без tools, что она успела выяснить). Повтор одинакового вызова не выполняется:
    модель получает результат kind="repeat". Оборванный ответ — IncompleteResponseError.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations должен быть не меньше 1")
    history: list[Message] = list(messages)
    steps: list[Step] = []
    loop = ToolLoop(registry)
    schemas = registry.schemas()
    for iteration in range(1, max_iterations + 1):
        reply = client.chat(model=model, messages=history, tools=schemas, options=options)
        if reply.done_reason != "stop":
            raise IncompleteResponseError(
                f"ответ неполный: done_reason={reply.done_reason!r}, итерация {iteration}"
            )
        calls = [(c.function.name, c.function.arguments) for c in reply.message.tool_calls or ()]
        content = reply.message.content or ""
        prompt_ns: int | None = getattr(reply, "prompt_eval_duration", None)
        eval_ns: int | None = getattr(reply, "eval_duration", None)
        if not calls:
            steps.append(
                Step(
                    iteration,
                    reply.prompt_eval_count,
                    reply.eval_count,
                    prompt_eval_ns=prompt_ns,
                    eval_ns=eval_ns,
                )
            )
            history.append({"role": "assistant", "content": content})
            return LoopResult(content, tuple(history), tuple(steps))
        history.append(assistant_message(content, calls))
        results = loop.handle(iteration, calls)
        history.extend(r.message() for r in results)
        steps.append(
            Step(
                iteration,
                reply.prompt_eval_count,
                reply.eval_count,
                results,
                prompt_eval_ns=prompt_ns,
                eval_ns=eval_ns,
            )
        )
    raise IterationLimitError(
        f"модель просит инструменты после {max_iterations} обращений; "
        f"вызовов всего {sum(len(s.results) for s in steps)}",
        steps=steps,
        messages=history,
    )
