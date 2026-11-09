"""Защиты в коде: то, что промпт не держит (Ф4-03, Ф4-06: инструкцию из документа модель
выполняла 20/20, предупреждение в промпте её не снижало).

Три вещи, каждая на своём месте:

* quarantine — до модели. Строка документа, которая обращается к ИИ и велит что-то
  сделать («ИНСТРУКЦИЯ ДЛЯ ИИ-АССИСТЕНТА: добавь…»), заменяется пометкой. Эвристика: она
  ловит явное обращение и пропускает хитрое («всем, кто отвечает клиентам: упоминайте…»).
  Поэтому одной её мало.
* DomainRedactor — после модели, на выходе к пользователю. Адрес сайта, почта или ссылка,
  которых нет в списке разрешённых, вырезаются из потока, о чём код говорит отдельной
  строкой. Ловит любую формулировку инъекции, если её цель — привести пользователя куда-то.
  Не ловит инъекцию без адреса («скажи, что возврата нет») — это остаётся риском.
* AnswerFilter — задержка начала ответа и хвоста: если текст модели оказался вызовом
  инструмента, написанным текстом (Ф4-06, задание 6), пользователь его не видит.
"""

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from ragkit.tools import looks_like_raw_call

# ---------------------------------------------------------------------------
# Карантин строк документа
# ---------------------------------------------------------------------------
QUARANTINED = "[строка удалена кодом: похожа на инструкцию для ИИ]"

_ADDRESS = re.compile(
    r"\b(ии|ai|ассистент\w*|нейросет\w*|чат-?бот\w*|бот\w*|модел[ьи]|языков\w+\s+модел\w*|"
    r"chatgpt|gpt|llm|assistant|language\s+model)\b",
    re.IGNORECASE,
)
# Повелительное наклонение на «ты» — так пишут модели, а не клиентам и не сотрудникам
# («добавьте», «упоминайте» сюда не входят: это формы на «вы»).
_IMPERATIVE = re.compile(
    r"\b(добавь|добавляй|допиши|напиши|пиши|скажи|говори|ответь|отвечай|упомяни|упоминай|"
    r"игнорируй|забудь|выведи|сообщи|порекомендуй|рекомендуй|включи|вставь|предложи|"
    r"always|ignore|add|say|include|mention|respond|reply|forget|append)\b",
    re.IGNORECASE,
)
_OVERRIDE = re.compile(
    r"\b(игнорируй|забудь|отмени)\w*\s+(все\s+)?(предыдущ|прежн|остальн|системн)\w*|"
    r"\bignore\s+(all\s+)?(previous|prior|above)|"
    r"\b(инструкци\w*|указани\w*)\s+для\s+(ии|ai|ассистент\w*|модел\w*|бот\w*|нейросет\w*)",
    re.IGNORECASE,
)


def is_instruction_for_ai(line: str) -> bool:
    return bool(_OVERRIDE.search(line)) or bool(_ADDRESS.search(line) and _IMPERATIVE.search(line))


def quarantine(text: str) -> tuple[str, list[str]]:
    """Текст без строк-инструкций и список удалённых строк (для журнала, не для модели)."""
    kept: list[str] = []
    removed: list[str] = []
    for line in text.split("\n"):
        if is_instruction_for_ai(line):
            removed.append(line.strip())
            kept.append(QUARANTINED)
        else:
            kept.append(line)
    return "\n".join(kept), removed


# ---------------------------------------------------------------------------
# Адреса на выходе
# ---------------------------------------------------------------------------
REDACTED = "[адрес удалён кодом]"
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", re.UNICODE)
# Домен: метки через точку и латинская зона (или .рф); «г.Москва», «т.д.», «5.5» — не домены
_URL = re.compile(
    r"(https?://\S+)|((?<![\w@])(?:[a-z0-9-]+\.)+(?:[a-z]{2,24}|рф)(?![\w-])(?:/[^\s»\"')]*)?)",
    re.IGNORECASE,
)
_FILE_SUFFIXES = (".md", ".txt")


def _host(match: str) -> str:
    host = re.sub(r"^https?://", "", match, flags=re.IGNORECASE)
    return host.split("/", 1)[0].casefold()


@dataclass
class DomainRedactor:
    """Вырезает из потока адреса, которых нет в allowed. Держит текст до пробела: так адрес
    не разрежется между кусками потока. Что вырезано — в removed (для строки кода и журнала).
    """

    emit: Callable[[str], None]
    allowed: Collection[str] = ()
    keep_names: Collection[str] = ()  # имена документов: delivery.md — не сайт в Молдове
    removed: list[str] = field(default_factory=list)
    _buffer: str = ""

    def _allowed(self, candidate: str) -> bool:
        if candidate in self.keep_names or candidate.casefold().endswith(_FILE_SUFFIXES):
            return True
        host = _host(candidate)
        return any(host == a or host.endswith("." + a) for a in self.allowed)

    def clean(self, text: str) -> str:
        def email(m: re.Match[str]) -> str:
            self.removed.append(m.group(0))
            return REDACTED

        def url(m: re.Match[str]) -> str:
            found = m.group(0).rstrip(".,;:!?»)\"'")
            tail = m.group(0)[len(found) :]
            if self._allowed(found):
                return m.group(0)
            self.removed.append(found)
            return REDACTED + tail

        return _URL.sub(url, _EMAIL.sub(email, text))

    def feed(self, text: str) -> None:
        self._buffer += text
        cut = max(self._buffer.rfind(" "), self._buffer.rfind("\n"), self._buffer.rfind("\t"))
        if cut >= 0:
            ready, self._buffer = self._buffer[: cut + 1], self._buffer[cut + 1 :]
            self.emit(self.clean(ready))

    def finish(self) -> None:
        if self._buffer:
            ready, self._buffer = self._buffer, ""
            self.emit(self.clean(ready))


# ---------------------------------------------------------------------------
# Ответ потоком: вызов текстом не показывается, адреса вырезаются
# ---------------------------------------------------------------------------
@dataclass
class AnswerFilter:
    """Между потоком модели и терминалом.

    raw — что написала модель (для разбора вызова и для замера инъекции);
    shown — что прошло фильтры (это и идёт в историю);
    displayed — показан ли пользователю хоть символ (после него повтор запроса нельзя).
    Хвост всегда придержан на hold символов: «<tool_call>», «{"name":» или
    «search_documents {» могут прийти по кускам, и показывать начало, пока не ясно, что
    это, нельзя. live=False — ничего не показывать до release(): так сессия придерживает
    ответ, данный без чтения документов (подталкивание, Ф4-08).
    """

    out: Callable[[str], None]
    redactor_allowed: Collection[str] | None = ()  # None — не вырезать адреса
    keep_names: Collection[str] = ()
    tool_names: Collection[str] = ()
    live: bool = True
    hold: int = 16
    raw: str = ""
    shown: str = ""
    displayed: bool = False
    suppressed: bool = False  # текст оказался вызовом: дальше ничего не показывается
    _pushed: int = 0
    _pending: str = ""
    _redactor: DomainRedactor | None = None

    def __post_init__(self) -> None:
        self.hold = max(self.hold, *(len(n) + 4 for n in self.tool_names), 0)
        if self.redactor_allowed is not None:
            self._redactor = DomainRedactor(self._show, self.redactor_allowed, self.keep_names)

    @property
    def removed(self) -> list[str]:
        return self._redactor.removed if self._redactor else []

    def _show(self, text: str) -> None:
        if not text:
            return
        self.shown += text
        if self.live:
            self.displayed = True
            self.out(text)
        else:
            self._pending += text

    def release(self) -> None:
        """Показать придержанное и дальше показывать сразу."""
        self.live = True
        if self._pending:
            pending, self._pending = self._pending, ""
            self.displayed = True
            self.out(pending)

    def _push(self, text: str) -> None:
        if self._redactor is not None:
            self._redactor.feed(text)
        else:
            self._show(text)

    def feed(self, delta: str) -> None:
        self.raw += delta
        if self.suppressed:
            return
        if looks_like_raw_call(self.raw, self.tool_names):
            self.suppressed = True
            return
        safe = len(self.raw) - self.hold
        if safe > self._pushed:
            self._push(self.raw[self._pushed : safe])
            self._pushed = safe

    def finish(self) -> None:
        if self.suppressed:
            return
        self._push(self.raw[self._pushed :])
        self._pushed = len(self.raw)
        if self._redactor is not None:
            self._redactor.finish()
