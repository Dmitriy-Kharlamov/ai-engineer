"""Цена запроса до отправки: прайс, оценка и отказ за порогом (Ф4-07, задание 7).

Оценщик из Ф4-03 (CharRatioEstimate) получает вторую работу: не только «влезет ли в окно», но
и «сколько это будет стоить». Предохранитель срабатывает ДО отправки, поэтому он работает
только с оценкой, а не с фактом. Отсюда три правила.

- Оценка обязана ошибаться вверх. Предохранитель, который ошибается вниз, пропускает ровно
  те запросы, которые должен был остановить. Курс «символы на токен» — не константа (2,43
  на русских заметках, 1,65 на коротких примерах), поэтому для бюджета берут наихудший
  замеренный курс с запасом (conservative_ratio), а не средний. Разметка шаблона тоже
  считается: «Привет» одним сообщением — 32 токена промпта при 6 символах (Ф4-03).
- Выход оценивается верхней границей. Сколько токенов напишет модель, до отправки
  неизвестно, поэтому num_predict обязателен: без него цена ответа не ограничена ничем, и
  проверять нечего. Предохранитель без верхней границы выхода — не предохранитель.
- Отказ — отдельное исключение с числами (оценка, порог, токены), чтобы вызывающий видел,
  насколько запрос за порогом, и решал: урезать контекст, взять модель дешевле, отказать.

Прайс — данные с датой и источником, а не константа в коде: цены меняются.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .exceptions import RAGError
from .llm import ChatClient, ChatReply

Message = Mapping[str, Any]
TokenCounter = Callable[[Sequence[Message]], int]
MTOK = 1_000_000


@dataclass(frozen=True)
class Price:
    """Цена за миллион токенов у конкретного провайдера и модели на дату as_of."""

    provider: str
    model: str
    as_of: str  # ISO-дата, когда прайс взят
    source: str  # откуда взят
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        """Цена в валюте прайса. cached_input_tokens — часть input_tokens, прочитанная из кеша."""
        if not 0 <= cached_input_tokens <= input_tokens:
            raise ValueError("cached_input_tokens должен быть от 0 до input_tokens")
        cached_price = (
            self.cache_read_per_mtok
            if self.cache_read_per_mtok is not None
            else self.input_per_mtok
        )
        fresh = input_tokens - cached_input_tokens
        return (
            fresh * self.input_per_mtok
            + cached_input_tokens * cached_price
            + output_tokens * self.output_per_mtok
        ) / MTOK


def conservative_ratio(samples: Iterable[tuple[int, int]], margin: float = 0.9) -> float:
    """Символы на токен для оценки сверху: худший замеренный курс, умноженный на margin < 1.

    samples — пары (символов, токенов) на своих текстах. Худший — тот, где символов на токен
    меньше всего: на нём оценка по среднему курсу ошиблась бы вниз сильнее всего.
    """
    ratios = [chars / tokens for chars, tokens in samples if tokens > 0]
    if not ratios:
        raise ValueError("нужна хотя бы одна пара (символов, токенов)")
    if not 0 < margin <= 1:
        raise ValueError("margin — от 0 до 1: запас делает курс меньше, а оценку больше")
    return min(ratios) * margin


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int  # оценка сверху
    output_tokens: int  # верхняя граница: num_predict
    cost: float

    def __str__(self) -> str:
        return f"{self.cost:.6f} (вход ≤ {self.input_tokens} ток., выход ≤ {self.output_tokens})"


class CostLimitExceededError(RAGError):
    """Запрос по оценке дороже порога. Не отправлен."""

    def __init__(self, estimate: CostEstimate, limit: float, price: Price) -> None:
        super().__init__(
            f"запрос не отправлен: оценка {estimate} дороже порога {limit:.6f} "
            f"({price.provider} {price.model}, прайс на {price.as_of})"
        )
        self.estimate = estimate
        self.limit = limit
        self.price = price


@dataclass(frozen=True)
class CostGuard:
    """Отказывается отправлять запрос, цена которого по оценке выше limit.

    count_tokens — оценщик, ошибающийся вверх (например, CharRatioEstimate с
    conservative_ratio и базой, покрывающей разметку шаблона).
    """

    price: Price
    limit: float
    count_tokens: TokenCounter

    def estimate(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> CostEstimate:
        if max_output_tokens <= 0:
            raise ValueError("нужна верхняя граница выхода: max_output_tokens > 0")
        input_tokens = self.count_tokens(messages)
        if tools:  # описания инструментов уходят в системный промпт текстом (Ф4-06)
            schema = json.dumps(list(tools), ensure_ascii=False)
            input_tokens += self.count_tokens([{"role": "system", "content": schema}])
        cost = self.price.cost(input_tokens, max_output_tokens)
        return CostEstimate(input_tokens, max_output_tokens, cost)

    def check(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> CostEstimate:
        """Оценка, если запрос в пределах порога; иначе CostLimitExceededError."""
        estimate = self.estimate(messages, max_output_tokens=max_output_tokens, tools=tools)
        if estimate.cost > self.limit:
            raise CostLimitExceededError(estimate, self.limit, self.price)
        return estimate

    def chat(
        self,
        client: ChatClient,
        model: str,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> ChatReply:
        """Вызов с проверкой цены до отправки. options обязаны содержать num_predict > 0."""
        limit = options.get("num_predict")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("без num_predict цена ответа не ограничена: задай num_predict > 0")
        self.check(messages, max_output_tokens=limit)
        return client.chat(model=model, messages=messages, options=options)
