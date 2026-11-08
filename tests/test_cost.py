"""Бюджет в коде: запрос за порогом не уходит, под порогом уходит, оценщик ошибается вверх."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ragkit.conversation import CharRatioEstimate
from ragkit.cost import CostGuard, CostLimitExceededError, Price, conservative_ratio

FIXTURES = Path(__file__).parent / "fixtures"
PRICE = Price(
    provider="Anthropic",
    model="Claude Sonnet 5.5",
    as_of="2026-11-08",
    source="https://platform.claude.com/docs/en/about-claude/pricing",
    input_per_mtok=2.0,
    output_per_mtok=10.0,
    cache_read_per_mtok=0.10,
)


def measured() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / "token_counts.json").read_text())
    return data


def safe_estimator() -> CharRatioEstimate:
    """Курс — худший замеренный с запасом 10 %; база покрывает разметку шаблона qwen2.5."""
    pairs = [(s["chars"], s["tokens"]) for s in measured()["text_samples"]]
    return CharRatioEstimate(chars_per_token=conservative_ratio(pairs), per_message=5, base=30)


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeReply:
    done_reason: str | None = "stop"
    message: FakeMessage = field(default_factory=lambda: FakeMessage("ok"))


@dataclass
class RecordingClient:
    sent: list[Sequence[Mapping[str, Any]]] = field(default_factory=list)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> FakeReply:
        self.sent.append(messages)
        return FakeReply()


def user(chars: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": "я" * chars}]


# ---------------------------------------------------------------------------
# Цена
# ---------------------------------------------------------------------------
def test_price_per_million_and_cache() -> None:
    assert PRICE.cost(1_000_000, 0) == pytest.approx(2.0)
    assert PRICE.cost(0, 1_000_000) == pytest.approx(10.0)
    assert PRICE.cost(1_000_000, 0, cached_input_tokens=1_000_000) == pytest.approx(0.10)


def test_cached_tokens_cannot_exceed_input() -> None:
    with pytest.raises(ValueError, match="cached_input_tokens"):
        PRICE.cost(10, 0, cached_input_tokens=11)


# ---------------------------------------------------------------------------
# Предохранитель
# ---------------------------------------------------------------------------
def test_request_over_limit_is_not_sent_and_error_has_numbers() -> None:
    client = RecordingClient()
    guard = CostGuard(PRICE, limit=0.001, count_tokens=safe_estimator())
    with pytest.raises(CostLimitExceededError) as info:
        guard.chat(client, "m", user(10_000), {"num_predict": 200})
    assert client.sent == []  # не ушёл
    error = info.value
    assert error.estimate.cost > error.limit == 0.001
    assert error.estimate.output_tokens == 200
    assert "0.001000" in str(error) and f"{error.estimate.cost:.6f}" in str(error)


def test_request_under_limit_is_sent() -> None:
    client = RecordingClient()
    guard = CostGuard(PRICE, limit=0.01, count_tokens=safe_estimator())
    reply = guard.chat(client, "m", user(100), {"num_predict": 64})
    assert reply.message.content == "ok"
    assert len(client.sent) == 1


def test_no_output_bound_no_request() -> None:
    client = RecordingClient()
    guard = CostGuard(PRICE, limit=1.0, count_tokens=safe_estimator())
    with pytest.raises(ValueError, match="num_predict"):
        guard.chat(client, "m", user(10), {"temperature": 0})
    assert client.sent == []


def test_tool_descriptions_are_counted() -> None:
    guard = CostGuard(PRICE, limit=1.0, count_tokens=safe_estimator())
    tool = {"type": "function", "function": {"name": "get_weather", "description": "я" * 300}}
    bare = guard.estimate(user(10), max_output_tokens=1)
    with_tool = guard.estimate(user(10), max_output_tokens=1, tools=[tool])
    assert with_tool.input_tokens > bare.input_tokens + 100


# ---------------------------------------------------------------------------
# Направление ошибки оценщика: на всех замерах — не меньше факта
# ---------------------------------------------------------------------------
def test_estimator_errs_upward_on_every_measured_text() -> None:
    ratio = safe_estimator().chars_per_token
    for sample in measured()["text_samples"]:
        estimate = math.ceil(sample["chars"] / ratio)
        assert estimate >= sample["tokens"], sample["source"]


def test_estimator_errs_upward_on_every_measured_request() -> None:
    estimate = safe_estimator()
    for request in measured()["requests"]:
        assert estimate(request["messages"]) >= request["tokens"], request["source"]


def test_average_ratio_would_err_downward() -> None:
    """Без этого теста предыдущие ничего не доказывают: средний курс ошибается вниз."""
    samples = measured()["text_samples"]
    prose = CharRatioEstimate(chars_per_token=2.43, per_message=5, base=3)  # Ф4-03
    under = [s for s in samples if math.ceil(s["chars"] / prose.chars_per_token) < s["tokens"]]
    assert under, "средний курс нигде не ошибся вниз — замеры слишком однородны"
    hello = measured()["requests"][0]
    assert prose(hello["messages"]) < hello["tokens"]


def test_conservative_ratio_validates_input() -> None:
    with pytest.raises(ValueError):
        conservative_ratio([])
    with pytest.raises(ValueError):
        conservative_ratio([(10, 5)], margin=1.5)
