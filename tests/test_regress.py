"""Регрессионный набор (Ф5.5-02): кассеты, базовые линии, карантин, отчёт — без модели."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ragkit.regress import (
    Baseline,
    CassetteClient,
    CassetteMiss,
    Fingerprint,
    ItemResult,
    Quarantine,
    RunBudget,
    RunBudgetExceeded,
    RunResult,
    Suite,
    compare,
    first_difference,
    min_detectable,
    request_key,
    sample_size,
)

FP = Fingerprint("qwen2.5:7b", "sha256:aaa", "0.40.14", {"temperature": 0}, None, "Darwin-arm64")
MSGS = [{"role": "user", "content": "Сколько стоит доставка?"}]


def reply(content: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "qwen2.5:7b",
        "created_at": "2026-11-14T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 100,
        "eval_count": 10,
        **extra,
    }


@dataclass
class FakeInner:
    """Настоящая модель в миниатюре: отвечает по счётчику, поток — тремя кусками."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    async def chat(self, **kwargs: Any) -> Any:
        from ollama import ChatResponse

        self.calls.append(kwargs)
        n = len(self.calls)
        if not kwargs.get("stream"):
            return ChatResponse.model_validate(reply(f"ответ {n}"))

        async def gen() -> AsyncIterator[Any]:
            for part in ("290 ", "рублей", ""):
                done = part == ""
                data = reply(part) if done else {**reply(part), "done": False, "done_reason": None}
                yield ChatResponse.model_validate(data)

        return gen()


async def drain(chunks: AsyncIterator[Any]) -> str:
    return "".join([c.message.content or "" async for c in chunks])


# ---------------------------------------------------------------------------
# Мощность
# ---------------------------------------------------------------------------
def test_sample_size_for_five_points_from_095() -> None:
    # (1,645·√0,0475 + 0,84·√0,09)² / 0,05² = 0,6105² / 0,0025 ≈ 149,1
    assert sample_size(0.95, 0.90) == 150


def test_min_detectable_inverts_sample_size() -> None:
    p1 = min_detectable(0.95, 40)
    assert 0.7 < p1 < 0.9
    assert sample_size(0.95, p1 - 0.001) <= 40 < sample_size(0.95, p1 + 0.005)


@pytest.mark.parametrize(("p0", "p1"), [(0.9, 0.95), (1.0, 0.5), (0.5, 0.0)])
def test_sample_size_rejects_bad_proportions(p0: float, p1: float) -> None:
    with pytest.raises(ValueError):
        sample_size(p0, p1)


# ---------------------------------------------------------------------------
# Ключ кассеты
# ---------------------------------------------------------------------------
BASE_REQUEST: dict[str, Any] = {
    "model": "qwen2.5:7b",
    "messages": MSGS,
    "tools": None,
    "options": {"temperature": 0},
    "format": None,
    "stream": False,
}


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model", "qwen2.5:14b"),
        ("messages", [{"role": "user", "content": "Сколько стоит доставка?!"}]),
        ("tools", [{"type": "function", "function": {"name": "f"}}]),
        ("options", {"temperature": 0.7}),
        ("format", {"type": "object"}),
        ("stream", True),
    ],
)
def test_key_changes_with_every_field_that_changes_the_answer(field_name: str, value: Any) -> None:
    assert request_key({**BASE_REQUEST, field_name: value}, "d") != request_key(BASE_REQUEST, "d")


def test_key_changes_with_model_digest_and_not_with_dict_order() -> None:
    assert request_key(BASE_REQUEST, "d1") != request_key(BASE_REQUEST, "d2")
    reordered = dict(reversed(list(BASE_REQUEST.items())))
    assert request_key(reordered, "d") == request_key(BASE_REQUEST, "d")


def test_key_refuses_request_without_a_field() -> None:
    with pytest.raises(AssertionError, match="format"):
        request_key({k: v for k, v in BASE_REQUEST.items() if k != "format"}, "d")


def test_first_difference_names_the_path() -> None:
    a = {"messages": [{"content": "a"}, {"content": "b"}]}
    b = {"messages": [{"content": "a"}, {"content": "B"}]}
    assert first_difference(a, b) == "messages[1].content@0"
    assert first_difference(a, a) is None


# ---------------------------------------------------------------------------
# Кассета: запись и воспроизведение
# ---------------------------------------------------------------------------
async def test_record_then_replay_gives_the_same_answers_without_the_model(tmp_path: Path) -> None:
    path = tmp_path / "q1.jsonl"
    inner = FakeInner()
    recorder = CassetteClient(path, "record", inner, FP)
    first = await recorder.chat(model="qwen2.5:7b", messages=MSGS, options={"seed": 1})
    streamed = await drain(
        await recorder.chat(model="qwen2.5:7b", messages=MSGS, options={"seed": 1}, stream=True)
    )
    player = CassetteClient(path, "replay")
    again = await player.chat(model="qwen2.5:7b", messages=MSGS, options={"seed": 1})
    again_stream = await drain(
        await player.chat(model="qwen2.5:7b", messages=MSGS, options={"seed": 1}, stream=True)
    )
    assert len(inner.calls) == 2  # воспроизведение к модели не ходило
    assert (again.message.content, again.prompt_eval_count) == (first.message.content, 100)
    assert again_stream == streamed == "290 рублей"
    assert player.fingerprint == FP


async def test_same_request_twice_is_replayed_in_recorded_order(tmp_path: Path) -> None:
    path = tmp_path / "loop.jsonl"
    recorder = CassetteClient(path, "record", FakeInner(), FP)
    for _ in range(2):
        await recorder.chat(model="m", messages=MSGS)
    player = CassetteClient(path, "replay")
    got = [(await player.chat(model="m", messages=MSGS)).message.content for _ in range(2)]
    assert got == ["ответ 1", "ответ 2"]


async def test_changed_request_is_a_miss_with_the_path(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    await CassetteClient(path, "record", FakeInner(), FP).chat(model="m", messages=MSGS)
    player = CassetteClient(path, "replay")
    changed = [{"role": "user", "content": "Сколько стоит самовывоз?"}]
    with pytest.raises(CassetteMiss, match=r"messages\[0\]\.content") as caught:
        await player.chat(model="m", messages=changed)
    assert "самовывоз" in str(caught.value) and "доставка" in str(caught.value)


async def test_extra_call_beyond_the_record_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    await CassetteClient(path, "record", FakeInner(), FP).chat(model="m", messages=MSGS)
    player = CassetteClient(path, "replay")
    await player.chat(model="m", messages=MSGS)
    with pytest.raises(CassetteMiss, match="записи больше нет"):
        await player.chat(model="m", messages=MSGS)


async def test_parameter_outside_the_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    client = CassetteClient(path, "record", FakeInner(), FP)
    with pytest.raises(AssertionError, match="keep_alive"):
        await client.chat(model="m", messages=MSGS, keep_alive="5m")


async def test_edited_request_in_file_breaks_the_key_assert(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    await CassetteClient(path, "record", FakeInner(), FP).chat(model="m", messages=MSGS)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["request"]["messages"][0]["content"] = "другое"
    path.write_text(lines[0] + "\n" + json.dumps(record, ensure_ascii=False) + "\n", "utf-8")
    with pytest.raises(AssertionError, match="ключ"):
        CassetteClient(path, "replay")


async def test_edited_answer_in_file_is_replayed_silently(tmp_path: Path) -> None:
    """Кассета проверяет код, а не модель: подменённый ответ проигрывается как настоящий."""
    path = tmp_path / "q.jsonl"
    await CassetteClient(path, "record", FakeInner(), FP).chat(model="m", messages=MSGS)
    text = path.read_text(encoding="utf-8").replace("ответ 1", "ответ, которого модель не давала")
    path.write_text(text, encoding="utf-8")
    got = await CassetteClient(path, "replay").chat(model="m", messages=MSGS)
    assert got.message.content == "ответ, которого модель не давала"


def test_record_needs_a_real_client(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CassetteClient(tmp_path / "x.jsonl", "record")


# ---------------------------------------------------------------------------
# Отпечаток, набор, карантин
# ---------------------------------------------------------------------------
def test_fingerprint_diff_names_what_changed() -> None:
    other = Fingerprint("qwen2.5:7b", "sha256:bbb", "0.41.0", {"temperature": 0})
    changes = FP.diff(other)
    assert any(c.startswith("digest:") for c in changes)
    assert any(c.startswith("ollama:") for c in changes)
    assert FP.diff(FP) == []


def suite_data() -> dict[str, Any]:
    return {
        "name": "docqa",
        "version": "1",
        "items": [{"id": "a"}, {"id": "b"}],
        "baselines": {
            "факт": {
                "runs": [19, 20, 19, 18, 19],
                "n": 20,
                "fingerprint": FP.to_dict(),
                "measured": "2026-11-14 abc",
            }  # fmt: skip
        },
        "quarantine": [{"item": "b", "until": "2026-12-01", "reason": "нестабилен"}],
    }


def test_suite_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite_data(), ensure_ascii=False), encoding="utf-8")
    suite = Suite.load(path)
    suite.save(path)
    again = Suite.load(path)
    assert again.baselines["факт"].runs == [19, 20, 19, 18, 19]
    assert again.quarantine == [Quarantine("b", date(2026, 12, 1), "нестабилен")]


def test_suite_rejects_quarantine_of_unknown_item(tmp_path: Path) -> None:
    data = suite_data()
    data["quarantine"][0]["item"] = "нет-такого"
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="несуществующие"):
        Suite.load(path)


def test_baseline_threshold_lower_bound_and_sensitivity() -> None:
    base = Baseline([19, 20, 19, 18, 19], 20, FP, "x")
    assert base.point == pytest.approx(0.95)
    low, high = base.interval()
    assert low < 0.95 < high
    assert base.threshold("point") == pytest.approx(0.95)
    assert base.threshold("lower") == pytest.approx(low)
    assert base.sensitivity == pytest.approx(0.95 - low)


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------
def run(layer: str, results: list[ItemResult], fp: Fingerprint | None = FP) -> RunResult:
    return RunResult(layer, "abc", "2026-11-14", fp, results)


def suite_obj(tmp_path: Path) -> Suite:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite_data(), ensure_ascii=False), encoding="utf-8")
    return Suite.load(path)


def test_report_shows_flip_direction_kind_and_reason(tmp_path: Path) -> None:
    before = run("replay", [ItemResult("a", "цикл", {"turn_failed": True})])
    after = run("replay", [ItemResult("a", "цикл", {"turn_failed": False})])
    report = compare(suite_obj(tmp_path), before, after, date(2026, 11, 14))
    assert report.red
    text = report.render()
    assert "[цикл]" in text and "a: прошёл → упал — turn_failed" in text


def test_quarantined_item_does_not_gate_until_expired(tmp_path: Path) -> None:
    suite = suite_obj(tmp_path)
    after = run("replay", [ItemResult("a", "x", {"ok": True}), ItemResult("b", "y", {"ok": False})])
    assert not compare(suite, None, after, date(2026, 11, 30)).red
    late = compare(suite, None, after, date(2026, 12, 2))
    assert late.red and late.expired and "просрочен карантин" in late.render()


def test_metrics_gate_only_in_live_layer(tmp_path: Path) -> None:
    suite = suite_obj(tmp_path)
    low = [ItemResult(f"q{i}", "вопрос", {"ok": True}, {"факт": i < 10}) for i in range(20)]
    replay = compare(suite, None, run("replay", low), date(2026, 11, 14))
    live = compare(suite, None, run("live", low), date(2026, 11, 14))
    assert not replay.red and live.red
    assert "НИЖЕ ПОРОГА" in live.render() and "не видят падения меньше" in live.render()


def test_fingerprint_change_is_reported(tmp_path: Path) -> None:
    other = Fingerprint("qwen2.5:7b", "sha256:bbb", "0.40.14", {"temperature": 0})
    before = run("live", [ItemResult("a", "x", {"ok": True})])
    after = run("live", [ItemResult("a", "x", {"ok": True})], other)
    text = compare(suite_obj(tmp_path), before, after, date(2026, 11, 14)).render()
    assert "digest: 'sha256:aaa' → 'sha256:bbb'" in text


def test_run_result_round_trip(tmp_path: Path) -> None:
    result = run("live", [ItemResult("a", "x", {"ok": True}, {"факт": True}, 1.5, "", 0.01)])
    result.save(tmp_path / "r.json")
    again = RunResult.load(tmp_path / "r.json")
    assert again.items[0].metrics == {"факт": True} and again.fingerprint == FP


def test_run_budget_stops_before_the_item_that_does_not_fit() -> None:
    budget = RunBudget(0.10)
    assert budget.before("a", 0.02) == pytest.approx(0.10)
    budget.add(0.09)
    with pytest.raises(RunBudgetExceeded, match="перед b"):
        budget.before("b", 0.02)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('"detail": "PermissionError: [Errno 13] denied: \'/Users/dmitry/docs/a.md\'"', "путь"),
        ('"detail": "1 validation error for Args\\n  input_value={\'file\': 1}"', "pydantic"),
        ("Traceback (most recent call last):\n  File", "трассировка"),
        ('"key": "sk-abcdefghijklmnopqrstuv"', "ключ"),
    ],
)
def test_scan_secrets_finds_what_must_not_be_logged(text: str, kind: str) -> None:
    from ragkit.regress import scan_secrets

    found = scan_secrets(text)
    assert found and kind in found[0]


def test_scan_secrets_passes_clean_log() -> None:
    from ragkit.regress import scan_secrets

    clean = '{"event": "tool", "detail": "PermissionError: [Errno 13] Permission denied: a.md"}'
    assert scan_secrets(clean) == []
