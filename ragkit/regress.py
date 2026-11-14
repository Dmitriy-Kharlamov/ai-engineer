"""Регрессионный набор (Ф5.5-02): набор как данные, кассеты, пороги по нижней границе.

Три слоя, по честности и цене:

* фейковый клиент — модель не зовут вовсе; детерминирован, ловит поломки кода (pytest);
* кассеты — один раз сходили к модели, записали обращения и ответы, дальше проигрываем.
  Детерминированы и быстры, но проверяют только код: запись не знает, что модель с тех пор
  стала отвечать иначе. Изменённый запрос (другой промпт, другая история) — промах кассеты:
  «код спросил не то, что записано», с путём до первого различия;
* живая модель — честно и медленно, каждый раз немного по-разному. Воротами её не ставят:
  метрика сравнивается с порогом по НИЖНЕЙ границе интервала базовой линии, и рядом с
  результатом печатается предел чувствительности — меньшую регрессию ворота не видят.

Набор — файл JSON, а не код: пункты, ожидания, базовые линии метрик с отпечатком среды
(цифровой отпечаток модели, версия ollama, параметры генерации, версия рубрики судьи) и
карантин со сроком. Просроченный карантин роняет ворота: карантин без срока — удаление
проверки, оформленное как забота.

Модуль не знает docqa (правило слоёв): исполнение пунктов — в docqa/regress.py.
"""

import hashlib
import json
import math
import os
import platform
from collections import Counter, deque
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .exceptions import RAGError
from .judge import wilson

# ---------------------------------------------------------------------------
# Сколько пунктов нужно (задание 3)
# ---------------------------------------------------------------------------
Z_ALPHA = 1.645  # односторонний 5 %
Z_BETA = 0.84  # мощность 80 %


def sample_size(p0: float, p1: float, z_alpha: float = Z_ALPHA, z_beta: float = Z_BETA) -> int:
    """Пунктов, чтобы с мощностью 1−β заметить падение доли с p0 до p1 на уровне α.

    n ≈ (z_α·√(p₀(1−p₀)) + z_β·√(p₁(1−p₁)))² / (p₀ − p₁)²
    """
    if not 0 < p1 < p0 < 1:
        raise ValueError("нужно 0 < p1 < p0 < 1")
    top = z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))
    return math.ceil(top * top / (p0 - p1) ** 2)


def min_detectable(p0: float, n: int, z_alpha: float = Z_ALPHA, z_beta: float = Z_BETA) -> float:
    """Наименьшее p1, падение до которого набор из n пунктов заметит с той же мощностью.

    Обратная задача к sample_size: ищем p1 < p0, при котором формула даёт ровно n.
    """
    if n <= 0 or not 0 < p0 < 1:
        raise ValueError("нужно n > 0 и 0 < p0 < 1")

    def needed(p1: float) -> float:
        top = z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))
        return top * top / (p0 - p1) ** 2

    lo, hi = 1e-9, p0 - 1e-9  # needed(lo) мало, needed(hi) → ∞
    if needed(lo) > n:
        return 0.0  # даже падение до нуля не видно
    for _ in range(200):
        mid = (lo + hi) / 2
        if needed(mid) > n:
            hi = mid
        else:
            lo = mid
    return lo


# ---------------------------------------------------------------------------
# Отпечаток среды
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fingerprint:
    """Всё, что меняет ответ модели, кроме кода и запроса. Разные отпечатки — разные среды:
    базовую линию одной нельзя переносить на другую."""

    model: str
    digest: str  # цифровой отпечаток модели из ollama: другой файл весов — другая модель
    ollama: str
    options: Mapping[str, Any]
    rubric: str | None = None  # тег рубрики судьи, если метрика судейская
    platform: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "options": dict(self.options)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fingerprint":
        return cls(
            data["model"],
            data["digest"],
            data["ollama"],
            dict(data.get("options", {})),
            data.get("rubric"),
            data.get("platform", ""),
        )

    def diff(self, other: "Fingerprint") -> list[str]:
        """Что изменилось: «digest: a → b». Пусто — та же среда."""
        mine, theirs = self.to_dict(), other.to_dict()
        out = []
        for key in mine:
            if mine[key] != theirs[key]:
                out.append(f"{key}: {mine[key]!r} → {theirs[key]!r}")
        return out


def ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    if host.count(":") < 2:
        host += ":11434"
    return host.rstrip("/")


def probe_fingerprint(
    model: str, options: Mapping[str, Any], rubric: str | None = None, host: str | None = None
) -> Fingerprint:
    """Отпечаток живой среды: digest модели и версия сервера — у самого ollama."""
    import httpx

    base = host or ollama_host()
    version = httpx.get(f"{base}/api/version", timeout=10).json().get("version", "?")
    tags = httpx.get(f"{base}/api/tags", timeout=10).json().get("models", [])
    digest = next((m.get("digest", "?") for m in tags if m.get("name") == model), "нет модели")
    return Fingerprint(
        model, digest, version, dict(options), rubric, f"{platform.system()}-{platform.machine()}"
    )


# ---------------------------------------------------------------------------
# Кассеты
# ---------------------------------------------------------------------------
KEY_FIELDS = ("model", "messages", "tools", "options", "format", "stream")


def plain(value: Any) -> Any:
    """Значение без обёрток: pydantic-модели ollama, MappingProxy, кортежи — в JSON-форму."""
    if isinstance(value, BaseModel):
        return plain(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_key(request: Mapping[str, Any], digest: str) -> str:
    """Ключ записи: всё, от чего зависит ответ, — поля запроса и отпечаток модели."""
    missing = set(KEY_FIELDS) - set(request)
    if missing:
        raise AssertionError(f"в ключе кассеты нет полей {sorted(missing)}")
    return hashlib.sha256(canonical({"request": request, "digest": digest}).encode()).hexdigest()


def first_difference(recorded: Any, actual: Any, path: str = "") -> str | None:
    """Путь до первого различия двух JSON-значений: «messages[3].content». None — равны."""
    if isinstance(recorded, dict) and isinstance(actual, dict):
        for key in sorted(set(recorded) | set(actual)):
            if key not in recorded or key not in actual:
                return f"{path}.{key}".lstrip(".")
            found = first_difference(recorded[key], actual[key], f"{path}.{key}")
            if found is not None:
                return found.lstrip(".")
        return None
    if isinstance(recorded, list) and isinstance(actual, list):
        for i, (a, b) in enumerate(zip(recorded, actual, strict=False)):
            found = first_difference(a, b, f"{path}[{i}]")
            if found is not None:
                return found
        if len(recorded) != len(actual):
            shorter = min(len(recorded), len(actual))
            return f"{path}[{shorter}]  (длина {len(recorded)} → {len(actual)})"
        return None
    if recorded == actual:
        return None
    if isinstance(recorded, str) and isinstance(actual, str):
        i = next((i for i, (a, b) in enumerate(zip(recorded, actual, strict=False)) if a != b),
                 min(len(recorded), len(actual)))  # fmt: skip
        return f"{path or '<значение>'}@{i}"
    return path or "<значение>"


def _at(value: Any, path: str) -> Any:
    """Значение по пути из first_difference — для отчёта о промахе; у строки — окрестность
    первого различного символа («…@120»)."""
    import re

    where, _, char = path.split("  ")[0].partition("@")
    current = value
    for name, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", where):
        try:
            current = current[int(index)] if index else current[name]
        except (KeyError, IndexError, TypeError):
            return "<нет>"
    if char and isinstance(current, str):
        i = int(char)
        return ("…" if i > 30 else "") + current[max(0, i - 30) : i + 50] + "…"
    return current


class CassetteMiss(RAGError):
    """Запроса нет в записи: код спросил модель не о том, что записано."""

    def __init__(self, path: Path, ordinal: int, where: str | None, recorded: Any, actual: Any):
        def short(v: Any) -> str:
            text = json.dumps(v, ensure_ascii=False)
            return text if len(text) <= 160 else text[:159] + "…"

        if where is None:
            message = f"кассета {path.name}: обращение №{ordinal} — записи больше нет"
        else:
            message = (
                f"кассета {path.name}: обращение №{ordinal} не совпало с записью в {where}: "
                f"записано {short(recorded)}, сейчас {short(actual)}"
            )
        super().__init__(message)
        self.where = where


@dataclass
class _Record:
    key: str
    request: dict[str, Any]
    response: dict[str, Any] | None = None  # обычный ответ
    stream: list[dict[str, Any]] | None = None  # куски потока
    seconds: float = 0.0


class Cassette:
    """Файл JSONL: заголовок с отпечатком, затем по строке на обращение — в порядке хода."""

    def __init__(self, path: Path, fingerprint: Fingerprint, records: list[_Record]) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.records = records

    @classmethod
    def load(cls, path: Path) -> "Cassette":
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"{path}: пустая кассета")
        header = json.loads(lines[0])
        fingerprint = Fingerprint.from_dict(header["fingerprint"])
        records = []
        for number, line in enumerate(lines[1:], start=1):
            data = json.loads(line)
            record = _Record(**data)
            # ключ держится на ассерте: запись, чей ключ не сходится с её же запросом, —
            # битая или отредактированная руками в запросе
            expected = request_key(record.request, fingerprint.digest)
            if record.key != expected:
                raise AssertionError(
                    f"{path.name}, запись {number}: ключ {record.key[:12]} не соответствует "
                    f"запросу ({expected[:12]}) — запись правили или ключ считается иначе"
                )
            records.append(record)
        return cls(path, fingerprint, records)

    @classmethod
    def start(cls, path: Path, fingerprint: Fingerprint) -> "Cassette":
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"cassette": 1, "fingerprint": fingerprint.to_dict()}
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return cls(path, fingerprint, [])

    def append(self, record: _Record) -> None:
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _response(data: Mapping[str, Any]) -> Any:
    from ollama import ChatResponse

    return ChatResponse.model_validate(data)


async def _replay_stream(chunks: Sequence[Mapping[str, Any]]) -> AsyncIterator[Any]:
    for chunk in chunks:
        yield _response(chunk)


class CassetteClient:
    """Обёртка над ollama.AsyncClient: запись или воспроизведение обращений chat().

    Принимает ровно те параметры, что входят в ключ. Любой другой (think, keep_alive,
    logprobs…) — AssertionError: от него зависит ответ, а ключ его не видит, и
    воспроизведение молча вернуло бы ответ на другой запрос.
    """

    def __init__(
        self,
        path: Path,
        mode: Literal["record", "replay"],
        inner: Any = None,
        fingerprint: Fingerprint | None = None,
    ) -> None:
        self.mode = mode
        self.inner = inner
        if mode == "record":
            if inner is None or fingerprint is None:
                raise ValueError("запись: нужен настоящий клиент и отпечаток среды")
            self.cassette = Cassette.start(path, fingerprint)
        else:
            self.cassette = Cassette.load(path)
        self._by_key: dict[str, deque[int]] = {}
        for i, record in enumerate(self.cassette.records):
            self._by_key.setdefault(record.key, deque()).append(i)
        self._used: set[int] = set()
        self.calls = 0

    @property
    def fingerprint(self) -> Fingerprint:
        return self.cassette.fingerprint

    def _take(self, key: str, request: dict[str, Any]) -> _Record:
        self.calls += 1
        queue = self._by_key.get(key)
        if queue:
            index = queue.popleft()
            self._used.add(index)
            return self.cassette.records[index]
        unused = [i for i in range(len(self.cassette.records)) if i not in self._used]
        if not unused:
            raise CassetteMiss(self.cassette.path, self.calls, None, None, None)
        nearest = self.cassette.records[unused[0]].request
        where = first_difference(nearest, plain(request))
        raise CassetteMiss(
            self.cassette.path, self.calls, where, _at(nearest, where or ""),
            _at(plain(request), where or ""),
        )  # fmt: skip

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
        format: Any = None,  # noqa: A002 — имя параметра у ollama
        **extra: Any,
    ) -> Any:
        if extra:
            raise AssertionError(
                f"параметры {sorted(extra)} не входят в ключ кассеты, а ответ от них зависит"
            )
        request = plain({"model": model, "messages": messages, "tools": tools,
                         "options": options, "format": format, "stream": stream})  # fmt: skip
        key = request_key(request, self.fingerprint.digest)
        if self.mode == "replay":
            record = self._take(key, request)
            if stream:
                return _replay_stream(record.stream or [])
            return _response(record.response or {})
        return await self._record(key, request)

    async def _record(self, key: str, request: dict[str, Any]) -> Any:
        import time

        started = time.monotonic()
        kwargs = {k: v for k, v in request.items() if v is not None}
        reply = await self.inner.chat(**kwargs)
        if not request["stream"]:
            self.cassette.append(
                _Record(key, request, response=plain(reply), seconds=time.monotonic() - started)
            )
            return reply
        return self._tee(reply, key, request, started)

    async def _tee(
        self, chunks: AsyncIterator[Any], key: str, request: dict[str, Any], started: float
    ) -> AsyncIterator[Any]:
        """Поток насквозь, с записью кусков; оборванный поток записывается как оборванный."""
        import time

        got: list[dict[str, Any]] = []
        try:
            async for chunk in chunks:
                got.append(plain(chunk))
                yield chunk
        finally:
            self.cassette.append(
                _Record(key, request, stream=got, seconds=time.monotonic() - started)
            )


# ---------------------------------------------------------------------------
# Набор как данные
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Quarantine:
    """Пункт убран из ворот до даты until. После неё — снова в воротах, а пока не убран
    из файла — ворота красные: «срок карантина истёк»."""

    item: str
    until: date
    reason: str

    def expired(self, today: date) -> bool:
        return today > self.until


@dataclass
class Baseline:
    """Базовая линия метрики: значения чистых прогонов в одной среде."""

    runs: list[int]  # k в каждом чистом прогоне
    n: int  # пунктов в метрике
    fingerprint: Fingerprint
    measured: str  # дата и коммит

    @property
    def point(self) -> float:
        return sum(self.runs) / (len(self.runs) * self.n)

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        """Интервал одного прогона из n при доле point: так пляшет метрика сама по себе."""
        return wilson(round(self.point * self.n), self.n, z)

    def threshold(self, rule: Literal["point", "lower"] = "lower") -> float:
        return self.point if rule == "point" else self.interval()[0]

    @property
    def sensitivity(self) -> float:
        """Предел чувствительности ворот: падение меньше point − low они не видят."""
        return self.point - self.interval()[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "n": self.n,
            "fingerprint": self.fingerprint.to_dict(),
            "measured": self.measured,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Baseline":
        return cls(
            list(data["runs"]), int(data["n"]), Fingerprint.from_dict(data["fingerprint"]),
            str(data["measured"]),
        )  # fmt: skip


@dataclass
class Suite:
    name: str
    version: str
    items: list[dict[str, Any]]  # исполнитель (docqa/regress.py) знает их поля
    baselines: dict[str, Baseline] = field(default_factory=dict)
    quarantine: list[Quarantine] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # прочие поля файла — как есть

    @classmethod
    def load(cls, path: Path) -> "Suite":
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = [item["id"] for item in data["items"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: повторяются id пунктов")
        quarantine = [
            Quarantine(q["item"], date.fromisoformat(q["until"]), q["reason"])
            for q in data.get("quarantine", [])
        ]
        unknown = {q.item for q in quarantine} - set(ids)
        if unknown:
            raise ValueError(f"{path}: в карантине несуществующие пункты {sorted(unknown)}")
        known = {"name", "version", "items", "baselines", "quarantine"}
        return cls(
            data["name"],
            data["version"],
            data["items"],
            {k: Baseline.from_dict(v) for k, v in data.get("baselines", {}).items()},
            quarantine,
            {k: v for k, v in data.items() if k not in known},
        )

    def save(self, path: Path) -> None:
        data = {
            "name": self.name,
            "version": self.version,
            **self.extra,
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
            "quarantine": [
                {"item": q.item, "until": q.until.isoformat(), "reason": q.reason}
                for q in self.quarantine
            ],
            "items": self.items,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def quarantined(self, today: date) -> dict[str, Quarantine]:
        """Действующий карантин: пункт → запись. Просроченные сюда не входят."""
        return {q.item: q for q in self.quarantine if not q.expired(today)}

    def expired(self, today: date) -> list[Quarantine]:
        return [q for q in self.quarantine if q.expired(today)]


# ---------------------------------------------------------------------------
# Результаты прогона и отчёт о расхождениях
# ---------------------------------------------------------------------------
@dataclass
class ItemResult:
    id: str
    kind: str  # вид отказа из таблицы или «вопрос»
    checks: dict[str, bool]  # ожидания кода: имя → выполнено
    metrics: dict[str, bool] = field(default_factory=dict)  # вклад в метрики модели
    seconds: float = 0.0
    error: str = ""  # промах кассеты, падение исполнителя — что помешало проверке
    cost_usd: float = 0.0
    details: dict[str, str] = field(default_factory=dict)  # к ожиданию — что увидено

    @property
    def passed(self) -> bool:
        return not self.error and all(self.checks.values())

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]


@dataclass
class RunResult:
    layer: str  # replay | record | live
    commit: str
    started: str
    fingerprint: Fingerprint | None
    items: list[ItemResult]
    seconds: float = 0.0

    def metric(self, name: str) -> tuple[int, int]:
        values = [r.metrics[name] for r in self.items if name in r.metrics]
        return sum(values), len(values)

    def metric_names(self) -> list[str]:
        return sorted({name for r in self.items for name in r.metrics})

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "commit": self.commit,
            "started": self.started,
            "seconds": round(self.seconds, 2),
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
            "items": [asdict(r) for r in self.items],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunResult":
        fp = data.get("fingerprint")
        return cls(
            data["layer"],
            data["commit"],
            data["started"],
            Fingerprint.from_dict(fp) if fp else None,
            [ItemResult(**r) for r in data["items"]],
            float(data.get("seconds", 0.0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=1)
        path.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunResult":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Flip:
    item: str
    kind: str
    was: str  # «прошёл» / «упал» / «нет»
    now: str
    why: str  # не выполненные ожидания или ошибка


@dataclass(frozen=True)
class MetricLine:
    name: str
    k: int
    n: int
    threshold: float | None
    rule: str
    baseline: Baseline | None

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def ok(self) -> bool:
        return self.threshold is None or self.value >= self.threshold - 1e-12


@dataclass
class Report:
    layer: str
    flips: list[Flip]
    failing: list[ItemResult]  # не прошли сейчас (вне карантина)
    metrics: list[MetricLine]
    fingerprint_changes: list[str]
    quarantined: list[tuple[Quarantine, bool]]  # запись, прошёл ли пункт сейчас
    expired: list[Quarantine]
    gate_metrics: bool  # метрики модели — ворота (только live); в replay они копия записи

    @property
    def red(self) -> bool:
        if self.failing or self.expired:
            return True
        return self.gate_metrics and not all(m.ok for m in self.metrics)

    def render(self) -> str:
        out: list[str] = []
        state = "КРАСНЫЙ" if self.red else "зелёный"
        out.append(f"прогон слоя {self.layer}: {state}")
        if self.expired:
            out.append("просрочен карантин — ворота красные, пока запись не убрать или продлить:")
            out += [f"   {q.item}: до {q.until}, {q.reason}" for q in self.expired]
        if self.flips:
            out.append("изменили вердикт (было → стало), по видам отказа:")
            kinds: dict[str, list[Flip]] = {}
            for flip in self.flips:
                kinds.setdefault(flip.kind, []).append(flip)
            for kind, flips in sorted(kinds.items()):
                out.append(f"   [{kind}]")
                for f in flips:
                    reason = f" — {f.why}" if f.why else ""
                    out.append(f"      {f.item}: {f.was} → {f.now}{reason}")
        else:
            out.append("вердикты пунктов не изменились")
        broken = Counter(name for r in self.failing for name in r.failed_checks)
        broken.update("промах кассеты" for r in self.failing if r.error.startswith("промах"))
        if broken:
            out.append(
                "не выполнено, по ожиданиям: "
                + "; ".join(f"{name} — {count}" for name, count in broken.most_common())
            )
        still = [r for r in self.failing if r.id not in {f.item for f in self.flips}]
        if still:
            out.append("падают и падали раньше:")
            out += [f"   {r.id} [{r.kind}]: {why(r)}" for r in still]
        note = {
            "live": "",
            "record": " (запись: живая модель, но один прогон — не ворота)",
        }.get(self.layer, " (в слое кассет — копия записи, не замер; не ворота)")
        out.append(f"метрики модели{note}:")
        for m in self.metrics:
            if m.baseline is None or m.threshold is None:
                out.append(f"   {m.name}: {m.k}/{m.n} — базовой линии нет")
                continue
            lo, hi = m.baseline.interval()
            mark = "ок" if m.ok else f"НИЖЕ ПОРОГА на {(m.threshold - m.value) * 100:.1f} п. п."
            out.append(
                f"   {m.name}: {m.k}/{m.n} = {m.value:.0%}; база {m.baseline.point:.0%} "
                f"[{lo:.0%}–{hi:.0%}] по {len(m.baseline.runs)} прогонам; порог ({m.rule}) "
                f"{m.threshold:.0%} — {mark}; ворота не видят падения меньше "
                f"{m.baseline.sensitivity * 100:.0f} п. п."
            )
        if self.fingerprint_changes:
            out.append("отпечаток среды изменился: " + "; ".join(self.fingerprint_changes))
        else:
            out.append("отпечаток среды тот же")
        if self.quarantined:
            out.append("в карантине (не ворота):")
            for q, passed in self.quarantined:
                state = "прошёл" if passed else "упал"
                out.append(f"   {q.item}: до {q.until} — {q.reason}; сейчас {state}")
        return "\n".join(out)


def why(result: ItemResult) -> str:
    if result.error:
        return result.error
    parts = [
        f"{name} ({result.details[name]})" if name in result.details else name
        for name in result.failed_checks
    ]
    return ", ".join(parts) or "—"


def compare(
    suite: Suite,
    before: RunResult | None,
    after: RunResult,
    today: date,
    rule: Literal["point", "lower"] = "lower",
) -> Report:
    """Отчёт: что изменило вердикт, метрики против порогов, среда, карантин."""
    quarantine = suite.quarantined(today)
    was = {r.id: r for r in before.items} if before else {}
    flips = []
    for r in after.items:
        old = was.get(r.id)
        old_state = "нет" if old is None else ("прошёл" if old.passed else "упал")
        new_state = "прошёл" if r.passed else "упал"
        if old_state != new_state:
            flips.append(Flip(r.id, r.kind, old_state, new_state, "" if r.passed else why(r)))
    failing = [r for r in after.items if not r.passed and r.id not in quarantine]
    metrics = []
    for name in after.metric_names():
        k, n = after.metric(name)
        base = suite.baselines.get(name)
        metrics.append(MetricLine(name, k, n, base.threshold(rule) if base else None, rule, base))
    changes: list[str] = []
    if before and before.fingerprint and after.fingerprint:
        changes = before.fingerprint.diff(after.fingerprint)
    results = {r.id: r for r in after.items}
    quarantined = [(q, results[q.item].passed) for q in quarantine.values() if q.item in results]
    return Report(
        after.layer, flips, failing, metrics, changes, quarantined, suite.expired(today),
        gate_metrics=after.layer == "live",
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Журнал без секретов (остаток Ф3-12)
# ---------------------------------------------------------------------------
SECRET_PATTERNS: dict[str, str] = {
    "абсолютный путь с именем пользователя": r"/(?:Users|home)/[^/\s\"']+/|[A-Za-z]:\\\\Users",
    "сырой текст pydantic": r"input_value=|errors\.pydantic\.dev|validation errors? for ",
    "трассировка": r"Traceback \(most recent call last\)",
    "ключ или токен": r"\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{20,}",
}


def scan_secrets(text: str) -> list[str]:
    """Что из запрещённого нашлось в журнале: «вид: …кусок…». Пусто — журнал чист."""
    import re

    found = []
    for name, pattern in SECRET_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            start = max(0, match.start() - 30)
            found.append(f"{name}: …{text[start : match.end() + 30]}…")
    return found


# ---------------------------------------------------------------------------
# Бюджет всего прогона
# ---------------------------------------------------------------------------
class RunBudgetExceeded(RAGError):
    def __init__(self, spent: float, limit: float, item: str) -> None:
        super().__init__(
            f"бюджет прогона исчерпан перед {item}: потрачено {spent:.4f} $ из {limit:.4f} $ — "
            "остальные пункты не запускаются"
        )


@dataclass
class RunBudget:
    """CostGuard на весь прогон: каждому пункту — остаток как порог его сессии."""

    limit_usd: float
    spent_usd: float = 0.0

    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def before(self, item: str, minimum: float) -> float:
        """Остаток для пункта; меньше minimum — RunBudgetExceeded, пункт не запускается."""
        left = self.remaining()
        if left < minimum:
            raise RunBudgetExceeded(self.spent_usd, self.limit_usd, item)
        return left

    def add(self, cost: float) -> None:
        self.spent_usd += cost


def summarize(values: Iterable[int], n: int) -> str:
    vals = list(values)
    return f"{min(vals)}–{max(vals)} из {n}" if vals else "—"
