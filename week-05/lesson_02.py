"""Ф5-02: эмбеддинги и косинусная близость на практике.

Корпус — corpus/polka (16 документов «Полки», 56 вопросов: 50 с ответом, 6 без ответа в корпусе).
Генератор и судья — как в Ф5-01: qwen2.5:7b, temperature 0, seed 42; судья fact@v1 со своим кешем.
Модель эмбеддингов — одна, bge-m3 (многоязычная, 1024 измерения; по карточке приставки query:/
passage: не нужны). Сравнение моделей — Ф5-03, сюда не мешаю.

    ollama pull bge-m3
    python -u week-05/lesson_02.py --task 1 2>&1 | tee week-05/lesson_02_task1.txt   # оракул, ~6 мин
    python -u week-05/lesson_02.py --task 2 2>&1 | tee week-05/lesson_02_task2.txt
    python -u week-05/lesson_02.py --task 3 2>&1 | tee week-05/lesson_02_task3.txt
    python -u week-05/lesson_02.py --task 4 2>&1 | tee week-05/lesson_02_task4.txt
    python -u week-05/lesson_02.py --task 5 2>&1 | tee week-05/lesson_02_task5.txt
    python -u week-05/lesson_02.py --task 6 2>&1 | tee week-05/lesson_02_task6.txt   # блок в наборе

Задания 2–6 не запускаются, пока задание 1 не записало, что корпус прошёл три условия приёмки.
Векторы — в кеше week-05/data/embed_cache.jsonl по ключу (отпечаток модели, текст): повторный
запуск в модель не ходит; задание 6 меряет цены на холодную, мимо кеша.

Чанки — абзацы подряд до 500 знаков (ragkit.pipeline.Paragraphs): 41 чанк. Золотой чанк вопроса —
чанк его документа, сильнее всего перекрывающий золотой диапазон.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import ollama

from docqa.documents import WORD, DocumentStore, stem
from ragkit.chunking import Chunk
from ragkit.embedding import (
    EmbeddingCache,
    OllamaEmbedder,
    Vectors,
    VectorSpace,
)
from ragkit.judge import Case, wilson
from ragkit.pipeline import (
    BlockContext,
    Doc,
    FolderLoader,
    OllamaGenerator,
    OracleRetriever,
    Paragraphs,
    Pipeline,
    WholeDocument,
)
from ragkit.regress import Fingerprint, Suite, paired, probe_fingerprint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = ROOT / "corpus" / "polka"
DOCS = CORPUS / "docs"
SUITE = ROOT / "regress" / "suite.json"
DATA = HERE / "data"
ORACLE = DATA / "lesson_02_oracle.json"
EMBED_CACHE = DATA / "embed_cache.jsonl"
MODEL = "qwen2.5:7b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}
EMBED_MODEL = "bge-m3"
CHUNK_CHARS = 500
KINDS = ("перефразировка", "синоним", "обозначение", "буквально")
# Условия приёмки корпуса (задание 1)
MIN_QUESTIONS = 40
MAX_ORACLE = 0.85
MAX_WORD_HIT = 0.70
# «Значимое слово» для проверки видов: всё, кроме вопросительных и служебных слов
STOP_WORDS = (
    "как какой какая какие каков сколько что где когда можно ли если мне меня мой моя мою мои "
    "это есть будет быть нет для при уже ещё еще нужно надо вас вам ваш вашего вашей вы его она "
    "они них там тут так тоже только чем чего кто кому почему зачем или над под без про через "
    "после него ним был была было были вот всё все весь этот эта эти того тот куда откуда очень "
    "себя свой сам сама иметь получится"
)
STOP = set(STOP_WORDS.split())

PREDICTIONS = {
    "1": "потолок по оракулу 78–86 % (44–48 из 56): с документом в контексте 7B ошибётся на "
    "счёте (600 баллов из 1000, суббота после 16:00), на исключениях (штамп у книги 1985 года, "
    "ребёнок 5 лет в мастерской 7–12) и на «можно ли» с запретом в другом абзаце; судья fact@v1 "
    "добавит 1–3 придирки к лишнему. Попадание поиском по словам я уже замерил при сборке "
    "корпуса — это условие приёмки, а код детерминирован: 31 из 50 = 62 % (перефразировка "
    "0/13, синоним 12/15, обозначение 9/12, буквально 10/10); прогноз — то же число",
    "2": "bge-m3: медиана «вопрос ↔ золотой чанк» 0,60–0,68; «два случайных чанка разных "
    "документов» 0,38–0,48 — фон далеко от нуля; «вопрос ↔ чанк другого документа» 0,35–0,45; "
    "текст ↔ он сам 1,000 (отклонение пакет/по одному < 1e-3); текст ↔ перефразировка 0,70–0,85",
    "3": "порог, сохраняющий 90 % релевантных пар, ≈ 0,52 (10-й перцентиль релевантных); "
    "нерелевантных при нём проходит 5–15 %; из 6 вопросов без ответа порог пропустит 4–6: "
    "их ближайший чанк не дальше, чем у вопросов с ответом",
    "4": "близость к вопросу: сам диапазон 0,62, в чанке на 500 знаков 0,60, весь документ "
    "(~900 знаков) 0,57, 2000 знаков (документ + чужой текст) 0,50; приставки query:/passage: на "
    "bge-m3 (не требует их) — медиана ±2 %, попадание ±1 вопрос; перестановка вопроса и чанка — "
    "ровно 0; пакет против отправки по одному — < 1e-3",
    "5": "попадание нужного документа первым результатом, слова → векторы: перефразировка 0/13 → "
    "9–12/13; синоним 12/15 → 13–15/15; обозначение 9/12 → 5–8/12 (соседние артикулы и даты "
    "векторы путают); буквально 10/10 → 9–10/10. Различимо попунктно — только перефразировка",
    "6": "корпус 41 чанк: эмбеддинг 2–6 с (плюс загрузка модели 2–5 с при первом вызове); один "
    "вопрос 30–80 мс; векторы 1024 × float32 = 4 КБ на чанк при ~600 байтах текста чанка → "
    "в 5–8 раз больше документов (в кеше base64 с JSON — ещё ×1,4)",
}


# ---------------------------------------------------------------------------
# Корпус
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    text: str
    doc: str | None  # где ответ; None — ответа в корпусе нет
    judge_doc: str  # документ, который видит судья
    start: int
    end: int
    fact: str


@dataclass(frozen=True)
class Corpus:
    docs: list[Doc]
    questions: list[Question]
    paraphrases: list[tuple[str, str]]
    signature: str  # версии документов и вопросы: другой корпус — другие замеры

    def text(self, name: str) -> str:
        return next(d.text for d in self.docs if d.name == name)

    @property
    def answerable(self) -> list[Question]:
        return [q for q in self.questions if q.doc is not None]


@cache
def corpus() -> Corpus:
    data = json.loads((CORPUS / "questions.json").read_text(encoding="utf-8"))
    docs = FolderLoader(DOCS).load()
    now = {d.name: d.version for d in docs}
    if now != data["versions"]:
        changed = sorted(
            n for n in set(now) | set(data["versions"]) if now.get(n) != data["versions"].get(n)
        )
        sys.exit(f"документы корпуса разошлись с разметкой: {changed} — пересмотри questions.json")  # fmt: skip
    questions = []
    for r in data["questions"]:
        if r.get("doc"):
            text = next(d.text for d in docs if d.name == r["doc"])
            if text[r["start"] : r["end"]] != r["span"]:
                sys.exit(f"{r['id']}: золотой диапазон не совпал с текстом {r['doc']}")
            questions.append(Question(r["id"], r["kind"], r["question"], r["doc"], r["doc"],
                                      r["start"], r["end"], r["fact"]))  # fmt: skip
        else:
            questions.append(Question(r["id"], r["kind"], r["question"], None, r["judge_doc"], 0, 0,
                                      r["fact"]))  # fmt: skip
    signature = hashlib.sha256(
        json.dumps([data["versions"], data["questions"]], ensure_ascii=False).encode()
    ).hexdigest()[:12]
    pairs = [(p["text"], p["paraphrase"]) for p in data["paraphrases"]]
    return Corpus(docs, questions, pairs, signature)


def content(text: str) -> set[str]:
    return {stem(w) for w in WORD.findall(text) if w.casefold() not in STOP}


@cache
def store() -> DocumentStore:
    return DocumentStore(DOCS)


def word_docs(question: str, k: int = 1) -> list[str]:
    """Поиск по словам docqa (search_documents) по тексту вопроса, как подстраховка в Ф5-01."""
    hits = store().search(" ".join(question.split())[:200])["results"]
    return [str(h["name"]) for h in hits[:k]]


@cache
def chunks() -> list[Chunk]:
    return Paragraphs(CHUNK_CHARS).chunk(corpus().docs)


def span_of(i: int) -> tuple[int, int]:
    start, end = chunks()[i].metadata["start"], chunks()[i].metadata["end"]
    assert isinstance(start, int) and isinstance(end, int)
    return start, end


def gold_chunk(q: Question) -> int:
    """Индекс чанка документа q, сильнее всего перекрывающего золотой диапазон."""
    best, overlap = -1, -1
    for i, c in enumerate(chunks()):
        if c.source != q.doc:
            continue
        start, end = span_of(i)
        here = min(end, q.end) - max(start, q.start)
        if here > overlap:
            best, overlap = i, here
    return best


# ---------------------------------------------------------------------------
# Общие мелочи
# ---------------------------------------------------------------------------
def predict(task: str) -> None:
    print(f"   прогноз (до запуска): {PREDICTIONS[task]}")


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]" if n else "—"


def dist(values: Sequence[float]) -> str:
    v = sorted(values)
    q = np.quantile(v, [0.1, 0.9])
    return (f"n={len(v)}: мин {v[0]:.3f}, p10 {q[0]:.3f}, медиана {statistics.median(v):.3f}, "
            f"p90 {q[1]:.3f}, макс {v[-1]:.3f}")  # fmt: skip


def entry(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "point": round(k / n, 3) if n else None, "low": round(lo, 3),
            "high": round(hi, 3), "sensitivity": round(k / n - lo, 3) if n else None}  # fmt: skip


def load_f55_lesson_01() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "f55_lesson_01", ROOT / "week-05.5" / "lesson_01.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["f55_lesson_01"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Задание 1. Корпус, который способен различать
# ---------------------------------------------------------------------------
def oracle_runs() -> dict[str, Any]:
    """Оракул на всех вопросах + судья; кеш — ORACLE, пока не сменился корпус."""
    c = corpus()
    if ORACLE.exists():
        saved = json.loads(ORACLE.read_text(encoding="utf-8"))
        if saved.get("signature") == c.signature:
            print(f"   оракул: из {ORACLE.relative_to(ROOT)} (корпус {c.signature})")
            return dict(saved)
    gold = {q.text: q.doc for q in c.questions}
    pipe = Pipeline(FolderLoader(DOCS), WholeDocument(), OracleRetriever(gold), BlockContext(),
                    OllamaGenerator(ollama.Client(), MODEL, OPTIONS))  # fmt: skip
    pipe.build_index()
    l1 = load_f55_lesson_01()
    runner = l1.Runner(None)
    rows = []
    for q in c.questions:
        a = pipe.answer(q.text)
        text = a.generation.text.strip()
        result = runner.run(l1.FACT_V1, Case(q.text, c.text(q.judge_doc), (text,)))
        verdict = None if result is None else result.verdict == "верно"
        regex = re.search(q.fact, text, re.IGNORECASE) is not None
        rows.append({"id": q.id, "answer": text, "judge": verdict, "regex": regex,
                     "prompt_tokens": a.generation.prompt_tokens,
                     "seconds": round(a.generation.seconds, 2)})  # fmt: skip
        print(f"   оракул {q.id} [{q.kind}]: судья {'верно' if verdict else 'НЕВЕРНО' if verdict is False else '—'}, "
              f"регулярка {'да' if regex else 'нет'}; {a.generation.seconds:.1f} с", flush=True)  # fmt: skip
    print(f"   судья fact@v1: обращений {runner.calls}, из кеша {runner.hits}, ошибок разбора {runner.failures}")  # fmt: skip
    saved = {"signature": c.signature, "model": MODEL, "options": OPTIONS, "rows": rows}
    DATA.mkdir(parents=True, exist_ok=True)
    ORACLE.write_text(json.dumps(saved, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return saved


def task1() -> None:
    print("\n== Задание 1. Корпус, который способен различать")
    predict("1")
    c = corpus()
    by_kind = {k: [q for q in c.questions if q.kind == k] for k in (*KINDS, "нет ответа")}
    print(f"   документов {len(c.docs)}, {sum(len(d.text) for d in c.docs)} знаков; вопросов "
          f"{len(c.questions)}: " + ", ".join(f"{k} {len(v)}" for k, v in by_kind.items()))  # fmt: skip
    # виды проверяются кодом, а не на слово
    para = [
        (q.id, sorted(content(q.text) & content(c.text(q.doc or ""))))
        for q in by_kind["перефразировка"]
    ]
    leaks = [p for p in para if p[1]]
    print(f"   перефразировка: значимых слов вопроса в документе — у {len(para) - len(leaks)} из "
          f"{len(para)} ноль" + (f"; утечки: {leaks}" if leaks else ""))  # fmt: skip
    literal = [(q.id, sorted(content(q.text) - content(c.text(q.doc or "")[q.start : q.end])))
               for q in by_kind["буквально"]]  # fmt: skip
    gaps = [p for p in literal if p[1]]
    print(f"   буквально: все значимые слова вопроса в золотом диапазоне — у {len(literal) - len(gaps)} "
          f"из {len(literal)}" + (f"; не хватает: {gaps}" if gaps else ""))  # fmt: skip
    pairs = {"РТ-4401": "РТ-4402", "14 декабря": "15 декабря", "Гарипова": "Гарифуллин"}
    for mine, neighbour in pairs.items():
        where = [d.name for d in c.docs if mine in d.text]
        there = [d.name for d in c.docs if neighbour in d.text]
        print(f"   соседние обозначения: {mine} — {where}, {neighbour} — {there}")

    # условие 3: поиск по словам
    print("\n   попадание нужного документа поиском по словам (первый результат search_documents):")
    hits_all = 0
    for kind in KINDS:
        qs = by_kind[kind]
        hits = [q for q in qs if word_docs(q.text) == [q.doc]]
        hits_all += len(hits)
        miss = [f"{q.id}→{(word_docs(q.text) or ['—'])[0]}" for q in qs if q not in hits]
        print(
            f"      {kind:<15} {share(len(hits), len(qs))}"
            + (f"; мимо: {', '.join(miss)}" if miss else "")
        )
    answerable = len(c.answerable)
    word_hit = hits_all / answerable

    # условие 2: оракул
    print(
        "\n   оракул — пайплайн Ф5-01 с документом из золотой разметки (вопрос без ответа — пустой контекст):"
    )
    saved = oracle_runs()
    rows = {r["id"]: r for r in saved["rows"]}
    for kind in (*KINDS, "нет ответа"):
        qs = by_kind[kind]
        ok = sum(rows[q.id]["judge"] is True for q in qs)
        print(
            f"      {kind:<15} судья {share(ok, len(qs))}, регулярка {sum(rows[q.id]['regex'] for q in qs)}/{len(qs)}"
        )
    right = sum(r["judge"] is True for r in saved["rows"])
    ceiling = right / len(c.questions)
    for q in c.questions:
        r = rows[q.id]
        if r["judge"] is not True or not r["regex"]:
            mark = (
                "судья НЕВЕРНО"
                if r["judge"] is False
                else "судья —"
                if r["judge"] is None
                else "судья верно, регулярка нет"
            )
            print(f"      {q.id} {mark}: {q.text} → «{r['answer'][:150]}»")

    print("\n   УСЛОВИЯ ПРИЁМКИ:")
    checks = {
        f"вопросов с золотой разметкой ≥ {MIN_QUESTIONS}": (answerable >= MIN_QUESTIONS, f"{answerable} (и {len(c.questions) - answerable} без ответа)"),
        f"потолок по оракулу ≤ {MAX_ORACLE:.0%}": (ceiling <= MAX_ORACLE, share(right, len(c.questions))),
        f"попадание поиском по словам ≤ {MAX_WORD_HIT:.0%}": (word_hit <= MAX_WORD_HIT, share(hits_all, answerable)),
    }  # fmt: skip
    for name, (ok, value) in checks.items():
        print(f"      {'✓' if ok else '✗'} {name}: {value}")
    passed = all(ok for ok, _ in checks.values())
    saved["conditions"] = {"passed": passed, "ceiling": [right, len(c.questions)],
                           "word_hit": [hits_all, answerable]}  # fmt: skip
    ORACLE.write_text(json.dumps(saved, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        "   → корпус принят"
        if passed
        else "   → НЕ ПРИНЯТ: доделывай корпус, задания 2–6 не запускаются"
    )


def require_accepted() -> dict[str, Any]:
    if not ORACLE.exists():
        sys.exit("нет week-05/data/lesson_02_oracle.json: сначала задание 1")
    saved: dict[str, Any] = json.loads(ORACLE.read_text(encoding="utf-8"))
    if saved.get("signature") != corpus().signature or not saved.get("conditions", {}).get(
        "passed"
    ):
        sys.exit("корпус не прошёл условия приёмки (или изменился после задания 1) — задание 1")
    return saved


# ---------------------------------------------------------------------------
# Векторы
# ---------------------------------------------------------------------------
@cache
def embed_fingerprint() -> Fingerprint:
    fp = probe_fingerprint(EMBED_MODEL, {"normalize": "при сохранении", "truncate": False})
    if fp.digest == "нет модели":
        sys.exit(f"в ollama нет {EMBED_MODEL}: ollama pull {EMBED_MODEL}")
    return fp


@cache
def embedder() -> OllamaEmbedder:
    return OllamaEmbedder(ollama.Client(), EMBED_MODEL, embed_fingerprint().digest)


@cache
def space(query_prefix: str = "", passage_prefix: str = "") -> VectorSpace:
    return VectorSpace(embedder(), EmbeddingCache(EMBED_CACHE), query_prefix, passage_prefix)


@dataclass(frozen=True)
class Measured:
    q: Vectors  # вопросы, в порядке corpus().questions
    c: Vectors  # чанки, в порядке chunks()
    sims: np.ndarray  # вопрос × чанк


@cache
def measured(query_prefix: str = "", passage_prefix: str = "") -> Measured:
    s = space(query_prefix, passage_prefix)
    q = s.queries([x.text for x in corpus().questions])
    c = s.passages([x.text for x in chunks()])
    return Measured(q, c, q.cosine(c))


def qi(q: Question) -> int:
    return corpus().questions.index(q)


def relevant(m: Measured) -> list[float]:
    return [float(m.sims[qi(q), gold_chunk(q)]) for q in corpus().answerable]


def q_other(m: Measured) -> list[float]:
    """Вопрос ↔ каждый чанк других документов (для вопроса без ответа — всех)."""
    out = []
    for q in corpus().answerable:
        out += [float(m.sims[qi(q), j]) for j, ch in enumerate(chunks()) if ch.source != q.doc]
    return out


def chunk_pairs(m: Measured) -> list[float]:
    cs, sims = chunks(), m.c.cosine(m.c)
    return [float(sims[i, j]) for i in range(len(cs)) for j in range(i + 1, len(cs))
            if cs[i].source != cs[j].source]  # fmt: skip


def top_docs(m: Measured, q: Question, k: int = 1) -> list[str]:
    order = np.argsort(-m.sims[qi(q)], kind="stable")
    out: list[str] = []
    for j in order:
        name = chunks()[int(j)].source
        if name not in out:
            out.append(name)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# Задание 2. Шкала косинуса на этих данных
# ---------------------------------------------------------------------------
def task2() -> None:
    require_accepted()
    print("\n== Задание 2. Какая на этих данных шкала у косинуса")
    predict("2")
    fp = embed_fingerprint()
    print(f"   модель {fp.model}, digest {fp.digest[:19]}…, ollama {fp.ollama}; чанков {len(chunks())} "
          f"(абзацы до {CHUNK_CHARS} знаков), вопросов {len(corpus().questions)}")  # fmt: skip
    m = measured()
    straddle = [q.id for q in corpus().answerable
                if not (span_of(gold_chunk(q))[0] <= q.start and q.end <= span_of(gold_chunk(q))[1])]  # fmt: skip
    print(f"   золотой диапазон целиком в одном чанке — у {len(corpus().answerable) - len(straddle)} из "
          f"{len(corpus().answerable)}" + (f" (через границу: {straddle})" if straddle else ""))  # fmt: skip

    # текст ↔ он сам: тот же текст отдельным вызовом, мимо кеша
    alone = OllamaEmbedder(ollama.Client(), EMBED_MODEL, fp.digest, batch_size=1)
    texts = [ch.text for ch in chunks()][:12]
    single = VectorSpace(alone, EmbeddingCache()).passages(texts)
    batch = m.c.rows(range(len(texts)))
    self_sims = [float(x) for x in np.sum(single.matrix * batch.matrix, axis=1)]
    print("\n   пять видов пар (косинус):")
    print(f"      вопрос ↔ его золотой чанк              {dist(relevant(m))}")
    print(f"      вопрос ↔ чанк другого документа        {dist(q_other(m))}")
    print(f"      два чанка разных документов            {dist(chunk_pairs(m))}")
    print(f"      текст ↔ он сам (пакетом и по одному)   {dist(self_sims)}")
    s = space()
    para = [float(s.passages([a]).cosine(s.passages([b]))[0, 0]) for a, b in corpus().paraphrases]
    print(f"      текст ↔ его перефразировка             {dist(para)}")
    shuffled = [b for _, b in corpus().paraphrases[1:] + corpus().paraphrases[:1]]
    foreign = [float(s.passages([a]).cosine(s.passages([b]))[0, 0])
               for (a, _), b in zip(corpus().paraphrases, shuffled, strict=True)]  # fmt: skip
    print(f"      (для сравнения: фраза ↔ чужая перефразировка {dist(foreign)})")
    norms = alone.raw_norms
    print(f"\n   нормы векторов от сервера до нормировки: {min(norms):.4f}–{max(norms):.4f} "
          f"({'сервер уже нормирует' if max(abs(n - 1) for n in norms) < 1e-3 else 'сервер не нормирует'}; "
          "нормирую всё равно — один раз, при сохранении)")  # fmt: skip
    background = statistics.median(chunk_pairs(m))
    rel = statistics.median(relevant(m))
    print(f"   ФОН (медиана «два чанка разных документов»): {background:.3f}; медиана релевантных "
          f"{rel:.3f}; зазор {rel - background:+.3f}")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 3. Порог и его цена
# ---------------------------------------------------------------------------
def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Доля пар (релевантная, нерелевантная), где релевантная выше: 0,5 — не различает вовсе."""
    p, n = np.asarray(pos), np.asarray(neg)
    return float(
        ((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum())
        / (len(p) * len(n))
    )


def task3() -> None:
    require_accepted()
    print("\n== Задание 3. Где ставить порог и чем за него платишь")
    predict("3")
    m = measured()
    rel, irr = relevant(m), q_other(m)
    print(f"   релевантные (вопрос ↔ золотой чанк): {dist(rel)}")
    print(f"   нерелевантные (вопрос ↔ чанк другого документа): {dist(irr)}")
    print(f"   перекрытие: AUC {auc(rel, irr):.3f}; нерелевантных выше худшей релевантной — "
          f"{sum(x >= min(rel) for x in irr) / len(irr):.1%}; релевантных ниже лучшей нерелевантной — "
          f"{sum(x <= max(irr) for x in rel) / len(rel):.0%}")  # fmt: skip
    hard = [float(max(m.sims[qi(q), j] for j, ch in enumerate(chunks()) if ch.source != q.doc))
            for q in corpus().answerable]  # fmt: skip
    beaten = sum(h > r for h, r in zip(hard, rel, strict=True))
    print(f"   самый близкий чужой чанк (для каждого вопроса): {dist(hard)}; ближе золотого — у {beaten} "
          f"из {len(rel)} вопросов")  # fmt: skip
    none = [q for q in corpus().questions if q.doc is None]
    top_none = [float(m.sims[qi(q)].max()) for q in none]
    top_yes = [float(m.sims[qi(q)].max()) for q in corpus().answerable]
    print(f"   лучший чанк у вопросов С ответом: {dist(top_yes)}")
    print(f"   лучший чанк у вопросов БЕЗ ответа: {dist(top_none)}")
    p10 = float(np.quantile(rel, 0.10))
    candidates = sorted({round(x, 2) for x in (*np.quantile(rel, [0.05, 0.10, 0.25, 0.5]),
                                                *np.quantile(irr, [0.9, 0.99]), max(irr))})  # fmt: skip
    print(
        "\n   порог | релевантных сохраняет | нерелевантных пропускает | вопросов без ответа "
        "пропускает | верный первый результат выше порога"
    )
    for t in candidates:
        keep = sum(x >= t for x in rel) / len(rel)
        leak = sum(x >= t for x in irr) / len(irr)
        nn = sum(x >= t for x in top_none)
        first = sum(
            top_docs(m, q) == [q.doc] and m.sims[qi(q)].max() >= t for q in corpus().answerable
        )
        print(f"   {t:.2f}  | {keep:>6.0%}                | {leak:>6.1%}                   | "
              f"{nn}/{len(none)}                         | {first}/{len(rel)}")  # fmt: skip
    print(f"   порог для 90 % релевантных: {p10:.3f}; нерелевантных при нём проходит "
          f"{sum(x >= p10 for x in irr) / len(irr):.1%}, вопросов без ответа — "
          f"{sum(x >= p10 for x in top_none)} из {len(none)}")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 4. Длина и асимметрия
# ---------------------------------------------------------------------------
def window(text: str, start: int, end: int, size: int) -> str:
    if end - start >= size or len(text) <= size:
        return text if end - start < size else text[start:end]
    mid = (start + end) // 2
    left = max(0, min(len(text) - size, mid - size // 2))
    return text[left : left + size]


def task4() -> None:
    require_accepted()
    print("\n== Задание 4. Длина и асимметрия")
    predict("4")
    c, s = corpus(), space()
    rng = random.Random(2)
    rows: dict[str, list[float]] = {"сам диапазон": [], "окно 500 знаков": [], "весь документ": [],
                                    "2000 знаков (документ + чужой текст)": []}  # fmt: skip
    lengths: dict[str, list[int]] = {k: [] for k in rows}
    for q in c.answerable:
        doc = c.text(q.doc or "")
        others = [d.text for d in c.docs if d.name != q.doc]
        rng.shuffle(others)
        wrappers = {
            "сам диапазон": doc[q.start : q.end],
            "окно 500 знаков": window(doc, q.start, q.end, 500),
            "весь документ": doc,
            "2000 знаков (документ + чужой текст)": (doc + "\n\n" + "\n\n".join(others))[:2000],
        }
        qv = s.queries([q.text])
        for name, text in wrappers.items():
            rows[name].append(float(qv.cosine(s.passages([text]))[0, 0]))
            lengths[name].append(len(text))
    print("   (а) один и тот же ответ в разной обёртке, косинус с вопросом:")
    for name, values in rows.items():
        print(f"      {name:<38} длина ~{statistics.median(lengths[name]):>5.0f}; {dist(values)}")
    names = list(rows)
    falls = sum(all(rows[names[i]][k] >= rows[names[i + 1]][k] for i in range(len(names) - 1))
                for k in range(len(c.answerable)))  # fmt: skip
    print(f"      падает на каждом шаге — у {falls} из {len(c.answerable)} вопросов; медиана разности "
          f"«диапазон − 2000»: {statistics.median(a - b for a, b in zip(rows[names[0]], rows[names[-1]], strict=True)):+.3f}")  # fmt: skip

    print(
        "\n   (б) асимметрия. Карточка bge-m3: приставки для поиска не нужны — вопрос и чанк "
        "отправляются как есть"
    )
    m = measured()
    swapped = m.c.cosine(m.q).T
    print(
        f"      перестановка (вопрос, чанк) → (чанк, вопрос): макс. |Δ| = {float(np.abs(m.sims - swapped).max()):.2e}"
    )
    alone = OllamaEmbedder(ollama.Client(), EMBED_MODEL, embed_fingerprint().digest, batch_size=1)
    single = VectorSpace(alone, EmbeddingCache()).queries([q.text for q in c.questions])
    delta = np.abs(single.cosine(m.c) - m.sims)
    print(f"      вопросы по одному против пакета: макс. |Δ косинуса| = {float(delta.max()):.2e}, "
          f"смен первого результата — {sum(top_docs(m, q) != [chunks()[int(np.argmax(single.cosine(m.c)[qi(q)]))].source] for q in c.answerable)}")  # fmt: skip
    pm = measured("query: ", "passage: ")
    rel0, rel1 = statistics.median(relevant(m)), statistics.median(relevant(pm))
    hit0 = sum(top_docs(m, q) == [q.doc] for q in c.answerable)
    hit1 = sum(top_docs(pm, q) == [q.doc] for q in c.answerable)
    bg0, bg1 = statistics.median(chunk_pairs(m)), statistics.median(chunk_pairs(pm))
    print(f"      с приставками query:/passage: (как у e5; bge-m3 их не просит): медиана релевантных "
          f"{rel0:.3f} → {rel1:.3f} ({(rel1 - rel0) / rel0:+.1%}), фон {bg0:.3f} → {bg1:.3f} "
          f"({(bg1 - bg0) / bg0:+.1%}), попадание первым {hit0} → {hit1} из {len(c.answerable)}")  # fmt: skip
    print(f"      отпечатки пространств: без приставок {m.q.fingerprint}; с ними {pm.q.fingerprint} — "
          "сравнить вектор одного с вектором другого Vectors.cosine() не даст")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 5. Что эмбеддинги покупают и что теряют
# ---------------------------------------------------------------------------
def hits_by_kind(m: Measured) -> dict[str, tuple[list[bool], list[bool], list[bool], list[bool]]]:
    out = {}
    for kind in KINDS:
        qs = [q for q in corpus().answerable if q.kind == kind]
        out[kind] = (
            [word_docs(q.text) == [q.doc] for q in qs],
            [top_docs(m, q) == [q.doc] for q in qs],
            [q.doc in word_docs(q.text, 3) for q in qs],
            [q.doc in top_docs(m, q, 3) for q in qs],
        )
    return out


def task5() -> None:
    require_accepted()
    print("\n== Задание 5. Что эмбеддинги покупают и что теряют")
    predict("5")
    m = measured()
    table = hits_by_kind(m)
    print(
        "   попадание нужного документа; слова — search_documents docqa, векторы — косинус по чанкам "
        "(документ лучшего чанка); попунктно — paired(), A = слова, B = векторы"
    )
    print(
        f"   {'вид':<15} {'n':>3} | {'слова @1':>9} | {'векторы @1':>10} | {'слова @3':>8} | {'векторы @3':>10} | попунктно @1"
    )
    allw: list[bool] = []
    allv: list[bool] = []
    for kind, (w1, v1, w3, v3) in table.items():
        allw += w1
        allv += v1
        verdict = paired(w1, v1)
        print(f"   {kind:<15} {len(w1):>3} | {sum(w1):>4}/{len(w1):<4} | {sum(v1):>5}/{len(v1):<4} | "
              f"{sum(w3):>3}/{len(w3):<4} | {sum(v3):>5}/{len(v3):<4} | {verdict.text('слова', 'векторы')}")  # fmt: skip
    print(f"   {'всего':<15} {len(allw):>3} | {sum(allw):>4}/{len(allw):<4} | {sum(allv):>5}/{len(allv):<4} | "
          f"{'':>8} | {'':>10} | {paired(allw, allv).text('слова', 'векторы')}")  # fmt: skip
    print(
        "\n   промахи векторов первым результатом (какой документ встал первым и с каким косинусом):"
    )
    for q in corpus().answerable:
        top = top_docs(m, q)
        if top != [q.doc]:
            best = float(m.sims[qi(q)].max())
            print(f"      {q.id} [{q.kind}] {q.text} → {top[0]} ({best:.3f}; золотой чанк "
                  f"{float(m.sims[qi(q), gold_chunk(q)]):.3f}); слова → {(word_docs(q.text) or ['—'])[0]}")  # fmt: skip


# ---------------------------------------------------------------------------
# Задание 6. Модуль, цена и отпечаток
# ---------------------------------------------------------------------------
def task6() -> None:
    saved = require_accepted()
    print("\n== Задание 6. Модуль, цена и отпечаток")
    predict("6")
    fp = embed_fingerprint()
    texts = [ch.text for ch in chunks()]
    warm = OllamaEmbedder(ollama.Client(), EMBED_MODEL, fp.digest)
    warm.embed(["прогрев: модель в памяти"])
    cold = OllamaEmbedder(ollama.Client(), EMBED_MODEL, fp.digest, batch_size=32)
    began = time.monotonic()
    vectors = VectorSpace(cold, EmbeddingCache()).passages(texts)
    corpus_s = time.monotonic() - began
    one = OllamaEmbedder(ollama.Client(), EMBED_MODEL, fp.digest, batch_size=1)
    began = time.monotonic()
    VectorSpace(one, EmbeddingCache()).passages(texts)
    one_s = time.monotonic() - began
    per_q = []
    qspace = VectorSpace(OllamaEmbedder(ollama.Client(), EMBED_MODEL, fp.digest), EmbeddingCache())
    for q in corpus().questions[:12]:
        began = time.monotonic()
        qv = qspace.queries([q.text])
        per_q.append(time.monotonic() - began)
    began = time.monotonic()
    for _ in range(1000):
        qv.cosine(vectors)
    search_ms = time.monotonic() - began  # секунд на 1000 повторов = мс на один
    raw = len(texts) * vectors.dim * 4
    doc_bytes = sum(len(d.text.encode("utf-8")) for d in corpus().docs)
    cache_file = EmbeddingCache(EMBED_CACHE)
    per_vector = cache_file.file_bytes / len(cache_file) if len(cache_file) else 0.0
    print(f"   офлайн, эмбеддинг корпуса ({len(texts)} чанков, пакеты по 32, модель уже в памяти): "
          f"{corpus_s:.2f} с, обращений {cold.calls}, токенов {cold.prompt_tokens}; по одному чанку — "
          f"{one_s:.2f} с ({one_s / corpus_s:.1f}×)")  # fmt: skip
    print(f"   онлайн, один вопрос: медиана {statistics.median(per_q) * 1000:.0f} мс "
          f"({min(per_q) * 1000:.0f}–{max(per_q) * 1000:.0f}); поиск — умножение матрицы {len(texts)}×"
          f"{vectors.dim} на вектор: {search_ms:.3f} мс")  # fmt: skip
    print(f"   на диске: векторы float32 {raw / 1e6:.3f} МБ = {raw / doc_bytes:.1f}× от документов "
          f"({doc_bytes / 1e3:.1f} КБ UTF-8); в кеше JSONL с base64 — {per_vector / 1e3:.1f} КБ на вектор "
          f"({per_vector * len(texts) / doc_bytes:.1f}× от документов)")  # fmt: skip

    m = measured()
    table = hits_by_kind(m)
    gen = probe_fingerprint(MODEL, OPTIONS, "fact@v1")
    swapped_alone = OllamaEmbedder(ollama.Client(), EMBED_MODEL, fp.digest, batch_size=1)
    single = VectorSpace(swapped_alone, EmbeddingCache()).queries(
        [q.text for q in corpus().questions]
    )
    noise = float(np.abs(single.cosine(m.c) - m.sims).max())
    right, total = saved["conditions"]["ceiling"]
    whit, wn = saved["conditions"]["word_hit"]
    block: dict[str, Any] = {
        "about": "Ф5-02: корпус corpus/polka, 41 чанк (абзацы до 500 знаков); попадание — нужный "
        "документ первым результатом; слова — search_documents docqa, векторы — косинус по "
        "чанкам; предел чувствительности — точка минус нижняя граница Уилсона; базовая линия — один "
        "прогон: вектор детерминирован с точностью до «шума пакета»",
        "fingerprint": {"embedder": fp.to_dict(), "generator": gen.to_dict(), "space": m.q.fingerprint},
        "corpus": {"signature": corpus().signature, "documents": len(corpus().docs),
                   "questions": len(corpus().questions), "chunks": len(texts)},
        "косинус": {
            "фон (медиана, два чанка разных документов)": round(statistics.median(chunk_pairs(m)), 3),
            "медиана, вопрос ↔ чанк другого документа": round(statistics.median(q_other(m)), 3),
            "медиана релевантных (вопрос ↔ золотой чанк)": round(statistics.median(relevant(m)), 3),
            "порог для 90 % релевантных": round(float(np.quantile(relevant(m), 0.1)), 3),
            "шум пакета (макс. |Δ|, по одному против пакета)": float(f"{noise:.2e}"),
        },
        "оракул (судья fact@v1)": entry(right, total),
        "попадание": {kind: {"слова": entry(sum(w1), len(w1)), "векторы": entry(sum(v1), len(v1))}
                      for kind, (w1, v1, _, _) in table.items()},  # fmt: skip
        "попадание, всего": {"слова": entry(whit, wn),
                             "векторы": entry(sum(sum(v[1]) for v in table.values()), wn)},  # fmt: skip
        "цена": {"секунд на корпус": round(corpus_s, 2), "мс на вопрос": round(statistics.median(per_q) * 1000),
                 "МБ векторов": round(raw / 1e6, 3), "векторы / документы": round(raw / doc_bytes, 2)},
    }  # fmt: skip
    suite = Suite.load(SUITE)
    suite.extra["embedding"] = block
    suite.save(SUITE)
    print(f"\n   блок «embedding» в {SUITE.relative_to(ROOT)}: отпечаток {block['fingerprint']['space']} "
          f"рядом с генератором {gen.model}@{gen.digest[:12]}")  # fmt: skip
    for name, value in block["косинус"].items():
        print(f"      {name}: {value}")
    for kind, pair in block["попадание"].items():
        print(f"      {kind}: слова {pair['слова']['k']}/{pair['слова']['n']}, векторы "
              f"{pair['векторы']['k']}/{pair['векторы']['n']} (предел {pair['векторы']['sensitivity'] * 100:.0f} п. п.)")  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=range(1, 7), required=True)
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip
    print(
        f"коммит {head}; генератор {MODEL}, {OPTIONS}; эмбеддинги {EMBED_MODEL}; задание {args.task}"
    )
    tasks: dict[int, Callable[[], None]] = {
        1: task1,
        2: task2,
        3: task3,
        4: task4,
        5: task5,
        6: task6,
    }
    tasks[args.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
