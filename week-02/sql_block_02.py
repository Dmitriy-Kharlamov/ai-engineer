import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "sql_block_02.db"
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA foreign_keys = ON")

cur.execute("""
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    tokens INTEGER
)
""")
cur.execute("""
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    author TEXT,
    dense_score REAL,             -- оценка векторного поиска
    sparse_score REAL,            -- оценка BM25
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
""")

cur.executemany(
    "INSERT INTO documents VALUES (?, ?, ?, ?)",
    [
        (1, "Attention is all you need", "arxiv", 8700),
        (2, "BERT", "arxiv", 7200),
        (3, "Внутренний регламент", "confluence", 1500),
        (4, "Пустой черновик", "confluence", 0),
        (5, "README ragkit", "github", None),
        (6, "Гайд по деплою", "github", 3000),
    ],
)

CHUNKS = [
    (1, "Введение", 1000, "Vaswani"),
    (1, "Архитектура", 1500, "Vaswani"),  # ничья с Self-attention (1500)
    (1, "Self-attention", 1500, None),
    (1, "Multi-head", 1800, "Shazeer"),
    (1, "Эксперименты", 2000, None),
    (1, "Выводы", 900, "Parmar"),
    (2, "Pre-training", 3000, "Devlin"),
    (2, "Fine-tuning", 3500, None),
    (2, "Pre-training", 3000, "Devlin"),  # явный дубликат (2, Pre-training)
    (3, "Правила доступа", 1500, None),
    (5, "Установка", 400, "Dmitriy"),
    (5, "Примеры", 600, None),
    (6, "Docker", 2500, "Dmitriy"),
    (None, "Сирота без документа", 300, None),
]

random.seed(42)  # фиксированный seed → воспроизводимо
rows = [
    (doc, text, tokens, author, round(random.random(), 3), round(random.random(), 3))
    for doc, text, tokens, author in CHUNKS
]
cur.executemany(
    """
    INSERT INTO chunks (document_id, text, tokens, author, dense_score, sparse_score)
    VALUES (?, ?, ?, ?, ?, ?)
""",
    rows,
)
conn.commit()


def show(title, sql, params=()):
    print(f"\n{title}")
    for row in cur.execute(sql, params):
        print(row)


def plan(title, sql):
    print(f"\nплан: {title}")
    for row in cur.execute("EXPLAIN QUERY PLAN " + sql):
        print("   ", row[3])


# 1. Задание 6 прошлого урока через оконную функцию
Q_CORR = """
    SELECT c.id, d.title, c.tokens,
           (SELECT AVG(c2.tokens) FROM chunks c2
            WHERE c2.document_id = c.document_id) AS doc_avg
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.tokens > (SELECT AVG(c2.tokens) FROM chunks c2
                      WHERE c2.document_id = c.document_id)
    ORDER BY c.id
"""
Q_WINDOW = """
    WITH ranked AS (
        SELECT c.id, c.document_id, c.tokens,
               AVG(c.tokens) OVER (PARTITION BY c.document_id) AS doc_avg
        FROM chunks c
    )
    SELECT r.id, d.title, r.tokens, r.doc_avg
    FROM ranked r
    JOIN documents d ON d.id = r.document_id
    WHERE r.tokens > r.doc_avg
    ORDER BY r.id
"""
show("1. чанки больше среднего по документу (окно)", Q_WINDOW)
assert cur.execute(Q_WINDOW).fetchall() == cur.execute(Q_CORR).fetchall()
plan("коррелированный", Q_CORR)
plan("оконный", Q_WINDOW)
# Результат совпал с коррелированным вариантом (assert прошёл).
# Что изменилось в плане:
#   было:  две строки CORRELATED SCALAR SUBQUERY, в каждой SCAN c2 — среднее
#          пересчитывается для каждого чанка, причём дважды (SELECT и WHERE),
#          порядка n × n строк.
#   стало: коррелированных подзапросов нет. Один SCAN c, затем
#          USE TEMP B-TREE FOR ORDER BY — строки сортируются по document_id,
#          чтобы разложить их по окнам, и среднее каждого окна считается за
#          один проход. Порядка n log n (сортировка) вместо n².


# 2. ROW_NUMBER, RANK, DENSE_RANK
Q2 = """
    SELECT text, tokens,
           ROW_NUMBER() OVER (ORDER BY tokens DESC) AS rn,
           RANK()       OVER (ORDER BY tokens DESC) AS rnk,
           DENSE_RANK() OVER (ORDER BY tokens DESC) AS dense
    FROM chunks
"""
show("2. три функции ранжирования", Q2)
Q2_STABLE = Q2.replace(
    "ROW_NUMBER() OVER (ORDER BY tokens DESC)", "ROW_NUMBER() OVER (ORDER BY tokens DESC, id)"
)
show("2. ROW_NUMBER с вторым ключом id", Q2_STABLE)
# Ничья: три чанка по 1500 токенов (Архитектура, Self-attention, Правила доступа).
#   ROW_NUMBER = 7, 8, 9 — всегда уникален, просто нумерует строки подряд.
#   RANK       = 7, 7, 7 — одинаковый для равных; следующий (Введение) получает 10:
#                три места 7, 8, 9 «съедены» ничьей, номер прыгает.
#   DENSE_RANK = 6, 6, 6 — одинаковый для равных, без прыжка: следующий — 7.
#                Он 6, а не 7, потому что выше ещё одна ничья (два Pre-training
#                по 3000), и DENSE_RANK считает уровни, а не строки.
# Порядок ничейных строк в ROW_NUMBER: ORDER BY tokens его не определяет —
#   SQLite отдаёт их в том порядке, в каком ему удобно прочитать (здесь совпало
#   с порядком вставки, но это не гарантия: другой план или другая база — другой
#   порядок). Воспроизводимым он становится со вторым уникальным ключом:
#   ORDER BY tokens DESC, id — при равных токенах решает id.


# 3. Top-2 чанка каждого документа
Q3 = """
    WITH ranked AS (
        SELECT c.document_id, c.id, c.text, c.tokens,
               ROW_NUMBER() OVER (
                   PARTITION BY c.document_id
                   ORDER BY c.tokens DESC, c.id
               ) AS rn
        FROM chunks c
    )
    SELECT d.title, r.text, r.tokens, r.rn
    FROM ranked r
    JOIN documents d ON d.id = r.document_id
    WHERE r.rn <= 2
    ORDER BY d.id, r.rn
"""
show("3. top-2 по документу", Q3)
Q3_GROUP = """
    SELECT d.title, c.text, MAX(c.tokens)
    FROM chunks c JOIN documents d ON d.id = c.document_id
    GROUP BY c.document_id
    ORDER BY d.id
"""
show("3. попытка через GROUP BY", Q3_GROUP)
# Почему через GROUP BY не получается: GROUP BY превращает каждую группу в ОДНУ
#   строку — видно по выводу: по одному чанку на документ, второго взять неоткуда.
#   К тому же нужен сам текст чанка, а не агрегат; text в такой группировке —
#   «голый» столбец, который SQLite разрешает по особому правилу, а PostgreSQL
#   отвергнет с ошибкой. Окно же оставляет все строки и нумерует их внутри
#   документа, а отбор rn <= 2 делается снаружи.


# 4. LAG: предыдущий чанк и разница
Q4 = """
    SELECT document_id, id, text, tokens,
           LAG(tokens) OVER (PARTITION BY document_id ORDER BY id) AS prev_tokens,
           tokens - LAG(tokens, 1, 0) OVER (PARTITION BY document_id ORDER BY id) AS delta
    FROM chunks
    WHERE document_id IS NOT NULL
    ORDER BY document_id, id
"""
show("4. LAG и разница", Q4)
for doc_id, cid, text, tokens, prev, delta in cur.execute(Q4).fetchall():
    if prev is None:  # первый чанк документа
        assert delta == tokens
# LAG(tokens) у первого чанка документа — None: строки выше в этом окне нет.
# LAG(tokens, 1, 0): сдвиг 1, а вместо отсутствующей строки — 0. Поэтому
#   delta первого чанка = tokens - 0 = его собственные токены (1000, 3000, ...),
#   что и проверяет assert.
# PARTITION BY document_id важен: без него первый чанк документа 2 сравнился бы
#   с последним чанком документа 1.


# 5. RANGE против ROWS
Q5 = """
    SELECT document_id, id, tokens,
           SUM(tokens) OVER (PARTITION BY document_id ORDER BY tokens) AS sum_range,
           SUM(tokens) OVER (PARTITION BY document_id ORDER BY tokens
                             ROWS UNBOUNDED PRECEDING) AS sum_rows
    FROM chunks
    WHERE document_id IN (1, 2)
    ORDER BY document_id, tokens, id
"""
show("5. кумулятивная сумма: RANGE против ROWS", Q5)
r5 = cur.execute(Q5).fetchall()
assert any(row[3] != row[4] for row in r5)  # обязаны разойтись
# Разошлись на ничьих:
#   документ 1, два чанка по 1500 (id 2 и 3): RANGE 4900, 4900; ROWS 3400, 4900.
#   документ 2, два чанка по 3000 (id 7 и 9): RANGE 6000, 6000; ROWS 3000, 6000.
# Почему: с ORDER BY рамка по умолчанию — RANGE BETWEEN UNBOUNDED PRECEDING
#   AND CURRENT ROW. RANGE понимает «до текущей строки» по ЗНАЧЕНИЮ ключа и
#   включает всех равных (peers): для первого 1500 в рамку попадает и второй
#   1500 → 900 + 1000 + 1500 + 1500 = 4900. Получается плато.
#   ROWS считает физические строки: до первого 1500 — только он сам и строки
#   выше → 900 + 1000 + 1500 = 3400, сумма растёт построчно.
# Там, где ключ уникален (900, 1000, 1800, 2000), варианты совпадают.


# 6. Скользящее среднее по трём
Q6 = """
    SELECT document_id, id, tokens,
           AVG(tokens) OVER (
               PARTITION BY document_id ORDER BY id
               ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
           ) AS moving_avg_3
    FROM chunks
    WHERE document_id = 1
    ORDER BY id
"""
show("6. скользящее среднее (документ 1)", Q6)
r6 = {row[1]: row[3] for row in cur.execute(Q6)}
assert r6[1] == 1000 / 1  # начало: только сам чанк
assert r6[2] == (1000 + 1500) / 2  # две строки
assert r6[4] == (1500 + 1500 + 1800) / 3  # середина: id 2, 3, 4
# Ручная проверка:
#   начало, id 1: строк выше нет, в рамке только он → 1000 / 1 = 1000.
#   начало, id 2: выше одна строка → (1000 + 1500) / 2 = 1250.
#   середина, id 4: рамка — id 2, 3, 4 → (1500 + 1500 + 1800) / 3 = 1600.
# В начале окна строк меньше трёх, и AVG делит на столько, сколько есть,
#   а не на 3.


# 7. Дедупликация через ROW_NUMBER
Q7 = """
    WITH numbered AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY document_id, text
                   ORDER BY id
               ) AS rn
        FROM chunks
    )
    SELECT id, document_id, text, tokens FROM numbered
    WHERE rn = 1
    ORDER BY id
"""
before = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
after = len(cur.execute(Q7).fetchall())
show("7. после дедупликации", Q7)
print("строк до:", before, " после:", after)
assert before - after == 1
# PARTITION BY document_id, text — каждое окно — одна пара (документ, текст).
#   У уникальных пар окно из одной строки, rn = 1. У дубликата (2, Pre-training)
#   две строки: id 7 получает rn = 1, id 9 — rn = 2 (ORDER BY id → меньший id
#   первым). Фильтр rn = 1 оставляет id 7 и выкидывает id 9: было 14, стало 13.


# 8. Оконная функция в WHERE
Q8_BAD = """
    SELECT text, ROW_NUMBER() OVER (ORDER BY tokens) AS rn
    FROM chunks
    WHERE ROW_NUMBER() OVER (ORDER BY tokens) = 1
"""
print("\n8. окно в WHERE")
try:
    cur.execute(Q8_BAD)
except sqlite3.OperationalError as e:
    print("ошибка:", e)
Q8_GOOD = """
    WITH numbered AS (
        SELECT text, tokens, ROW_NUMBER() OVER (ORDER BY tokens, id) AS rn
        FROM chunks
    )
    SELECT text, tokens, rn FROM numbered WHERE rn = 1
"""
show("8. исправлено через CTE", Q8_GOOD)
# Точный текст ошибки: misuse of window function ROW_NUMBER()
# Почему: порядок выполнения FROM → WHERE → GROUP BY → HAVING → SELECT → ...
#   Оконные функции вычисляются в фазе SELECT, а WHERE работает раньше —
#   в момент фильтрации номеров строк ещё не существует.
# Исправление: окно считается в CTE (там это фаза SELECT внутреннего запроса),
#   а внешний запрос получает rn как обычный готовый столбец и фильтрует его
#   в своём WHERE.


# 9. RRF
Q9 = """
    WITH ranks AS (
        SELECT id, text, dense_score, sparse_score,
               RANK() OVER (ORDER BY dense_score DESC)  AS dense_rank,
               RANK() OVER (ORDER BY sparse_score DESC) AS sparse_rank
        FROM chunks
    )
    SELECT id, text, dense_rank, sparse_rank,
           ROUND(1.0 / (:k + dense_rank) + 1.0 / (:k + sparse_rank), 5) AS rrf
    FROM ranks
    ORDER BY rrf DESC, id
    LIMIT 5
"""
show("9. RRF, k = 60", Q9, {"k": 60})
show("9. RRF, k = 1", Q9, {"k": 1})
# Почему ранги, а не оценки: dense_score (косинусная близость) и sparse_score
#   (BM25) живут в разных шкалах — косинус обычно в [0, 1], BM25 может быть
#   10 или 25. Сложить их напрямую — значит дать перевес той, у которой шкала
#   крупнее. Ранг же у обеих одинаковый по смыслу: «какое место в своём списке».
#   RRF берёт только места, и шкалы перестают иметь значение.
# k = 1 вместо 60: разница между верхними местами становится огромной —
#   1/(1+1) = 0.5 против 1/(1+2) = 0.33, а при k = 60: 0.0164 против 0.0161.
#   Малое k даёт решающий голос тому, кто высоко хотя бы в ОДНОМ списке.
#   В нашем выводе: Multi-head (ранги 2 и 11) при k = 1 обогнал Fine-tuning
#   (ранги 6 и 4) — второе место в dense перевесило то, что в sparse он 11-й.
#   При k = 60 выше Fine-tuning: стабильно неплох в обоих списках.
#   Большое k сглаживает и вознаграждает согласие двух списков.

conn.close()


# 10. Устно, потом письменно.
#   Что делает OVER: превращает агрегат (AVG, SUM, COUNT) или функцию
#     ранжирования в оконную — она считается по набору строк (окну), связанному
#     с текущей строкой, а результат пишется в каждую строку. PARTITION BY
#     задаёт окна, ORDER BY — порядок внутри окна, ROWS/RANGE — рамку.
#   Чем окно отличается от GROUP BY: GROUP BY склеивает группу в одну строку,
#     отдельные строки пропадают. Окно ничего не склеивает: все строки остаются,
#     и каждая дополнительно знает агрегат своей группы (задание 1: каждый чанк
#     видит среднее по своему документу) или своё место в ней (задание 3).
#   Почему нельзя в WHERE и можно в ORDER BY: порядок выполнения
#     FROM → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT.
#     Окна считаются в фазе SELECT. WHERE идёт раньше — значений ещё нет
#     (задание 8: misuse of window function). ORDER BY идёт позже — значения
#     уже посчитаны, по ним можно сортировать.
