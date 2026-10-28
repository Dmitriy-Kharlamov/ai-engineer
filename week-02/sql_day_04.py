import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "library_day4.db"
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA foreign_keys = ON")

cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    tokens INTEGER NOT NULL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
""")

cur.executemany(
    "INSERT INTO documents (id, title, source, tokens) VALUES (?, ?, ?, ?)",
    [
        (1, "Attention is all you need", "arxiv", 8400),
        (2, "BERT", "arxiv", 7200),
        (3, "Внутренний регламент", "confluence", 1500),
        (4, "Пустой черновик", "confluence", 0),
    ],
)

cur.executemany(
    "INSERT INTO chunks (document_id, text, tokens) VALUES (?, ?, ?)",
    [
        (1, "Hello", 50),
        (1, "How are you", 2000),
        (1, "I'm fine", 3000),
        (2, "I'm too", 40),
        (2, "Good", 70),
        (3, "Thank you", 5000),
    ],
)
conn.commit()


def show(title, sql, params=()):
    print(f"\n{title}")
    for row in cur.execute(sql, params):
        print(row)


# 1. UPDATE одного чанка
show("чанк 1 до", "SELECT id, text, tokens FROM chunks WHERE id = ?", (1,))
cur.execute("UPDATE chunks SET tokens = ? WHERE id = ?", (20, 1))
show("чанк 1 после", "SELECT id, text, tokens FROM chunks WHERE id = ?", (1,))

# 2. UPDATE с подзапросом: +10% токенов у чанков документов из arxiv
show("чанки до +10%", "SELECT id, document_id, tokens FROM chunks")
cur.execute("""
    UPDATE chunks
    SET tokens = CAST(tokens * 1.1 AS INTEGER)
    WHERE document_id IN (SELECT id FROM documents WHERE source = 'arxiv')
""")
show("чанки после +10%", "SELECT id, document_id, tokens FROM chunks")

# 3. DELETE с подсчётом удалённых строк
cur.execute("DELETE FROM chunks WHERE tokens < 50")
print("\nудалено чанков:", cur.rowcount)

conn.commit()

# 4. Транзакция с откатом
show("до транзакции", "SELECT id, tokens FROM chunks")
try:
    cur.execute("UPDATE chunks SET tokens = tokens + 1000 WHERE id = 2")  # изменение 1
    raise RuntimeError("сбой посередине транзакции")  # имитация ошибки
    cur.execute("UPDATE chunks SET tokens = tokens + 1000 WHERE id = 3")  # изменение 2
    conn.commit()
except RuntimeError as e:
    print("ошибка:", e, "→ откат")
    conn.rollback()
show("после отката", "SELECT id, tokens FROM chunks")

# 4б. То же, но без ошибки — с commit
cur.execute("UPDATE chunks SET tokens = tokens + 1000 WHERE id = 2")
cur.execute("UPDATE chunks SET tokens = tokens + 1000 WHERE id = 3")
conn.commit()
show("после commit", "SELECT id, tokens FROM chunks")

# 5. Предсказание: close() без commit не сохраняет изменение — останется значение
#    после последнего commit, (2, 3200), а не (2, 0).
#    Результат: подтвердилось. sqlite3 при close() без commit откатывает
#    незафиксированные изменения — они в файл не попадают.
other = sqlite3.connect(DB_PATH)
other.execute("UPDATE chunks SET tokens = 0 WHERE id = 2")
other.close()  # ни commit, ни rollback

check = sqlite3.connect(DB_PATH)
print(
    "\nчанк 2 после close без commit:",
    check.execute("SELECT id, tokens FROM chunks WHERE id = 2").fetchone(),
)
check.close()


# 6. Результаты одинаковые (документы 1, 2, 3), а планы различаются:
#    IN:   SCAN chunks один раз → LIST SUBQUERY (список id без дублей)
#          → SEARCH documents по первичному ключу. DISTINCT не нужен.
#    JOIN: SCAN chunks → SEARCH documents для каждого чанка → получаются
#          дубли (по строке на чанк) → USE TEMP B-TREE FOR DISTINCT,
#          чтобы их убрать. Это лишняя временная структура.
#    Вывод: когда нужны только строки из documents («есть ли чанки»),
#    IN выражает смысл точнее и обходится без шага DISTINCT. JOIN нужен,
#    когда в результат нужны и колонки из chunks.
Q_IN = """
    SELECT id, title FROM documents
    WHERE id IN (SELECT document_id FROM chunks)
"""
Q_JOIN = """
    SELECT DISTINCT documents.id, documents.title
    FROM documents
    JOIN chunks ON documents.id = chunks.document_id
"""

show("через IN (подзапрос)", Q_IN)
show("через JOIN + DISTINCT", Q_JOIN)
assert set(cur.execute(Q_IN).fetchall()) == set(cur.execute(Q_JOIN).fetchall())

print("\nплан IN:  ", cur.execute("EXPLAIN QUERY PLAN " + Q_IN).fetchall())
print("план JOIN:", cur.execute("EXPLAIN QUERY PLAN " + Q_JOIN).fetchall())

conn.close()
