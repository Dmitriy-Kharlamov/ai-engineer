# Почему индексы не создают на все столбцы:
#   индекс — отдельная отсортированная структура. При каждом INSERT база должна
#   вставить строку не только в таблицу, но и в структуру индексов, сохранив его
#   порядок; при UPDATE и DELETE — тоже обновить все индексы. Десять индексов —
#   десять лишних записей на каждую вставку. Плюс каждый индекс занимает место на диске.
#   Поэтому индекс ставят на столбцы, по которым часто ищут (WHERE, JOIN),
#   и лучше избирательные: здесь у document_id 4 значения, и ускорение лишь ~4×.


import random
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "library_day3.db"
if DB_PATH.exists():
    DB_PATH.unlink()                    # каждый запуск — с чистой базы

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL
)
""")

rows = [(random.randint(1, 4), f"чанк {i}", random.randint(10, 500))
        for i in range(200_000)]
cur.executemany("INSERT INTO chunks (document_id, text, tokens) VALUES (?, ?, ?)", rows)
conn.commit()

QUERY = "SELECT COUNT(*) FROM chunks WHERE document_id = 3"


def timed(query):
    start = time.perf_counter()
    result = cur.execute(query).fetchone()
    return result, time.perf_counter() - start


def plan(query):
    return cur.execute("EXPLAIN QUERY PLAN " + query).fetchall()


result, before = timed(QUERY)
print("до индекса:", result, f"{before:.5f} с")
print("план до:", plan(QUERY))


cur.execute("CREATE INDEX idx_chunks_document_id ON chunks(document_id)")

result, after = timed(QUERY)
print("после индекса:", result, f"{after:.5f} с")
print("план после:", plan(QUERY))
print(f"ускорение: в {before / after:.1f} раз")


# Часть 2. UNIQUE + INSERT OR IGNORE — отдельная таблица, чтобы автоиндекс UNIQUE
# не испортил эксперимент из части 1
cur.execute("""
CREATE TABLE chunks_unique (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    UNIQUE (document_id, text)
)
""")


def load_unique(data):
    cur.executemany(
        "INSERT OR IGNORE INTO chunks_unique (document_id, text, tokens) VALUES (?, ?, ?)",
        data,
    )
    conn.commit()
    return cur.execute("SELECT COUNT(*) FROM chunks_unique").fetchone()[0]


print("\nпервая загрузка, строк:", load_unique(rows))
print("повторная загрузка, строк:", load_unique(rows))   # те же данные ещё раз

# Для сравнения: обычный INSERT на дубле падает
try:
    cur.execute("INSERT INTO chunks_unique (document_id, text, tokens) VALUES (?, ?, ?)",
                rows[0])
except sqlite3.IntegrityError as e:
    print("обычный INSERT дубля:", e)

# Бонус: UNIQUE сам создал индекс — посмотри на план
print("план на chunks_unique:",
      plan("SELECT COUNT(*) FROM chunks_unique WHERE document_id = 3"))