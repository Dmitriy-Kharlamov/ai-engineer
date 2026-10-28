import os
import sqlite3

DB_PATH = "week-02/library_day2.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

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


# FOREIGN KEY: с PRAGMA foreign_keys = ON вставка чанка с document_id = 999 падает
# с IntegrityError, потому что документа с таким id нет. Без PRAGMA SQLite вставка чанка не проверяет связь,
# и в базе появляется чанк, который относится к несуществующему документу 999.
try:
    cur.execute(
        "INSERT INTO chunks (document_id, text, tokens) VALUES (?, ?, ?)", (999, "чанк-сирота", 50)
    )
except sqlite3.IntegrityError as e:
    print("FOREIGN KEY:", e)


def show(title, sql):
    print(f"\n{title}")
    for row in cur.execute(sql):
        print(row)


show(
    "1. Крупные чанки",
    """
    SELECT text, tokens
    FROM chunks
    WHERE tokens > 100
    ORDER BY tokens DESC
    LIMIT 3
""",
)


show(
    "2. INNER JOIN",
    """
    SELECT documents.title, chunks.text
    FROM chunks
    INNER JOIN documents ON chunks.document_id = documents.id
""",
)


# 3. INNER JOIN оставляет только строки, у которых нашлась пара в обеих таблицах,
#    поэтому документа без чанков в нём нет. LEFT JOIN оставляет все строки
#    левой таблицы (documents), а где чанка нет, подставляет Null — так
#    в результат попадает «Пустой черновик».
show(
    "3. LEFT JOIN",
    """
    SELECT documents.title, chunks.text
    FROM documents
    LEFT JOIN chunks ON documents.id = chunks.document_id
""",
)


# 4. COUNT(chunks.id), а не COUNT(*): у документа без чанков после LEFT JOIN
#    остаётся одна строка с NULL вместо чанка. COUNT(*) посчитает её и покажет 1,
#    а COUNT(chunks.id) пропускает NULL и даёт правильный 0.
show(
    "4. Чанков на документ",
    """
    SELECT documents.title, COUNT(chunks.id)
    FROM documents
    LEFT JOIN chunks ON documents.id = chunks.document_id
    GROUP BY documents.id
""",
)

# 5. WHERE нельзя, потому что WHERE фильтрует данные до группировки, а COUNT
#    появляется только после GROUP BY. HAVING фильтрует уже готовые сгруппированные данные.
show(
    "5. Больше одного чанка",
    """
    SELECT documents.title, COUNT(chunks.id)
    FROM documents
    LEFT JOIN chunks ON documents.id = chunks.document_id
    GROUP BY documents.id
    HAVING COUNT(chunks.id) > 1
""",
)


show(
    "6. Токены по источнику",
    """
    SELECT source, SUM(tokens)
    FROM documents
    GROUP BY source
    ORDER BY SUM(tokens) DESC
""",
)
