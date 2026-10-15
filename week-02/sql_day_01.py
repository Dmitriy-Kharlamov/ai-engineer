import sqlite3

conn = sqlite3.connect("week-02/library.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    tokens INTEGER NOT NULL
)
""")

cur.executemany(
    "INSERT INTO documents (title, source, tokens) VALUES (?, ?, ?)",
    [("Attention is all you need", "arxiv", 8400),
     ("BERT", "arxiv", 7200),
     ("Внутренний регламент", "confluence", 1500)]
)
conn.commit()

for row in cur.execute("SELECT title, tokens FROM documents ORDER BY tokens DESC"):
    print(row)

for row in cur.execute("SELECT source, COUNT(*) FROM documents GROUP BY source"):
    print(row)

conn.close()


# 1. Если подставлять значения прямо в строку запроса (f-строкой), то кавычка в данных
#    может закрыть строку SQL раньше времени. Тогда запрос сломается (например, на "O'Reilly"),
#    а злоумышленник может дописать свою команду, например DROP TABLE. Это SQL-инъекция.
#    С ? значения передаются в базу отдельно от текста запроса, поэтому они всегда
#    остаются данными и не могут выполниться как код.

# 2. conn.commit() завершает транзакцию и записывает изменения в файл library.db.
#    Если его не вызвать, при закрытии соединения все изменения после последнего commit
#    отменятся и в файл не попадут. При этом внутри того же соединения SELECT их ещё видит,
#    поэтому без проверки это легко не заметить.

# 3. GROUP BY source собирает строки в группы по значению source, а COUNT(*) считает
#    количество строк в каждой группе. На наших данных запрос вернёт:
#    ('arxiv', 2)
#    ('confluence', 1)