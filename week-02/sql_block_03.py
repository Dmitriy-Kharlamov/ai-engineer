import random
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "sql_block_03.db"
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA foreign_keys = ON")

cur.execute("""
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL
)
""")
cur.execute("""
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    processed INTEGER NOT NULL,      -- 0 = ещё не обработан (~5% строк)
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
""")

SOURCES = ["arxiv", "confluence", "github"]
TOPICS = ["attention", "bert", "retrieval", "docker", "policy"]

docs = [(i, f"Документ {i:02d}", SOURCES[i % 3]) for i in range(1, 51)]
docs += [(51, "Черновик A", "drafts"),        # источник целиком без чанков
         (52, "Черновик B", "drafts"),
         (53, "Пустой README", "github")]     # документ без чанков в живом источнике
cur.executemany("INSERT INTO documents VALUES (?, ?, ?)", docs)

random.seed(42)
N = 200_000
rows = []
for i in range(1, N + 1):
    topic = random.choice(TOPICS)
    rows.append((random.randint(1, 50),             # document_id от 1 до 50
                 f"{topic}-{i:06d}",                # например 'bert-000123'
                 random.randint(50, 1000),          # tokens
                 0 if random.random() < 0.05 else 1))
cur.executemany(
    "INSERT INTO chunks (document_id, text, tokens, processed) VALUES (?, ?, ?, ?)",
    rows)
conn.commit()
print("чанков:", cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


def show(title, sql, params=()):
    print(f"\n{title}")
    for row in cur.execute(sql, params):
        print(row)


def plan(title, sql, params=()):
    print(f"\nплан: {title}")
    for row in cur.execute("EXPLAIN QUERY PLAN " + sql, params):
        print("   ", row[3])


def best_of(sql, params=(), runs=7):
    """Минимальное время из runs прогонов — минимум меньше всего зашумлён."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        cur.execute(sql, params).fetchall()
        times.append(time.perf_counter() - start)
    return min(times)


# 2 (до). Замер ДО создания индекса — поэтому стоит раньше задания 1
Q2 = "SELECT * FROM chunks WHERE document_id = 7 ORDER BY tokens"
plan("2. до индекса", Q2)
t_before = best_of(Q2)
print(f"время до индекса:    {t_before:.5f} с")


# 1. Составной индекс и правило левого префикса
cur.execute("CREATE INDEX idx_chunks_doc_tokens ON chunks(document_id, tokens)")
plan("1a. document_id = 7",
     "SELECT * FROM chunks WHERE document_id = 7")
plan("1b. document_id = 7 AND tokens > 400",
     "SELECT * FROM chunks WHERE document_id = 7 AND tokens > 400")
plan("1c. document_id = 7 ORDER BY tokens",
     "SELECT * FROM chunks WHERE document_id = 7 ORDER BY tokens")
plan("1d. tokens > 400",
     "SELECT * FROM chunks WHERE tokens > 400")
# 1a: SEARCH (document_id=?) — document_id = левый префикс индекса, поиск сразу
#     в нужный участок дерева вместо перебора 200 000 строк.
# 1b: SEARCH (document_id=? AND tokens>?) — префикс плюс следующий столбец:
#     внутри document_id = 7 строки уже лежат по tokens, база прыгает к первому
#     tokens > 400 и читает до конца участка. Используются оба столбца.
# 1c: SEARCH (document_id=?) и НЕТ строки USE TEMP B-TREE FOR ORDER BY —
#     индекс закрыл и сортировку: внутри document_id = 7 строки в индексе уже
#     идут по возрастанию tokens, база просто читает их по порядку.
# 1d: SCAN chunks — индекс не используется. Индекс отсортирован сначала по
#     document_id, а tokens упорядочен только ВНУТРИ каждого document_id.
#     Без условия на document_id строки с tokens > 400 разбросаны по всем
#     50 участкам — это «все Сергеи» из телефонной книги по фамилии.


# 2 (после).
plan("2. после индекса", Q2)
t_after = best_of(Q2)
print(f"время после индекса: {t_after:.5f} с   (быстрее в {t_before / t_after:.1f} раз)")
# План: было SCAN chunks + USE TEMP B-TREE FOR ORDER BY, стало SEARCH ... без
#   TEMP B-TREE. Исчезли две работы: полный проход по 200 000 строк и
#   сортировка ~4000 найденных. Из этого и выигрыш во времени.
# Почему выигрыш не в сотни раз: SELECT * — индекс хранит только document_id
#   и tokens, остальные столбцы для каждой из ~4000 строк приходится доставать
#   из таблицы отдельным поиском по rowid. Если бы запрос выбирал только
#   индексные столбцы, план показал бы COVERING INDEX, и разрыв был бы больше.
# Замер ДО стоит выше задания 1, потому что индекс создаётся в задании 1.


# 3. Убийцы индекса
cur.execute("CREATE INDEX idx_chunks_text ON chunks(text)")
cur.execute("PRAGMA case_sensitive_like = ON")   # иначе LIKE в SQLite не берёт индекс

# 3.1 функция от столбца
plan("3.1 до: LOWER(text) = ...",
     "SELECT * FROM chunks WHERE LOWER(text) = 'bert-000123'")
cur.execute("CREATE INDEX idx_chunks_text_lower ON chunks(LOWER(text))")
plan("3.1 после: индекс по выражению LOWER(text)",
     "SELECT * FROM chunks WHERE LOWER(text) = 'bert-000123'")

# 3.2 ведущий %
plan("3.2 до: LIKE '%-000123'",
     "SELECT * FROM chunks WHERE text LIKE '%-000123'")
plan("3.2 после: LIKE 'bert-0001%'",
     "SELECT * FROM chunks WHERE text LIKE 'bert-0001%'")

# 3.3 арифметика на столбце
plan("3.3 до: tokens + 100 > 1000",
     "SELECT * FROM chunks WHERE document_id = 7 AND tokens + 100 > 1000")
plan("3.3 после: tokens > 900",
     "SELECT * FROM chunks WHERE document_id = 7 AND tokens > 900")
# 3.1: LOWER(text) — индекс idx_chunks_text хранит значения text, а не
#      LOWER(text); значения выражения в нём нет → SCAN. Лечение — индекс по
#      выражению: в нём лежат именно значения LOWER(text) → SEARCH (<expr>=?).
#      (Другой путь — хранить текст уже в нижнем регистре и не вызывать LOWER.)
# 3.2: LIKE '%-000123' — начало строки неизвестно, а индекс отсортирован с
#      первого символа: с какого места дерева начинать, непонятно → SCAN.
#      LIKE 'bert-0001%' — префикс известен, это диапазон
#      text >= 'bert-0001' AND text < 'bert-0002' → SEARCH (text>? AND text<?).
#      PRAGMA case_sensitive_like = ON нужна потому, что LIKE в SQLite по
#      умолчанию без учёта регистра, а индекс по text — с учётом; без неё
#      SQLite не может свести LIKE к диапазону индекса.
# 3.3: tokens + 100 > 1000 — в индексе лежит tokens, а не tokens + 100. Видно
#      точно: индекс использован только по document_id=?, часть про tokens
#      проверяется построчно. tokens > 900 — то же условие, но столбец «голый»
#      → SEARCH (document_id=? AND tokens>?), работают оба столбца.
# 3.4: document_id = '7' — в SQLite индекс НЕ потерялся: у столбца тип INTEGER,
#      и SQLite по правилам affinity превращает строку '7' в число 7 до поиска.
#      В PostgreSQL и MySQL несовпадение типов может выключить индекс — поэтому
#      этот пункт урока надо помнить, даже если SQLite его прощает.

# 3.4 (бонус) другой тип
plan("3.4 document_id = '7' (строка)",
     "SELECT * FROM chunks WHERE document_id = '7'")


# 4. Частичный индекс: необработанные чанки (~5%)
share = cur.execute("SELECT AVG(processed = 0) FROM chunks").fetchone()[0]
print(f"\n4. доля необработанных: {share:.1%}")
cur.execute("""
    CREATE INDEX idx_unprocessed ON chunks(tokens) WHERE processed = 0
""")
plan("4a. попадает: processed = 0",
     "SELECT id FROM chunks WHERE processed = 0 AND tokens > 500")
plan("4b. не попадает: processed = 1",
     "SELECT id FROM chunks WHERE processed = 1 AND tokens > 500")
# 4a: SEARCH ... COVERING INDEX idx_unprocessed (tokens>?) — условие запроса
#     содержит processed = 0, то же, что в WHERE индекса, значит все нужные
#     строки гарантированно в нём есть. COVERING — потому что нужен только id
#     (rowid хранится в любом индексе) и tokens, в таблицу ходить не надо.
# 4b: SCAN chunks — строк с processed = 1 в частичном индексе нет вообще, он
#     ничем не может помочь, и планировщик его не трогает.
# Цена и выгода: индекс хранит ~10 000 строк вместо 200 000 (см. sqlite_stat1
#   в задании 5) — меньше места, и запись строк с processed = 1 его не трогает.
# В SQLite условие запроса должно буквально содержать условие индекса:
#   processed = 0 — да; processed < 1 — уже нет, хотя по смыслу то же самое.


# 5. ANALYZE
plan("5. document_id = 7 до ANALYZE",
     "SELECT * FROM chunks WHERE document_id = 7")
cur.execute("ANALYZE")
plan("5. document_id = 7 после ANALYZE",
     "SELECT * FROM chunks WHERE document_id = 7")
show("5. sqlite_stat1", "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl, idx")
# План не изменился: и до, и после ANALYZE — SEARCH по idx_chunks_doc_tokens.
#   Для document_id = 7 есть один подходящий индекс, и эвристика без
#   статистики уже выбрала правильно. ANALYZE меняет план, когда есть ВЫБОР:
#   несколько индексов или индекс против полного перебора, и решение зависит
#   от того, сколько строк отберёт условие.
# sqlite_stat1 — по строке на индекс, первое число — сколько строк в индексе,
#   дальше — сколько в среднем строк приходится на одно значение префикса:
#   idx_chunks_doc_tokens '200000 4000 5': 200 000 строк; на одно значение
#     document_id ~4000 строк (200 000 / 50); на пару (document_id, tokens) ~5.
#     Отсюда планировщик знает: document_id = 7 отберёт ~2% таблицы — индекс
#     выгоден.
#   idx_chunks_text '200000 1': на одно значение text — 1 строка, текст уникален.
#   idx_unprocessed '9950 11': в частичном индексе всего 9950 строк (5%).
#   documents '53': у таблицы без индексов записано только число строк.


# 6. Итоговая задача блока
Q6 = """
    WITH doc_stats AS (                      -- шаг 1: каждый документ, даже без чанков
        SELECT d.id, d.title, d.source,
               COUNT(c.id)               AS chunk_count,
               COALESCE(SUM(c.tokens), 0) AS chunk_tokens
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id
    ),
    ranked AS (                              -- шаг 2: окна по источнику
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY source
                                  ORDER BY chunk_tokens DESC, id) AS rn,
               COUNT(*)          OVER (PARTITION BY source) AS doc_count,
               SUM(chunk_count)  OVER (PARTITION BY source) AS source_chunks,
               SUM(chunk_tokens) OVER (PARTITION BY source) AS source_tokens
        FROM doc_stats
    )
    SELECT source, doc_count, source_chunks, source_tokens,
           title AS leader,
           ROUND(100.0 * chunk_tokens / NULLIF(source_tokens, 0), 2) AS leader_pct
    FROM ranked
    WHERE rn = 1                             -- лидер источника
    ORDER BY source
"""
show("6. итог по источникам", Q6)
plan("6.", Q6)
# Что откуда:
#   CTE — два шага: doc_stats (по документу), ranked (окна по источнику).
#   LEFT JOIN + COALESCE — документы без чанков остаются с 0 чанков и 0 токенов
#     («Пустой README» в github, оба черновика в drafts). COUNT(c.id), а не
#     COUNT(*): у документа без чанков c.id = NULL, и счёт честно даёт 0.
#   ROW_NUMBER() OVER (PARTITION BY source ORDER BY chunk_tokens DESC, id) —
#     лидер источника; id — второй ключ, чтобы при ничьей выбор был
#     воспроизводимым (в drafts оба по 0, лидером стал Черновик A с меньшим id).
#   SUM/COUNT ... OVER (PARTITION BY source) без ORDER BY — рамка на всё окно,
#     в каждой строке итог по источнику, строки не склеиваются.
#   NULLIF(source_tokens, 0) — у drafts сумма 0, деление дало бы ошибку или
#     бессмыслицу; с NULLIF доля = NULL (None): «доли нет», а не «доля 0%».
#   Фильтр rn = 1 — снаружи, в основном запросе: окно в WHERE нельзя.
# План: SCAN d + SEARCH c USING COVERING INDEX idx_chunks_doc_tokens LEFT-JOIN —
#   для каждого документа его чанки находятся по индексу, и в таблицу ходить не
#   нужно (нужны только document_id и tokens — оба в индексе). Затем
#   TEMP B-TREE FOR ORDER BY для разбиения на окна и для итоговой сортировки.

conn.close()


# 7. Устно, потом письменно.
#   Порядок столбцов в составном индексе: индекс отсортирован по первому
#     столбцу, второй упорядочен только внутри равных значений первого.
#     Поэтому индекс (document_id, tokens) помогает WHERE document_id = 7 и
#     document_id = 7 AND tokens > 400, но не WHERE tokens > 400 (задание 1d).
#     Правило: сначала столбцы с точным равенством, потом диапазон и сортировка.
#   Почему LOWER(title) = ... не использует индекс по title: индекс хранит
#     значения title в отсортированном виде, а значений LOWER(title) в нём нет.
#     Чтобы проверить условие, нужно вычислить LOWER для каждой строки — это
#     и есть SCAN. Лечится индексом по выражению LOWER(title) (задание 3.1).
#   Что меняет ANALYZE: собирает статистику (sqlite_stat1) — сколько строк в
#     среднем приходится на значение каждого индекса. Без неё планировщик
#     выбирает по эвристикам, с ней — оценивает, сколько строк отберёт условие,
#     и осознанно выбирает между индексами или между индексом и полным
#     перебором. Поэтому планы на пустой учебной базе и на проде могут
#     расходиться.