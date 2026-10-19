import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "sql_block_01.db"
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
    tokens INTEGER                -- без NOT NULL: может быть неизвестно
)
""")

cur.execute("""
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,          -- без NOT NULL: допускаем сироту
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    author TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
""")

cur.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", [
    (1, "Attention is all you need", "arxiv", 8400),
    (2, "BERT", "arxiv", 7200),
    (3, "Внутренний регламент", "confluence", 1500),
    (4, "Пустой черновик", "confluence", 0),       # без чанков
    (5, "README ragkit", "github", None),           # tokens неизвестно
    (6, "Гайд по деплою", "github", 3000),
])

cur.executemany(
    "INSERT INTO chunks (document_id, text, tokens, author) VALUES (?, ?, ?, ?)", [
    (1, "Введение", 1000, "Vaswani"),
    (1, "Архитектура", 1200, "Vaswani"),
    (1, "Self-attention", 1500, None),
    (1, "Multi-head", 1800, "Shazeer"),
    (1, "Эксперименты", 2000, None),
    (1, "Выводы", 900, "Parmar"),
    (2, "Pre-training", 3000, "Devlin"),
    (2, "Fine-tuning", 3500, None),
    (3, "Правила доступа", 1500, None),
    (5, "Установка", 400, "Dmitriy"),
    (5, "Примеры", 600, None),
    (6, "Docker", 2500, "Dmitriy"),
    (None, "Сирота без документа", 300, None),     # сирота для задания 3
])
conn.commit()


def one(sql, params=()):
    """Выполнить запрос и вернуть одно значение."""
    return cur.execute(sql, params).fetchone()[0]


def show(title, sql, params=()):
    print(f"\n{title}")
    for row in cur.execute(sql, params):
        print(row)
def plan(title, sql):
    print(f"\nплан: {title}")
    for row in cur.execute("EXPLAIN QUERY PLAN " + sql):
        print("   ", row[3])          # row[3] — текст шага плана


# 1. Трёхзначная логика
print("\n1. Трёхзначная логика")
print("NULL = NULL   →", one("SELECT NULL = NULL"))
print("NULL <> NULL  →", one("SELECT NULL <> NULL"))
print("NULL IS NULL  →", one("SELECT NULL IS NULL"))
print("author = NULL  →", one("SELECT COUNT(*) FROM chunks WHERE author = NULL"))
print("author IS NULL →", one("SELECT COUNT(*) FROM chunks WHERE author IS NULL"))
# Объяснение:
#   NULL = NULL  → NULL (None): NULL значит «неизвестно»; равно ли неизвестное
#                  неизвестному — тоже неизвестно, результат UNKNOWN, а не TRUE.
#   NULL <> NULL → NULL (None): по той же причине; любое сравнение с NULL
#                  (=, <>, <, >) даёт UNKNOWN.
#   NULL IS NULL → 1: IS NULL — не сравнение, а проверка «значения нет»;
#                  она всегда даёт TRUE или FALSE, никогда UNKNOWN.
#   author = NULL  → 0: для каждой строки сравнение даёт UNKNOWN, даже где
#                  author пуст; WHERE пропускает только TRUE → не прошла ни одна.
#   author IS NULL → 6: проверка честно находит 6 чанков без автора
#                  (Self-attention, Эксперименты, Fine-tuning,
#                  Правила доступа, Примеры, Сирота).


# 2. Агрегаты и NULL
print("\n2. Агрегаты и NULL")
row = cur.execute("""
    SELECT COUNT(*), COUNT(tokens), AVG(tokens), AVG(COALESCE(tokens, 0))
    FROM documents
""").fetchone()
print("COUNT(*), COUNT(tokens), AVG, AVG(COALESCE):", row)
assert row[2] == 20100 / 5      # сумма известных tokens / 5 строк с не-NULL
assert row[3] == 20100 / 6      # та же сумма / все 6 строк (NULL стал 0)
# Объяснение:
#   COUNT(*) = 6      — считает строки, неважно что в них.
#   COUNT(tokens) = 5 — пропускает строку, где tokens NULL (документ 5).
#   Сумма известных: 8400 + 7200 + 1500 + 0 + 3000 = 20100.
#   AVG(tokens) = 20100 / 5 = 4020 — AVG игнорирует NULL: документ 5
#     не участвует ни в сумме, ни в делителе.
#   AVG(COALESCE(tokens, 0)) = 20100 / 6 = 3350 — COALESCE заменил NULL
#     на 0, строка стала обычной: в сумму она добавила 0, а в делитель +1.
#   Средние различаются, потому что делитель разный: 5 против 6.
#   Какое из них «правильное», зависит от смысла: если NULL — «не знаем»,
#   честнее 4020; если NULL — «на самом деле ноль», то 3350.


# 3. NOT IN против NOT EXISTS
Q_NOT_IN = """
    SELECT title FROM documents
    WHERE id NOT IN (SELECT document_id FROM chunks)
"""
Q_NOT_EXISTS = """
    SELECT title FROM documents d
    WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
"""
r_not_in = cur.execute(Q_NOT_IN).fetchall()
r_not_exists = cur.execute(Q_NOT_EXISTS).fetchall()
print("\n3. Документы без чанков")
print("NOT IN:    ", r_not_in)
print("NOT EXISTS:", r_not_exists)
assert r_not_in != r_not_exists          # зачёт требует, чтобы разошлись
# Объяснение:
#   NOT IN вернул [] — пусто, потому что среди document_id есть NULL
#   (чанк-сирота), и условие ни для одного документа не может стать TRUE.
#   Раскрытие: id NOT IN (1, 1, ..., 6, NULL) превращается в
#              id <> 1 AND id <> 2 AND ... AND id <> NULL
#   Для документа 4: 4 <> 1 → TRUE, 4 <> 2 → TRUE, ..., 4 <> NULL → UNKNOWN;
#   TRUE AND ... AND UNKNOWN = UNKNOWN, а WHERE пропускает только TRUE →
#   документ 4 отброшен, хотя чанков у него нет. Так с каждым документом.
#   NOT EXISTS вернул [('Пустой черновик',)] — правильно: для каждого документа
#   он спрашивает «есть ли хоть одна строка с c.document_id = d.id». У сироты
#   NULL = 4 → UNKNOWN, строка просто «не нашлась» и ничего не ломает.
#   Для документа 4 не нашлось ни одной → NOT EXISTS = TRUE.


# 4. ON против WHERE
Q_A = """
    SELECT d.title, c.text, c.tokens
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id AND c.tokens > 1500
"""
Q_B = """
    SELECT d.title, c.text, c.tokens
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    WHERE c.tokens > 1500
"""
show("4A. условие в ON", Q_A)
show("4B. условие в WHERE", Q_B)
print("строк A:", len(cur.execute(Q_A).fetchall()),
      "строк B:", len(cur.execute(Q_B).fetchall()))
# Объяснение (порог 1500 вместо 100 из урока: у нас все чанки > 100,
# и разница была бы всего в одну строку):
#   A — 8 строк. Условие в ON фильтрует то, ЧТО присоединяется: к документу
#   цепляются только чанки > 1500. Если таких нет, LEFT JOIN всё равно
#   оставляет документ, заполнив поля чанка NULL. Отсюда три строки с None:
#   «Внутренний регламент» (его чанк ровно 1500, не больше),
#   «Пустой черновик» (чанков нет), «README ragkit» (чанки 400 и 600).
#   B — 5 строк. Условие в WHERE работает ПОСЛЕ соединения и фильтрует
#   результат: в тех трёх строках c.tokens = NULL, NULL > 1500 → UNKNOWN →
#   строки отброшены. Разница 8 − 5 = 3 — ровно эти три документа.
#   Условие на правую таблицу в WHERE превращает LEFT JOIN в INNER JOIN.


# 5. Антисоединение третьим способом + сравнение планов
Q_LEFT = """
    SELECT d.title FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    WHERE c.id IS NULL
"""
print("\n5. LEFT JOIN ... IS NULL:", cur.execute(Q_LEFT).fetchall())
assert cur.execute(Q_LEFT).fetchall() == r_not_exists
plan("NOT IN", Q_NOT_IN)
plan("NOT EXISTS", Q_NOT_EXISTS)
plan("LEFT JOIN IS NULL", Q_LEFT)
# Почему c.id: это первичный ключ, у настоящего чанка он никогда не NULL.
#   Если после LEFT JOIN c.id IS NULL — пары не нашлось, документ без чанков.
# Планы:
#   NOT IN:     SCAN documents + LIST SUBQUERY (SCAN chunks один раз → список).
#               Быстрый, но даёт неверный ответ при NULL в списке.
#   NOT EXISTS: SCAN d + CORRELATED SCALAR SUBQUERY (SCAN c) — логически
#               подзапрос на каждый документ, без индекса по chunks.document_id
#               каждый раз полный проход по chunks.
#   LEFT JOIN:  SCAN d + SEARCH c USING AUTOMATIC COVERING INDEX — SQLite сам
#               построил временный индекс по document_id и ищет по нему,
#               а не сканирует chunks заново.
# Какой нравится больше: по плану на этих данных — LEFT JOIN (поиск по индексу
#   вместо повторного сканирования). Но писать стоит NOT EXISTS: он так же
#   корректен при NULL, точнее выражает смысл «нет ни одного», а с обычным
#   индексом на chunks.document_id его план тоже станет SEARCH.
#   NOT IN:     SCAN documents + LIST SUBQUERY (SCAN chunks один раз → список,
#               по нему CREATE BLOOM FILTER — быстрая проверка «точно нет»).
#               Быстрый, но даёт неверный ответ при NULL в списке.


# 6. Коррелированный подзапрос против WITH + JOIN
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
Q_CTE = """
    WITH doc_avg AS (
        SELECT document_id, AVG(tokens) AS avg_tokens
        FROM chunks
        GROUP BY document_id
    )
    SELECT c.id, d.title, c.tokens, a.avg_tokens
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    JOIN doc_avg a ON a.document_id = c.document_id
    WHERE c.tokens > a.avg_tokens
    ORDER BY c.id
"""
show("6. коррелированный подзапрос", Q_CORR)
show("6. WITH + JOIN", Q_CTE)
r_corr = cur.execute(Q_CORR).fetchall()
assert r_corr == cur.execute(Q_CTE).fetchall()
assert [row[0] for row in r_corr] == [3, 4, 5, 8, 11]
# Ручной расчёт средних:
#   док 1: 8400 / 6 = 1400 → больше: 1500 (id 3), 1800 (id 4), 2000 (id 5)
#   док 2: 6500 / 2 = 3250 → больше: 3500 (id 8)
#   док 3: 1500 / 1 = 1500 → никто (чанк не больше самого себя)
#   док 5: 1000 / 2 = 500  → больше: 600 (id 11)
#   док 6: 2500 / 1 = 2500 → никто
#   Сирота отпадает на JOIN documents: NULL = d.id не бывает TRUE.
plan("коррелированный", Q_CORR)
plan("WITH + JOIN", Q_CTE)
# Сравнение планов:
#   Коррелированный: две строки CORRELATED SCALAR SUBQUERY, в каждой SCAN c2 —
#     среднее пересчитывается для каждой внешней строки, причём дважды
#     (в SELECT и в WHERE). На n чанков это порядка n × n строк — O(n²).
#   WITH + JOIN: CO-ROUTINE doc_avg — средние считаются ОДИН раз за один проход
#     (SCAN chunks + TEMP B-TREE FOR GROUP BY), потом присоединяются по
#     автоматическому индексу (SEARCH a USING AUTOMATIC COVERING INDEX).
#     Порядка n строк плюс группировка — заметно дешевле на больших таблицах.
#   Результаты одинаковые (assert), но WITH-вариант и читается сверху вниз,
#   и не зависит от того, догадается ли планировщик переписать подзапрос.


# 7. CASE внутри агрегата
Q7 = """
    SELECT source,
           COUNT(*) AS total,
           SUM(CASE WHEN tokens > 5000 THEN 1 ELSE 0 END) AS big,
           SUM(CASE WHEN tokens IS NULL THEN 1 ELSE 0 END) AS unknown,
           ROUND(100.0 * SUM(CASE WHEN tokens > 5000 THEN 1 ELSE 0 END)
                 / NULLIF(COUNT(*), 0), 1) AS big_pct
    FROM documents
    GROUP BY source
    ORDER BY source
"""
show("7. по источникам: всего, крупных, неизвестных, % крупных", Q7)
# CASE внутри SUM — «условный подсчёт»: строка, подходящая под условие, даёт 1,
#   остальные 0; SUM складывает единицы. Три счётчика — один проход, один запрос.
# Почему 100.0, а не 100: в SQLite целое / целое = целое деление (как // в Python),
#   1 / 2 = 0. Множитель 100.0 делает выражение дробным.
# NULLIF(COUNT(*), 0): если делитель 0, превращает его в NULL, и деление даёт
#   NULL вместо ошибки. Здесь COUNT(*) в группе не бывает 0, но защита —
#   привычка: в другом запросе делитель может оказаться нулём.
# Документ 5 (tokens NULL): в big не попал (NULL > 5000 → UNKNOWN → ELSE 0),
#   зато посчитан в unknown.


# 8. CTE из двух шагов: рассинхрон documents.tokens и суммы чанков
Q8 = """
    WITH stats AS (
        SELECT document_id,
               COUNT(*) AS chunk_count,
               SUM(tokens) AS chunk_tokens
        FROM chunks
        GROUP BY document_id
    ),
    joined AS (
        SELECT d.id, d.title, d.tokens,
               COALESCE(s.chunk_count, 0) AS chunk_count,
               COALESCE(s.chunk_tokens, 0) AS chunk_tokens
        FROM documents d
        LEFT JOIN stats s ON s.document_id = d.id
    )
    SELECT * FROM joined
    WHERE chunk_tokens IS NOT tokens
    ORDER BY id
"""
show("8. рассинхрон (IS NOT)", Q8)
show("8. то же через <>", Q8.replace("IS NOT", "<>"))
# Шаг 1 (stats): число чанков и сумма токенов по каждому document_id.
# Шаг 2 (joined): LEFT JOIN, чтобы остались все документы, и COALESCE,
#   чтобы документ 4 без чанков сравнивался как 0 с 0 — и совпал.
# Сирота: его строка в stats (document_id NULL) ни к чему не присоединилась.
# Почему результаты различаются: у документа 5 tokens = NULL.
#   1000 <> NULL → UNKNOWN → WHERE отбрасывает, рассинхрон потерян.
#   1000 IS NOT NULL → TRUE: IS NOT сравнивает с учётом NULL
#   («значение есть против значения нет» — это различие).
#   Та же ловушка, что в задании 1: сравнение с NULL через = / <> молча
#   выкидывает строки.


# 9. UNION ALL против UNION
Q9_ALL = """
    SELECT 'документ', title FROM documents
    UNION ALL
    SELECT 'чанк', text FROM chunks
"""
Q9_UNION = Q9_ALL.replace("UNION ALL", "UNION")
r_all = cur.execute(Q9_ALL).fetchall()
r_union = cur.execute(Q9_UNION).fetchall()
show("9. UNION ALL", Q9_ALL)
print("строк UNION ALL:", len(r_all), " строк UNION:", len(r_union))
plan("UNION", Q9_UNION)
# Число строк не изменилось: 19 = 6 документов + 13 чанков. UNION убирает
#   только ПОЛНОСТЬЮ одинаковые строки, а у нас их нет: метка 'документ'/'чанк'
#   отличает одни от других, и все названия и тексты уникальны.
# Но план показывает UNION USING TEMP B-TREE: UNION всё равно строил временную
#   структуру, чтобы искать дубли, которых нет. Это лишняя работа — поэтому
#   по умолчанию UNION ALL, а UNION только когда дубли точно надо убрать.

# 10. Порядок выполнения SELECT:
#   FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
#
#   Следствие 1: алиас из SELECT нельзя использовать в WHERE, но можно в ORDER BY.
#     Пример: SELECT tokens * 2 AS doubled FROM chunks WHERE doubled > 100 —
#     ошибка: WHERE выполняется раньше SELECT, имени doubled ещё нет.
#     ORDER BY doubled — работает: ORDER BY идёт после SELECT.
#
#   Следствие 2: условие в WHERE после LEFT JOIN фильтрует уже готовый результат
#     соединения и выкидывает строки с NULL — LEFT JOIN становится INNER.
#     Пример: задание 4. Условие c.tokens > 1500 в ON (работает во время JOIN)
#     дало 8 строк; в WHERE (работает после JOIN) — 5: три документа без
#     крупных чанков пропали, потому что NULL > 1500 → UNKNOWN.
conn.close()   # пока в конце; дальше будем дописывать задания перед ним