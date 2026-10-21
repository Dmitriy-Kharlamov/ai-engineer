import sys
import time
import warnings

import numpy as np
import pandas as pd

print("numpy", np.__version__, "| pandas", pd.__version__)


def best_of(func, runs=7):
    """Минимальное время из runs прогонов."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


# ---------------------------------------------------------------------------
# 1. Память и скорость
# ---------------------------------------------------------------------------
print("\n1. Память и скорость")
N = 1_000_000
lst = [float(i) for i in range(N)]
arr = np.arange(N, dtype=np.float64)

mem_list = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)
mem_arr = arr.nbytes
print(f"список:  {mem_list / 1e6:.1f} МБ")
print(f"ndarray: {mem_arr / 1e6:.1f} МБ")
print(f"список / ndarray по памяти: {mem_list / mem_arr:.1f}")
assert mem_arr == N * 8
# Список — это массив указателей (8 байт на элемент) плюс отдельный объект
#   float на каждое число (24 байта: заголовок с типом и счётчиком ссылок +
#   само число). ndarray — только сами числа подряд, по 8 байт.

t_list = best_of(lambda: [x * 2 for x in lst])
t_vec = best_of(lambda: arr * 2)
print(f"\nсписок, comprehension: {t_list:.4f} с")
print(f"ndarray, arr * 2:      {t_vec:.5f} с")
print(f"comprehension / векторно: {t_list / t_vec:.0f}")
assert np.array_equal(np.array([x * 2 for x in lst]), arr * 2)


def loop_over_array():
    out = np.empty(N)
    for i in range(len(arr)):
        out[i] = arr[i] * 2
    return out


def loop_over_list():
    out = [0.0] * N
    for i in range(len(lst)):
        out[i] = lst[i] * 2
    return out


t_loop_arr = best_of(loop_over_array)
t_loop_list = best_of(loop_over_list)
print(f"\nfor по списку:  {t_loop_list:.4f} с")
print(f"for по ndarray: {t_loop_arr:.4f} с")
print(f"ndarray / список в цикле: {t_loop_arr / t_loop_list:.1f}")
# Почему for по ndarray медленнее for по списку: в списке lst[i] просто
#   отдаёт уже существующий объект float. В массиве лежат «голые» 8 байт, и
#   arr[i] каждый раз упаковывает их в новый объект numpy.float64 (выделение
#   памяти, заголовок), а out[i] = ... распаковывает обратно. На каждом шаге
#   два лишних преобразования между машинным числом и объектом Python.
#   Векторная операция делает этот цикл на C без единого объекта Python.


# ---------------------------------------------------------------------------
# 2. dtype
# ---------------------------------------------------------------------------
print("\n2. dtype")
m64 = np.zeros((10_000, 256), dtype=np.float64)
m32 = np.zeros((10_000, 256), dtype=np.float32)
print(f"float64: {m64.nbytes / 1e6:.2f} МБ,  float32: {m32.nbytes / 1e6:.2f} МБ")
assert m64.nbytes == 2 * m32.nbytes

n_vec, dim = 1_000_000, 1024
gb64 = n_vec * dim * 8 / 1e9
gb32 = n_vec * dim * 4 / 1e9
print(f"1 млн × 1024: float64 = {gb64:.1f} ГБ, float32 = {gb32:.1f} ГБ")
# 1 000 000 × 1024 × 8 байт = 8,19 ГБ; × 4 байта = 4,10 ГБ.

big = np.array([2**62], dtype=np.int64)
np_result = big * 4
py_result = 2**62 * 4
print(f"numpy int64: 2**62 * 4 = {np_result[0]}")
print(f"Python int:  2**62 * 4 = {py_result}")
assert np_result[0] != py_result and np_result[0] == 0
# 2**62 * 4 = 2**64, а int64 вмещает максимум 2**63 − 1. Результат «обрезается»
#   по модулю 2**64 и превращается в 0 — без исключения и без предупреждения.
#   Python-int растёт сколько нужно, поэтому там ответ верный.


# ---------------------------------------------------------------------------
# 3. Broadcasting: сначала предсказание, потом проверка
# ---------------------------------------------------------------------------
print("\n3. Broadcasting")
cases = [
    # (форма A, форма B, предсказание)
    ((3,),   (3,),   "(3,) — формы равны, поэлементно"),
    ((2, 3), (3,),   "(2, 3) — (3,) считается (1, 3) и растягивается на 2 строки"),
    ((2, 3), (2, 1), "(2, 3) — столбец (2, 1) растягивается на 3 столбца"),
    ((2, 3), (2,),   "ОШИБКА — справа 3 против 2, ни одна не равна 1"),
    ((3, 1), (1, 3), "(3, 3) — обе растягиваются: внешнее «произведение»"),
]
for shape_a, shape_b, predicted in cases:
    a = np.ones(shape_a)
    b = np.ones(shape_b)
    try:
        real = str((a + b).shape)
    except ValueError as e:
        real = f"ошибка: {e}"
    print(f"{str(shape_a):7} + {str(shape_b):7} | предсказание: {predicted}")
    print(f"{'':17}| реальность:   {real}")
# Правило: формы выравниваются по правому краю и сравниваются справа налево;
#   размерности совместимы, если равны или одна из них 1; недостающие слева
#   считаются 1. (2, 3) и (2,): правые размерности 3 и 2 — не равны и ни одна
#   не 1 → ошибка, хотя «по смыслу» хотелось прибавить по строкам — для этого
#   нужна форма (2, 1).


# ---------------------------------------------------------------------------
# 4. axis
# ---------------------------------------------------------------------------
print("\n4. axis")
m = np.array([[3, 7, 1, 0],
              [9, 2, 8, 4],
              [5, 5, 6, 1]])
print(m)
for name in ("sum", "mean", "max"):
    for axis in (None, 0, 1):
        res = getattr(m, name)(axis=axis)
        print(f"{name}(axis={axis}): {res}  форма {np.shape(res)}")
assert m.sum(axis=0).shape == (4,) and m.sum(axis=1).shape == (3,)
assert m.argmax(axis=1).tolist() == [1, 0, 2]
print("argmax(axis=1):", m.argmax(axis=1), "форма", m.argmax(axis=1).shape)
# axis — это ось, которая исчезает: у (3, 4) argmax(axis=1) пробегает по оси 1
#   (по 4 столбцам внутри каждой строки), и она пропадает → (3,): для каждой
#   из 3 строк номер столбца с максимумом: [1, 0, 2] (7, 9 и 6).


# ---------------------------------------------------------------------------
# 5. Представления против копий
# ---------------------------------------------------------------------------
print("\n5. Представления против копий")
lst5 = [1, 2, 3, 4]
part = lst5[1:3]
part[0] = 99
print("срез списка:   ", lst5, "← не изменился")
assert lst5 == [1, 2, 3, 4]

arr5 = np.array([1, 2, 3, 4])
view = arr5[1:3]
print("срез ndarray:   view.base is None →", view.base is None)
view[0] = 99
print("                ", arr5, "← изменился")
assert arr5.tolist() == [1, 99, 3, 4]

arr5 = np.array([1, 2, 3, 4])
masked = arr5[arr5 > 2]
print("булева маска:   masked.base is None →", masked.base is None)
masked[0] = 99
print("                ", arr5, "← не изменился")
assert arr5.tolist() == [1, 2, 3, 4]

fancy = arr5[[0, 2]]
print("список индексов: fancy.base is None →", fancy.base is None)
fancy[0] = 99
print("                ", arr5, "← не изменился")
assert arr5.tolist() == [1, 2, 3, 4]

copy = arr5[1:3].copy()
print(".copy() среза:   copy.base is None →", copy.base is None)
# Почему срез — представление: простой срез описывается началом, длиной и
#   шагом, и numpy может выдать «окно» в тот же блок памяти, не копируя ни
#   байта. Срез миллиона элементов — O(1) по времени и памяти. Для
#   больших данных это решающее преимущество, а опасность лечится явным
#   .copy(). Маска и список индексов выбирают элементы вразнобой — такое
#   окном не описать, поэтому numpy вынужден создать новый массив.


# ---------------------------------------------------------------------------
# 6. Косинусная близость и top-k
# ---------------------------------------------------------------------------
print("\n6. Косинусная близость")
rng = np.random.default_rng(42)
matrix = rng.standard_normal((50_000, 384))
query = rng.standard_normal(384)


def cosine_similarities(query, matrix):
    q = query / np.linalg.norm(query)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)   # (n, 1)
    return m @ q                                                  # (n, d) @ (d,) → (n,)


sims = cosine_similarities(query, matrix)
assert sims.shape == (50_000,)
assert np.all(sims >= -1 - 1e-9) and np.all(sims <= 1 + 1e-9)
self_sims = cosine_similarities(matrix[123], matrix)
assert np.isclose(self_sims[123], 1.0)
print(f"min {sims.min():.4f}, max {sims.max():.4f}; к самому себе: {self_sims[123]!r}")

k = 10


def top_k_sort():
    return np.argsort(-sims)[:k]


def top_k_partition():
    idx = np.argpartition(-sims, k)[:k]
    return idx[np.argsort(-sims[idx])]


assert np.array_equal(top_k_sort(), top_k_partition())
print("top-10:", top_k_partition())
t_sort = best_of(top_k_sort)
t_part = best_of(top_k_partition)
print(f"argsort:      {t_sort * 1000:.2f} мс")
print(f"argpartition: {t_part * 1000:.2f} мс")
print(f"argsort / argpartition: {t_sort / t_part:.1f}")
# argsort полностью сортирует 50 000 значений — O(n log n), хотя нужны 10.
#   argpartition только раскладывает: 10 лучших влево, остальное вправо, без
#   упорядочивания внутри — O(n) (алгоритм introselect). Потом досортировываем
#   лишь эти 10 — O(k log k), копейки. Это heapq.nlargest, но на C и без
#   объектов Python.


# ---------------------------------------------------------------------------
# 7. NaN
# ---------------------------------------------------------------------------
print("\n7. NaN")
full = np.arange(1, 1001, dtype=np.float64)        # 1..1000
with_nan = full.copy()
with_nan[500] = np.nan                              # было 501.0
print(f"sum {with_nan.sum()}, mean {with_nan.mean()}, max {with_nan.max()}")
print(f"nansum {np.nansum(with_nan)}, nanmean {np.nanmean(with_nan):.4f}")
print("nan == nan:", np.nan == np.nan, "| np.isnan находит:", np.flatnonzero(np.isnan(with_nan)))
assert np.isnan(with_nan.sum()) and np.isnan(with_nan.mean()) and np.isnan(with_nan.max())
assert not np.isnan(np.nansum(with_nan))
assert np.nan != np.nan

lost = len(with_nan) - np.count_nonzero(~np.isnan(with_nan))
print(f"mean полных данных: {full.mean()}, nanmean: {np.nanmean(with_nan):.4f}, "
      f"потеряно элементов: {lost}")
assert lost == 1
# nanmean посчитал среднее по 999 значениям: сумма без 501, делённая на 999.
#   Потерялся 1 элемент, и среднее сдвинулось с 500.5 до ~500.4995.
# Какая честнее: mean, возвращающий nan, честно говорит «данные неполные,
#   ответа нет» — его нельзя пропустить. nanmean молча считает по тому, что
#   есть, — это правильно, только если пропуски случайны и их мало. Если
#   пропусков много или они не случайны (например, nan там, где значение было
#   слишком большим), nanmean даст уверенно выглядящий, но смещённый ответ.
#   Поэтому: сначала посчитать np.isnan(x).sum(), и только потом выбирать.


# ---------------------------------------------------------------------------
# 8. Pandas
# ---------------------------------------------------------------------------
print("\n8. Pandas")
df = pd.DataFrame({
    "id":          [1, 2, 3, 4, 5, 6, 7, 8],
    "document_id": [1, 1, 1, 2, 2, 3, 3, 5],
    "source":      ["arxiv", "arxiv", "arxiv", "arxiv", "arxiv",
                    "confluence", "confluence", "github"],
    "tokens":      [50, 300, None, 700, 90, None, 1500, 400],
    "text":        ["Intro", "Model", "Pics", "Pretrain", "Notes",
                    "Rules", "Access", "Setup"],
})

print("\n8.1 info():")
df.info()
print("\n8.1 describe():")
print(df.describe())
# tokens стал float64: в целочисленном столбце numpy нет места для NaN.
#   info() показывает 6 non-null из 8; describe() считает count=6 — пропуски
#   в статистику не идут.

print("\n8.2 фильтрация tokens > 100")
big_df = df[df["tokens"] > 100]
print("индекс до reset_index:   ", big_df.index.tolist())
big_df = big_df.reset_index(drop=True)
print("индекс после reset_index:", big_df.index.tolist())
assert big_df.index.tolist() == list(range(len(big_df)))
# После фильтрации остались старые метки с дырками (1, 3, 6, 7). Если дальше
#   обратиться df.loc[0], получим KeyError — строки с меткой 0 уже нет.

print("\n8.3 loc против iloc")
print("df.loc[0:2]  строк:", len(df.loc[0:2]))
print("df.iloc[0:2] строк:", len(df.iloc[0:2]))
assert len(df.loc[0:2]) == 3 and len(df.iloc[0:2]) == 2
# loc работает по МЕТКАМ и включает правую границу (метки 0, 1, 2).
#   iloc работает по ПОЗИЦИЯМ, как срез Python, — правая граница не входит.

print("\n8.4 присваивание через цепочку и через loc")
chain = df.copy()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    chain[chain["tokens"] > 100]["source"] = "x"
print("цепочка: изменено строк:", (chain["source"] == "x").sum())
for w in caught:
    print(f"   предупреждение {w.category.__name__}: {str(w.message).splitlines()[0]}")
proper = df.copy()
proper.loc[proper["tokens"] > 100, "source"] = "x"
print("loc:     изменено строк:", (proper["source"] == "x").sum())
assert (chain["source"] == "x").sum() == 0
assert (proper["source"] == "x").sum() == 4
# chain[маска] создаёт НОВЫЙ временный DataFrame, ["source"] = "x" меняет
#   уже его, и временный объект тут же выбрасывается — исходный не тронут.
#   .loc[маска, "столбец"] — одна операция над самим df, без промежуточной
#   копии.

print("\n8.5 groupby")
stats = df.groupby("source").agg(chunks=("id", "size"), tokens=("tokens", "sum"))
print(stats)
assert stats.loc["arxiv", "chunks"] == 5 and stats.loc["arxiv", "tokens"] == 1140
# size считает строки (включая те, где tokens NaN), sum пропускает NaN —
#   как COUNT(*) и SUM(tokens) в SQL.

print("\n8.6 merge")
docs = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "title": ["Attention", "BERT", "Регламент", "Пустой черновик"],
})
merged = docs.merge(df, left_on="id", right_on="document_id",
                    how="left", validate="one_to_many", suffixes=("_doc", "_chunk"))
print(merged[["id_doc", "title", "id_chunk", "tokens"]])
assert len(merged) == 8      # 7 чанков документов 1–3 + 1 строка документа 4 с NaN
# how="left" — LEFT JOIN: «Пустой черновик» остался с NaN. Чанк документа 5
#   отпал: документа 5 в docs нет (в LEFT JOIN главная — левая таблица).
try:
    docs.merge(df, left_on="id", right_on="document_id", how="left",
               validate="one_to_one")
except pd.errors.MergeError as e:
    print("validate='one_to_one':", type(e).__name__, "—", e)
# validate проверяет уникальность ключей ДО соединения: у одного документа
#   несколько чанков, значит справа document_id повторяется — связь «один ко
#   многим», а не «один к одному», и pandas падает вместо того, чтобы тихо
#   размножить строки.


# ---------------------------------------------------------------------------
# 9. apply против векторизации
# ---------------------------------------------------------------------------
print("\n9. apply против .str")
rng9 = np.random.default_rng(42)
words = np.array(["Attention", "BERT", "Retrieval", "Chunk", "ЭМБЕДДИНГ"])
texts = pd.DataFrame({"text": words[rng9.integers(0, len(words), 200_000)]})
col = texts["text"]
print("dtype столбца:", col.dtype, "| хранение:", getattr(col.dtype, "storage", "object"))
assert col.str.lower().tolist() == col.apply(str.lower).tolist()
t_str = best_of(lambda: col.str.lower())
t_apply = best_of(lambda: col.apply(str.lower))
print(f".str.lower():      {t_str * 1000:.1f} мс")
print(f".apply(str.lower): {t_apply * 1000:.1f} мс")
print(f"apply / .str: {t_apply / t_str:.1f}")

obj = col.astype(object)                      # старый формат: объекты Python
t_str_obj = best_of(lambda: obj.str.lower())
t_apply_obj = best_of(lambda: obj.apply(str.lower))
print(f"для dtype=object: .str {t_str_obj * 1000:.1f} мс, "
      f"apply {t_apply_obj * 1000:.1f} мс, отношение {t_apply_obj / t_str_obj:.1f}")
# Разница зависит от того, КАК хранятся строки:
#   хранение pyarrow (pandas 3 при установленном pyarrow): строки лежат одним
#     буфером в формате Arrow, и .str.lower() — один проход на C по всему
#     столбцу. apply же для каждой из 200 000 строк достаёт объект Python,
#     вызывает str.lower и собирает результат обратно — цикл в интерпретаторе.
#     Отсюда разница в десятки раз.
#   хранение python или dtype=object: столбец — это массив объектов Python, и
#     .str.lower() внутри тоже вызывает str.lower на каждом объекте. Отличие от
#     apply — только в накладных расходах, поэтому отношение около 1–1,5.
# Вывод: «.str — это C» верно только для строкового dtype на pyarrow.
#   В pandas 3 он включается сам, если установлен pyarrow (pip install pyarrow).


# ---------------------------------------------------------------------------
# 10. Устно, потом письменно
# ---------------------------------------------------------------------------
# Почему ndarray быстрее списка — две причины:
#   1) Память: числа лежат одним непрерывным блоком одного типа, без
#      указателей и заголовков объектов (задание 1: ~4 раза меньше). Процессор
#      читает их подряд, кэш работает идеально.
#   2) Операция над массивом — один цикл на C над машинными числами. Python не
#      создаёт объект на каждый элемент и не интерпретирует байткод на каждом
#      шаге (задание 1: arr * 2 в десятки раз быстрее comprehension).
# Что означает axis=1 в sum: сумма вдоль оси 1, то есть по столбцам внутри
#   каждой строки. Ось 1 исчезает: (3, 4) → (3,), по одной сумме на строку.
# Срез numpy против среза списка: срез списка копирует элементы, срез массива —
#   представление (окно) в ту же память, и изменение через него меняет
#   оригинал (задание 5, base не None). Так сделано ради производительности:
#   окно описывается началом, шагом и длиной, создаётся за O(1) без копирования,
#   что критично на больших массивах. Копию делают явно через .copy().