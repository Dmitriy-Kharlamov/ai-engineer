import re
import time
import unicodedata


def best_of(func, *args, runs=7):
    """Минимальное время из runs прогонов."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return min(times)


# 1
print("1. Юникод")
nfc = unicodedata.normalize("NFC", "йод")  # «й» одним символом U+0439
nfd = "йод"  # «и» + комбинирующая краткая U+0306

print(f"{nfc=}  len={len(nfc)}  bytes={nfc.encode('utf-8')}")
print(f"{nfd=}  len={len(nfd)}  bytes={nfd.encode('utf-8')}")
print("nfc == nfd:", nfc == nfd)
print("hash равны:", hash(nfc) == hash(nfd))
assert len(nfc) == 3 and len(nfd) == 4
assert nfc != nfd and hash(nfc) != hash(nfd)

fixed = unicodedata.normalize("NFC", nfd)
print("после NFC:", fixed == nfc, len(fixed), fixed.encode("utf-8"))
assert fixed == nfc and hash(fixed) == hash(nfc)
# На экране одинаково, но для Python это разные строки: в nfd четыре символа,
# краткая — отдельный код U+0306 (байты cc 86). == и hash сравнивают коды,
# поэтому False и разные хеши → в словаре это два разных ключа.
# NFC «составляет» и + краткая в один символ й → строки совпадают.

lig = "ﬁ"  # лигатура «ﬁ»
print(f"\nлигатура: {lig!r} len={len(lig)}")
print("NFC: ", repr(unicodedata.normalize("NFC", lig)))
print("NFKC:", repr(unicodedata.normalize("NFKC", lig)))
assert unicodedata.normalize("NFC", lig) == lig  # NFC не трогает
assert unicodedata.normalize("NFKC", lig) == "fi"  # NFKC разворачивает в f + i
# NFC объединяет только канонически равные записи (одна буква — два способа
# записи). ﬁ и fi — не одно и то же написание, а «совместимые» варианты:
# это решает только NFKC. Для поискового индекса — NFKC, для показа — NFC.


# 2
print("\n2. casefold против lower")
pairs = [
    ("Straße", "STRASSE"),  # немецкая ß → ss
    ("λογος", "λογοσ"),  # греческая конечная ς → σ
    ("ﬁle", "FILE"),  # лигатура ﬁ → fi
]
print(f"{'a':<10}{'b':<10}{'lower ==':<10}{'casefold =='}")
for a, b in pairs:
    eq_lower = a.lower() == b.lower()
    eq_fold = a.casefold() == b.casefold()
    print(f"{a:<10}{b:<10}{str(eq_lower):<10}{eq_fold}")
    assert not eq_lower and eq_fold
# lower только меняет регистр буквы на строчную: ß, ς и ﬁ уже строчные,
# их lower оставляет как есть. casefold приводит к форме для СРАВНЕНИЯ:
# ß → ss, ς → σ (конечная и обычная сигма — одна буква), ﬁ → fi.


# 3
print("\n3. translate против replace")
REMOVE = "«»“”„‘’—–…"
TABLE = str.maketrans("", "", REMOVE)


def clean_translate(text):
    return text.translate(TABLE)


def clean_replace(text):
    for ch in REMOVE:  # по проходу на каждый символ
        text = text.replace(ch, "")
    return text


sample = "«Цитата» — “пример” с „кавычками“ и многоточием… ‘ещё’ – тире. "
assert clean_translate(sample) == clean_replace(sample)
print(repr(clean_translate(sample)))

big = (sample * (200_000 // len(sample) + 1))[:200_000]
assert len(big) == 200_000
assert clean_translate(big) == clean_replace(big)
t_tr = best_of(clean_translate, big)
t_rep = best_of(clean_replace, big)
print(f"translate: {t_tr:.5f} с")
print(f"replace:   {t_rep:.5f} с")
print(f"replace / translate: {t_rep / t_tr:.1f}")
# Результаты совпали (assert), а замер вышел НЕ таким, как обещает урок:
#   цепочка replace оказалась быстрее translate (см. отношение в выводе).
# Почему: replace ищет один конкретный символ быстрым поиском на C, почти
#   как memchr, и 10 таких проходов по 200 000 символов — это очень дёшево.
#   translate делает один проход, но для КАЖДОГО из 200 000 символов
#   ищет его в таблице-словаре. Для не-ASCII текста (кириллица) в CPython это
#   медленный путь, без оптимизаций, которые есть для ASCII.
# translate выигрывает, когда символов для замены много (десятки и сотни —
#   тогда replace понадобилось бы столько же проходов) или когда нужна замена
#   символ-в-символ по большой таблице. Для десятка удаляемых символов
#   цепочка replace честно быстрее. Вывод: верить замеру, а не правилу.


# 4
print("\n4. parse_config_line")


def parse_config_line(line):
    """'ключ = значение' → (ключ, значение); без '=' или без ключа → None."""
    line = line.strip().removeprefix("export ")  # 'export KEY=1' как в .env
    key, sep, value = line.partition("=")  # режет по ПЕРВОМУ '='
    if not sep:  # '=' не нашлось
        return None
    key = key.strip()
    if not key:  # '= значение' — ключа нет
        return None
    return key, value.strip()


cases = [
    ("host = localhost", ("host", "localhost")),  # обычный
    ("url=postgres://u:p@h/db?a=1", ("url", "postgres://u:p@h/db?a=1")),  # '=' в значении
    ("   name   =   ragkit   ", ("name", "ragkit")),  # пробелы вокруг
    ("empty =", ("empty", "")),  # пустое значение
    ("просто строка без знака", None),  # нет '='
    ("export TOKEN = abc", ("TOKEN", "abc")),  # removeprefix
    ("= orphan", None),  # нет ключа
]
for line, expected in cases:
    result = parse_config_line(line)
    print(f"{line!r:34} → {result!r}")
    assert result == expected, (line, result)

# Почему removeprefix, а не lstrip: lstrip("export ") удаляет любые символы
# из набора {e, x, p, o, r, t, ' '} с края, а не подстроку:
print("lstrip:      ", repr("export port=1".lstrip("export ")))
print("removeprefix:", repr("export port=1".removeprefix("export ")))
assert "export port=1".lstrip("export ") == "=1"  # съел и «port»!
assert "export port=1".removeprefix("export ") == "port=1"


# 5
print("\n5. Регулярные выражения")

# 5.1 все числа, включая дробные и отрицательные
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
text = "Модель на 7 млрд параметров, loss 0.37, температура -2.5 градуса."
nums = NUMBER.findall(text)
print("числа:", nums)
assert NUMBER.findall("было 3.14, стало 10 и -2.5") == ["3.14", "10", "-2.5"]
# Группа (?:\.\d+) незахватывающая — иначе findall вернул бы только дробные
# части ('.14', '', '.5') вместо чисел целиком.

# 5.2 email через fullmatch
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
for addr, ok in [
    ("dmitriy@example.com", True),
    ("a.b+tag@mail.co.uk", True),
    ("без собаки.com", False),
    ("x@localhost", False),
    ("a@b.com лишнее", False),
]:
    got = EMAIL.fullmatch(addr) is not None
    print(f"email {addr!r:24} → {got}")
    assert got == ok
# fullmatch требует, чтобы шаблону соответствовала ВСЯ строка: search нашёл бы
# 'a@b.com' внутри 'a@b.com лишнее' и сказал бы «да».

# 5.3 разбор лога именованными группами
LOG = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2}) "
    r"(?P<level>[A-Z]+) "
    r"(?P<message>.+)"
)
m = LOG.fullmatch("2026-10-19 18:31:07 ERROR ragkit: чанк слишком большой")
print("лог:", m.groupdict())
assert m.groupdict() == {
    "date": "2026-10-19",
    "time": "18:31:07",
    "level": "ERROR",
    "message": "ragkit: чанк слишком большой",
}

# 5.4 любые пробельные последовательности → один пробел
messy = "много   пробелов\tи\n\nпереносов  "
squashed = re.sub(r"\s+", " ", messy).strip()
print("пробелы:", repr(squashed))
assert squashed == "много пробелов и переносов"

# Жадный против ленивого
html = "<a><b>"
greedy = re.findall(r"<.+>", html)
lazy = re.findall(r"<.+?>", html)
print("\nжадный  <.+> :", greedy)
print("ленивый <.+?>:", lazy)
assert greedy == ["<a><b>"] and lazy == ["<a>", "<b>"]
# .+ сначала захватывает всё до конца строки и отступает назад только до
# последнего '>', поэтому совпадение одно и широкое. .+? берёт минимум и
# расширяется, пока не встретит первый '>', поэтому совпадений два.

# findall с захватывающими и незахватывающими группами
ranges = "страницы 10-20 и 30-40"
capturing = re.findall(r"(\d+)-(\d+)", ranges)
noncapturing = re.findall(r"(?:\d+)-(?:\d+)", ranges)
print("с группами:   ", capturing)
print("без захвата:  ", noncapturing)
assert capturing == [("10", "20"), ("30", "40")]
assert noncapturing == ["10-20", "30-40"]
# Если в шаблоне есть захватывающие группы, findall возвращает ГРУППЫ
# (кортежи), а не совпадение целиком. (?:...) группирует, но не захватывает.


# 6
print("\n6. Чистка текста из PDF")
RAW = "Трансфор-\nмеры  используют   self-attention.¹ Это\nпозволяет ﬁксировать зависимости."
EXPECTED = "Трансформеры используют self-attention. Это позволяет фиксировать зависимости."

FOOTNOTE = re.compile(r"[¹²³⁰-⁹]+")  # ¹ ² ³ ⁰ ⁴…⁹
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
LATIN_LIG_IN_CYRILLIC = re.compile(r"ﬁ(?=[а-яё])", re.IGNORECASE)


def clean_pdf_text(raw):
    # 1. Сноски — первым делом, пока они ещё надстрочные ¹
    text = FOOTNOTE.sub("", raw)
    # 2. Лигатура перед русскими буквами — это русское «фи», а не латинское fi
    text = LATIN_LIG_IN_CYRILLIC.sub("фи", text)
    # 3. NFKC: остальные лигатуры, неразрывные пробелы и прочие варианты
    text = unicodedata.normalize("NFKC", text)
    # 4. Перенос по дефису: «Трансфор-\nмеры» → «Трансформеры»
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    # 5. Оставшиеся переносы строк и лишние пробелы → один пробел
    text = re.sub(r"\s+", " ", text).strip()
    return text


cleaned = clean_pdf_text(RAW)
print(repr(cleaned))
assert cleaned == EXPECTED
# Почему именно такой порядок:
#   Сноска (1) раньше NFKC (3): NFKC превращает надстрочное ¹ в обычную 1, и
#     после него «attention.¹» станет «attention.1» — сноску уже не отличить от
#     настоящей цифры в тексте, регулярка по ¹ ничего не найдёт.
#   Лигатура в русском слове (2) раньше NFKC (3): NFKC разворачивает ﬁ только в
#     ЛАТИНСКИЕ f + i. Получится «fiксировать» — выглядит правильно, но это
#     смесь алфавитов, и поиск по «фиксировать» её не найдёт. Ровно та ловушка
#     из пункта про юникод: одинаково на экране, разное в байтах.
#   Перенос по дефису (4) раньше склейки строк (5): шаблон ищет «-\n». Если
#     поменять 4 и 5 местами, \n сначала станет пробелом, получится
#     «Трансфор- меры», и шаблон «-\n» уже не сработает — слово останется
#     разорванным навсегда.
#   Пробелы (5) последними: каждый предыдущий шаг может оставить двойные
#     пробелы (например, на месте удалённой сноски), и финальная склейка
#     убирает их все разом.
# Почему (\w)-\n(\w), а не просто «-\n»: дефис между буквами на стыке строк —
#   перенос. «self-attention» внутри строки шаблон не трогает: там нет \n.


# 7
print("\n7. truncate")
ELLIPSIS = "…"


def truncate(text, limit):
    """Обрезать до limit символов (… входит в лимит), не разрывая слова."""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    room = limit - 1  # одно место отдаём под …
    cut = text[:room]
    mid_word = not text[room].isspace()  # следующий символ — продолжение слова?
    if mid_word and " " in cut:
        cut = cut[: cut.rindex(" ")]  # откатываемся к концу целого слова
    # если пробела нет — первое слово длиннее лимита, режем его как есть
    return cut.rstrip() + ELLIPSIS


tests = [
    ("", 10, ""),  # пустая строка
    ("короткий текст", 50, "короткий текст"),  # короче лимита
    ("ровно", 5, "ровно"),  # ровно в лимит
    ("что угодно", 0, ""),  # лимит 0
    ("Трансформеры", 5, "Тран…"),  # одно длинное слово
    ("Трансформеры используют внимание", 20, "Трансформеры…"),  # не рвём слово
    ("один два три", 9, "один два…"),  # граница слова
    ("один два три", 1, "…"),  # место только под …
]
for text, limit, expected in tests:
    result = truncate(text, limit)
    print(f"truncate({text!r}, {limit}) → {result!r}")
    assert result == expected, (text, limit, result)
    assert len(result) <= max(limit, 0)  # никогда не длиннее лимита


# ---------------------------------------------------------------------------
# 8. Устно, потом письменно
# ---------------------------------------------------------------------------
# Зачем нормализовать юникод на входе: одна и та же буква может прийти в
#   разных записях (й одним символом или и + краткая). Python сравнивает коды,
#   а не картинку: == даёт False, хеши разные, и в словаре или индексе одно
#   слово ляжет двумя ключами — поиск «не находит слово, которое точно есть».
#   Нормализация в NFC (или NFKC для индекса) на границе системы приводит всё
#   к одной форме один раз, до любой обработки, и дальше этот класс ошибок
#   невозможен (задание 1).
# casefold против lower: lower только переводит буквы в строчные, а ß, ς и ﬁ
#   уже строчные — lower их не трогает. casefold приводит к форме для
#   сравнения: ß → ss, ς → σ, ﬁ → fi (задание 2). Для регистронезависимого
#   сравнения и поиска — casefold; для показа пользователю — lower, потому что
#   casefold меняет написание («Straße» стала бы «strasse»).
# Почему strip("abc") не отрезает подстроку "abc": аргумент strip — это НАБОР
#   символов, а не строка. strip удаляет с краёв любые символы из набора
#   {a, b, c} в любом порядке и количестве, пока не встретит другой символ:
#   "cabbage".strip("abc") → "ge". Подстроку отрезают removeprefix/removesuffix
#   (задание 4: lstrip("export ") съел и «port»).
