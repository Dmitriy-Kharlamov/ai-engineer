import random
import string
from collections import Counter
from ragkit import Chunk

PUNCT = string.punctuation + "«»—…"     # string.punctuation знает только английские знаки


def char_frequency(text):
    freq = {}
    for ch in text.lower():
        if ch == " ":
            continue
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def char_frequency_reference(text):
    return Counter(ch for ch in text.lower() if ch != " ")


def check_char_frequency(trials=500):
    alphabet = "aAbBcC  "
    for _ in range(trials):
        text = "".join(random.choice(alphabet) for _ in range(random.randint(0, 20)))
        assert char_frequency(text) == char_frequency_reference(text), f"вход {text!r}"
    print(f"char_frequency: {trials} проверок пройдено")


# 1. most_common сортирует только по частоте, а при равной частоте оставляет
#    символы в порядке их первого появления в тексте (в "bbaacd" первой встречается b).
#    top_chars сортирует по ключу (-частота, символ): при равной частоте
#    второй элемент кортежа упорядочивает символы по алфавиту.
def top_chars(text, n):
    freq = char_frequency(text)
    ordered = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:n]


# 2. Без нормализации 58 ключей, с нормализацией 51. Слились: python, код, словари, ошибки, ключи
#    (регистр и знаки по краям: «Код», «код,» → «код»). Тире «—» без нормализации
#    считалось словом, с нормализацией стало пустой строкой и отброшено.
#    Ограничение: разные формы слова не сливаются («ключи» и «ключей», «словари»
#    и «словарей» остались разными), а «O(1).» превратилось в «o(1».
def word_frequency(text, normalize=True):
    freq = {}
    for word in text.split():
        if normalize:
            word = word.lower().strip(PUNCT)   # регистр + отрезать знаки по краям
            if not word:
                continue                     # от "—" или "..." осталась пустая строка
        freq[word] = freq.get(word, 0) + 1
    return freq


# 3. Ключ — tuple(sorted(word)).
#    Почему одинаковый у анаграмм: анаграммы состоят из одинаковых букв в любом порядке,
#    а сортировка убирает различие в порядке — «кот» и «ток» оба дают ('к', 'о', 'т').
#    Почему хешируемый: sorted возвращает список, а он изменяемый и не может быть
#    ключом словаря; tuple — неизменяемый, поэтому у него есть хеш.
#    Детерминированность: порядок групп в словаре зависит от порядка слов на входе,
#    поэтому сортируем слова внутри групп и сами группы — тогда результат можно
#    сравнить с эталоном через ==.
def group_anagrams(words):
    groups = {}
    for word in words:
        key = tuple(sorted(word))
        groups.setdefault(key, []).append(word)
    # детерминированный порядок: слова внутри группы и сами группы — отсортированы
    return sorted(sorted(group) for group in groups.values())


def group_anagrams_reference(words):
    groups = []
    for word in words:
        for group in groups:
            if Counter(group[0]) == Counter(word):
                group.append(word)
                break
        else:
            groups.append([word])
    return sorted(sorted(group) for group in groups)


def check_group_anagrams(trials=500):
    for _ in range(trials):
        words = ["".join(random.choice("абв") for _ in range(random.randint(1, 3)))
                 for _ in range(random.randint(0, 10))]
        assert group_anagrams(words) == group_anagrams_reference(words), f"вход {words}"
    print(f"group_anagrams: {trials} проверок пройдено")


# 4. Одним проходом нельзя: в момент, когда мы видим символ, неизвестно,
#    встретится ли он еще раз (в "абракадабра" «б» на позиции 1 кажется уникальной,
#    но повторяется на позиции 8). Чтобы это знать, нужно сначала увидеть количество повторений каждого символа —
#    это первый проход (подсчёт). Второй проход идёт заново, чтобы найти
#    именно первый символ со счётом 1.
def first_unique(text):
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in text:
        if counts[ch] == 1:
            return ch
    return None


def first_unique_reference(text):
    for ch in text:
        if text.count(ch) == 1:
            return ch
    return None


def check_first_unique(trials=500):
    for _ in range(trials):
        text = "".join(random.choice("абвг") for _ in range(random.randint(0, 10)))
        assert first_unique(text) == first_unique_reference(text), f"вход {text!r}"
    print(f"first_unique: {trials} проверок пройдено")


# 5
DOCS = {
    1: "Python — язык для анализа данных",
    2: "Словари в Python работают быстро",
    3: "RAG ищет документы по запросу",
    4: "Обратный индекс ускоряет поиск документов",
    5: "Быстрый поиск — основа RAG",
    6: "Данные хранятся в словарях",
    7: "Поиск, поиск и ещё раз поиск: RAG без поиска не работает",
}


def tokenize(text):
    words = [w.lower().strip(PUNCT) for w in text.split()]
    return [w for w in words if w]           # выкидываем пустые (от «—»)


def build_index(docs):
    index = {}
    for doc_id, text in docs.items():
        for term in tokenize(text):
            index.setdefault(term, set()).add(doc_id)
    return index


def search(query, index):
    terms = tokenize(query)
    if not terms:
        return set()
    result = index.get(terms[0], set()).copy()
    for term in terms[1:]:
        result &= index.get(term, set())
    return result


def naive_search(query, docs):
    terms = set(tokenize(query))
    if not terms:
        return set()
    result = set()
    for doc_id, text in docs.items():
        doc_terms = set(tokenize(text))
        if terms <= doc_terms:
            result.add(doc_id)
    return result


def check_search(index, docs, trials=10):
    vocabulary = list(index) + ["java", "космос"]
    for _ in range(trials):
        query = " ".join(random.sample(vocabulary, random.randint(0, 3)))
        assert search(query, index) == naive_search(query, docs), f"запрос {query!r}"
    print(f"search vs naive_search: {trials} случайных запросов совпали")


# 6. TF = число вхождений терма / общее число термов документа.
#    Абсолютное число зависит от размера документа: в длинном тексте любое слово
#    встречается чаще просто потому, что слов больше. Например, «python» 2 раза
#    в документе из 10 слов (TF = 0.2) значит больше, чем 5 раз в документе
#    из 1000 слов (TF = 0.05), хотя абсолютное число у второго больше.
#    Доля убирает влияние длины и делает документы сравнимыми.
def term_frequency(text):
    terms = tokenize(text)
    if not terms:
        return {}
    counts = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    total = len(terms)
    return {term: count / total for term, count in counts.items()}


# 7. Хешируемость.
#    список — TypeError: список изменяемый, его содержимое можно поменять после того, как
#      он стал ключом; хеш бы изменился, и словарь не нашёл бы ключ. Поэтому
#      у list нет __hash__.
#    кортеж (1, 2) — OK: кортеж неизменяемый, и все его элементы хешируемые, поэтому
#      хеш считается из хешей элементов.
#    кортеж со списком — TypeError: хеш кортежа считается из хешей его элементов,
#      а у списка внутри хеша нет. Неизменяемости самого контейнера мало —
#      хешируемым должно быть и содержимое.
#    Chunk — TypeError, Chunk.__hash__ = None: в классе определён свой __eq__
#      (сравнивает по числу токенов), а __hash__ нет. В таком случае Python сам
#      ставит __hash__ = None.
#      Иначе остался бы хеш по умолчанию (по id объекта): два равных чанка
#      имели бы разные хеши, и словарь считал бы их разными ключами.
#    Исправление: добавил __hash__ = hash(self.tokens), согласованный с __eq__
#      (тот сравнивает по числу токенов). Теперь чанки работают как ключи.
#      Следствие: чанки с разным текстом, но одинаковым числом слов — один ключ
#      ({a, b, c} даёт 1 элемент). Если нужна дедупликация по содержимому,
#      __eq__ и __hash__ надо перевести на (text, source).
def try_as_key(label, make_key):
    try:
        d = {make_key(): "значение"}
        print(f"{label}: OK, ключ принят → {d}")
    except TypeError as e:
        print(f"{label}: TypeError — {e}")


def show_hashability():
    try_as_key("список", lambda: [1, 2])
    try_as_key("кортеж", lambda: (1, 2))
    try_as_key("кортеж со списком", lambda: (1, [2]))
    try_as_key("Chunk", lambda: Chunk("текст", "a.pdf"))
    print("Chunk.__hash__ =", Chunk.__hash__)


# 8. Ловушки словарей
def show_dict_traps():
    from collections import defaultdict

    # а) удаление ключей во время цикла по словарю
    d = {"a": 1, "b": 2, "c": 3}
    try:
        for key in d:
            if d[key] % 2 == 1:
                del d[key]
    except RuntimeError as e:
        print("а) ошибка:", e)
    d = {"a": 1, "b": 2, "c": 3}
    for key in list(d):
        if d[key] % 2 == 1:
            del d[key]
    print("а) через list(d):", d)
    #   Механизм: словарь запрещает менять размер во время цикла по нему — итератор
    #   теряет позицию, поэтому RuntimeError. list(d) заранее делает список ключей,
    #   цикл идёт по нему, а удаляем из самого словаря.

    # б) d.keys() — живое представление
    d = {"a": 1}
    keys = d.keys()
    print("б) до:", keys)
    d["b"] = 2
    print("б) после добавления:", keys)
    # Механизм: d.keys() — не копия, а ссылка на сам словарь: изменения в d
    #   сразу видны в keys. Нужна копия на момент — list(d.keys()).

    # в) defaultdict создаёт ключ при простом обращении
    dd = defaultdict(int)
    print("в) 'x' в dd до:", "x" in dd)
    if dd["x"] == 0:
        pass
    print("в) 'x' в dd после dd['x']:", "x" in dd, "→", dict(dd))
    # Механизм: defaultdict при обращении к отсутствующему ключу вызывает int(),
    #   получает 0 и записывает его в словарь. Даже простое чтение dd["x"] создаёт ключ.
    #   Проверять наличие надо через "x" in dd, а не через dd["x"].

    # г) Counter не создаёт ключ
    cnt = Counter()
    print("г) cnt['x'] =", cnt["x"])
    print("г) 'x' в cnt после:", "x" in cnt, "→", dict(cnt))
    # Механизм: Counter для отсутствующего ключа равен 0, но не кладет ключ в словарь,
    #   поэтому чтение не меняет счётчик.


# 9. Без кода.
#    1) Словарь — таблица ячеек. Хеш ключа сразу даёт номер ячейки, перебирать
#       ничего не нужно, поэтому в среднем O(1). Если у многих ключей хеши
#       попадают в одну ячейку (коллизии), Python проверяет их по очереди через ==;
#       когда в одну ячейку попали все n ключей — O(n).
#    2) Словарь сначала ищет по хешу, потом сравнивает через ==, поэтому должно
#       выполняться правило «a == b → hash(a) == hash(b)». Хеш по умолчанию
#       считается по id и разный у любых двух объектов, а свой __eq__ может
#       считать их равными — правило нарушится, и словарь молча не найдёт
#       равный ключ, а во множестве появятся дубли. TypeError лучше: ошибка
#       видна сразу, а не превращается в тихий баг.
#    3) Если бы хеш строк был постоянным, злоумышленник мог бы заранее подобрать
#       тысячи строк с одинаковым хешем и отправить их серверу — все попали бы
#       в одну ячейку, и каждая операция со словарём стала бы O(n) (атака hash
#       flooding, DoS). Случайная соль в каждом процессе не даёт подобрать
#       коллизии заранее. Побочный эффект: hash('python') отличается между
#       запусками, и порядок строк в set может меняться.


if __name__ == "__main__":
    check_char_frequency()
    text = "bbaacd"
    mine = top_chars(text, 2)
    theirs = Counter(ch for ch in text.lower() if ch != " ").most_common(2)
    print("top_chars:  ", mine)
    print("most_common:", theirs)
    assert mine == [("a", 2), ("b", 2)]
    assert mine != theirs
    paragraph = """Python — мой первый язык. На курсе мы пишем код каждый день, и код
    становится лучше. «Словари» в Python работают быстро: словари ищут ключ за O(1).
    Код без проверок — это не код, а надежда. Проверки, assert и тесты спасают от ошибок.
    Ошибки бывают разные; ошибки в индексах — самые частые. Python прощает многое,
    но python не прощает путаницу ключей. Ключи должны быть хешируемыми, ключи — основа словарей."""
    raw = word_frequency(paragraph, normalize=False)
    norm = word_frequency(paragraph)
    print("Без нормализации, ключей:", len(raw))
    print(sorted(raw.items(), key=lambda item: (-item[1], item[0]))[:10])
    print("С нормализацией, ключей:", len(norm))
    print(sorted(norm.items(), key=lambda item: (-item[1], item[0]))[:10])
    groups = {}
    for word in raw:
        key = word.lower().strip(PUNCT)
        groups.setdefault(key, []).append(word)
    for key, variants in groups.items():
        if len(variants) > 1:
            print(key, "←", variants)
    check_group_anagrams()
    print(group_anagrams(["кот", "ток", "окт", "дом", "мод", "сон"]))
    check_first_unique()
    print(first_unique("абракадабра"))
    print(first_unique("aabb"))
    index = build_index(DOCS)
    assert search("python", index) == {1, 2}
    assert search("быстро python", index) == {2}
    assert search("java", index) == set()
    assert search("", index) == set()
    print("search: проверки на 4 запросах пройдены")
    check_search(index, DOCS)
    for doc_id, text in DOCS.items():
        tf = term_frequency(text)
        assert abs(sum(tf.values()) - 1) < 1e-9, f"документ {doc_id}"
        top3 = sorted(tf.items(), key=lambda item: (-item[1], item[0]))[:3]
        print(doc_id, top3)
    assert term_frequency("") == {}
    print("term_frequency: проверки пройдены")
    show_hashability()
    print("разные тексты, равны?", Chunk("привет мир", "a.pdf") == Chunk("пока всем", "b.pdf"))
    a = Chunk("привет мир", "a.pdf")
    b = Chunk("привет мир", "a.pdf")            # другой объект, тот же текст
    c = Chunk("пока всем", "b.pdf")             # другой текст, но тоже 2 слова
    d = {a: "первый"}
    print("d[b] =", d[b])                        # нашли по равному объекту
    print("d[c] =", d[c])                        # тоже нашли: для словаря c == a
    print("уникальных в {a, b, c}:", len({a, b, c}))
    assert hash(a) == hash(b) == hash(c)
    assert len({a, b, c}) == 1
    show_dict_traps()