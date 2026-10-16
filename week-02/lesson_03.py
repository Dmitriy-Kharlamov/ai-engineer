import random
import string
from collections import Counter

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


# 3



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