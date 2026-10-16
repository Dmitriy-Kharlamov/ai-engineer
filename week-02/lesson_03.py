import random
from collections import Counter


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


if __name__ == "__main__":
    check_char_frequency()
    text = "bbaacd"
    mine = top_chars(text, 2)
    theirs = Counter(ch for ch in text.lower() if ch != " ").most_common(2)
    print("top_chars:  ", mine)
    print("most_common:", theirs)
    assert mine == [("a", 2), ("b", 2)]
    assert mine != theirs