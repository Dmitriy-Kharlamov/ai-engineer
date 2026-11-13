"""Признаки испорченного текста ответа — проверки кодом, без модели.

script_mess — из Ф4-01 (задание 8) и Ф4-02: когда модель уходит от уже использованных
токенов (repeat_penalty), она перескакивает на токены другой письменности. В Ф4-02 при
repeat_penalty=1.8 список городов дал 6 слов со смесью кириллицы и латиницы и 5 иероглифов,
при 1.0 — 1 и 0 (lesson_02_part2.txt). В Ф4-08 ответ по-китайски пришёл при temperature 0.7
(injection.txt, A, seed 1).

Отличие от Ф4-01: слова делятся и по дефису. «Wi-Fi-роутер» — законная смесь, «уточнитеdetails»
— нет; без дефиса первое не считается.
"""

import re
from dataclasses import dataclass

_WORD = re.compile(r"\w+")
_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
_LAT = re.compile(r"[a-z]", re.IGNORECASE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


@dataclass(frozen=True)
class ScriptMess:
    mixed: tuple[str, ...]  # слова, где кириллица и латиница вперемешку
    cjk: int  # иероглифов, каны и хангыля

    def __bool__(self) -> bool:
        return bool(self.mixed) or self.cjk > 0

    def describe(self) -> str:
        parts = []
        if self.mixed:
            parts.append(
                f"слов вперемешку кириллицей и латиницей: {len(self.mixed)} "
                f"({', '.join(self.mixed[:3])})"
            )
        if self.cjk:
            parts.append(f"иероглифов: {self.cjk}")
        return "; ".join(parts) or "чисто"


def script_mess(text: str) -> ScriptMess:
    mixed = tuple(w for w in _WORD.findall(text) if _CYR.search(w) and _LAT.search(w))
    return ScriptMess(mixed, len(_CJK.findall(text)))
