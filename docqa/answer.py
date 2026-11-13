"""Итоговый ответ — объект, а не текст: answer, sources, not_found (ragkit.structured).

Зачем: в свободном тексте источники не проверить. В прогонах присеста 2 модель ссылалась
на «…undry.md», которого нет (4 из 40, injection_natural.txt), на pickup.md, который не
читала (injection.txt, сид 1), и говорила «документ не содержит информации» о документе,
чтение которого упало (failures*.txt, № 3). Со списком источников и флагом это проверяет
код, до того как пользователь увидит ответ.

Как получен: когда цикл инструментов кончился ответом, этот черновик не показывается, а
модели уходит ещё одно обращение — с format=<схема DocAnswer> и сообщением кода, где схема
расписана словами (в Ф4-05 описания полей без схемы в промпте модель не видела: 0 из 20, со
схемой — 18–20 из 20) и названы документы, прочитанные и не прочитанные в этом ходе.

Проверки кодом (check_answer):
* nonexistent — источника нет в папке;
* unread — источник есть, но в этом ходе не прочитан: опереться на него ответ не мог;
* no_sources — ответ дан (not_found=false), а источников нет и в ходе не было действия;
* unverified — «не нашёл», но чтение какого-то документа упало: искать было негде;
* script — в ответе смесь письменностей (ragkit.quality.script_mess).
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ragkit.quality import script_mess

MAX_ANSWER = 2000
MAX_SOURCES = 6


class DocAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(
        min_length=1,
        max_length=MAX_ANSWER,
        description="ответ пользователю по-русски, только по прочитанным документам",
    )
    sources: list[str] = Field(
        max_length=MAX_SOURCES,
        description="имена файлов прочитанных документов, на которых основан ответ",
    )
    not_found: bool = Field(
        description="true, если в прочитанных документах ответа на вопрос нет",
    )


def instruction(
    read: Sequence[str],
    failed: Sequence[str],
    saved: Sequence[str],
    *,
    declined: Sequence[str] = (),
) -> str:
    """Сообщение кода для последнего обращения: схема словами и факты из журнала хода."""
    lines = [
        "[Сообщение программы, не пользователя.] Дай итоговый ответ объектом JSON с полями:",
        "- answer — ответ пользователю по-русски, коротко, только по тому, что написано в "
        "прочитанных документах;",
        "- sources — имена файлов документов из папки, на которых основан ответ, ровно как в "
        "read_document (например delivery.md); заметки — не источники; пустой список, если "
        "опереться не на что;",
        "- not_found — true, если в прочитанных документах ответа нет; тогда так и скажи в answer.",
        "Прочитаны в этом ходе: " + (", ".join(read) if read else "ни одного документа") + ".",
    ]
    if failed:
        lines.append(
            "Прочитать НЕ удалось (ошибка инструмента): "
            + ", ".join(failed)
            + ". Что в них написано, неизвестно — не утверждай, что в них нет ответа."
        )
    if saved:
        lines.append("Сохранены заметки: " + ", ".join(saved) + ".")
    if declined:
        lines.append(
            "Пользователь НЕ подтвердил запись заметки: "
            + ", ".join(declined)
            + ". Она не сохранена — так и скажи в answer."
        )
    return "\n".join(lines)


IssueKind = Literal["nonexistent", "unread", "no_sources", "unverified", "script"]


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    text: str  # для пользователя: строка кода перед ответом


def check_answer(
    answer: DocAnswer,
    *,
    existing: Collection[str],
    read: Collection[str],
    failed: Collection[str],
    acted: bool = False,
    notes: Collection[str] = (),
) -> list[Issue]:
    """acted — в ходе была попытка действия (save_note): ответ о нём источников не требует.
    notes — файлы заметок, сохранённых в ходе: модель называет их источником (failures5.txt,
    15b/15d/15e), это не ошибка и не документ — такие имена пропускаются."""
    issues: list[Issue] = []
    for name in dict.fromkeys(answer.sources):  # порядок, без повторов
        if name in notes:
            continue
        if name not in existing:
            issues.append(
                Issue("nonexistent", f"ответ ссылается на {name!r} — такого документа в папке нет")
            )
        elif name not in read:
            issues.append(
                Issue("unread", f"ответ ссылается на {name}, но в этом ходе он не прочитан")
            )
    valid = [n for n in answer.sources if n in read]
    if not answer.not_found and not valid and not acted:
        issues.append(
            Issue("no_sources", "у ответа нет ни одного прочитанного источника — он не проверен")
        )
    if answer.not_found and failed:
        issues.append(
            Issue(
                "unverified",
                f"прочитать {', '.join(failed)} не удалось — «ответа нет» не проверено: "
                "ответ мог быть в непрочитанном",
            )
        )
    mess = script_mess(answer.answer)
    if mess:
        issues.append(Issue("script", f"в ответе смесь письменностей — {mess.describe()}"))
    return issues


def valid_sources(answer: DocAnswer, read: Collection[str]) -> list[str]:
    return [n for n in dict.fromkeys(answer.sources) if n in read]
