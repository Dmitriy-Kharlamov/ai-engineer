"""Ф5.5-01: ручная разметка 60 ответов по той же рубрике, что у судьи.

    python week-05.5/label.py                        # data/answers60.jsonl → data/gold.jsonl

Показывает вопрос, документ и ответ целиком; подсказка — метка регулярки из answers.py.
  Enter — согласиться с подсказкой;  в — верно;  н — неверно;  q — выйти (продолжить потом).
После буквы можно написать заметку: «н выдумал срок продления».
Размеченное дописывается в gold.jsonl сразу; при повторном запуске размеченные пропускаются.
"""

import argparse
import json
import sys
from pathlib import Path

from ragkit.judge import RULES

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docqa" / "sample_docs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path, default=HERE / "data" / "answers60.jsonl")
    parser.add_argument("gold", nargs="?", type=Path, default=HERE / "data" / "gold.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()]
    done: set[str] = set()
    if args.gold.exists():
        done = {json.loads(x)["id"] for x in args.gold.read_text(encoding="utf-8").splitlines()}
    todo = [r for r in records if r["id"] not in done]
    print(f"Рубрика (та же, что у судьи):\n{RULES}\n")
    print(f"размечено {len(done)}, осталось {len(todo)} из {len(records)}")
    shown_doc: str | None = "—"
    for number, record in enumerate(todo, start=len(done) + 1):
        if record["doc"] != shown_doc:
            shown_doc = record["doc"]
            text = (
                (DOCS / shown_doc).read_text(encoding="utf-8")
                if shown_doc
                else (
                    "(документа с ответом нет: верно, если ответ прямо говорит, что такой услуги "
                    "или информации нет)"
                )
            )
            print(f"\n{'=' * 78}\nДокумент {shown_doc or '—'}:\n{text.strip()}\n{'=' * 78}")
        hint = "в" if record["regex_fact"] else "н"
        print(f"\n[{number}/{len(records)}] {record['id']}  вопрос: {record['question']}")
        print(f"ответ:\n  {record['answer'].strip().replace(chr(10), chr(10) + '  ')}")
        while True:
            raw = input(f"Enter = {hint} (регулярка) | в | н | q > ").strip()
            if raw.lower() == "q":
                print(f"сохранено в {args.gold}")
                return 0
            label = (raw[:1].lower() or hint) if raw[:1].lower() in ("", "в", "н") else ""
            if label:
                break
            print("  ответь в, н, Enter или q")
        note = raw[1:].strip()
        with args.gold.open("a", encoding="utf-8") as f:
            row = {**record, "human": label == "в", "note": note}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nготово: {args.gold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
