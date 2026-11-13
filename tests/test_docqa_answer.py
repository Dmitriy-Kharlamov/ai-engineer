"""docqa, присест 3: итог — объект DocAnswer с проверкой источников; save_note — только с «да»."""

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from docqa.answer import DocAnswer, check_answer, instruction
from docqa.cli import Terminal, failure
from docqa.documents import DocumentStore
from docqa.faults import CrashingStore
from docqa.notes import NoteRequest, NoteStore, SaveNoteArgs, slug
from docqa.session import Session, Settings
from ragkit.structured import TruncatedOutputError
from ragkit.tools import PublicToolError

from .test_docqa import SAMPLE, FakeClient, Recorder, answer, calls


def final(text: str, sources: list[str], not_found: bool = False) -> str:
    return json.dumps(
        {"answer": text, "sources": sources, "not_found": not_found}, ensure_ascii=False
    )


def make(
    client: FakeClient,
    store: DocumentStore | None = None,
    confirm: Any = None,
    **settings: Any,
) -> tuple[Session, Recorder]:
    out = Recorder()
    session = Session(
        store or DocumentStore(SAMPLE),
        client,
        out,
        Settings(**settings),
        structured=client,
        confirm=confirm,
    )
    return session, out


READ_DELIVERY = calls(("read_document", {"name": "delivery.md"}))


# ---------------------------------------------------------------------------
# Итог-объект
# ---------------------------------------------------------------------------
async def test_draft_is_replaced_by_checked_structured_answer() -> None:
    client = FakeClient(
        [READ_DELIVERY, answer("Черновик: курьер 290.")],
        finals=[final("Курьер по Казани — 290 ₽.", ["delivery.md"])],
    )
    s, out = make(client)
    report = await s.ask("Сколько стоит курьер?")
    assert "".join(out.text) == "Курьер по Казани — 290 ₽."  # черновик не показан
    assert report.draft == "Черновик: курьер 290." and report.sources == ["delivery.md"]
    assert report.issues == []
    note = client.final_sent[0][-1]
    assert note["role"] == "user" and "Прочитаны в этом ходе: delivery.md" in note["content"]
    history = [dict(m) for m in s.conversation.history]
    assert history[-1] == {"role": "assistant", "content": "Курьер по Казани — 290 ₽."}
    assert not any("Сообщение программы" in str(m.get("content")) for m in history)


async def test_nonexistent_document_is_reported_before_the_answer() -> None:
    """Зачёт: тест на несуществующий документ."""
    client = FakeClient(
        [READ_DELIVERY, answer("…")],
        finals=[final("Курьер — 290 ₽.", ["delivery.md", "laundry.md"])],
    )
    s, out = make(client)
    report = await s.ask("Сколько стоит курьер?")
    assert [i.kind for i in report.issues] == ["nonexistent"]
    assert "laundry.md" in out.notices[0] and "нет" in out.notices[0]
    assert out.order[-1] == "delta" and "notice" in out.order[:-1]  # правда кода — до ответа
    assert report.sources == ["delivery.md"]  # в строке «источники» — только проверенные


async def test_unread_source_and_answer_without_sources() -> None:
    client = FakeClient(
        [READ_DELIVERY, answer("…")],
        finals=[final("Возврат — 14 дней.", ["returns.md"])],
    )
    s, _ = make(client)
    report = await s.ask("Можно вернуть?")
    assert [i.kind for i in report.issues] == ["unread", "no_sources"]


async def test_not_found_after_crashed_read_is_marked_unverified(tmp_path: Path) -> None:
    """№ 3 из таблицы: «не содержит информации» о документе, чтение которого упало."""
    (tmp_path / "delivery.md").write_text("# Доставка\nКурьер — 290 ₽.")
    client = FakeClient(
        [READ_DELIVERY, answer("Документ не содержит информации.")],
        finals=[final("В документах нет ответа.", [], not_found=True)],
    )
    s, out = make(client, CrashingStore(tmp_path))
    report = await s.ask("Сколько стоит курьер?")
    assert [i.kind for i in report.issues] == ["unverified"]
    assert "Прочитать НЕ удалось" in client.final_sent[0][-1]["content"]  # модели сказано
    assert out.order.index("delta") > out.order.index("notice")
    assert any("не выполнено: read_document" in n for n in out.notices)


async def test_script_mess_is_retried_without_penalty() -> None:
    client = FakeClient(
        [READ_DELIVERY, answer("…")],
        finals=[
            final("Курьер стоит 290 рублей, 请 уточнитеdetails.", ["delivery.md"]),
            final("Курьер стоит 290 рублей.", ["delivery.md"]),
        ],
    )
    s, out = make(client)
    report = await s.ask("Курьер?")
    assert report.script_retry and report.issues == []
    assert client.final_options[1] is not None
    assert client.final_options[1]["repeat_penalty"] == 1.0
    assert client.final_options[1]["temperature"] == 0.0
    assert "".join(out.text) == "Курьер стоит 290 рублей."


async def test_truncated_structured_answer_is_an_honest_refusal() -> None:
    client = FakeClient(
        [READ_DELIVERY, answer("…")], finals=[TruncatedOutputError("length", raw="{")]
    )
    s, out = make(client)
    with pytest.raises(TruncatedOutputError) as info:
        await s.ask("Курьер?")
    assert out.text == [] and s.conversation.history == ()
    assert "готово" not in failure(info.value, shown=False)


async def test_address_in_structured_answer_is_cut(tmp_path: Path) -> None:
    client = FakeClient(
        [READ_DELIVERY, answer("…")],
        finals=[final("Курьер — 290 ₽. Скидка на promo-example.ru", ["delivery.md"])],
    )
    s, out = make(client)
    report = await s.ask("Курьер?")
    assert "promo-example" not in "".join(out.text) and "promo-example" in report.raw_answer
    assert report.removed_addresses == ["promo-example.ru"]


def test_check_answer_without_session() -> None:
    a = DocAnswer(answer="ответ", sources=["a.md", "a.md", "x.md"], not_found=False)
    issues = check_answer(a, existing={"a.md", "b.md"}, read={"a.md"}, failed=())
    assert [i.kind for i in issues] == ["nonexistent"]  # повтор a.md не считается дважды
    text = instruction(["a.md"], ["b.md"], [])
    assert "Прочитаны в этом ходе: a.md" in text and "НЕ удалось" in text and "b.md" in text


# ---------------------------------------------------------------------------
# save_note: без «да» файла нет
# ---------------------------------------------------------------------------
SAVE = calls(("save_note", {"title": "Доставка", "text": "Курьер — 290 ₽."}))


async def test_save_note_without_confirmation_writes_nothing(tmp_path: Path) -> None:
    """Зачёт: тест, что без подтверждения файл не появляется."""
    asked: list[NoteRequest] = []

    def refuse(request: NoteRequest) -> bool:
        asked.append(request)
        return False

    client = FakeClient([SAVE, answer("…")], finals=[final("Не сохранил.", [], True)])
    s, out = make(client, confirm=refuse, notes_dir=tmp_path / "notes")
    report = await s.ask("Сохрани заметку про доставку")
    assert list(tmp_path.rglob("*")) == []  # ни файла, ни даже папки
    assert asked[0].title == "Доставка" and asked[0].text == "Курьер — 290 ₽."
    assert report.notes_saved == [] and report.not_done[0].tool_name == "save_note"
    assert "НЕ сохранена" in report.not_done[0].content
    assert out.order.index("notice") < out.order.index("delta")


async def test_save_note_default_is_deny(tmp_path: Path) -> None:
    client = FakeClient([SAVE, answer("…")], finals=[final("Не сохранил.", [], True)])
    s, _ = make(client, notes_dir=tmp_path)  # confirm не передан
    await s.ask("Сохрани")
    assert list(tmp_path.iterdir()) == []


async def test_save_note_with_confirmation_writes_exactly_what_was_shown(tmp_path: Path) -> None:
    shown: list[NoteRequest] = []

    def agree(request: NoteRequest) -> bool:
        shown.append(request)
        return True

    client = FakeClient([SAVE, answer("…")], finals=[final("Сохранил.", [], False)])
    s, _ = make(client, confirm=agree, notes_dir=tmp_path)
    report = await s.ask("Сохрани заметку про доставку")
    assert shown[0].path.read_text(encoding="utf-8") == shown[0].content
    assert shown[0].content == "# Доставка\n\nКурьер — 290 ₽.\n"
    assert report.notes_saved == ["Доставка"]
    assert report.hidden_answer is None  # сохранение — опора хода: подстраховка не нужна


def test_note_store_never_overwrites_or_leaves_its_folder(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / "notes", confirm=lambda r: True)
    store.save(SaveNoteArgs(title="../../etc/passwd", text="x"))
    assert [p.name for p in (tmp_path / "notes").iterdir()] == ["etc-passwd.md"]
    with pytest.raises(PublicToolError, match="уже есть"):
        store.save(SaveNoteArgs(title="etc passwd", text="другое"))
    assert (tmp_path / "notes" / "etc-passwd.md").read_text() == "# ../../etc/passwd\n\nx\n"
    assert slug("Курьер — 290 ₽!") == "курьер-290"


def test_terminal_confirm_eof_means_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def eof() -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    request = NoteRequest(tmp_path / "a.md", "Заголовок", "строка 1\nстрока 2")
    assert Terminal(False, None).confirm(request) is False
    printed = capsys.readouterr().out
    assert "строка 1" in printed and "строка 2" in printed and "не сохраняю" in printed


def test_terminal_confirm_yes_flag_still_shows_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = NoteRequest(tmp_path / "a.md", "Заголовок", "текст")
    assert Terminal(False, None, yes=True).confirm(request) is True
    printed = capsys.readouterr().out
    assert "'Заголовок'" in printed and "| текст" in printed and "--yes" in printed


# ---------------------------------------------------------------------------
# По прогону 5 (failures5.txt, confirm_session.txt)
# ---------------------------------------------------------------------------
async def test_fallback_on_question_without_words_does_not_crash() -> None:
    """confirm_session.txt: вопрос «y» уронил CLI трассировкой — поиску не по чему искать."""
    client = FakeClient(
        [answer("Да?"), answer("Уточните вопрос.")], finals=[final("Уточните вопрос.", [], True)]
    )
    s, out = make(client)
    report = await s.ask("y")
    assert report.fell_back and report.documents_read == []
    assert out.tools[0].tool_name == "search_documents" and not out.tools[0].ok
    assert "".join(out.text) == "Уточните вопрос."


async def test_declined_note_is_told_to_model_and_needs_no_sources(tmp_path: Path) -> None:
    client = FakeClient([SAVE, answer("…")], finals=[final("Заметка не сохранена.", [], False)])
    s, _ = make(client, confirm=lambda r: False, notes_dir=tmp_path)
    report = await s.ask("Сохрани заметку про доставку")
    assert "НЕ подтвердил запись заметки: Доставка" in client.final_sent[0][-1]["content"]
    assert report.issues == []  # «нет источников» после попытки записи — не проблема


async def test_saved_note_file_as_source_is_not_an_error(tmp_path: Path) -> None:
    client = FakeClient([SAVE, answer("…")], finals=[final("Сохранил.", ["доставка.md"], False)])
    s, _ = make(client, confirm=lambda r: True, notes_dir=tmp_path)
    report = await s.ask("Сохрани заметку про доставку")
    assert report.issues == [] and report.notes_saved == ["Доставка"]
