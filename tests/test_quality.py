from ragkit.quality import script_mess


def test_clean_russian_with_latin_names_is_not_mess() -> None:
    assert not script_mess("Курьер по Казани — 290 ₽ (delivery.md, СДЭК, Wi-Fi-роутер, qwen2.5).")


def test_mixed_words_and_cjk_are_counted() -> None:
    mess = script_mess("Уточнитеdetails, пожалуйста 请问")
    assert mess.mixed == ("Уточнитеdetails",) and mess.cjk == 2 and bool(mess)
    assert "иероглифов: 2" in mess.describe()
