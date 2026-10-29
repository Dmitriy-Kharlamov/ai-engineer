# ai-engineer

Учебный репозиторий курса AI-инженера: задания по неделям и пакет `ragkit`, который растёт вместе с курсом.

## Структура

| Путь | Что там |
|---|---|
| `ragkit/` | пакет: `Chunk`, `PDFChunk`, `Document`, эмбеддеры, исключения, настройки (`config.py`) |
| `api/` | HTTP-сервис на FastAPI поверх `ragkit` |
| `tests/` | тесты (`pytest`) |
| `week-01/` | неделя 1 — основы Python, ООП, файлы, исключения |
| `week-02/` | неделя 2 — алгоритмы, хеш-таблицы, SQL, строки, сортировки, NumPy/Pandas, Linux |
| `week-03/` | неделя 3 — HTTP, asyncio, надёжность, конфигурация |
| `todo/` | отдельное учебное CLI-приложение со списком задач |
| `demo.py` | демонстрация `ragkit` |

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .              # ragkit ставится в режиме разработки
cp .env.example .env          # и заполнить APP_API_KEY
```

`pip install -e .` делает `ragkit` импортируемым из любого каталога: больше не нужен `PYTHONPATH=.`.

## Запуск

```bash
python demo.py                    # демо пакета
python -m ragkit                  # то же через пакет
python week-03/lesson_04.py       # любой урок
```

## Настройки

Читаются один раз через `ragkit.config.get_settings()`.
Порядок старшинства: аргументы кода > переменные окружения > `.env` > умолчания.
Все переменные имеют префикс `APP_`, а вложенные группы разделяются через `__` (`APP_LLM__TIMEOUT=60`).
Список ключей лежит в `.env.example`. Сам `.env` в git не попадает.

## API

```bash
uvicorn api.main:app --reload     # http://127.0.0.1:8000/docs
pytest -q                         # тесты
mypy && ruff check .              # строгие типы для ragkit/, api/, tests/ и линтер
```

| Эндпоинт | Что делает |
|---|---|
| `GET /health` | версия ragkit, время старта, сколько раз загружен эмбеддер |
| `GET /health/config` | несекретные настройки (через `Depends(get_settings)`) |
| `POST /chunks` | нарезать текст на чанки (`mode`: `words` / `sentences`) |
| `POST /chunks/documents` | собрать документ и найти самый большой чанк |
| `POST /chunks/embed` | эмбеддинги текстов (модель загружается один раз в `lifespan`) |
| `/lab/*` | учебные эндпоинты урока Ф3-06 |
