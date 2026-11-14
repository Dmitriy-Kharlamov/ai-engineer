# ai-engineer

Учебный репозиторий курса AI-инженера: задания по неделям и пакет `ragkit`, который растёт вместе с курсом.

## Структура

| Путь | Что там |
|---|---|
| `ragkit/` | пакет: `Chunk`, `PDFChunk`, `Document`, эмбеддеры, исключения, настройки (`config.py`) |
| `api/` | HTTP-слой на FastAPI: роутеры, схемы, зависимости, безопасность |
| `services/` | бизнес-логика: пользователи, ключи, документы, ограничитель; про HTTP не знает |
| `tools/` | `check_layers.py` — проверка направления импортов `api → services → ragkit` |
| `tests/` | тесты (`pytest`) |
| `week-01/` | неделя 1 — основы Python, ООП, файлы, исключения |
| `week-02/` | неделя 2 — алгоритмы, хеш-таблицы, SQL, строки, сортировки, NumPy/Pandas, Linux |
| `week-03/` | неделя 3 — HTTP, asyncio, надёжность, конфигурация |
| `todo/` | отдельное учебное CLI-приложение со списком задач |
| `docqa/` | CLI: вопросы к папке документов через локальную модель с инструментами (чек-поинт Ф4-08) — [docqa/README.md](docqa/README.md) |
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
uvicorn api.main:create_app --factory --reload   # http://127.0.0.1:8000/docs
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
| `POST /auth/register`, `POST /auth/login` | регистрация (bcrypt) и вход (JWT на 15 минут) |
| `GET /me`, `GET /me/key` | кто я: по JWT / по API-ключу (с ограничением частоты) |
| `POST /documents`, `GET /documents/{id}` | свои документы; чужой — 404 |
| `/admin/*` | только API-ключ с ролью `admin` (защита на уровне роутера) |
| `/lab/*` | учебные эндпоинты; есть только при `APP_DEBUG=true` |

## Регрессионный набор и CI (Ф5.5-02)

Набор — `regress/suite.json`: 49 пунктов (q01–q20 — вопросы по `docqa/sample_docs`, 01–15e —
таблица отказов Ф4-08), базовые линии метрик с отпечатком среды и карантин со сроком. Бегун —
`python -m docqa.regress`, машинерия — `ragkit/regress.py`.

| слой | что | модель | ворота |
|---|---|---|---|
| фейковый клиент | `pytest` | нет | да |
| кассеты | `python -m docqa.regress --layer replay` | нет: запись в `regress/cassettes/` | да |
| живая модель | `python -m docqa.regress --layer live --seed N` | да | нет — задача (issue) |

Кассеты перезаписываются, когда меняется запрос к модели (промпт, схема, история):
`python -m docqa.regress --layer record --accept --save regress/runs/record.json`.

Два потока:

* `.github/workflows/ci.yml` — на каждый push: ruff, слои, mypy, тесты, набор на кассетах (там же
  проверка журнала на сырые ошибки и срок карантина). Ни одного обращения к модели.
* `.github/workflows/nightly.yml` — каждую ночь на self-hosted раннере (Mac с ollama, метки
  `self-hosted, macOS, ollama`): живой слой с порогами по нижней границе, бюджет прогона.
  Красный — создаётся issue с отчётом. Репозиторий публичный: в Settings → Actions → General
  включено «Require approval for all outside collaborators», иначе чужой pull request мог бы
  запустить код на этом Mac.

Карантин: пункт можно убрать из ворот только записью `{"item", "until", "reason"}` в
`quarantine` набора; после `until` блокирующий поток красный, пока запись не убрана или не
продлена.

Время и цена потоков (замер — Ф5.5-02, задание 8):

| поток | время | цена |
|---|---|---|
| блокирующий (GitHub, ubuntu) | — замер после первого прогона | 0 (публичный репозиторий) |
| ночной (Mac, qwen2.5:7b) | — замер после первого прогона | — по прайсу CostGuard |
