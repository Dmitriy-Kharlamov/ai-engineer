"""Ф3-08. FastAPI: зависимости, аутентификация, структура проекта.

Запуск: python week-03/lesson_08.py      (около 15 с: bcrypt с rounds=12 медленный нарочно)
Сервис: uvicorn api.main:create_app --factory --reload
Прогноз к заданию 1 записан в коде до первого запуска.
"""

import base64
import json
import shutil
import statistics
import tempfile
import time
import timeit
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import bcrypt
import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api import deps
from api.errors import safe_error
from api.main import create_app
from ragkit.config import Settings, get_settings
from services.apikeys import ApiKeyStore, key_digest
from services.db import Database, Session
from tools.check_layers import find_violations
from tools.check_layers import main as check_layers_main

ROOT = Path(__file__).resolve().parent.parent


def lesson_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "api_key": SecretStr("lesson-key"),
        "debug": True,
        "jwt_secret": SecretStr("lesson-jwt-secret-at-least-32-bytes!!"),
        "bcrypt_rounds": 4,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def keys(app: FastAPI) -> ApiKeyStore:
    store: ApiKeyStore = app.state.api_keys
    return store


def bearer(client: TestClient, user: str, password: str = "password-123") -> dict[str, str]:
    client.post("/auth/register", json={"username": user, "password": password})
    r = client.post("/auth/login", json={"username": user, "password": password})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ===========================================================================
# 1. Кеш зависимости
# ===========================================================================
def task1() -> None:
    print("\n1. Кеш зависимости в пределах запроса")
    print("   прогноз: /two — 1 вызов; /nocache — 2 (кешированный + некешированный);")
    print("            два запроса подряд — 2 (кеш не переживает запрос)")
    calls = {"n": 0}

    def settings_dep() -> str:
        calls["n"] += 1
        return "настройки"

    def repo_a(s: Annotated[str, Depends(settings_dep)]) -> str:
        return "a"

    def repo_b(s: Annotated[str, Depends(settings_dep)]) -> str:
        return "b"

    def repo_fresh(s: Annotated[str, Depends(settings_dep, use_cache=False)]) -> str:
        return "fresh"

    demo = FastAPI()

    @demo.get("/two")
    def two(
        a: Annotated[str, Depends(repo_a)],
        b: Annotated[str, Depends(repo_b)],
        s: Annotated[str, Depends(settings_dep)],
    ) -> dict[str, int]:
        return {"calls": calls["n"]}

    @demo.get("/nocache")
    def nocache(
        a: Annotated[str, Depends(repo_a)], f: Annotated[str, Depends(repo_fresh)]
    ) -> dict[str, int]:
        return {"calls": calls["n"]}

    c = TestClient(demo)
    for title, path, repeat in [
        ("/two (нужен трижды)", "/two", 1),
        ("/nocache (use_cache=False)", "/nocache", 1),
        ("два запроса /two подряд", "/two", 2),
    ]:
        calls["n"] = 0
        for _ in range(repeat):
            c.get(path)
        print(f"   {title:28}: вызовов всего: {calls['n']}")

    print("   то же на настоящей «базе»: три сессии вместо одной")
    db = Database()
    one, two_, three = db.session(), db.session(), db.session()
    doc = one.add("alice", "a.md", "текст")
    print(
        f"      сессий открыто: {db.stats.opened}; сессия 2 видит документ сессии 1 "
        f"до commit: {two_.get(doc.id) is not None}"
    )
    one.commit()
    three.rollback()
    print(
        f"      после commit одной и rollback другой: документов {len(db.documents)}, "
        f"коммитов {db.stats.committed}, откатов {db.stats.rolled_back}"
    )


# Прогноз совпал: 1, 2, 2. FastAPI кеширует результат зависимости на один запрос: repo_a,
#   repo_b и обработчик получили один и тот же объект. use_cache=False у repo_fresh
#   заставил вызвать settings_dep второй раз. Между запросами кеша нет.
# Что сломается, если get_db вызывать напрямую из get_repo, из сервиса и из обработчика:
#   три вызова = три Session = три независимые транзакции (выше: сессий открыто 3).
#   Документ, добавленный через первую, вторая не видит до commit; одна может
#   зафиксироваться, другая откатиться — и в базе останется половина операции. Через
#   Depends(get_db) все три получают ОДНУ сессию запроса, то есть одну транзакцию.


# ===========================================================================
# 2. yield-зависимость как транзакция
# ===========================================================================
class LoggingSession(Session):
    def __init__(self, db: Database, url: str, log: list[str]) -> None:
        super().__init__(db, url)
        self.log = log
        log.append("open")

    def commit(self) -> None:
        super().commit()
        self.log.append("commit")

    def rollback(self) -> None:
        super().rollback()
        self.log.append("rollback")

    def close(self) -> None:
        super().close()
        self.log.append("close")


class LoggingDatabase(Database):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def session(self, url: str = "memory://") -> Session:
        self.stats.opened += 1
        return LoggingSession(self, url, self.log)


def task2() -> None:
    print("\n2. Настоящий get_db: порядок событий в трёх случаях")
    log: list[str] = []
    database = LoggingDatabase(log)
    demo = FastAPI()
    settings = lesson_settings()
    demo.dependency_overrides[get_settings] = lambda: settings  # не сама функция: у неё **kwargs
    demo.dependency_overrides[deps.get_database] = lambda: database
    SessionDep = Annotated[Session, Depends(deps.get_db)]

    @demo.middleware("http")
    async def mark(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        log.append("ответ готов к отправке")
        return response

    @demo.post("/ok")
    def ok(session: SessionDep) -> dict[str, bool]:
        log.append("обработчик")
        session.add("alice", "a.md", "текст")
        return {"ok": True}

    @demo.post("/http")
    def http_error(session: SessionDep) -> dict[str, bool]:
        log.append("обработчик")
        session.add("alice", "b.md", "текст")
        raise HTTPException(418, "я чайник")

    @demo.post("/boom")
    def boom(session: SessionDep) -> dict[str, bool]:
        log.append("обработчик")
        session.add("alice", "c.md", "текст")
        raise RuntimeError("необработанное")

    c = TestClient(demo, raise_server_exceptions=False)
    for path in ("/ok", "/http", "/boom"):
        log.clear()
        r = c.post(path)
        print(f"   {path:6} → {r.status_code}: {' → '.join(log)}")
    s = database.stats
    print(
        f"   итог: сессий {s.opened}, commit {s.committed}, rollback {s.rolled_back}, "
        f"close {s.closed}; документов в базе: {len(database.documents)} (только из /ok)"
    )


# До обработчика: open (код до yield). После: commit при успехе, rollback при любом
#   исключении — и HTTPException (418), и необработанном RuntimeError (500); close — всегда,
#   в finally. В базе один документ: изменения /http и /boom откатились.
# Тонкость, которую показал замер на fastapi 0.162: при УСПЕХЕ commit и close идут ПОСЛЕ
#   того, как ответ уже готов к отправке («→ ответ готов» стоит раньше commit). Значит,
#   клиент может получить 200 раньше, чем commit случился, — а если commit упадёт, клиент
#   об этом не узнает. В настоящем сервисе commit делают явно в обработчике или сервисе,
#   до return, а yield-зависимость оставляют для rollback и close. При исключении порядок
#   обратный: rollback и close — до ответа.
# Про «объект тянет данные из закрытой сессии»: здесь сериализация ответа идёт ДО выхода из
#   зависимости, так что для обычного JSON-ответа это не проявляется. Проявится для
#   StreamingResponse и фоновых задач — они работают уже после close.


# ===========================================================================
# 3. Цепочка get_settings → get_db → get_repo
# ===========================================================================
def task3() -> None:
    print("\n3. Цепочка зависимостей на /documents")
    print("   get_settings ─┬─> get_db ──> get_repo ──> create_document")
    print("                 └──────────────^")
    app = create_app(lesson_settings())
    calls = {"n": 0}

    def counting_settings() -> Settings:
        calls["n"] += 1
        return lesson_settings()

    with TestClient(app) as c:
        headers = bearer(c, "alice")
        app.dependency_overrides[get_settings] = counting_settings
        calls["n"] = 0
        r = c.post("/documents", json={"name": "a.md", "text": "текст"}, headers=headers)
        print(
            f"   POST /documents → {r.status_code}; get_settings вызван за запрос: {calls['n']} раз"
        )
    assert calls["n"] == 1


# get_settings объявлен параметром и у get_db, и у get_repo, но FastAPI строит граф
#   зависимостей запроса и вызывает каждый узел один раз: get_repo и get_db получили один и
#   тот же объект Settings.


# ===========================================================================
# 4. Защита роутера целиком
# ===========================================================================
def task4() -> None:
    print("\n4. /admin закрыт на уровне роутера")
    app = create_app(lesson_settings())
    with TestClient(app) as c:
        user_key = keys(app).issue("bob", frozenset({"user"}))
        admin_key = keys(app).issue("root", frozenset({"admin"}))
        print(f"   GET /health без ключа: {c.get('/health').status_code}")
        for path in ("/admin/stats", "/admin/new-report"):
            codes = [
                c.get(path).status_code,
                c.get(path, headers={"X-API-Key": user_key}).status_code,
                c.get(path, headers={"X-API-Key": admin_key}).status_code,
            ]
            print(f"   GET {path:18} без ключа / ключ user / ключ admin: {codes}")


# /admin/new-report добавлен последним, и в его коде нет ни Depends, ни проверки ключа
#   (см. api/routers/admin.py). Он всё равно отвечает 401 без ключа, 403 с ключом
#   пользователя и 200 только админу — защищает сам факт попадания в роутер с
#   dependencies=[Depends(require_admin)]. Забыть защиту для нового эндпоинта невозможно.


# ===========================================================================
# 5. API-ключ и compare_digest
# ===========================================================================
def task5() -> None:
    import secrets

    print("\n5. == против compare_digest: ключ 63 символа, 200 000 сравнений, минимум из 5")
    key = "k" * 63
    first = "X" + key[1:]
    last = key[:-1] + "X"
    n = 200_000
    results = {}
    for name, stmt in [
        ("== , отличается в 1-м", lambda: key == first),
        ("== , отличается в последнем", lambda: key == last),
        ("compare_digest, в 1-м", lambda: secrets.compare_digest(key, first)),
        ("compare_digest, в последнем", lambda: secrets.compare_digest(key, last)),
    ]:
        best = min(timeit.repeat(stmt, number=n, repeat=5)) / n * 1e9
        results[name] = best
        print(f"   {name:30}: {best:6.1f} нс")
    eq_gap = results["== , отличается в последнем"] - results["== , отличается в 1-м"]
    cd_gap = results["compare_digest, в последнем"] - results["compare_digest, в 1-м"]
    print(f"   разница «последний − первый»: == {eq_gap:+.1f} нс, compare_digest {cd_gap:+.1f} нс")

    big = "k" * 1_000_000  # тот же опыт на длинной строке: механизм становится виден
    big_first, big_last = "X" + big[1:], big[:-1] + "X"
    for name, stmt in [
        ("== , 1 МБ, отличие в 1-м", lambda: big == big_first),
        ("== , 1 МБ, отличие в последнем", lambda: big == big_last),
        ("compare_digest, 1 МБ, в 1-м", lambda: secrets.compare_digest(big, big_first)),
        ("compare_digest, 1 МБ, в последнем", lambda: secrets.compare_digest(big, big_last)),
    ]:
        best = min(timeit.repeat(stmt, number=200, repeat=5)) / 200 * 1e6
        print(f"   {name:34}: {best:8.2f} мкс")

    store = ApiKeyStore()
    raw = store.issue("bob", frozenset({"user"}))
    print(f"   выпущен ключ: {raw[:10]}… ; на сервере хранится: {store.stored_digests()[0][:16]}…")
    print(
        f"   sha256(ключа) == хранимое: {key_digest(raw) == store.stored_digests()[0]}; "
        f"verify(ключ) → {store.verify(raw)}; verify(чужой) → {store.verify(raw + 'x')}"
    )


# Честно о моих числах: на ключе из 63 символов разница «последний − первый» у == вышла
#   на уровне шума — от прогона к прогону она у меня бывала и +1 нс, и −20 нс, то есть
#   «различие в конце» иногда даже быстрее. Устойчивых 6 нс, как в уроке, я не получил.
#   Строки в CPython сравниваются через memcmp, и 63 байта он проверяет за несколько
#   машинных инструкций, где бы ни было различие; сколько это стоит — зависит от
#   процессора. Абсолютные числа включают вызов лямбды (десятки нс).
# Механизм при этом настоящий — его показывает строка в 1 МБ: == с различием в первом
#   символе выходит мгновенно, с различием в последнем — просматривает весь мегабайт.
#   compare_digest в обоих случаях проходит строку целиком: время не зависит от места
#   различия. Это и есть «постоянное время».
# Масштаб: на коротком ключе и по сети сигнал неизмерим — сетевой разброс на порядки
#   больше. Но решение всё равно однозначное: compare_digest стоит десятки наносекунд, а
#   правило «любой секрет — compare_digest» избавляет от необходимости в каждом новом
#   месте доказывать, что утечка по времени не важна. Длина секрета, реализация сравнения
#   и железо меняются; правило — нет.
# Правильный вариант: ключ выдаётся один раз (rk_…), на сервере лежит только sha256. Утечка
#   хранилища не даёт действующих ключей. sha256, а не bcrypt: ключ — 256 случайных бит,
#   перебор бессмыслен, а медленный хеш нужен паролям, которые придумывают люди.


# ===========================================================================
# 6. Пароли и перебор логинов
# ===========================================================================
def median_ms(func: Callable[[], object], runs: int = 7) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def task6() -> None:
    print("\n6. bcrypt rounds=12: время ответа на вход, медиана из 7")
    app = create_app(lesson_settings(bcrypt_rounds=12))
    with TestClient(app) as c:
        c.post("/auth/register", json={"username": "alice", "password": "correct-horse"})
        cases = [
            ("верный пароль", {"username": "alice", "password": "correct-horse"}),
            ("неверный пароль", {"username": "alice", "password": "wrong-password"}),
            ("нет такого логина", {"username": "nobody", "password": "wrong-password"}),
        ]
        for title, path in (
            ("ДО (без выравнивания)", "/lab/login-naive"),
            ("ПОСЛЕ (фиктивный хеш)", "/auth/login"),
        ):
            print(f"   {title}:")
            for name, body in cases:
                ms = median_ms(partial(c.post, path, json=body))
                code = c.post(path, json=body).status_code
                print(f"      {name:18}: {ms:6.1f} мс  (код {code})")

    print("   цена одного хеша:")
    for rounds in (4, 12):
        ms = median_ms(partial(bcrypt.hashpw, b"pw", bcrypt.gensalt(rounds)), runs=5)
        print(f"      rounds={rounds:2}: {ms:7.1f} мс")


# ДО: верный и неверный пароль стоят одинаково (bcrypt считается в обоих случаях), а
#   несуществующий логин отвечает почти мгновенно — проверки хеша нет. Разница в сотни
#   миллисекунд видна по сети без всякой статистики: по времени ответа можно перебрать
#   список логинов и узнать, какие зарегистрированы.
# ПОСЛЕ: для несуществующего логина прогоняем checkpw против фиктивного хеша той же
#   стоимости — все три случая стоят одинаково, ответ и текст ошибки тоже одинаковые.
# rounds — это log2 числа итераций: +1 раунд = вдвое дороже, 12 против 4 — в 2^8 = 256 раз
#   (в выводе отношение того же порядка). Выбирают так: максимальное число, при котором
#   хеш на ПРОДАКШН-железе считается ~100–300 мс — пользователь при входе не заметит, а
#   перебор утечки становится в сотни тысяч раз дороже, чем с sha256. Пересматривают, когда
#   меняется железо. В тестах ставят 4: стоимость там не нужна.


# ===========================================================================
# 7. JWT
# ===========================================================================
def b64json(part: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def task7() -> None:
    print("\n7. JWT")
    app = create_app(lesson_settings())
    with TestClient(app) as c:
        token = bearer(c, "alice")["Authorization"].removeprefix("Bearer ")
        secret: str = app.state.jwt_secret
        print(f"   выпущен: {token[:40]}…")
        print(
            f"   GET /me с токеном: {c.get('/me', headers={'Authorization': f'Bearer {token}'}).json()}"
        )

        header, payload, _ = token.split(".")
        print(f"   прочитано БЕЗ ключа: header={b64json(header)} payload={b64json(payload)}")

        tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
        r = c.get("/me", headers={"Authorization": f"Bearer {tampered}"})
        print(f"   испорченная подпись: {r.status_code} {r.json()}")

        forged = b64json(payload) | {"roles": ["admin", "user"]}
        forged_payload = base64.urlsafe_b64encode(json.dumps(forged).encode()).rstrip(b"=").decode()
        r = c.get(
            "/me",
            headers={"Authorization": f"Bearer {header}.{forged_payload}.{token.split('.')[2]}"},
        )
        print(f"   нагрузка подменена (roles += admin), подпись старая: {r.status_code} {r.json()}")

        expired = jwt.encode(
            {"sub": "alice", "exp": datetime.now(UTC) - timedelta(seconds=1)},
            secret,
            algorithm="HS256",
        )
        r = c.get("/me", headers={"Authorization": f"Bearer {expired}"})
        print(
            f"   просроченный: {r.status_code} {r.json()}, WWW-Authenticate: {r.headers['WWW-Authenticate']}"
        )
        try:
            jwt.decode(token, secret)
        except jwt.DecodeError as e:
            print(f"   jwt.decode без algorithms: {type(e).__name__}: {e}")


# Нагрузка — это base64url, а не шифр: sub, roles, iat, exp прочитаны без ключа. Отсюда:
#   в токен кладут только идентификаторы и сроки — никаких email, ролей «для красоты» с
#   секретными названиями, тем более паролей и ключей.
# Подпись защищает от ПОДДЕЛКИ: испорченная подпись и подменённая нагрузка (дописал себе
#   admin) — 401. От ЧТЕНИЯ она не защищает.
# Просроченный — 401 «токен просрочен» с WWW-Authenticate: Bearer: jwt.decode по умолчанию
#   проверяет exp. Отозвать выпущенный токен нельзя — поэтому срок 15 минут.
# jwt.decode без algorithms в PyJWT 2.x не работает вовсе — DecodeError: алгоритм должен
#   выбирать сервер, а не заголовок токена, иначе подмена alg (например, на none или на
#   HS256 с публичным ключом RSA вместо секрета) открывает подделку.


# ===========================================================================
# 8. 401, 403, 404
# ===========================================================================
def task8() -> None:
    print("\n8. 401 / 403 / 404")
    app = create_app(lesson_settings())
    with TestClient(app) as c:
        r = c.get("/admin/stats")
        print(
            f"   без ключа:              {r.status_code} {r.json()} WWW-Authenticate: {r.headers.get('WWW-Authenticate')}"
        )
        user_key = keys(app).issue("bob", frozenset({"user"}))
        r = c.get("/admin/stats", headers={"X-API-Key": user_key})
        print(f"   ключ без роли admin:    {r.status_code} {r.json()}")
        alice, bob = bearer(c, "alice"), bearer(c, "bob")
        doc_id = c.post("/documents", json={"name": "a.md", "text": "т"}, headers=alice).json()[
            "id"
        ]
        foreign = c.get(f"/documents/{doc_id}", headers=bob)
        missing = c.get("/documents/424242", headers=bob)
        print(f"   чужой документ {doc_id}:     {foreign.status_code} {foreign.json()}")
        print(f"   несуществующий 424242:  {missing.status_code} {missing.json()}")


# 401 — «не знаю, кто ты»: заголовок WWW-Authenticate говорит, какой пропуск принести.
# 403 — «знаю, но нельзя»: ключ понят, роли admin нет, другой ключ того же человека не поможет.
# 404 вместо 403 для чужого документа: ответы на чужой и на несуществующий id
#   неотличимы — и статус, и тело. 403 подтвердил бы, что документ с таким id существует,
#   и перебором id можно было бы узнать, сколько документов в системе и какие id заняты.
#   Это решение: для владельца-не-хозяина объекта «нет» — честный ответ, потому что для
#   него объекта действительно нет.


# ===========================================================================
# 9. Направление зависимостей
# ===========================================================================
def task9() -> None:
    print("\n9. Проверка направления api → services → ragkit")
    print(f"   репозиторий: нарушений {len(find_violations(ROOT))}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(ROOT / "ragkit", root / "ragkit")
        shutil.copytree(ROOT / "services", root / "services")
        (root / "ragkit" / "leak.py").write_text("from api.deps import get_db\n", encoding="utf-8")
        (root / "services" / "leak.py").write_text("import api.security\n", encoding="utf-8")
        print("   после того как в копию добавлены два нарушения:")
        code = check_layers_main(["check_layers", str(root)])
        print(f"   код возврата: {code}")


# tools/check_layers.py разбирает импорты через ast и падает с кодом 1, если ragkit
#   импортирует api или services, или services импортирует api. Тот же код — в
#   tests/test_layers.py, так что нарушение не пройдёт pytest.
# Пример из жизни этого урока: хеширование паролей нужно services/users.py, поэтому оно в
#   services/passwords.py, а не в api/security.py, как в схеме урока. Положи я его в api —
#   сервис импортировал бы api, и проверка упала бы. Направление важнее «логичного» места.


# ===========================================================================
# 10. Ограничение частоты
# ===========================================================================
def task10() -> None:
    print("\n10. Не больше 5 запросов в минуту с одного ключа")
    app = create_app(lesson_settings(rate_limit_per_minute=5))
    with TestClient(app) as c:
        key = keys(app).issue("dave", frozenset())
        other = keys(app).issue("erin", frozenset())
        codes = []
        retry = None
        for _ in range(7):
            r = c.get("/me/key", headers={"X-API-Key": key})
            codes.append(r.status_code)
            retry = r.headers.get("Retry-After", retry)
        print(f"   7 запросов dave: {codes}; Retry-After: {retry}")
        print(
            f"   erin в то же время: {c.get('/me/key', headers={'X-API-Key': other}).status_code}"
        )


# Пять — 200, шестой и седьмой — 429 с Retry-After. Лимит считается по ключу: erin не
#   страдает от того, что dave упёрся в потолок.
# Какое значение Retry-After: не «60 на всякий случай», а сколько секунд осталось до
#   момента, когда самый старый запрос в окне выйдет из него, округлённое вверх (здесь
#   почти 60, потому что все пять запросов были только что). Связь с Ф3-03: там мой клиент
#   ждал ровно Retry-After и экономил запросы. Если сервер пишет слишком мало — клиент
#   вернётся рано и получит ещё 429; слишком много — будет простаивать зря. Точное число
#   выгодно обеим сторонам.


# ===========================================================================
# 11. Долги
# ===========================================================================
def task11() -> None:
    print("\n11. Долги")
    for debug in (False, True):
        with TestClient(create_app(lesson_settings(debug=debug))) as c:
            r = c.post("/lab/params/1", json={"note": "x"})
            print(f"   debug={debug!s:5}: POST /lab/params/1 → {r.status_code}")
    with TestClient(create_app(lesson_settings())) as c:
        r = c.post(
            "/chunks/documents", json={"name": "d", "max_chunk_tokens": 2, "texts": ["a b c"]}
        )
        print(f"   ChunkTooLargeError → {r.status_code} {r.json()}")
    err = {
        "type": "bad_key_prefix",
        "loc": ("key",),
        "msg": "неверный префикс",
        "input": "sk-секрет",
        "ctx": {"got": "sk-секрет", "min_length": 12},
    }
    print(f"   незнакомый ключ ctx 'got': {safe_error(err)}")


# lab: подключается в create_app только при settings.debug — в проде /lab не существует (404).
# Стабильный код: клиент сверяет "code": "chunk_too_large", а не имя класса Python. Имя
#   класса — деталь реализации: переименуешь исключение — сломаешь всех клиентов.
# ctx: белый список с запретом по умолчанию — ключ got не в списке и не проходит, а
#   min_length из списка проходит.
# 413: пересмотрел, теперь 422. 413 Content Too Large — про размер тела запроса целиком; его
#   выдают прокси и сервер до разбора, и балансировщики обрабатывают его особо. У нас тело
#   принято и разобрано, а не прошёл смысловой лимит одного чанка — это ошибка содержимого,
#   422 с кодом chunk_too_large.


# ===========================================================================
# 12. Устно, потом письменно
# ===========================================================================
# Зачем Depends, если зависимость можно вызвать: FastAPI строит граф и кеширует узлы на
#   запрос (одна сессия на запрос — задание 1), выполняет уборку yield-зависимостей,
#   позволяет подменить зависимость в тестах (dependency_overrides) и показывает её в
#   схеме (/docs знает про X-API-Key и Bearer). Прямой вызов не даёт ничего из этого.
# Что гарантирует кеш и чего нет: гарантирует — в пределах ОДНОГО запроса каждая
#   зависимость вызвана один раз, и все её потребители получили один объект. Не
#   гарантирует — ничего между запросами (каждый запрос заново) и ничего при
#   use_cache=False; это не кеш между пользователями и не lru_cache.
# Когда выполняется уборка в yield-зависимости: код после yield — после обработчика; при
#   исключении — до отправки ответа (rollback, close), при успехе в fastapi 0.162 — уже
#   после того, как ответ готов (задание 2). finally выполняется всегда.
# 401 / 403 / 404: 401 — не знаем, кто ты (+ WWW-Authenticate); 403 — знаем, но нельзя; 404
#   — для чужого объекта, чтобы не подтверждать его существование.
# Почему sha256 не годится для паролей: он быстрый — миллиарды попыток в секунду на GPU.
#   Хеш пароля должен быть медленным и с параметром стоимости (bcrypt, argon2), чтобы
#   перебор утёкшей базы стоил годы, а не часы.
# Что защищает подпись JWT: целостность — нагрузку нельзя изменить без секрета (задание 7).
#   Не защищает: конфиденциальность (нагрузка читается без ключа) и отзыв (токен живёт до
#   exp).
# Почему циклический импорт — симптом: он появляется, когда нижний слой захотел знать о
#   верхнем, то есть направление api → services → ragkit нарушено. Локальный импорт внутри
#   функции убирает ошибку, но оставляет нарушенную архитектуру; лечится переносом кода в
#   правильный слой (как пароли в services/passwords.py).


def main() -> None:
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()
    task10()
    task11()


if __name__ == "__main__":
    main()
