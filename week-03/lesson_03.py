import asyncio
import heapq
import json
import random
import time
import traceback
import urllib.parse
import uuid
from collections import Counter, defaultdict
from http import HTTPStatus
from itertools import pairwise
from types import SimpleNamespace

import httpx

T0 = time.perf_counter()


def now():
    return time.perf_counter()


# ===========================================================================
# Управляемый сервер
# ===========================================================================
class ControlServer:
    """Локальный HTTP-сервер со сценариями сбоев.

    /flaky/<имя>?fail=N&code=503&delay=S  — первые N запросов отвечает code, потом 200
    /limited/<имя>?times=N&retry_after=S  — N «окон» 429 с заголовком Retry-After
    /orders/<имя>?fail=N   (POST)         — считает заказы по Idempotency-Key;
                                            первые N ответов «теряются» (503)
    /breaker/<имя>                        — 200, если сервис «жив», иначе 503
    /slow?delay=S                         — ответ через S секунд
    /trickle?seconds=N&interval=I         — N байт, по одному раз в I секунд
    /work?service=S                       — работа S секунд; не больше max_conn
                                            одновременно, лишним — сразу 503
    """

    def __init__(self, max_conn=1000):
        self.max_conn = max_conn
        self.active = 0
        self.hits = Counter()  # имя сценария → число запросов
        self.times = defaultdict(list)  # имя → моменты прихода запросов
        self.keys = defaultdict(list)  # имя → ключи идемпотентности по запросам
        self.blocked_until = {}
        self.windows = Counter()
        self.alive = {}

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handle(self, reader, writer):
        admitted = False
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
            lines = head.split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {
                k.strip().lower(): v.strip()
                for k, v in (line.split(":", 1) for line in lines[1:] if ":" in line)
            }
            await reader.readexactly(int(headers.get("content-length", 0)))
            if self.active >= self.max_conn:  # лимит одновременных
                await self.respond(writer, 503, {"error": "busy"})
                return
            self.active += 1
            admitted = True
            parts = urllib.parse.urlsplit(target)
            q = dict(urllib.parse.parse_qsl(parts.query))
            await self.route(method, parts.path.strip("/").split("/"), q, headers, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # клиент ушёл — нормально
        finally:
            if admitted:
                self.active -= 1
            writer.close()

    async def route(self, method, path, q, headers, writer):
        kind, name = path[0], (path[1] if len(path) > 1 else path[0])
        n = self.hits[name]
        self.hits[name] += 1
        self.times[name].append(now())

        if kind == "flaky":
            await asyncio.sleep(float(q.get("delay", 0)))
            code = int(q.get("code", 503)) if n < int(q.get("fail", 0)) else 200
            await self.respond(writer, code, {"request": n + 1})

        elif kind == "limited":
            ra = float(q.get("retry_after", 2))
            t = now()
            if t < self.blocked_until.get(name, 0):  # рано пришёл
                left = self.blocked_until[name] - t
                await self.respond(writer, 429, {}, {"Retry-After": f"{left:.0f}"})
            elif self.windows[name] < int(q.get("times", 0)):  # новое окно лимита
                self.windows[name] += 1
                self.blocked_until[name] = t + ra
                await self.respond(writer, 429, {}, {"Retry-After": f"{ra:.0f}"})
            else:
                await self.respond(writer, 200, {"ok": True})

        elif kind == "orders":
            self.keys[name].append(headers.get("idempotency-key"))
            created = len(set(self.keys[name]))  # новый ключ = новый заказ
            code = 503 if n < int(q.get("fail", 0)) else 200  # заказ создан, ответ «потерян»
            await self.respond(writer, code, {"orders_created": created})

        elif kind == "breaker":
            await self.respond(writer, 200 if self.alive.get(name) else 503, {})

        elif kind == "slow":
            await asyncio.sleep(float(q.get("delay", 1)))
            await self.respond(writer, 200, {"slow": True})

        elif kind == "trickle":
            seconds, interval = int(q.get("seconds", 10)), float(q.get("interval", 1))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % seconds
            )
            await writer.drain()
            for _ in range(seconds):
                await asyncio.sleep(interval)
                writer.write(b"x")
                await writer.drain()

        elif kind == "work":
            await asyncio.sleep(float(q.get("service", 0.1)))
            await self.respond(writer, 200, {"done": True})

        else:
            await self.respond(writer, 404, {})

    @staticmethod
    async def respond(writer, code, body, extra=None):
        payload = json.dumps(body).encode()
        head = [
            f"HTTP/1.1 {code} {HTTPStatus(code).phrase}",
            "Content-Type: application/json",
            f"Content-Length: {len(payload)}",
            "Connection: close",
        ]
        head += [f"{k}: {v}" for k, v in (extra or {}).items()]
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + payload)
        await writer.drain()


# ===========================================================================
# 1. Классификация
# ===========================================================================
IDEMPOTENT = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"}
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
MAYBE_DONE = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.RemoteProtocolError)


def should_retry(outcome, method):
    """outcome — код ответа (int) или класс исключения httpx."""
    if isinstance(outcome, int):
        return outcome in {408, 425, 429, 500, 502, 503, 504}
    if issubclass(outcome, NOT_SENT):  # запрос точно не ушёл на сервер
        return True
    if issubclass(outcome, MAYBE_DONE):  # мог выполниться — только идемпотентный
        return method.upper() in IDEMPOTENT
    return False


def task1():
    print("\n1. should_retry")
    cases = [
        (httpx.ConnectError, "POST", True, "не дошло до сервера — повтор безопасен даже для POST"),
        (httpx.ReadTimeout, "GET", True, "GET идемпотентен: повтор ничего не испортит"),
        (httpx.ReadTimeout, "POST", False, "запрос мог выполниться — повтор создаст дубль"),
        (429, "POST", True, "лимит: сервер не выполнял запрос; паузу берём из Retry-After"),
        (500, "GET", True, "сервер сломался, может пройти"),
        (502, "GET", True, "сломался шлюз/прокси перед сервером"),
        (503, "POST", True, "сервис недоступен — обычно до выполнения"),
        (504, "GET", True, "шлюз не дождался сервера"),
        (408, "GET", True, "сервер сам сказал: не дождался запроса, пришли ещё раз"),
        (425, "POST", True, "сервер сам просит повторить позже (Too Early)"),
        (400, "POST", False, "запрос кривой — от повтора он не исправится"),
        (401, "GET", False, "нужен другой токен, а не повтор"),
        (403, "GET", False, "прав нет — повтор не даст прав"),
        (404, "GET", False, "ресурса нет и через секунду не появится"),
        (405, "GET", False, "метод не поддерживается — это баг в коде"),
        (422, "POST", False, "данные не прошли валидацию — нужно чинить данные"),
        (409, "PUT", False, "конфликт состояния — его разрешают, а не перезапрашивают"),
        # три своих случая
        (httpx.ConnectTimeout, "POST", True, "своё: соединение не установилось — запроса не было"),
        (
            httpx.RemoteProtocolError,
            "POST",
            False,
            "своё: сервер оборвал ответ посередине — как ReadTimeout",
        ),
        (
            httpx.ReadTimeout,
            "PUT",
            True,
            "своё: PUT идемпотентен — «заменить на X» дважды = один раз",
        ),
    ]
    for outcome, method, expected, why in cases:
        got = should_retry(outcome, method)
        name = outcome if isinstance(outcome, int) else outcome.__name__
        print(f"   {str(name):20} {method:5} → {'повторять' if got else 'нет':9} | {why}")
        assert got == expected, (outcome, method)


# Почему ConnectError можно повторять для POST, а ReadTimeout нельзя:
#   ConnectError — соединение не установилось (DNS, отказ, сеть). Байты запроса
#   не ушли, сервер о нём не знает, значит, ничего не выполнил: повтор — это
#   первая настоящая попытка. ReadTimeout — запрос УЖЕ отправлен целиком, и мы
#   не дождались ответа. Сервер мог его получить, создать заказ и не успеть
#   ответить. Повтор POST в этом случае = второй заказ. Для GET это не важно:
#   прочитать дважды — то же, что один раз.
# Оговорка к таблице: 500 на POST тоже «не знаю» — сервер мог упасть ПОСЛЕ
#   записи в базу. Таблица говорит «повторять», и это верно, только если у
#   POST есть ключ идемпотентности (задание 7).


# ===========================================================================
# 2. Retry руками
# ===========================================================================
async def retry_request(
    client,
    method,
    url,
    *,
    attempts=5,
    base=0.5,
    cap=8.0,
    jitter=0.5,
    respect_retry_after=True,
    log=print,
    **kw,
):
    start = now()
    for attempt in range(1, attempts + 1):
        response = error = None
        try:
            response = await client.request(method, url, **kw)
            if response.status_code < 400:
                log(f"      попытка {attempt}: {response.status_code} за {now() - start:.2f} с")
                return response
            outcome, reason = response.status_code, f"HTTP {response.status_code}"
        except httpx.TransportError as e:
            error, outcome, reason = e, type(e), type(e).__name__
        if attempt == attempts or not should_retry(outcome, method):
            log(f"      попытка {attempt}: {reason} — сдаюсь через {now() - start:.2f} с")
            if error:
                raise error
            response.raise_for_status()
        delay = min(cap, base * 2 ** (attempt - 1)) + random.uniform(0, jitter)
        source = "формула"
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if respect_retry_after and retry_after:
            delay, source = float(retry_after), "Retry-After"
        log(
            f"      попытка {attempt}: {reason} → пауза {delay:.2f} с ({source}), "
            f"прошло {now() - start:.2f} с"
        )
        await asyncio.sleep(delay)
    raise ValueError(f"attempts должно быть ≥ 1, получено {attempts}")


async def task2(srv, client):
    print("\n2. Retry руками: сервер падает первые 3 раза")
    random.seed(1)
    low, high = 0.5 + 1 + 2, (0.5 + 1 + 2) + 3 * 0.5
    print(
        f"   прогноз: паузы 0.5 + 1 + 2 = {low} с плюс джиттер 3 × [0, 0.5) "
        f"→ от {low} до {high} с, в среднем {low + 3 * 0.25} с"
    )
    start = now()
    await retry_request(client, "GET", f"{srv.url}/flaky/t2?fail=3")
    total = now() - start
    print(f"   итого: {total:.2f} с, запросов на сервере: {srv.hits['t2']}")
    assert srv.hits["t2"] == 4 and low <= total <= high + 0.3


# Сервер падает 3 раза, значит, нужно 4 попытки и 3 паузы. Паузы растут
#   экспонентой от base = 0.5: 0.5, 1, 2 (потолок cap = 8 ещё не достигнут),
#   к каждой — джиттер от 0 до 0.5 с. Итог попал в предсказанный диапазон;
#   отличие от 3.5 — это сумма трёх джиттеров плюс миллисекунды на запросы.


# ===========================================================================
# 3. Джиттер по двум метрикам
# ===========================================================================
BASE, CAP, CLIENTS, RETRIES = 0.5, 20.0, 200, 5


def make_pause(kind):
    def pause(retry_index):
        exp = min(CAP, BASE * 2**retry_index)
        if kind == "без джиттера":
            return exp
        if kind == "аддитивный":
            return exp + random.uniform(0, 1.0)
        return random.uniform(0, exp)  # full jitter

    return pause


KINDS = ["без джиттера", "аддитивный", "full jitter"]


def peak_metric(kind, seed=0):
    """Все 200 упали в t=0; у каждого 5 повторов. Пик повторов в окне 100 мс."""
    random.seed(seed)
    pause = make_pause(kind)
    retries = []
    for _ in range(CLIENTS):
        t = 0.0
        for i in range(RETRIES):
            t += pause(i)
            retries.append(t)
    windows = Counter(int(t / 0.1) for t in retries)
    return max(windows.values()), len(windows)


def drain_metric(kind, max_retries, seed=0, capacity=10, service=0.1):
    """Событийная симуляция: сервер держит capacity одновременно, лишним — отказ.
    Возвращает: сколько клиентов прошло, когда закончил последний, всего запросов."""
    random.seed(seed)
    pause = make_pause(kind)
    events = [(0.0, c, 0) for c in range(CLIENTS)]  # (время, клиент, № повтора)
    heapq.heapify(events)
    busy, served, last_done, requests = [], 0, 0.0, 0
    while events:
        t, client, retry = heapq.heappop(events)
        requests += 1
        busy = [end for end in busy if end > t]  # кто ещё обслуживается
        if len(busy) < capacity:
            busy.append(t + service)
            served += 1
            last_done = max(last_done, t + service)
        elif retry < max_retries:
            heapq.heappush(events, (t + pause(retry), client, retry + 1))
    return served, last_done, requests


async def task3():
    print("\n3. Джиттер: 200 клиентов, 5 повторов, base=0.5, cap=20, seed=0")
    print(
        f"   {'формула':14} | {'пик в 100 мс':>12} | {'окон':>4} || "
        f"{'5 повторов: прошло':>18} | {'конец':>6} || {'без лимита: все за':>18} | {'запросов':>8}"
    )
    rows = {}
    for kind in KINDS:
        peak, used = peak_metric(kind)
        served5, end5, _ = drain_metric(kind, max_retries=5)
        served_all, end_all, reqs = drain_metric(kind, max_retries=10_000)
        rows[kind] = (peak, used, served5, end5, end_all)
        print(
            f"   {kind:14} | {peak:>12} | {used:>4} || {served5:>14}/200 | "
            f"{end5:>5.1f}с || {end_all:>16.1f} с | {reqs:>8}"
        )
        assert served_all == CLIENTS
    print(
        "   устойчивость к seed (пик, среднее по seed 0–19):",
        {k: round(sum(peak_metric(k, s)[0] for s in range(20)) / 20, 1) for k in KINDS},
    )
    assert rows["без джиттера"][0] == 200 and rows["без джиттера"][1] == 5
    assert rows["аддитивный"][0] < rows["full jitter"][0]  # пик: аддитивный лучше
    assert rows["full jitter"][4] < rows["аддитивный"][4]  # проход: full лучше


# Метрика 1 (пик). Мои числа: без джиттера 200 и 5 окон — совпало с уроком.
#   Аддитивный 27 / 107 и full jitter 65 / 120 против 24 / 108 и 66 / 123 в
#   уроке — не сошлось до единицы. Порядок величин и вывод те же: точные
#   числа зависят от того, в каком порядке тянутся случайные числа (по
#   клиентам или по попыткам) и как округляется окно. Средние по 20 seed
#   показывают, что разница устойчива, а не случайна.
#   Без джиттера все 200 ждут одинаково и приходят одновременно: 5 повторов
#   = 5 окон по 200 запросов. Аддитивный размазывает каждую волну на секунду
#   и даёт самый низкий пик. Full jitter тянет ранние паузы к нулю — часть
#   клиентов повторяет почти сразу, и пик выше.
# Метрика 2 (проход через сервер на 10 одновременных, обслуживание 0.1 с).
#   Это событийная симуляция того же сервера, что /work: в реальном времени
#   прогон без лимита повторов шёл бы минуты. Сервер пропускает максимум
#   100 запросов/с, так что 200 клиентов физически нельзя обслужить быстрее
#   ~2 с.
#   С 5 повторами без джиттера каждая волна — 200 одновременных, и сервер
#   берёт из неё только 10: 6 волн (первая попытка + 5 повторов) × 10 = 60 из
#   200, остальные сдаются; последняя волна в 15.5 с, конец в 15.6 с. Без
#   лимита повторов то же самое тянется 20 волн — 291.6 с и 2100 запросов.
#   С джиттером проходят все 200 даже за 5 повторов: клиенты растащены во
#   времени, и сервер всё время занят полезной работой.
#   Full jitter по времени прохода лучше: 5.0 с против 6.0 у аддитивного —
#   его клиенты занимают и ранние «пустые» окна, которые аддитивный (пауза не
#   меньше exp) оставляет простаивать. Но платит за это числом запросов:
#   647 против 492 — ранние повторы чаще натыкаются на занятый сервер и
#   получают отказ. То есть даже по второй метрике ответ зависит от того,
#   что мерить: время прохода — full jitter, лишняя нагрузка — аддитивный.
# Вывод — как в уроке: формула выбирается под метрику аварии. Боимся
#   мгновенного пика (сервер ляжет от удара) — аддитивный. Боимся, что очередь
#   не рассосётся (ограниченный ресурс) — full jitter. Без джиттера — плохо
#   по обеим.


async def task3_server_check():
    print("   проверка сервера: 30 одновременных запросов при лимите 10")
    async with ControlServer(max_conn=10) as srv, httpx.AsyncClient(trust_env=False) as c:
        codes = await asyncio.gather(*[c.get(f"{srv.url}/work?service=0.3") for _ in range(30)])
        counts = Counter(r.status_code for r in codes)
        print(f"   ответы: {dict(counts)}")
        assert counts[200] == 10 and counts[503] == 20


# Сервер принял ровно 10 (лимит K), остальным 20 сразу ответил 503 — тот же
#   отказ «занято», который моделирует симуляция в метрике 2.


# ===========================================================================
# 4. 429 и Retry-After
# ===========================================================================
async def task4(srv, client):
    print("\n4. 429 и Retry-After (сервер: 2 окна лимита по 2 с)")
    random.seed(4)
    print("   а) уважаем Retry-After:")
    start = now()
    await retry_request(client, "GET", f"{srv.url}/limited/ra_ok?times=2&retry_after=2", jitter=0)
    t_ok = now() - start
    print(f"   итого {t_ok:.2f} с, запросов на сервере: {srv.hits['ra_ok']}")

    print("   б) игнорируем Retry-After (паузы по формуле 0.5, 1, 2, 4):")
    start = now()
    await retry_request(
        client,
        "GET",
        f"{srv.url}/limited/ra_bad?times=2&retry_after=2",
        jitter=0,
        respect_retry_after=False,
    )
    t_bad = now() - start
    print(f"   итого {t_bad:.2f} с, запросов на сервере: {srv.hits['ra_bad']}")
    assert srv.hits["ra_ok"] == 3 and srv.hits["ra_bad"] == 5 and t_bad > t_ok


# а) Обе паузы ровно 2.00 с с пометкой «Retry-After» — код взял паузу из
#    заголовка, а не из формулы (по формуле было бы 0.5 и 1). Сервер открыл
#    второе окно лимита, мы его выждали, третья попытка прошла: 3 запроса,
#    ~4 с.
# б) Без заголовка паузы 0.5 и 1 короче окна: попытки 2 и 3 пришли, пока лимит
#    ещё действовал, и получили лишние 429 впустую. Попытка 4 (через 3.5 с)
#    открыла второе окно, и пришлось ждать по формуле ещё 4 с. Итог: 5 запросов
#    вместо 3 и почти вдвое дольше. Игнорирование Retry-After не ускоряет, а
#    замедляет и нагружает API; у LLM-провайдеров за это могут и заблокировать.


# ===========================================================================
# 5. Дедлайн поверх цикла
# ===========================================================================
def where(exc):
    """Последняя строка ЭТОГО файла в трассировке исключения."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == __file__]
    return f"строка {frames[-1].lineno}: {frames[-1].line}" if frames else "?"


async def task5(srv, client):
    print("\n5. Дедлайн поверх цикла")
    print("   без дедлайна было бы: паузы 1 + 2 + 4 + 8 = 15 с + 5 попыток × 0.2 с ≈ 16 с")
    for title, url in [
        ("а) дедлайн попал в паузу", f"{srv.url}/flaky/t5a?fail=99&delay=0.2"),
        ("б) дедлайн попал в попытку", f"{srv.url}/slow?delay=10"),
    ]:
        print(f"   {title}:")
        start = now()
        try:
            async with asyncio.timeout(3):
                await retry_request(
                    client,
                    "GET",
                    url,
                    attempts=5,
                    base=1,
                    cap=8,
                    jitter=0,
                    timeout=httpx.Timeout(30.0),
                )
        except TimeoutError as e:
            cancelled = e.__cause__ or e.__context__
            print(f"   → {type(e).__name__} через {now() - start:.2f} с")
            print(f"     внутри было: {type(cancelled).__name__}, {where(cancelled)}")


# а) Попытка 1 (0–0.2 с), пауза 1 с, попытка 2 (1.2–1.4 с), пауза 2 с должна была
#    закончиться в 3.4 с, но в 3.0 asyncio.timeout отменил задачу: CancelledError
#    вылетел из await asyncio.sleep(delay), и блок превратил его в TimeoutError.
#    Цикл не дошёл до попыток 3–5 — дедлайн ограничивает ВСЁ, включая паузы.
# б) Сервер отвечает через 10 с, а read-таймаут 30 с его не остановит. Через 3 с
#    отменилась сама попытка — CancelledError вылетел из await client.request,
#    httpx закрыл соединение. Важно: для сервера это обрыв клиента, а запрос
#    мог быть уже принят и выполняться. Отменённый по дедлайну POST — такое же
#    «не знаю», как ReadTimeout.


# ===========================================================================
# 6. Два таймаута одновременно
# ===========================================================================
async def task6(srv, client):
    print("\n6. Таймаут попытки (read=1) + общий дедлайн (asyncio.timeout(5))")
    per_attempt = httpx.Timeout(10.0, read=1.0)
    for title, url in [
        ("а) сервер молчит 2 с", f"{srv.url}/slow?delay=2"),
        (
            "б) сервер шлёт по байту раз в 0.5 с, 20 байт",
            f"{srv.url}/trickle?seconds=20&interval=0.5",
        ),
    ]:
        print(f"   {title}:")
        start = now()
        try:
            async with asyncio.timeout(5):
                await retry_request(
                    client, "GET", url, attempts=3, base=0.2, jitter=0, timeout=per_attempt
                )
        except httpx.ReadTimeout:
            print(f"   → сработал таймаут ПОПЫТКИ: ReadTimeout через {now() - start:.2f} с")
        except TimeoutError:
            print(f"   → сработал общий ДЕДЛАЙН: TimeoutError через {now() - start:.2f} с")


# а) Сервер молчит 2 с, а read=1: каждая попытка падает с ReadTimeout через 1 с.
#    3 попытки по 1 с + паузы 0.2 и 0.4 = ~3.6 с — меньше 5, и до дедлайна дело
#    не дошло: retry-цикл исчерпал попытки и отдал последний ReadTimeout.
# б) Байт приходит каждые 0.5 с — пауза между порциями ни разу не больше 1 с,
#    read-таймаут не срабатывает никогда, а весь ответ шёл бы 10 с. Первая же
#    попытка ещё читала, когда в 5.0 с сработал общий дедлайн.
#    Таймаут попытки ловит «сервер молчит», дедлайн — «сервер тянет время».
#    Нужны оба.


# ===========================================================================
# 7. Ключ идемпотентности
# ===========================================================================
async def place_order_buggy(client, url, attempts=3):
    for _ in range(attempts):
        key = str(uuid.uuid4())  # ОШИБКА: новый ключ на каждую попытку
        r = await client.post(url, json={"item": "book"}, headers={"Idempotency-Key": key})
        if r.status_code < 400:
            return r
        await asyncio.sleep(0.1)
    raise RuntimeError(f"заказ не подтверждён за {attempts} попыток")


async def place_order_fixed(client, url):
    key = str(uuid.uuid4())  # один раз, СНАРУЖИ цикла
    return await retry_request(
        client,
        "POST",
        url,
        base=0.1,
        jitter=0,
        json={"item": "book"},
        headers={"Idempotency-Key": key},
        log=lambda *_: None,
    )


async def task7(srv, client):
    print("\n7. Ключ идемпотентности (сервер создаёт заказ, а первые 2 ответа «теряет»)")
    r = await place_order_buggy(client, f"{srv.url}/orders/bug?fail=2")
    print(
        f"   uuid4 внутри цикла:  запросов {srv.hits['bug']}, разных ключей "
        f"{len(set(srv.keys['bug']))} → создано заказов: {r.json()['orders_created']}"
    )
    r = await place_order_fixed(client, f"{srv.url}/orders/fix?fail=2")
    print(
        f"   uuid4 снаружи цикла: запросов {srv.hits['fix']}, разных ключей "
        f"{len(set(srv.keys['fix']))} → создано заказов: {r.json()['orders_created']}"
    )
    assert srv.keys and len(set(srv.keys["bug"])) == 3 and len(set(srv.keys["fix"])) == 1

    r1 = await place_order_buggy(client, f"{srv.url}/orders/happy_bug")
    r2 = await place_order_fixed(client, f"{srv.url}/orders/happy_fix")
    print(
        f"   счастливый путь (без сбоев): с ошибкой — {r1.json()['orders_created']} заказ, "
        f"исправленный — {r2.json()['orders_created']} заказ"
    )


# С ошибкой: 3 попытки — 3 ключа, сервер видит три РАЗНЫЕ операции и честно
#   создаёт 3 заказа (первые два — с «потерянным» ответом). Исправленный: один
#   ключ на все 3 попытки, сервер узнаёт повтор — 1 заказ.
# Почему тест на счастливом пути не ловит: без сбоев попытка ровно одна, и
#   ключ генерируется один раз в обоих вариантах — результат одинаковый
#   (1 и 1). Ошибка проявляется только при повторе, то есть только во время
#   сбоя. Ловить её надо тестом с сервером, который «теряет» ответ, — как здесь.


# ===========================================================================
# 8. Проверка чисел про urllib3 (и tenacity)
# ===========================================================================
def task8_sync(srv_url, srv):
    from importlib.metadata import version

    import requests
    import tenacity
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry

    print(f"\n8. urllib3 {urllib3.__version__}, tenacity {version('tenacity')}")
    r = Retry()
    print("   DEFAULT_ALLOWED_METHODS:", sorted(Retry.DEFAULT_ALLOWED_METHODS))
    print(
        f"   backoff_factor={r.backoff_factor}  backoff_jitter={r.backoff_jitter}  "
        f"backoff_max={r.backoff_max}  respect_retry_after_header={r.respect_retry_after_header}"
    )

    r = Retry(total=5, backoff_factor=1)
    backoffs = []
    for _ in range(4):
        r = r.increment(method="GET", url="/x", error=urllib3.exceptions.ConnectTimeoutError())
        backoffs.append(r.get_backoff_time())
    print("   get_backoff_time после 1-й, 2-й, 3-й, 4-й ошибки (backoff_factor=1):", backoffs)

    with requests.Session() as s:
        s.trust_env = False
        s.mount(
            "http://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3, backoff_factor=1, status_forcelist=[503], raise_on_status=False
                )
            ),
        )
        resp = s.get(f"{srv_url}/flaky/u3?fail=3", timeout=5)
    times = srv.times["u3"]
    gaps = [round(b - a, 2) for a, b in pairwise(times)]
    print(
        f"   реальный прогон через requests: статус {resp.status_code}, "
        f"запросов {len(times)}, паузы между ними: {gaps}"
    )
    assert gaps[0] < 0.1 and abs(gaps[1] - 2) < 0.3 and abs(gaps[2] - 4) < 0.3

    random.seed(8)
    st = SimpleNamespace(attempt_number=3)
    ej = tenacity.wait_exponential_jitter(initial=1, jitter=1)
    rx = tenacity.wait_random_exponential(multiplier=1, max=60)
    a = [ej(st) for _ in range(10_000)]
    b = [rx(st) for _ in range(10_000)]
    print(
        f"   tenacity, 3-я попытка: wait_exponential_jitter ∈ [{min(a):.2f}, {max(a):.2f}]"
        f" | wait_random_exponential ∈ [{min(b):.2f}, {max(b):.2f}]"
    )


# urllib3: методы, backoff_factor = 0, backoff_jitter = 0.0, backoff_max = 120 —
#   совпало с уроком. Версия — в первой строке задания 8 (урок проверял 2.7.0;
#   при подготовке я проверял на 2.8.0 — значения те же). Если у тебя другая
#   версия и другие числа — правильные твои.
# get_backoff_time: после первой ошибки 0, потом 2, 4, 8 = backoff_factor ×
#   2^(n−1), а первый повтор всегда уходит без паузы — подтвердилось и на
#   реальном прогоне: паузы между запросами ~0, ~2, ~4 с. Обрати внимание:
#   при backoff_factor=1 вторая пауза не 1, а 2 — формула сдвинута.
# tenacity: wait_exponential_jitter на 3-й попытке даёт [4, 5] — экспонента
#   1·2² = 4 плюс до 1 с сверху, то есть аддитивный джиттер. wait_random_exponential
#   даёт [0, 4] — full jitter. Имена врут ровно так, как сказано в уроке.


# ===========================================================================
# 9. Предохранитель
# ===========================================================================
class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, threshold=3, open_seconds=1.0, clock=time.monotonic):
        self.threshold, self.open_seconds, self.clock = threshold, open_seconds, clock
        self.state, self.failures, self.opened_at = "closed", 0, 0.0
        self.probe_in_flight = False

    def _before_call(self):
        if self.state == "open":
            if self.clock() - self.opened_at < self.open_seconds:
                raise CircuitOpenError("предохранитель открыт")
            self.state = "half-open"
        if self.state == "half-open":
            if self.probe_in_flight:
                raise CircuitOpenError("пробник уже в полёте")
            self.probe_in_flight = True

    def _open(self):
        self.state, self.opened_at = "open", self.clock()

    async def call(self, func, *args, **kwargs):
        self._before_call()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            self.probe_in_flight = False
            self.failures += 1
            if self.state == "half-open" or self.failures >= self.threshold:
                self._open()
            raise
        self.state, self.failures, self.probe_in_flight = "closed", 0, False
        return result


async def task9(srv, client):
    print("\n9. Предохранитель (порог 3 ошибки, открыт 1 с)")
    breaker = CircuitBreaker(threshold=3, open_seconds=1.0)
    url = f"{srv.url}/breaker/svc"
    srv.alive["svc"] = False

    async def call_service():
        r = await client.get(url)
        r.raise_for_status()
        return r.status_code

    async def attempt(label):
        before = srv.hits["svc"]
        try:
            result = f"ответ {await breaker.call(call_service)}"
        except CircuitOpenError as e:
            result = f"отказ без запроса ({e})"
        except httpx.HTTPStatusError as e:
            result = f"ошибка {e.response.status_code}"
        touched = srv.hits["svc"] - before
        print(
            f"   {label:28} → {result:42} | состояние: {breaker.state:9} | "
            f"сервер получил: {touched} (всего {srv.hits['svc']})"
        )
        return touched

    for i in range(3):
        await attempt(f"сервис лежит, вызов {i + 1}")
    assert breaker.state == "open"
    for i in range(4):
        assert await attempt(f"открыт, вызов {i + 1}") == 0
    await asyncio.sleep(1.1)
    await attempt("пауза прошла, пробник")  # сервис всё ещё лежит
    assert breaker.state == "open"
    await asyncio.sleep(1.1)
    srv.alive["svc"] = True
    await attempt("сервис ожил, пробник")
    assert breaker.state == "closed"
    for i in range(2):
        await attempt(f"закрыт, вызов {i + 1}")
    assert srv.hits["svc"] == 3 + 1 + 1 + 2


# 3 ошибки подряд — предохранитель открылся. 4 следующих вызова получили отказ
#   мгновенно, и счётчик на сервере не сдвинулся (получил: 0) — упавший сервис
#   не трогали. Через 1 с (half-open) пропущен один пробник; сервис ещё лежал —
#   пробник упал, и предохранитель снова открылся на 1 с. После оживления
#   пробник прошёл — состояние closed, дальше запросы идут как обычно. Всего
#   сервер увидел 7 запросов: 3 + 1 + 1 + 2, а не 3 + 4 + … при слепых повторах.


# ===========================================================================
# 10. Устно, потом письменно
# ===========================================================================
# Почему Timeout опаснее ConnectionError: ConnectionError — «не дошло», сервер
#   ничего не сделал, повтор безопасен. Timeout на чтении — «не знаю»: запрос
#   ушёл, сервер мог его выполнить и не успеть ответить. Для POST повтор после
#   таймаута может создать дубль, а не повторять — потерять операцию.
# Почему 5 попыток могут быть хуже 2: с экспонентой 5 попыток — это 1+2+4+8 =
#   15 с пауз. Если эндпоинт обещает ответ за 3 с, пользователь давно ушёл, а
#   мы продолжаем грузить упавшую зависимость. Считать надо бюджет времени
#   (дедлайн поверх цикла, задание 5), а не число попыток.
# Усиление повторов: если 3 слоя сервисов повторяют по 3 раза, нижний получает
#   3 × 3 × 3 = 27 запросов на один пользовательский — именно тогда, когда он и
#   так не справляется. Повторять должен один слой, обычно самый внешний;
#   остальные пробрасывают ошибку. Либо общий бюджет повторов на весь трафик.
# Ключ идемпотентности снаружи цикла: сервер узнаёт повтор по ключу. Новый
#   ключ в каждой попытке — это для сервера новая операция, и защита
#   выключена ровно при повторе, когда она нужна (задание 7: 3 заказа против 1).
# Предохранитель против backoff: backoff — решение одного клиента «подождать
#   подольше»; запросы к упавшему сервису всё равно идут, просто реже.
#   Предохранитель перестаёт ходить к сервису совсем: после порога ошибок
#   все вызовы отказывают мгновенно, не создавая нагрузки (задание 9: 0
#   запросов на сервере), и только редкий пробник проверяет, ожил ли он.
#   Это разрывает петлю «перегружен → таймауты → повторы → ещё больше
#   перегружен» и даёт сервису встать.


async def main():
    task1()
    timeout = httpx.Timeout(10.0)
    async with (
        ControlServer() as srv,
        httpx.AsyncClient(trust_env=False, timeout=timeout) as client,
    ):
        await task2(srv, client)
        await task3()
        await task3_server_check()
        await task4(srv, client)
        await task5(srv, client)
        await task6(srv, client)
        await task7(srv, client)
        await asyncio.to_thread(task8_sync, srv.url, srv)
        await task9(srv, client)
    print(f"\nвсё заняло {now() - T0:.1f} с")


if __name__ == "__main__":
    asyncio.run(main())
