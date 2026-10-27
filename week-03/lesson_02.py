import asyncio
import gc
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import httpx

BASE = os.environ.get("HTTPBIN_URL", "https://httpbin.org").rstrip("/")
DELAY_URL = f"{BASE}/delay/1"
N = 10


def stamp(start):
    """Секунды от старта, для отметок времени."""
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# 1. Корутина без await
# ---------------------------------------------------------------------------
async def say(name):
    print(f"   say({name!r}) выполнилась")


async def task1():
    print("\n1. Корутина без await")
    coro = say("без await")          # объект создан, тело НЕ выполнилось
    print("   после say('без await'):", coro)
    del coro                         # объект выброшен → предупреждение
    gc.collect()
    await say("с await")
# Вывод: строки «say('без await') выполнилась» нет вообще; вместо неё —
#   RuntimeWarning: coroutine 'say' was never awaited (в stderr, в момент,
#   когда объект корутины уничтожается). Вызов async-функции лишь создаёт
#   объект корутины — «план работы»; выполнить его может только await (или
#   asyncio.run / create_task). Это тот же класс ошибки, что int(raw) без
#   присваивания: выражение вычислено, результат выброшен, программа идёт
#   дальше, и ошибки нет.


# ---------------------------------------------------------------------------
# 2. Последовательно против вместе
# ---------------------------------------------------------------------------
async def fetch(client, url):
    r = await client.get(url)
    r.raise_for_status()
    return r.status_code


async def task2(client):
    print("\n2. Последовательно / gather / TaskGroup")

    start = time.perf_counter()
    for _ in range(N):
        await fetch(client, DELAY_URL)
    t_seq = stamp(start)

    start = time.perf_counter()
    await asyncio.gather(*[fetch(client, DELAY_URL) for _ in range(N)])
    t_gather = stamp(start)

    start = time.perf_counter()
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(client, DELAY_URL)) for _ in range(N)]
    assert all(t.result() == 200 for t in tasks)
    t_tg = stamp(start)

    print(f"   в цикле с await: {t_seq:.2f} с")
    print(f"   gather:          {t_gather:.2f} с")
    print(f"   TaskGroup:       {t_tg:.2f} с")
# В цикле каждый await ждёт ответа, и только потом цикл создаёт следующий
#   запрос: 10 × (1 с задержки + сеть) ≈ 10+ с. gather и TaskGroup сначала
#   запускают все 10 корутин; каждая отправляет запрос и на await отдаёт
#   управление циклу, тот запускает следующую. Через мгновение в полёте все
#   десять, и ждут они одновременно: общее время ≈ один самый медленный
#   запрос ≈ 1+ с.


# ---------------------------------------------------------------------------
# 3–4. Блокирующий вызов в корутине
# ---------------------------------------------------------------------------
async def run_mixed(client, slow_kind):
    """10 задач, у задачи 0 — задержка 3 с одного из трёх видов."""
    start = time.perf_counter()
    finished = {}

    async def job(i):
        if i == 0:
            if slow_kind == "time.sleep":
                time.sleep(3)                         # блокирует ВЕСЬ цикл
            elif slow_kind == "asyncio.sleep":
                await asyncio.sleep(3)                # отдаёт управление
            elif slow_kind == "to_thread":
                await asyncio.to_thread(time.sleep, 3)  # блокирует только поток
        await fetch(client, DELAY_URL)
        finished[i] = stamp(start)

    await asyncio.gather(*[job(i) for i in range(N)])
    total = stamp(start)
    others = statistics.median(finished[i] for i in range(1, N))
    print(f"   {slow_kind:14}: всего {total:.2f} с | "
          f"задача 0 — {finished[0]:.2f} с | остальные 9 — медиана {others:.2f} с")
    return total, others


async def task3_4(client):
    print("\n3. Блокирующий вызов в корутине  /  4. asyncio.to_thread")
    await run_mixed(client, "time.sleep")
    await run_mixed(client, "asyncio.sleep")
    await run_mixed(client, "to_thread")
# Общее время во всех трёх вариантах ≈ 4 с: задаче 0 нужно 3 с паузы и 1 с
#   запроса, а gather ждёт самую медленную. Разница видна по ОСТАЛЬНЫМ девяти:
# time.sleep(3): задача 0 запускается первой и держит единственный поток
#   цикла 3 секунды. Никто другой не может даже отправить свой запрос —
#   остальные девять стартуют только после паузы и заканчивают через ~4 с,
#   а не через ~1. Одна корутина остановила всех.
# asyncio.sleep(3): задача 0 на await отдаёт управление, и цикл сразу
#   запускает остальные — они заканчивают через ~1 с, как будто паузы нет.
# 4. asyncio.to_thread(time.sleep, 3): блокирующий вызов уехал в отдельный
#   поток из пула, а корутина ждёт его на await — цикл свободен, остальные
#   девять снова ~1 с. Снаружи то же, что asyncio.sleep, но механизм другой:
#   asyncio.sleep — это таймер внутри цикла, без потоков, почти бесплатно.
#   to_thread занимает настоящий поток ОС (пул по умолчанию —
#   min(32, ядра + 4) потоков). Это выход для ЧУЖОГО блокирующего кода
#   (синхронная библиотека, чтение файла), а не замена asyncio.sleep:
#   тысячу таких вызовов пул одновременно не потянет.


# ---------------------------------------------------------------------------
# 5. Semaphore
# ---------------------------------------------------------------------------
async def task5(client):
    print("\n5. Semaphore")
    total_requests, limit = 30, 5
    semaphore = asyncio.Semaphore(limit)
    in_flight = max_in_flight = 0

    async def fetch_limited():
        nonlocal in_flight, max_in_flight
        async with semaphore:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                return await fetch(client, DELAY_URL)
            finally:
                in_flight -= 1

    expected = total_requests / limit * 1.0
    print(f"   прогноз с семафором: {total_requests} / {limit} = "
          f"{total_requests // limit} волн × ~1 с ≈ {expected:.0f}+ с")

    start = time.perf_counter()
    await asyncio.gather(*[fetch_limited() for _ in range(total_requests)])
    t_sem = stamp(start)

    start = time.perf_counter()
    await asyncio.gather(*[fetch(client, DELAY_URL) for _ in range(total_requests)])
    t_free = stamp(start)

    print(f"   с семафором на {limit}: {t_sem:.2f} с (одновременно было максимум {max_in_flight})")
    print(f"   без семафора:     {t_free:.2f} с")
    assert max_in_flight == limit
# С семафором внутри блока не больше 5 запросов: 30 запросов идут шестью
#   волнами по ~1 с → ≈ 6 с, что и показал замер. Без семафора все 30 летят
#   сразу → ≈ 1 с.
# Зачем ограничивать, если без него быстрее: на 30 запросах к httpbin это
#   работает, на 10 000 чанков к API эмбеддингов — нет. Все 10 000 полетят
#   разом: API ответит 429 почти на всё (лимиты считаются в запросах и токенах
#   в минуту), мы упрёмся в лимит открытых соединений и файловых
#   дескрипторов, а у LLM-провайдера — ещё и в счёт. Семафор задаёт скорость,
#   которую выдерживает другая сторона; число берут из лимитов API.


# ---------------------------------------------------------------------------
# 6. asyncio.timeout против timeout в httpx
# ---------------------------------------------------------------------------
TRICKLE_SECONDS = 10


async def trickle_handler(reader, writer):
    """Отвечает на любой запрос телом из 10 байт — по байту раз в секунду."""
    try:
        await reader.readuntil(b"\r\n\r\n")             # дочитали заголовки запроса
        writer.write(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: text/plain\r\n"
                     b"Content-Length: %d\r\n"
                     b"Connection: close\r\n\r\n" % TRICKLE_SECONDS)
        await writer.drain()
        for _ in range(TRICKLE_SECONDS):
            await asyncio.sleep(1)
            writer.write(b"x")
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass                                             # клиент ушёл — это нормально
    finally:
        writer.close()


async def task6():
    print("\n6. asyncio.timeout против timeout в httpx")
    server = await asyncio.start_server(trickle_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"   локальный сервер: {url} — {TRICKLE_SECONDS} байт, по одному в секунду")

    async with server, httpx.AsyncClient(trust_env=False) as client:
        start = time.perf_counter()
        try:
            r = await client.get(url, timeout=2)
            print(f"   httpx timeout=2: УСПЕХ через {stamp(start):.1f} с, "
                  f"тело {r.content!r} — таймаут не сработал")
        except httpx.TimeoutException as e:
            print(f"   httpx timeout=2: {type(e).__name__} через {stamp(start):.1f} с")

        start = time.perf_counter()
        try:
            async with asyncio.timeout(3):
                await client.get(url, timeout=2)
            print(f"   asyncio.timeout(3): успех через {stamp(start):.1f} с")
        except TimeoutError:
            print(f"   asyncio.timeout(3): TimeoutError через {stamp(start):.1f} с")
# timeout=2 в httpx (как и в requests) — это предел ожидания ОДНОЙ порции
#   данных: «между двумя пакетами не больше 2 с». Сервер шлёт байт каждую
#   секунду, пауза ни разу не превышает 2 с — и запрос благополучно живёт
#   все 10 с. Медленный (или злонамеренный) сервер может так держать
#   соединение сколько угодно.
# asyncio.timeout(3) — срок на весь блок целиком, по часам: через 3 с
#   задача получает CancelledError, где бы она ни была, и блок бросает
#   TimeoutError. Сработает всегда, как бы сервер ни дробил ответ.


# ---------------------------------------------------------------------------
# 7. Ошибки: gather против TaskGroup
# ---------------------------------------------------------------------------
async def worker(i, done):
    try:
        await asyncio.sleep(0.1 * (i + 1))
        if i == 1:
            raise ValueError(f"задача {i} упала")
        done.append(i)
        print(f"      задача {i} завершилась")
        return i
    except asyncio.CancelledError:
        print(f"      задача {i} ОТМЕНЕНА")
        raise


async def task7():
    print("\n7. gather против TaskGroup (5 задач, задача 1 падает на 0.2 с)")

    print("   а) gather без return_exceptions")
    done = []
    try:
        await asyncio.gather(*[worker(i, done) for i in range(5)])
    except ValueError as e:
        print(f"   → исключение {e!r}; к этому моменту завершились: {done}")
    await asyncio.sleep(0.5)          # даём «осиротевшим» задачам доработать
    print(f"   → а через 0.5 с завершились уже: {done} — их результаты никто не забрал")

    print("   б) gather с return_exceptions=True")
    done = []
    results = await asyncio.gather(*[worker(i, done) for i in range(5)],
                                   return_exceptions=True)
    print(f"   → вернулось: {results}")

    print("   в) TaskGroup")
    done = []
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(5):
                tg.create_task(worker(i, done))
    except* ValueError as eg:
        print(f"   → ExceptionGroup: {eg.exceptions}; завершились: {done}")

    print("   г) TaskGroup: что видно снаружи без except*")
    done = []
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(5):
                tg.create_task(worker(i, done))
    except Exception as e:
        print(f"   → {type(e).__name__}: {e}")
# а) gather бросил ValueError сразу, на 0.2 с, когда успела только задача 0.
#    Но задачи 2, 3, 4 НЕ остановились: они дорабатывают в фоне, и их
#    результаты (и возможные ошибки) теряются — никто их больше не ждёт.
# б) return_exceptions=True: исключение вернулось как значение на своём
#    месте в списке, остальные четыре отработали — удобно, когда частичный
#    провал допустим.
# в) TaskGroup при первой ошибке ОТМЕНИЛ задачи 2, 3, 4 (видно «ОТМЕНЕНА»),
#    дождался их отмены и только потом бросил ExceptionGroup. После выхода
#    из блока не остаётся ни одной висящей задачи.
# г) Без except* снаружи видна ExceptionGroup — обёртка над всеми ошибками
#    группы: их может быть несколько, если упали сразу несколько задач.
# Для сервиса TaskGroup лучше: запрос пользователя упал — значит, вся работа
#   по нему отменена, а не продолжает тратить соединения и деньги на API в
#   фоне; ошибки не теряются.


# ---------------------------------------------------------------------------
# 8. as_completed
# ---------------------------------------------------------------------------
async def task8():
    print("\n8. as_completed")
    random.seed(42)
    delays = [round(random.uniform(0.1, 2.0), 2) for _ in range(10)]
    start = time.perf_counter()

    async def job(i, delay):
        await asyncio.sleep(delay)
        return i, delay

    tasks = [asyncio.create_task(job(i, d)) for i, d in enumerate(delays)]
    order = []
    for next_done in asyncio.as_completed(tasks):
        i, delay = await next_done
        order.append(i)
        print(f"   {stamp(start):5.2f} с: задача {i} (задержка {delay})")
    print("   порядок запуска:   ", list(range(10)))
    print("   порядок готовности:", order)
    assert order != list(range(10))
    assert order == sorted(range(10), key=lambda i: delays[i])
# Задачи приходят в порядке готовности — по возрастанию задержки, а не в
#   порядке запуска. gather вернул бы всё разом через 2 с (по самой медленной)
#   и в порядке аргументов.
# as_completed нужнее, когда результат полезен сразу: показывать пользователю
#   найденные документы по мере прихода, писать на диск уже скачанное,
#   досрочно остановиться, когда набралось достаточно ответов.


# ---------------------------------------------------------------------------
# 9. Вычисления
# ---------------------------------------------------------------------------
def sum_squares(n=10_000_000):
    return sum(i * i for i in range(n))


async def task9():
    print("\n9. Вычисления: 4 задачи sum_squares(10_000_000)")
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"   Python {sys.version.split()[0]}, GIL включён: {gil}, ядер: {os.cpu_count()}")
    loop = asyncio.get_running_loop()

    start = time.perf_counter()
    for _ in range(4):
        sum_squares()
    t_seq = stamp(start)

    async def as_coroutine():
        return sum_squares()                 # ни одного await внутри

    start = time.perf_counter()
    await asyncio.gather(*[as_coroutine() for _ in range(4)])
    t_gather = stamp(start)

    start = time.perf_counter()
    await asyncio.gather(*[asyncio.to_thread(sum_squares) for _ in range(4)])
    t_thread = stamp(start)

    with ProcessPoolExecutor(max_workers=4) as pool:
        await loop.run_in_executor(pool, sum_squares, 10)   # прогрев: запуск процессов
        start = time.perf_counter()
        await asyncio.gather(*[loop.run_in_executor(pool, sum_squares)
                               for _ in range(4)])
        t_proc = stamp(start)

    print(f"   последовательно:     {t_seq:.2f} с")
    print(f"   gather:              {t_gather:.2f} с")
    print(f"   asyncio.to_thread:   {t_thread:.2f} с")
    print(f"   ProcessPoolExecutor: {t_proc:.2f} с")
# Последовательно — 4 × время одной задачи, база для сравнения.
# gather — столько же: в sum_squares нет ни одного await, корутина ни разу не
#   отдаёт управление, и цикл выполняет четыре задачи по очереди. asyncio
#   ускоряет ожидание, а здесь ждать нечего — только считать.
# to_thread — столько же (или чуть хуже из-за переключений): потоков четыре,
#   но GIL пускает исполнять байткод Python только один поток за раз. Потоки
#   по очереди забирают GIL, суммарная работа та же.
# ProcessPoolExecutor — примерно в 4 раза быстрее (если ядер ≥ 4): у каждого
#   процесса свой интерпретатор и свой GIL, и четыре задачи считаются
#   по-настоящему параллельно на разных ядрах. Плата — запуск процессов
#   (его мы вынесли в прогрев) и пересылка аргументов и результатов между
#   процессами; для коротких задач она съест выигрыш.


# ---------------------------------------------------------------------------
# 10. Устно, потом письменно
# ---------------------------------------------------------------------------
# Почему вызов корутины её не запускает: async def при вызове возвращает
#   объект корутины — приостановленное вычисление. Выполнять его начинает
#   только цикл событий: через await, create_task/TaskGroup или asyncio.run.
#   Без этого объект просто выбрасывается (задание 1: RuntimeWarning, тело не
#   выполнилось).
# Почему два await подряд не дают конкурентности: await значит «остановись
#   здесь, пока не будет результата». Второй await начинается только после
#   первого, и ожидания складываются (задание 2: ~10 с). Конкурентность
#   появляется, когда несколько задач запущены ДО ожидания — gather, TaskGroup.
# requests.get в асинхронном обработчике: это блокирующий вызов, он не
#   отдаёт управление циклу. Пока он ждёт ответа, весь сервис в этом потоке
#   стоит — не обрабатываются ни другие запросы, ни новые соединения
#   (задание 3: девять задач ждали чужой time.sleep). Под нагрузкой
#   асинхронный сервис становится последовательным; лечится httpx.AsyncClient
#   или asyncio.to_thread.
# asyncio.timeout против timeout в httpx: timeout httpx ограничивает каждую
#   фазу отдельно — соединение, ожидание очередной порции данных, запись,
#   пул. Сервер, присылающий по байту раз в секунду, его не нарушает
#   никогда. asyncio.timeout — общий срок на весь блок по часам: по
#   истечении задача отменяется, что бы она ни делала (задание 6: 10 с
#   против 3 с).


async def main():
    print("httpbin:", BASE)
    await task1()
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=30.0)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await fetch(client, f"{BASE}/get")      # прогрев: соединение и TLS
        await task2(client)
        await task3_4(client)
        await task5(client)
    await task6()
    await task7()
    await task8()
    await task9()


if __name__ == "__main__":         # обязательно: ProcessPoolExecutor на macOS
    asyncio.run(main())            # запускает дочерние процессы через spawn