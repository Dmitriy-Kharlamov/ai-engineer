import hashlib
import os
import time
from pathlib import Path

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout

# Если httpbin.org недоступен — можно указать другой адрес:
#   HTTPBIN_URL=https://другой-адрес python week-03/lesson_01.py
BASE = os.environ.get("HTTPBIN_URL", "https://httpbin.org").rstrip("/")
HERE = Path(__file__).parent
print("httpbin:", BASE)


def best_of(func, runs=5):
    """Минимальное время из runs прогонов."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


# ---------------------------------------------------------------------------
# 1. Настоящий API: GitHub
# ---------------------------------------------------------------------------
print("\n1. GitHub API")
try:
    r = requests.get("https://api.github.com/repos/Dmitriy-Kharlamov/ai-engineer", timeout=10)
    print("status:", r.status_code)
    print("Content-Type:", r.headers.get("Content-Type"))
    if r.ok:
        repo = r.json()
        print("звёзд:", repo["stargazers_count"])
        print("создан:", repo["created_at"])
        print("размер:", repo["size"], "КБ")
    else:
        print("ответ:", r.json().get("message"))
    print("X-RateLimit-Limit:    ", r.headers.get("X-RateLimit-Limit"))
    print("X-RateLimit-Remaining:", r.headers.get("X-RateLimit-Remaining"))
    print("X-RateLimit-Reset:    ", r.headers.get("X-RateLimit-Reset"))
except RequestException as e:
    print("GitHub недоступен:", e)
# X-RateLimit-Limit — сколько запросов разрешено за окно (без токена GitHub
#   даёт 60 в час на IP-адрес, с токеном — 5000).
# X-RateLimit-Remaining — сколько осталось в текущем окне; каждый запуск
#   скрипта уменьшает его на 1.
# X-RateLimit-Reset — когда окно обнулится (Unix-время в секундах).
# Когда Remaining дойдёт до 0, GitHub начнёт отвечать 403/429, пока не наступит
#   Reset. Нормальный клиент смотрит на эти заголовки и сам притормаживает,
#   а не ждёт отказа. 404 вместо 200 значит, что репозиторий приватный:
#   без токена GitHub не признаётся, что он вообще существует.


# ---------------------------------------------------------------------------
# 2. Коды состояния
# ---------------------------------------------------------------------------
print("\n2. Коды состояния")
print(f"{'код':>4} | {'status':>6} | {'ok':5} | {'тело':>5} | raise_for_status()")
print("-" * 80)
for code in [200, 201, 204, 301, 400, 401, 403, 404, 429, 500]:
    r = requests.get(f"{BASE}/status/{code}", timeout=10, allow_redirects=False)
    try:
        r.raise_for_status()
        raised = "не бросил"
    except HTTPError as e:
        raised = f"HTTPError: {e}"
    print(f"{code:>4} | {r.status_code:>6} | {str(r.ok):5} | {len(r.content):>5} | {raised}")
# r.ok и raise_for_status() смотрят на одно и то же: код < 400 — успех.
#   Поэтому 301 тоже ok=True и не бросает: перенаправление — не ошибка.
#   У 204 тело всегда пустое — «No Content» по определению.
# allow_redirects=False в цикле — чтобы 301 показался как есть (см. ниже).

print("\n301 с allow_redirects=False:")
r = requests.get(f"{BASE}/status/301", timeout=10, allow_redirects=False)
print("   status:", r.status_code, "| Location:", r.headers.get("Location"))
print("301 по умолчанию (allow_redirects=True):")
r = requests.get(f"{BASE}/status/301", timeout=10)
print("   status:", r.status_code, "| итоговый URL:", r.url)
print("   цепочка history:", [(h.status_code, h.headers.get("Location")) for h in r.history])
# Без перехода requests возвращает сам ответ 301: тело пустое, а адрес, куда
#   идти, лежит в заголовке Location (у httpbin — /redirect/1).
# По умолчанию requests сам идёт по Location — и дальше, если там снова
#   перенаправление (/redirect/1 → /get), — и возвращает финальный ответ 200.
#   Промежуточные ответы сохраняются в r.history.


# ---------------------------------------------------------------------------
# 3. timeout — главное задание
# ---------------------------------------------------------------------------
print("\n3. timeout")
start = time.perf_counter()
try:
    requests.get(f"{BASE}/delay/5", timeout=2)
except Timeout as e:
    print(f"timeout=2:  {type(e).__name__} через {time.perf_counter() - start:.1f} с")
    print("   ", e)

start = time.perf_counter()
r = requests.get(f"{BASE}/delay/5", timeout=10)
print(f"timeout=10: status {r.status_code} через {time.perf_counter() - start:.1f} с")

start = time.perf_counter()
r = requests.get(f"{BASE}/delay/3")  # БЕЗ timeout — только для демонстрации
print(f"без timeout: status {r.status_code} через {time.perf_counter() - start:.1f} с")
# timeout=2: через ~2 с requests сдался и бросил ReadTimeout — соединение
#   установилось, но ответ не пришёл за отведённое время.
# timeout=10: сервер ответил через ~5 с, это меньше 10 — запрос прошёл.
# Без timeout: ждали ровно столько, сколько отвечал сервер, ~3 с. Если бы
#   сервер не ответил НИКОГДА (завис, сеть потеряла пакет и не сообщила),
#   запрос висел бы бесконечно: у requests нет значения по умолчанию.
#   В сервисе такой запрос навсегда занимает поток; когда зависнут все
#   потоки, сервис перестанет отвечать, хотя ни одной ошибки в логах не будет.
# timeout=(3, 30) — 3 с на установку соединения и 30 с на ожидание данных:
#   так и надо для LLM API, где соединиться быстро, а генерация долгая.


# ---------------------------------------------------------------------------
# 4. params против json против data
# ---------------------------------------------------------------------------
print("\n4. params / json / data")
payload = {"q": "rag", "k": "5"}

r = requests.get(f"{BASE}/get", params=payload, timeout=10)
body = r.json()
print("params=: url  =", body["url"])
print("         args =", body["args"])

r = requests.post(f"{BASE}/post", json=payload, timeout=10)
body = r.json()
print("json=:   json =", body["json"], "| form =", body["form"])
print("         Content-Type:", body["headers"]["Content-Type"])

r = requests.post(f"{BASE}/post", data=payload, timeout=10)
body = r.json()
print("data=:   form =", body["form"], "| json =", body["json"])
print("         Content-Type:", body["headers"]["Content-Type"])
assert r.json()["form"] == payload
# params= — данные идут в строку запроса (?q=rag&k=5), тела нет; httpbin
#   показывает их в args.
# json=  — тело в формате JSON, requests сам сериализует и ставит
#   Content-Type: application/json; httpbin разобрал его в поле json.
# data=  — со словарём тело кодируется как HTML-форма (q=rag&k=5),
#   Content-Type: application/x-www-form-urlencoded; это поле form, а json
#   пустой (None). API, которое ждёт JSON, такое тело не поймёт → 400/422.


# ---------------------------------------------------------------------------
# 5. Session против отдельных вызовов
# ---------------------------------------------------------------------------
print("\n5. Session")
URL = f"{BASE}/get"
N = 20


def separate_calls():
    for _ in range(N):
        requests.get(URL, timeout=10).raise_for_status()


def with_session():
    with requests.Session() as s:
        for _ in range(N):
            s.get(URL, timeout=10).raise_for_status()


t_separate = best_of(separate_calls)
t_session = best_of(with_session)
print(f"requests.get × {N}: {t_separate:.2f} с")
print(f"Session × {N}:      {t_session:.2f} с")
print(f"отдельные / Session: {t_separate / t_session:.1f}")
# requests.get каждый раз открывает новое TCP-соединение, проходит
#   TLS-рукопожатие (обмен ключами — несколько поездок туда и обратно до
#   сервера) и закрывает соединение. Session держит соединение открытым
#   (keep-alive) и шлёт следующий запрос по уже установленному каналу.
#   Чем дальше сервер (больше задержка сети), тем заметнее выигрыш: на
#   httpbin.org за океаном каждое рукопожатие — сотни миллисекунд.


# ---------------------------------------------------------------------------
# 6. Исключения
# ---------------------------------------------------------------------------
print("\n6. Исключения")
try:
    requests.get("https://this-host-does-not-exist-12345.com", timeout=5)
except RequestsConnectionError as e:
    print("ConnectionError:", e)

try:
    requests.get(f"{BASE}/delay/5", timeout=1)
except Timeout as e:
    print("Timeout:", e)

try:
    r = requests.get(f"{BASE}/status/500", timeout=10)
    r.raise_for_status()
except HTTPError as e:
    print("HTTPError:", e)
    print("   код из e.response:", e.response.status_code)
# ConnectionError и Timeout — ответа НЕ было. ConnectionError: не нашли хост
#   (DNS) или нас не пустили — запрос, скорее всего, не дошёл. Timeout:
#   запрос мог дойти и даже выполниться, просто ответ не успел вернуться.
#   Для POST это значит «неизвестно, создалась ли запись»: повторишь
#   вслепую — можешь создать дубль. Нужно проверять или слать ключ
#   идемпотентности.
# HTTPError — ответ БЫЛ: сервер получил запрос, обработал и явно сказал, что
#   не так (код в e.response.status_code). Состояние известно: на 4xx/5xx
#   запись не создана, и решение о повторе можно принять по коду.


# ---------------------------------------------------------------------------
# 7. Стриминг
# ---------------------------------------------------------------------------
print("\n7. Стриминг")
BYTES_URL = f"{BASE}/bytes/1048576?seed=42"
plain_path = HERE / "bytes_plain.bin"
stream_path = HERE / "bytes_stream.bin"

r = requests.get(BYTES_URL, timeout=30)
r.raise_for_status()
plain_path.write_bytes(r.content)  # всё тело уже в памяти

with requests.get(BYTES_URL, stream=True, timeout=30) as r:
    r.raise_for_status()
    with open(stream_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):  # по 8 КБ за раз
            f.write(chunk)

h_plain = hashlib.sha256(plain_path.read_bytes()).hexdigest()
h_stream = hashlib.sha256(stream_path.read_bytes()).hexdigest()
print("размер:", plain_path.stat().st_size, "байт")
print("sha256 обычный:", h_plain[:16], "…")
print("sha256 stream: ", h_stream[:16], "…")
assert h_plain == h_stream

plain_path.unlink()
stream_path.unlink()

with requests.get(f"{BASE}/stream/20", stream=True, timeout=30) as r:
    lines = [line for line in r.iter_lines(decode_unicode=True) if line]
print("stream/20: строк пришло:", len(lines))
assert len(lines) == 20
# ?seed=42: /bytes на каждый запрос генерирует НОВЫЕ случайные байты, и два
#   скачивания без seed почти наверняка не совпали бы — сравнение хешей
#   проверяло бы не способ скачивания, а удачу. С seed генератор выдаёт одну
#   и ту же последовательность, и совпадение хешей доказывает, что stream и
#   обычный get получили одинаковые данные.
# Размер: httpbin ограничивает /bytes сотней килобайт (102 400 байт), поэтому
#   вместо мегабайта пришло 100 КБ. Механизм от размера не зависит.
# Память: обычный get кладёт всё тело в r.content целиком — для файла в 1 ГБ
#   это гигабайт оперативной памяти. stream=True + iter_content держит в
#   памяти только текущий кусок в 8 КБ и сразу пишет его на диск — память
#   постоянна при любом размере файла.
# iter_lines отдаёт тело построчно по мере прихода — так LLM API присылает
#   ответ по токенам (Server-Sent Events).


# ---------------------------------------------------------------------------
# 8. Секрет в URL
# ---------------------------------------------------------------------------
print("\n8. Секрет в URL")
SECRET = "supersecret123"
r = requests.get(f"{BASE}/get?token={SECRET}", timeout=10)
body = r.json()
print("в URL:       url  =", body["url"])
print("             args =", body["args"])

r = requests.get(f"{BASE}/headers", headers={"Authorization": f"Bearer {SECRET}"}, timeout=10)
print("в заголовке: Authorization =", r.json()["headers"]["Authorization"])
# В первом случае токен — часть адреса: он виден в url и args, а значит
#   попадёт в лог доступа веб-сервера, в логи прокси и балансировщиков, в
#   историю браузера, в заголовок Referer при переходе на другой сайт, в
#   системы мониторинга. Всё это хранится долго и читается многими людьми.
# Во втором — токен только в заголовке; сервер его видит (так и задумано),
#   но стандартные логи доступа записывают строку запроса, а не заголовки.
# История оболочки: запуск `curl "https://httpbin.org/get?token=..."`
#   попадает в ~/.zsh_history открытым текстом и остаётся там навсегда —
#   его видит любой, кто прочитает файл, и он уезжает в бэкапы. Поэтому
#   секреты передают через переменные окружения (Ф2-22), а не в команде.


# ---------------------------------------------------------------------------
# 9. Устно, потом письменно
# ---------------------------------------------------------------------------
# Идемпотентность: повторный вызов даёт тот же результат, что и один.
#   GET, PUT, DELETE идемпотентны: удалить уже удалённое — ничего не меняет.
#   POST — нет: два вызова создают две записи. Когда запрос упал по сети
#   (Timeout, ConnectionError), неизвестно, выполнился ли он. Идемпотентный
#   можно просто повторить — хуже не станет. Неидемпотентный повторять
#   вслепую нельзя: нужен ключ идемпотентности, по которому сервер узнает
#   повтор и не выполнит действие второй раз.
# 401 против 403: 401 — «я не знаю, кто ты»: токена нет или он неверный;
#   правильный токен поможет. 403 — «я знаю, кто ты, и тебе нельзя»:
#   аутентификация прошла, но прав нет; другой токен того же пользователя
#   не поможет.
# Почему timeout обязателен: у requests нет таймаута по умолчанию — без него
#   запрос к зависшему серверу ждёт вечно (задание 3). Каждый такой запрос
#   держит поток, и сервис умирает не от ошибок, а от накопившихся
#   зависаний, без единой строки в логе.
# raise_for_status: если код 4xx или 5xx — бросает HTTPError, иначе ничего не
#   делает. Без него requests НЕ считает 404 или 500 ошибкой: ответ получен,
#   исключения нет, и код спокойно идёт дальше — например, вызывает r.json()
#   на странице ошибки или сохраняет её как результат (задание 2: у 500
#   r.ok = False, но исключение появилось только после raise_for_status()).
