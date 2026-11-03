"""Зависимый сервис (Ф3-11, задание 8): сразу после старта один раз стучится в api.

Повторов нет намеренно: так видно, в каком состоянии api был в момент старта worker.
Только стандартная библиотека — запускается в любом образе с Python.
"""

import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("API_URL", "http://api:8000") + "/health/ready"

started = time.monotonic()
try:
    with urllib.request.urlopen(URL, timeout=5) as response:
        body = response.read().decode()
        print(f"worker: {URL} -> {response.status} {body}", flush=True)
        code = 0
except urllib.error.HTTPError as e:  # ответил, но не 2xx — например 503 «не готов»
    print(f"worker: {URL} -> {e.code} {e.read().decode()}", flush=True)
    code = 1
except OSError as e:  # не ответил вовсе: соединение отклонено, имя не найдено
    print(f"worker: {URL} -> ОШИБКА СОЕДИНЕНИЯ: {e!r}", flush=True)
    code = 2
print(f"worker: попытка заняла {time.monotonic() - started:.3f} с, выхожу с кодом {code}")
sys.exit(code)
