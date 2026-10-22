import os
import signal
import sys
import time


def on_term(signum, frame):
    print("получен SIGTERM — завершаюсь корректно", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, on_term)
print(f"slow.py запущен, PID {os.getpid()}", flush=True)

counter = 0
while True:
    counter += 1
    print(f"счётчик: {counter}", flush=True)
    time.sleep(1)
