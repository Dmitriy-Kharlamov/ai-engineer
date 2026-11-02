"""Ограничитель частоты в памяти: скользящее окно по каждому ключу. Про HTTP не знает."""

import math
import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window = window
        self.clock = clock
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> int | None:
        """None — можно; иначе через сколько целых секунд освободится место."""
        now = self.clock()
        hits = self._hits[key]
        while hits and hits[0] <= now - self.window:  # выкидываем вышедшие из окна
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1, math.ceil(hits[0] + self.window - now))
        hits.append(now)
        return None
