"""In-process limits for POST /v1/extract.

Applied only after the service token is accepted and before the provider
runs. Rejected calls return 429; this process never logs the token or the
citizen message.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class ExtractLimiter:
    def __init__(self, rate_per_minute: int, max_inflight: int, retry_after_s: int) -> None:
        if rate_per_minute < 1 or max_inflight < 1 or retry_after_s < 1:
            raise ValueError("rate limits must be positive")
        self.rate_per_minute = rate_per_minute
        self.max_inflight = max_inflight
        self.retry_after_s = retry_after_s
        self._lock = threading.Lock()
        self._hits: deque[float] = deque()
        self._inflight = 0

    def try_acquire(self) -> tuple[bool, str]:
        """Return (allowed, reason). reason is empty, 'rate_limited', or 'concurrency_limited'."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - 60.0
            while self._hits and self._hits[0] <= cutoff:
                self._hits.popleft()
            if len(self._hits) >= self.rate_per_minute:
                return False, "rate_limited"
            if self._inflight >= self.max_inflight:
                return False, "concurrency_limited"
            self._hits.append(now)
            self._inflight += 1
            return True, ""

    def release(self) -> None:
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1
