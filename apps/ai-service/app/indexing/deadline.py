"""One monotonic deadline for a single index request."""

from __future__ import annotations

import threading
import time

from app.indexing.errors import IndexFailure


class Deadline:
    """Absolute end time. Stages call remaining() instead of restarting the budget."""

    def __init__(self, seconds: float, *, now: float | None = None) -> None:
        start = time.monotonic() if now is None else now
        self.deadline_at = start + seconds
        self._cancelled = False
        self._closers: list = []
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def remaining(self) -> float:
        if self._cancelled:
            raise IndexFailure("timeout")
        left = self.deadline_at - time.monotonic()
        if left <= 0:
            raise IndexFailure("timeout")
        return left

    def check(self) -> None:
        self.remaining()

    def timeout_for_io(self, cap: float | None = None) -> float:
        left = self.remaining()
        if cap is None:
            return left
        return min(left, cap)

    def bind(self, closer) -> None:
        """Register a callback that cancel() runs so an in-flight socket stops."""
        with self._lock:
            if self._cancelled:
                closer()
                raise IndexFailure("timeout")
            self._closers.append(closer)

    def unbind(self, closer) -> None:
        with self._lock:
            try:
                self._closers.remove(closer)
            except ValueError:
                return

    def cancel(self) -> None:
        """Expire this request. Closers run off the caller so a blocking close cannot stall the event loop."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self.deadline_at = time.monotonic()
            closers = list(self._closers)
        if not closers:
            return
        thread = threading.Thread(target=_drain_closers, args=(closers, 0.5), name="deadline-closer")
        with _closer_lock:
            _closer_threads.append(thread)
        thread.start()


_closer_lock = threading.Lock()
_closer_threads: list[threading.Thread] = []


def _drain_closers(closers, timeout_s: float) -> None:
    for closer in closers:
        worker = threading.Thread(target=_safe_close, args=(closer,), name="deadline-close-one")
        worker.start()
        worker.join(timeout_s)


def _safe_close(closer) -> None:
    try:
        closer()
    except Exception:
        return


def join_closers(timeout_s: float) -> None:
    with _closer_lock:
        threads = list(_closer_threads)
    deadline = time.monotonic() + max(0.0, timeout_s)
    for thread in threads:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        thread.join(left)
