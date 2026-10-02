"""One monotonic deadline for a single index request."""

from __future__ import annotations

import time

from app.indexing.errors import IndexFailure


class Deadline:
    """Absolute end time. Stages call remaining() instead of restarting the budget."""

    def __init__(self, seconds: float, *, now: float | None = None) -> None:
        start = time.monotonic() if now is None else now
        self.deadline_at = start + seconds

    def remaining(self) -> float:
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
