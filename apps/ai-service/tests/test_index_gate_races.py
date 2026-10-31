"""IndexGate races found by the CI flake of
test_four_inflight_timeouts_finish_inside_one_grace_window.

1. Process.join(timeout) can return before the child is waitable, so one join is
   not proof of exit (join_exited polls).
2. A slot whose tree could not be confirmed gone left the gate "failed", and the
   next successful recovery of another slot overwrote that with "ready".
3. A slot that cancel() is still stopping counts as recovering, not as failed.
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import pytest

from app.indexing.procgroup import join_exited
from app.indexing.supervisor import IndexGate


def _expire(_signum, _frame):
    raise TimeoutError("gate race test exceeded 45s")


@pytest.fixture(autouse=True)
def _bound_test_runtime():
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, 45)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class _EarlyJoinProc:
    """join() returns at once; is_alive() stays True for the first `alive_polls` calls."""

    def __init__(self, alive_polls: int) -> None:
        self.left = alive_polls
        self.joins = 0

    def join(self, _timeout=None) -> None:
        self.joins += 1

    def is_alive(self) -> bool:
        if self.left > 0:
            self.left -= 1
            return True
        return False


def test_join_exited_waits_past_an_early_join():
    proc = _EarlyJoinProc(alive_polls=5)
    assert join_exited(proc, 1.0) is True
    assert proc.joins == 1


def test_join_exited_is_false_for_a_process_that_stays_alive():
    proc = _EarlyJoinProc(alive_polls=10**9)
    started = time.monotonic()
    assert join_exited(proc, 0.2) is False
    assert time.monotonic() - started < 1.0


def _hang_two(gate: IndexGate, tmp_path: Path):
    handles, paths = [], []
    for index in range(2):
        path = tmp_path / f"pids-{index}"
        paths.append(path)
        handles.append(gate.submit({"op": "resistant_hang", "pids_path": str(path), "seconds": 30}, timeout_s=10))
    for path in paths:
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists()
    return handles


def _wait(predicate, seconds: float = 6.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_a_failed_slot_is_not_hidden_by_another_recovery(tmp_path: Path):
    gate = IndexGate(2, grace_s=0.2, recovery_timeout_s=4, kind="stage")
    gate.start(4)
    real = gate._kill_tree
    try:
        first, second = _hang_two(gate, tmp_path)
        failing = first.slot

        def unconfirmed(identity, slot):
            confirmed = real(identity, slot)
            return False if slot is failing else confirmed

        gate._kill_tree = unconfirmed
        gate.cancel(first)
        assert gate.status == "failed"
        gate.cancel(second)
        assert _wait(lambda: second.slot.ready and not second.slot.recovering)
        # Before the fix the recovery of `second` set "ready" here.
        assert gate.status == "failed"
        assert gate.health_problem(type("S", (), {"embedding_provider": "fake", "ocr_provider": "fake"})()) == "worker_failed"
    finally:
        gate._kill_tree = real
        gate.shutdown(timeout_s=5)


def test_a_slot_still_being_stopped_keeps_the_gate_recovering(tmp_path: Path):
    gate = IndexGate(2, grace_s=0.2, recovery_timeout_s=4, kind="stage")
    gate.start(4)
    real = gate._kill_tree
    release = threading.Event()
    try:
        first, second = _hang_two(gate, tmp_path)
        slow = first.slot

        def held(identity, slot):
            if slot is slow:
                release.wait(10)
            return real(identity, slot)

        gate._kill_tree = held
        stopper = threading.Thread(target=gate.cancel, args=(first,))
        stopper.start()
        assert _wait(lambda: slow.stopping)
        gate.cancel(second)
        assert _wait(lambda: second.slot.ready and not second.slot.recovering)
        # `first` is neither ready nor failed yet: the gate must say recovering.
        assert gate.status == "recovering"
        release.set()
        stopper.join(10)
        assert _wait(lambda: gate.status == "ready")
        assert len(gate.worker_pids()) == 2
        assert not slow.stopping
    finally:
        release.set()
        gate._kill_tree = real
        gate.shutdown(timeout_s=5)
