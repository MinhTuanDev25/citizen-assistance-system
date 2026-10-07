"""Bundle 10: parallel group cleanup and idle pipeline recovery."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from app.indexing.errors import IndexFailure
from app.indexing.procgroup import GroupKiller, become_group_leader, capture, process_alive
from app.indexing.supervisor import IndexGate


def _expire(_signum, _frame):
    raise TimeoutError("bundle10 test exceeded 45s")


@pytest.fixture(autouse=True)
def _bound_test_runtime():
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, 45)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _ignore_leader(conn) -> None:
    become_group_leader()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    conn.send(os.getpid())
    time.sleep(30)


def _plain_leader(conn) -> None:
    become_group_leader()
    conn.send(os.getpid())
    time.sleep(30)


def _settings():
    return type("S", (), {"embedding_provider": "fake", "ocr_provider": "fake"})()


def test_four_cleanups_share_the_grace_period():
    ctx = mp.get_context("spawn")
    procs = []
    identities = []
    try:
        for _ in range(4):
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_ignore_leader, args=(child,))
            proc.start()
            child.close()
            assert parent.poll(2)
            identity = capture(parent.recv(), timeout_s=1)
            assert identity is not None
            identities.append(identity)
            procs.append(proc)
        killer = GroupKiller()
        barrier = threading.Barrier(4)
        entered = []

        def _entered(key):
            entered.append(key)
            barrier.wait(2)

        killer.on_grace = _entered
        errors = []

        def _kill(identity):
            try:
                if not killer.kill(identity, 0.3):
                    errors.append("not-confirmed")
            except Exception as exc:  # noqa: BLE001 - the assertion reports it
                errors.append(repr(exc))

        started = time.monotonic()
        threads = [threading.Thread(target=_kill, args=(identity,)) for identity in identities]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        elapsed = time.monotonic() - started
        assert not errors
        assert len(entered) == 4
        assert elapsed < 0.9
        for identity in identities:
            assert not process_alive(identity[0])
        again = time.monotonic()
        assert killer.kill(identities[0], 0.3) is True
        assert time.monotonic() - again < 0.1
    finally:
        for proc in procs:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGKILL)
            proc.join(0.5)


def test_grace_ends_when_the_group_is_already_gone():
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_plain_leader, args=(child,))
    proc.start()
    child.close()
    try:
        assert parent.poll(2)
        identity = capture(parent.recv(), timeout_s=1)
        started = time.monotonic()
        assert GroupKiller().kill(identity, 1.0) is True
        assert time.monotonic() - started < 0.4
        assert not process_alive(identity[0])
    finally:
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
        proc.join(0.5)


def test_four_inflight_timeouts_finish_inside_one_grace_window(tmp_path: Path):
    grace = 0.4
    gate = IndexGate(4, grace_s=grace, recovery_timeout_s=4, kind="stage")
    gate.start(4)
    try:
        handles = []
        paths = []
        for index in range(4):
            path = tmp_path / f"pids-{index}"
            paths.append(path)
            handles.append(gate.submit({"op": "resistant_hang", "pids_path": str(path), "seconds": 30}, timeout_s=3))
        old = []
        for path in paths:
            deadline = time.monotonic() + 2
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            old.extend(int(line) for line in path.read_text(encoding="utf-8").split() if line.strip())
        assert len(old) == 8
        for pid in old:
            assert process_alive(pid)
        started = time.monotonic()
        threads = [threading.Thread(target=gate.cancel, args=(handle,)) for handle in handles]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        elapsed = time.monotonic() - started
        assert elapsed < grace + 0.7
        assert elapsed < grace * 4
        assert gate.pending() == 0
        for pid in old:
            assert not process_alive(pid)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and gate.status != "ready":
            time.sleep(0.02)
        assert gate.status == "ready"
        fresh = gate.worker_pids()
        assert len(fresh) == 4
        assert len(set(fresh)) == 4
        assert not set(fresh) & set(old)
        for pid in fresh:
            assert process_alive(pid)
        release = tmp_path / "release"
        started_paths = [tmp_path / f"started-{index}" for index in range(4)]
        replies = []
        errors = []

        def _hang(path: Path) -> None:
            try:
                replies.append(
                    gate.call(
                        {"op": "hang", "started_path": str(path), "release_path": str(release), "seconds": 5},
                        4,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(repr(exc))

        workers = [threading.Thread(target=_hang, args=(path,)) for path in started_paths]
        for thread in workers:
            thread.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not all(path.exists() for path in started_paths):
            time.sleep(0.02)
        release.write_text("1", encoding="utf-8")
        for thread in workers:
            thread.join(4)
        assert not errors
        assert len(replies) == 4
        assert {reply["pid"] for reply in replies} == set(fresh)
    finally:
        gate.shutdown(3)


def test_idle_pipeline_death_is_recovered(tmp_path: Path):
    gate = IndexGate(1, grace_s=0.2, recovery_timeout_s=5, kind="pipeline")
    gate.start(6)
    try:
        path = tmp_path / "pids"
        reply = gate.call({"op": "park_children", "pids_path": str(path)}, timeout_s=4)
        assert reply["event"] == "result"
        old = [int(line) for line in path.read_text(encoding="utf-8").split() if line.strip() and int(line) > 0]
        assert len(old) >= 3
        leader, _onnx, ocr = old[0], old[1], old[2]
        previous = gate._slots[0].identity
        assert previous is not None and previous[0] == leader
        assert process_alive(ocr)
        seen = []
        reasons = []
        stop = threading.Event()

        def _watch() -> None:
            while not stop.is_set():
                seen.append(gate.status)
                reasons.append(gate.health_problem(_settings()))
                time.sleep(0.01)

        watcher = threading.Thread(target=_watch)
        watcher.start()
        os.kill(leader, signal.SIGKILL)
        assert process_alive(ocr)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            slot = gate._slots[0]
            if gate.status == "ready" and slot.identity not in (None, previous) and slot.proc is not None and slot.proc.is_alive():
                break
            time.sleep(0.02)
        stop.set()
        watcher.join(1)
        seen.append(gate.status)
        reasons.append(gate.health_problem(_settings()))
        slot = gate._slots[0]
        assert "recovering" in seen
        assert gate.status == "ready"
        assert "worker_recovering" in reasons
        assert reasons[-1] is None
        assert slot.identity != previous
        for pid in old:
            assert not process_alive(pid)
        pong = gate.call({"op": "ping"}, 3)
        assert pong["event"] == "pong"
        assert pong["pid"] != leader
        assert gate.boot_count == 2
        assert gate._recovery_started == 1
        replacement = pong["pid"]
        boot = gate.boot_count
        os.kill(replacement, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while gate.status != "failed" and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.25)
        assert gate.status == "failed"
        assert gate.boot_count == boot
        assert gate.health_problem(_settings()) == "worker_failed"
        with pytest.raises(IndexFailure):
            gate.call({"op": "ping"}, 0.4)
    finally:
        gate.shutdown(3)


def _assert_no_spawn_after_return(gate: IndexGate, returned_at: float) -> None:
    assert gate._shutdown_returned_at is not None
    assert returned_at <= gate._shutdown_returned_at
    if gate._proc_started_at is not None:
        assert gate._proc_started_at <= returned_at
    assert gate.status in ("stopped", "failed")
    assert gate._monitor is None or not gate._monitor.is_alive()
    assert all(slot.proc is None or not slot.proc.is_alive() for slot in gate._slots)
    assert all(not thread.is_alive() for thread in gate._recovery)


def test_shutdown_blocks_a_spawn_that_passed_the_closed_check():
    """Recovery is inside the pre-start window. Shutdown must win or reap first."""
    gate = IndexGate(1, grace_s=0.05, recovery_timeout_s=2, kind="stage")
    gate.start(2)
    entered = threading.Event()
    release = threading.Event()
    observed = {}

    def _before_start() -> None:
        observed["boot"] = gate.boot_count
        entered.set()
        release.wait(3)

    gate._before_proc_start = _before_start
    slot = gate._slots[0]
    os.kill(slot.proc.pid, signal.SIGKILL)
    slot.proc.join(1)
    with gate._lock:
        slot.ready = False
        slot.recovering = True
    launcher = threading.Thread(target=gate._launch_recovery, args=(slot,))
    launcher.start()
    assert entered.wait(2)
    stopper = threading.Thread(target=gate.shutdown, kwargs={"timeout_s": 3})
    stopper.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not gate._closed:
        time.sleep(0.001)
    assert gate._closed
    assert stopper.is_alive()
    release.set()
    stopper.join(3)
    launcher.join(1)
    assert not stopper.is_alive()
    _assert_no_spawn_after_return(gate, gate._shutdown_returned_at)
    if gate.boot_count > observed["boot"]:
        assert gate._proc_started_at <= gate._shutdown_returned_at
    boot = gate.boot_count
    gate.shutdown(1)
    assert gate.boot_count == boot
    assert gate.status in ("stopped", "failed")


def test_shutdown_cannot_miss_a_recovery_thread_registered_before_start():
    gate = IndexGate(1, grace_s=0.05, recovery_timeout_s=2, kind="stage")
    gate.start(2)
    registered = threading.Event()
    release_register = threading.Event()
    at_spawn = threading.Event()
    release_spawn = threading.Event()

    def _before_thread(thread: threading.Thread) -> None:
        assert thread in gate._recovery
        assert thread.is_alive() is False
        registered.set()
        release_register.wait(3)

    def _before_start() -> None:
        at_spawn.set()
        release_spawn.wait(3)

    gate._before_thread_start = _before_thread
    gate._before_proc_start = _before_start
    slot = gate._slots[0]
    os.kill(slot.proc.pid, signal.SIGKILL)
    slot.proc.join(1)
    with gate._lock:
        slot.ready = False
        slot.recovering = True
    launcher = threading.Thread(target=gate._launch_recovery, args=(slot,))
    launcher.start()
    assert registered.wait(2)
    stopper = threading.Thread(target=gate.shutdown, kwargs={"timeout_s": 3})
    stopper.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not stopper.is_alive():
        time.sleep(0.001)
    assert stopper.is_alive()
    assert gate._closed is False
    release_register.set()
    if at_spawn.wait(1):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not gate._closed:
            time.sleep(0.001)
        assert gate._closed
        assert stopper.is_alive()
    release_spawn.set()
    stopper.join(3)
    launcher.join(1)
    assert gate._thread_started_at is not None
    assert gate._thread_started_at <= gate._shutdown_returned_at
    _assert_no_spawn_after_return(gate, gate._shutdown_returned_at)
    assert gate._recovery_started == 1


def test_shutdown_returns_while_a_registered_recovery_stays_blocked():
    gate = IndexGate(1, grace_s=0.05, recovery_timeout_s=2, kind="stage")
    entered = threading.Event()
    release = threading.Event()
    seen = {}

    def _remember(thread: threading.Thread) -> None:
        seen["thread"] = thread

    def _block() -> None:
        entered.set()
        release.wait(5)

    gate._before_thread_start = _remember
    gate._before_recover = _block
    launcher = threading.Thread(target=gate._launch_recovery, args=(gate._slots[0],))
    launcher.start()
    try:
        assert entered.wait(2)
        launcher.join(1)
        assert seen["thread"].daemon is True
        started = time.monotonic()
        gate.shutdown(0.05)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert gate._closed is True
        assert gate.status == "failed"
        assert gate.boot_count == 0
        assert gate._proc_started_at is None or gate._proc_started_at <= gate._shutdown_returned_at
        assert gate.status != "ready"
    finally:
        release.set()
        if "thread" in seen:
            seen["thread"].join(2)
    assert not seen["thread"].is_alive()
    assert gate._closed is True
    assert gate.status == "failed"
    assert gate.boot_count == 0
    assert gate._proc_started_at is None or gate._proc_started_at <= gate._shutdown_returned_at


def test_shutdown_spawn_race_over_many_interleavings():
    for _ in range(50):
        gate = IndexGate(1, grace_s=0.05, recovery_timeout_s=1.5, kind="stage")
        gate.start(2)
        os.kill(gate._slots[0].proc.pid, signal.SIGKILL)
        gate.shutdown(2)
        _assert_no_spawn_after_return(gate, gate._shutdown_returned_at)
        boot = gate.boot_count
        gate.shutdown(1)
        assert gate.boot_count == boot
        assert gate._proc_started_at is None or gate._proc_started_at <= gate._shutdown_returned_at
        assert gate.status in ("stopped", "failed")
