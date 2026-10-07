"""Bundle 8: process identity without ps, and fail-closed group cleanup."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

from app.indexing.errors import IndexFailure
from app.indexing.procgroup import GroupKiller, IdentityError, become_group_leader, capture, process_alive, read_process
from app.indexing.supervisor import IndexGate


def _expire(_signum, _frame):
    raise TimeoutError("bundle8 test exceeded 20s")


@pytest.fixture(autouse=True)
def _bound_test_runtime():
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, 20)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _gone(pid: int, timeout: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if read_process(pid) is None and not process_alive(pid):
            return True
        time.sleep(0.02)
    return read_process(pid) is None and not process_alive(pid)


def _forbid_ps(monkeypatch):
    real_check = subprocess.check_output
    real_run = subprocess.run
    real_call = subprocess.call
    real_popen = subprocess.Popen

    def _command(args):
        if isinstance(args, (list, tuple)) and args:
            return os.path.basename(str(args[0]))
        return os.path.basename(str(args).split()[0]) if args else ""

    def check_output(args, *pos, **kwargs):
        if _command(args) == "ps":
            raise FileNotFoundError("ps is not installed")
        return real_check(args, *pos, **kwargs)

    def run(args, *pos, **kwargs):
        if _command(args) == "ps":
            raise FileNotFoundError("ps is not installed")
        return real_run(args, *pos, **kwargs)

    def call(args, *pos, **kwargs):
        if _command(args) == "ps":
            raise FileNotFoundError("ps is not installed")
        return real_call(args, *pos, **kwargs)

    class GuardPopen(real_popen):
        def __init__(self, args, *pos, **kwargs):
            if _command(args) == "ps":
                raise FileNotFoundError("ps is not installed")
            super().__init__(args, *pos, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "call", call)
    monkeypatch.setattr(subprocess, "Popen", GuardPopen)


def _resistant(path: str) -> None:
    become_group_leader()
    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
        os._exit(0)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(pid))
    time.sleep(60)


def test_recovered_ocr_worker_outlives_the_recovery_thread():
    from app.indexing.ocr_supervisor import OcrSupervisor

    supervisor = OcrSupervisor(1, {"kind": "fake", "behavior": "crash", "recover": "ok"})
    try:
        supervisor.start(2)
        with pytest.raises(IndexFailure):
            supervisor.read_page(b"x", 1, timeout_s=2)
        assert supervisor.wait_settled(2) == "ready"
        thread = supervisor._recovery_thread
        assert thread is None or thread.is_alive() is False
        time.sleep(0.3)
        assert supervisor.alive_workers() == 1
        assert supervisor.read_page(b"x", 2, timeout_s=2) == "ok"
    finally:
        supervisor.shutdown()


def test_supervision_does_not_call_ps():
    source = Path(capture.__code__.co_filename).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert '["ps"' not in source


def test_timeout_without_ps_kills_the_worker_and_replaces_it(tmp_path, monkeypatch):
    _forbid_ps(monkeypatch)
    started = str(tmp_path / "started")
    gate = IndexGate(1, grace_s=0.15, recovery_timeout_s=3)
    gate.start(3)
    try:
        handle = gate.submit({"op": "hang", "seconds": 30, "started_path": started}, 4)
        assert handle.identity is not None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not os.path.exists(started):
            time.sleep(0.02)
        assert os.path.exists(started)
        pid = handle.identity[0]
        with pytest.raises(IndexFailure) as raised:
            gate.wait(handle, 0.25)
        assert raised.value.code == "timeout"
        assert gate.pending() == 0
        assert _gone(pid)
        assert gate.call({"op": "ping"}, 3)["event"] == "pong"
        assert gate.pending() == 0
    finally:
        gate.shutdown()


def test_capture_failure_is_fail_closed_and_reaps(monkeypatch):
    monkeypatch.setattr("app.indexing.supervisor.capture", lambda *_args, **_kwargs: None)
    gate = IndexGate(1, grace_s=0.1, recovery_timeout_s=1)
    with pytest.raises(IndexFailure) as raised:
        gate.start(2)
    assert raised.value.code == "worker_failed"
    assert gate.status == "failed"
    assert gate.pending() == 0
    for slot in gate._slots:
        assert slot.ready is False
        assert slot.proc is None or slot.proc.is_alive() is False
        if slot.proc is not None and slot.proc.pid:
            assert _gone(slot.proc.pid)
    gate.shutdown()


def test_transient_identity_failure_is_retried(tmp_path):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_resistant, args=(str(tmp_path / "unused"),))
    proc.start()
    killer = GroupKiller()
    try:
        identity = capture(proc.pid, timeout_s=2)
        assert identity is not None
        calls = {"n": 0}
        real = read_process

        def flaky(pid: int):
            calls["n"] += 1
            if calls["n"] == 1:
                raise IdentityError("transient")
            return real(pid)

        import app.indexing.procgroup as procgroup

        procgroup.read_process = flaky
        try:
            assert killer.kill(identity, 0.05) is False
            assert identity not in killer._done
            assert process_alive(proc.pid)
            procgroup.read_process = real
            assert killer.kill(identity, 0.2) is True
        finally:
            procgroup.read_process = real
        proc.join(1)
        assert _gone(identity[0])
        assert killer.kill(identity, 0.05) is True
    finally:
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(1)


def test_sigterm_kills_leader_and_sigkill_kills_resistant_descendant(tmp_path):
    path = str(tmp_path / "child")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_resistant, args=(path,))
    proc.start()
    killer = GroupKiller()
    identity = None
    try:
        identity = capture(proc.pid, timeout_s=2)
        assert identity is not None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not os.path.exists(path):
            time.sleep(0.02)
        child = int(Path(path).read_text(encoding="utf-8"))
        assert process_alive(child)
        assert killer.kill(identity, 0.25) is True
        proc.join(1)
        assert _gone(identity[0])
        assert _gone(child)
        assert killer.kill(identity, 0.05) is True
    finally:
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(1)
        if identity is not None:
            killer.kill(identity, 0.05)


def test_timeout_disconnect_and_shutdown_do_not_cross_kill(tmp_path):
    gate = IndexGate(2, grace_s=0.15, recovery_timeout_s=3, kind="pipeline")
    gate.start(4)
    pids_path = str(tmp_path / "pids")
    started_a = str(tmp_path / "a")
    started_b = str(tmp_path / "b")
    try:
        handle_a = gate.submit(
            {"op": "nested_hang", "seconds": 30, "pids_path": pids_path, "started_path": started_a},
            6,
        )
        handle_b = gate.submit({"op": "hang", "seconds": 0.4, "started_path": started_b}, 6)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not (os.path.exists(pids_path) and os.path.exists(started_b)):
            time.sleep(0.02)
        old_pids = [int(line) for line in open(pids_path, encoding="utf-8") if line.strip()]
        assert len(old_pids) >= 4
        assert handle_a.identity is not None
        assert handle_b.identity is not None
        assert handle_a.identity[0] != handle_b.identity[0]
        box = {}

        def _wait_a():
            try:
                gate.wait(handle_a, 0.2)
            except IndexFailure as exc:
                box["a"] = exc.code

        def _disconnect_a():
            time.sleep(0.05)
            gate.cancel(handle_a)

        def _wait_b():
            try:
                box["b"] = gate.wait(handle_b, 3)
            except IndexFailure as exc:
                box["b"] = exc

        threads = [threading.Thread(target=target) for target in (_wait_a, _disconnect_a, _wait_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(4)
        assert [thread.is_alive() for thread in threads] == [False, False, False]
        assert box["a"] == "timeout"
        assert box["b"]["event"] == "result"
        assert box["b"]["pid"] == handle_b.identity[0]
        assert gate.pending() == 0
        assert all(_gone(pid) for pid in old_pids)
        assert process_alive(handle_b.identity[0])
        kept = set(gate.worker_pids())
        assert gate._groups.kill(handle_a.identity, 0.1) is True
        assert set(gate.worker_pids()) == kept
        assert process_alive(handle_b.identity[0])
        stopper = threading.Thread(target=gate.shutdown)
        again = threading.Thread(target=lambda: gate.cancel(handle_a))
        stopper.start()
        again.start()
        stopper.join(4)
        again.join(4)
        assert stopper.is_alive() is False
        assert again.is_alive() is False
        assert gate.pending() == 0
        assert all(_gone(pid) for pid in old_pids)
        assert _gone(handle_b.identity[0])
    finally:
        gate.shutdown()
