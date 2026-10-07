"""Killable pipeline workers. A hung job is a process, so it can be terminated."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import socket
import threading
import time

from app.indexing.errors import IndexFailure
from app.indexing.procgroup import GroupKiller, IdentityError, become_group_leader, capture, read_process, reap_spawned

log_name = "cas.index"


def _publish_ocr(ocr, ocr_state) -> None:
    if ocr_state is None:
        return
    bind = getattr(ocr, "bind_state", None)
    if callable(bind):
        bind(ocr_state)
        return
    ocr_state.value = 1


def slot_main(conn, kind: str, ocr_state) -> None:
    """One persistent worker. Models load once per process, not once per request."""
    become_group_leader()
    models = None
    settings = None
    try:
        if kind == "pipeline":
            from app.config import load_settings
            from app.indexing.runtime import build_models

            settings = load_settings()
            models = build_models(settings)
            _publish_ocr(models[1], ocr_state)
        conn.send({"event": "ready", "pid": os.getpid(), "report": _report(models)})
        while True:
            msg = conn.recv()
            if not isinstance(msg, dict) or msg.get("op") == "shutdown":
                conn.send({"event": "bye"})
                return
            op = msg.get("op")
            if op == "ping":
                conn.send({"event": "pong", "pid": os.getpid()})
                continue
            if op == "hang":
                _hang(msg)
                conn.send({"event": "result", "stage": msg.get("stage"), "pid": os.getpid()})
                continue
            if op == "ipc_block":
                _ipc_block(msg)
                conn.send({"event": "result", "stage": "ipc", "pid": os.getpid()})
                continue
            if op == "socket_block":
                _socket_block(msg)
                conn.send({"event": "result", "stage": msg.get("stage"), "pid": os.getpid()})
                continue
            if op == "nested_hang":
                _nested_hang(msg)
                conn.send({"event": "result", "stage": "nested", "pid": os.getpid()})
                continue
            if op == "park_children":
                _park_children(msg)
                conn.send({"event": "result", "stage": "parked", "pid": os.getpid()})
                continue
            if op == "resistant_hang":
                _resistant_hang(msg)
                conn.send({"event": "result", "stage": "resistant", "pid": os.getpid()})
                continue
            if op == "pipeline":
                if settings is None or models is None:
                    conn.send({"event": "error", "code": "embedding_model_missing"})
                    continue
                try:
                    from app.indexing.contract import IndexRequestV2
                    from app.indexing.deadline import Deadline
                    import app.main as app_main

                    app_main._models = models
                    request = IndexRequestV2.model_validate(msg.get("body") or {})
                    clock = Deadline(float(msg.get("timeout_s") or settings.pipeline_timeout_s))
                    result = app_main._run_v2(request, settings, clock)
                    conn.send({"event": "result", "body": result.model_dump(mode="json"), "pid": os.getpid()})
                except IndexFailure as exc:
                    conn.send({"event": "error", "code": exc.code})
                except Exception:
                    conn.send({"event": "error", "code": "worker_failed"})
                continue
            conn.send({"event": "error", "code": "worker_failed"})
    except (EOFError, KeyboardInterrupt):
        return
    finally:
        for obj in models or ():
            stop = getattr(obj, "shutdown", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    continue


def _report(models) -> dict:
    if not models:
        return {}
    embedder, ocr = models[0], models[1]
    alive = ocr.alive_workers() if hasattr(ocr, "alive_workers") else 0
    return {
        "embedding_status": getattr(embedder, "status", "ready"),
        "cpu_compatible": getattr(embedder, "cpu_compatible", None),
        "ocr_status": getattr(ocr, "status", "ready"),
        "ocr_workers_alive": int(alive),
        "ocr_workers_expected": int(getattr(ocr, "slots", 0) or 0),
    }


def _mark(path) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("1")


def _wait_path(path, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path and os.path.exists(path):
            return
        time.sleep(0.02)


def _hang(msg: dict) -> None:
    _mark(msg.get("started_path"))
    if msg.get("release_path"):
        _wait_path(msg.get("release_path"), float(msg.get("seconds") or 30))
    else:
        time.sleep(float(msg.get("seconds") or 30))
    _mark(msg.get("wrote_path"))


def _ipc_block(msg: dict) -> None:
    _mark(msg.get("started_path"))
    _left, right = mp.Pipe(duplex=False)
    right.recv()


def _nested_hang(msg: dict) -> None:
    """Real child topology: this pipeline process owns a hung ONNX and OCR process."""
    from app.indexing.embed import OnnxProcess
    from app.indexing.ocr_supervisor import OcrSupervisor

    onnx = OnnxProcess("", behavior="hang")
    onnx.start(5)
    ocr = OcrSupervisor(1, {"kind": "fake", "behavior": "hang"})
    ocr.start(5)
    booting = OcrSupervisor(1, {"kind": "fake", "behavior": "init_hang"})

    def _boot() -> None:
        try:
            booting.start(30)
        except Exception:
            return

    def _stick(fn) -> None:
        try:
            fn()
        except Exception:
            return

    threading.Thread(target=_boot, name="nested-boot").start()
    threading.Thread(target=lambda: _stick(lambda: onnx.count_tokens("hang", timeout_s=30)), name="nested-onnx").start()
    threading.Thread(target=lambda: _stick(lambda: ocr.read_page(None, 1, timeout_s=30)), name="nested-ocr").start()
    end = time.monotonic() + 2
    boot_pid = None
    while time.monotonic() < end:
        with booting._guard:
            live = [proc.pid for proc in booting._booting if proc.pid and proc.is_alive()]
        if live and ocr.active_count() >= 1:
            boot_pid = live[0]
            break
        time.sleep(0.01)
    pids = [os.getpid(), onnx._proc.pid]
    if ocr._active:
        pids.append(ocr._active[0][0].pid)
    if boot_pid:
        pids.append(boot_pid)
    path = msg.get("pids_path")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(str(pid) for pid in pids))
    _mark(msg.get("started_path"))
    time.sleep(float(msg.get("seconds") or 30))
    _mark(msg.get("wrote_path"))


def _park_children(msg: dict) -> None:
    """Leave a fake ONNX process and a fake OCR process running, then return.

    The pipeline slot is idle afterward. The children stay in this process
    group. OCR does not arm parent-death, so killing only this leader leaves
    that descendant alive until the supervisor cleans the snapshotted group.
    """
    from app.indexing.embed import OnnxProcess
    from app.indexing.ocr_supervisor import OcrSupervisor

    onnx = OnnxProcess("", behavior="hang")
    onnx.start(5)
    ocr = OcrSupervisor(1, {"kind": "fake", "behavior": "hang"})
    ocr.start(5)
    threading.Thread(target=lambda: _stick(onnx.count_tokens, "hang"), name="park-onnx").start()
    threading.Thread(target=lambda: _stick(ocr.read_page, None, 1), name="park-ocr").start()
    end = time.monotonic() + 2
    while time.monotonic() < end and ocr.active_count() < 1:
        time.sleep(0.01)
    pids = [os.getpid(), onnx._proc.pid if onnx._proc is not None else 0]
    if ocr._active:
        pids.append(ocr._active[0][0].pid)
    path = msg.get("pids_path")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(str(pid) for pid in pids))


def _stick(fn, *args) -> None:
    try:
        fn(*args, timeout_s=30)
    except Exception:
        return


def _resistant_hang(msg: dict) -> None:
    """Ignore SIGTERM in this group until the supervisor sends SIGKILL."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
        os._exit(0)
    path = msg.get("pids_path")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n{child}\n")
    time.sleep(float(msg.get("seconds") or 30))


def _socket_block(msg: dict) -> None:
    _mark(msg.get("started_path"))
    sock = socket.create_connection((msg["host"], int(msg["port"])), timeout=30)
    try:
        sock.recv(16)
    finally:
        sock.close()


class _Slot:
    def __init__(self) -> None:
        self.proc: mp.Process | None = None
        self.conn = None
        self.ready = False
        self.busy = False
        self.recovering = False
        self.failures = 0
        self.state = None
        self.identity = None
        self.handle = None


class CallHandle:
    """One in-flight call. Cancellation kills only this slot's process tree."""

    def __init__(self, call_id: int, slot: _Slot) -> None:
        self.call_id = call_id
        self.slot = slot
        self.identity = slot.identity
        self.cancelled = False
        self.finished = False


class IndexGate:
    """A pool of supervised processes. Shutdown and timeout kill the job process."""

    def __init__(self, limit: int, *, grace_s: float = 0.2, recovery_timeout_s: float = 2.0, kind: str = "stage") -> None:
        self.limit = max(1, int(limit))
        self.grace_s = grace_s
        self.recovery_timeout_s = recovery_timeout_s
        self.kind = kind
        self.boot_count = 0
        self.started = False
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._slots: list[_Slot] = [_Slot() for _ in range(self.limit)]
        self._pending = 0
        self._closed = False
        self._status = "idle"
        self._recovery: set[threading.Thread] = set()
        self._recovery_started = 0
        self._before_thread_start = None
        self._before_recover = None
        self._before_proc_start = None
        self._proc_started_at: float | None = None
        self._thread_started_at: float | None = None
        self._shutdown_returned_at: float | None = None
        self._monitor: threading.Thread | None = None
        self._hold: dict | None = None
        self.report: dict = {}
        self._ids = 0
        self._groups = GroupKiller()

    @property
    def status(self) -> str:
        return self._status

    @property
    def serving(self) -> bool:
        if self._closed:
            return False
        if not self.started:
            return True
        return self._status == "ready" and self._alive_slots() > 0

    def blocked_reason(self) -> str:
        if self._closed:
            return "pipeline_busy"
        if self._status in ("recovering", "initializing", "terminating"):
            return "worker_recovering"
        return "worker_failed"

    def ocr_census(self) -> dict:
        """Live OCR state for every slot. A missing value is fail-closed."""
        alive = recovering = failed = missing = 0
        with self._lock:
            slots = list(self._slots)
        for slot in slots:
            if slot.state is None:
                missing += 1
                continue
            try:
                code = int(slot.state.value)
            except Exception:
                missing += 1
                continue
            process_alive = slot.proc is not None and slot.proc.is_alive()
            if code == 1 and process_alive:
                alive += 1
            elif code == 2:
                recovering += 1
            else:
                failed += 1
        return {
            "expected": self.limit,
            "alive": alive,
            "recovering": recovering,
            "failed": failed,
            "missing": missing,
        }

    def health_problem(self, settings) -> str | None:
        if not self.started:
            return None
        if self._closed:
            return "pipeline_busy"
        if self._status in ("recovering", "initializing", "terminating"):
            return "worker_recovering"
        if self._status != "ready" or self._alive_slots() < 1:
            return "worker_failed"
        if getattr(settings, "embedding_provider", "") == "onnx" and self.report.get("embedding_status") not in (None, "ready"):
            return "embedding_model_missing"
        if self.report.get("cpu_compatible") is False:
            return "embedding_cpu_unsupported"
        if getattr(settings, "ocr_provider", "") == "paddle":
            census = self.ocr_census()
            if census["missing"] or census["failed"]:
                return "ocr_model_missing"
            if census["recovering"]:
                return "worker_recovering"
            if census["alive"] < self.limit:
                return "ocr_model_missing"
        return None

    def start(self, timeout_s: float | None = None) -> None:
        if self.started:
            return
        budget = self.recovery_timeout_s if timeout_s is None else timeout_s
        self._status = "initializing"
        try:
            for slot in self._slots:
                self._spawn_into(slot, budget)
        except IndexFailure:
            self.shutdown(timeout_s=self.grace_s)
            self._status = "failed"
            raise
        self.started = True
        self._status = "ready"
        self._monitor = threading.Thread(target=self._monitor_loop, name="index-slot-monitor")
        self._monitor.start()

    def ensure_started(self) -> None:
        if self._closed:
            raise IndexFailure("pipeline_busy")
        if not self.started:
            self.start(self.recovery_timeout_s)

    def arm_hold(self, started_path: str, release_path: str, seconds: float = 30) -> None:
        self._hold = {"started_path": started_path, "release_path": release_path, "seconds": seconds}

    def pending(self) -> int:
        with self._lock:
            return self._pending

    def worker_pids(self) -> list[int]:
        with self._lock:
            return [slot.proc.pid for slot in self._slots if slot.proc is not None and slot.proc.pid]

    def submit(self, message: dict, timeout_s: float | None = None) -> CallHandle:
        self.ensure_started()
        if message.get("op") == "pipeline" and self._hold is not None and self.kind != "pipeline":
            hold = self._hold
            self._hold = None
            message = {"op": "hang", "stage": "handler", **hold}
        budget = self.recovery_timeout_s if timeout_s is None else min(float(timeout_s), self.recovery_timeout_s)
        slot = self._checkout(budget)
        with self._lock:
            self._ids += 1
            handle = CallHandle(self._ids, slot)
            slot.handle = handle
        try:
            slot.conn.send(message)
        except (EOFError, OSError) as exc:
            self.cancel(handle)
            raise IndexFailure("timeout") from exc
        return handle

    def wait(self, handle: CallHandle, timeout_s: float) -> dict:
        end = time.monotonic() + float(timeout_s)
        while True:
            if handle.cancelled:
                raise IndexFailure("timeout")
            if self._closed:
                self.cancel(handle)
                raise IndexFailure("pipeline_busy")
            left = end - time.monotonic()
            if left <= 0:
                self.cancel(handle)
                raise IndexFailure("timeout")
            conn = handle.slot.conn
            try:
                ready = conn is not None and conn.poll(min(0.02, left))
            except (EOFError, OSError) as exc:
                self.cancel(handle)
                raise IndexFailure("timeout") from exc
            if not ready:
                continue
            try:
                reply = conn.recv()
            except (EOFError, OSError) as exc:
                self.cancel(handle)
                raise IndexFailure("timeout") from exc
            with self._lock:
                if handle.cancelled:
                    raise IndexFailure("timeout")
                handle.finished = True
                if handle.slot.busy:
                    handle.slot.busy = False
                    self._pending = max(0, self._pending - 1)
            if not isinstance(reply, dict):
                raise IndexFailure("worker_failed")
            if reply.get("event") == "error":
                raise IndexFailure(str(reply.get("code") or "worker_failed"))
            return reply

    def cancel(self, handle: CallHandle | None) -> None:
        """Stop one call. A second call with the same handle does not signal a new group."""
        if handle is None:
            return
        with self._lock:
            if handle.finished or handle.cancelled:
                return
            handle.cancelled = True
            slot = handle.slot
            identity = handle.identity
            if slot.busy:
                slot.busy = False
                self._pending = max(0, self._pending - 1)
            slot.ready = False
            if not self._closed:
                self._status = "recovering"
        cleaned = self._kill_tree(identity, slot)
        if self._closed:
            return
        if cleaned:
            self._begin_recovery(slot)
            return
        with self._lock:
            slot.ready = False
            slot.recovering = False
            self._status = "failed"

    def call(self, message: dict, timeout_s: float, cancel_event=None) -> dict:
        handle = self.submit(message, timeout_s)
        watcher = None
        if cancel_event is not None:
            def _watch() -> None:
                while not handle.finished and not handle.cancelled:
                    if cancel_event.is_set():
                        self.cancel(handle)
                        return
                    time.sleep(0.01)

            watcher = threading.Thread(target=_watch, name="index-cancel")
            watcher.start()
        try:
            return self.wait(handle, timeout_s)
        finally:
            if watcher is not None:
                watcher.join(0.2)

    def shutdown(self, timeout_s: float = 5.0, models=None) -> int:
        with self._lock:
            self._closed = True
            if self._status not in ("stopped", "failed"):
                self._status = "terminating"
            threads = [thread for thread in self._recovery if thread is not threading.current_thread()]
            slots = list(self._slots)
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread() and monitor.is_alive():
            monitor.join(min(1.0, timeout_s))
        for slot in slots:
            handle = slot.handle
            if handle is not None and not handle.finished and not handle.cancelled and slot.busy:
                self.cancel(handle)
            else:
                self._kill_tree(slot.identity, slot)
                slot.ready = False
        for item in models or ():
            stop = getattr(item, "shutdown", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    continue
        deadline = time.monotonic() + timeout_s
        for thread in threads:
            left = deadline - time.monotonic()
            thread.join(left if left > 0 else 0)
        with self._lock:
            slots = list(self._slots)
        dirty = any(thread.is_alive() for thread in threads)
        for slot in slots:
            if not self._kill_tree(slot.identity, slot):
                dirty = True
        from app.indexing.deadline import join_closers

        join_closers(max(0.0, deadline - time.monotonic()))
        with self._lock:
            if dirty:
                self._status = "failed"
            elif self._status != "failed":
                self._status = "stopped"
            pending = self._pending
            self._shutdown_returned_at = time.monotonic()
        return pending

    def _checkout(self, wait_s: float) -> _Slot:
        end = time.monotonic() + max(0.0, wait_s)
        while True:
            self._sweep_dead_slots()
            with self._lock:
                recovering = False
                busy_live = False
                for slot in self._slots:
                    if slot.recovering:
                        recovering = True
                    if slot.ready and not slot.busy and slot.proc is not None and slot.proc.is_alive():
                        slot.busy = True
                        self._pending += 1
                        return slot
                    if slot.busy and slot.proc is not None and slot.proc.is_alive():
                        busy_live = True
            if busy_live and not recovering:
                raise IndexFailure("pipeline_busy")
            if self._closed or time.monotonic() >= end:
                raise IndexFailure("pipeline_busy")
            time.sleep(0.01)

    def _release(self, slot: _Slot) -> None:
        with self._lock:
            if slot.busy:
                slot.busy = False
                self._pending = max(0, self._pending - 1)

    def _kill_tree(self, identity, slot: _Slot) -> bool:
        """Stop this slot. True only when its process tree is confirmed gone."""
        proc = slot.proc
        if identity is None and (proc is None or not proc.is_alive()):
            self._close_slot(slot)
            slot.ready = False
            slot.proc = None
            return True
        confirmed = False
        if identity:
            confirmed = self._groups.kill(identity, self.grace_s)
        if not confirmed and proc is not None:
            try:
                confirmed = reap_spawned(proc)
            except IdentityError:
                confirmed = False
        self._close_slot(slot)
        if proc is not None:
            proc.join(0.5)
            if proc.is_alive():
                confirmed = False
        if identity and confirmed and not self._groups.kill(identity, 0):
            confirmed = False
        if identity:
            try:
                if read_process(identity[0]) is not None:
                    confirmed = False
            except IdentityError:
                confirmed = False
        slot.ready = False
        if confirmed:
            slot.proc = None
            slot.identity = None
        return confirmed

    def _close_slot(self, slot: _Slot) -> None:
        try:
            if slot.conn is not None:
                slot.conn.close()
        except OSError:
            pass
        slot.conn = None

    def _stop_process(self, proc: mp.Process, conn, identity=None) -> bool:
        slot = _Slot()
        slot.proc = proc
        slot.conn = conn
        slot.identity = identity
        return self._kill_tree(identity, slot)

    def _monitor_loop(self) -> None:
        while not self._closed:
            self._sweep_dead_slots()
            time.sleep(0.05)

    def _sweep_dead_slots(self) -> None:
        with self._lock:
            slots = list(self._slots)
        for slot in slots:
            if self._closed:
                return
            action = self._claim_dead(slot)
            if action == "recover":
                self._launch_recovery(slot)
            elif action == "seal":
                self._kill_tree(slot.identity, slot)
                self._fail_slot(slot)

    def _claim_dead(self, slot: _Slot) -> str | None:
        """One caller owns a ready slot whose leader has died.

        The first death starts one replacement. A later death is cleaned and
        left failed so a crash loop cannot spawn forever.
        """
        with self._lock:
            if self._closed or slot.recovering or slot.busy or not slot.ready:
                return None
            proc = slot.proc
            if proc is None or proc.is_alive():
                return None
            slot.ready = False
            if slot.failures >= 1:
                self._status = "failed"
                return "seal"
            slot.recovering = True
            slot.failures += 1
            self._status = "recovering"
            return "recover"

    def _begin_recovery(self, slot: _Slot) -> None:
        with self._lock:
            if self._closed:
                self._status = "stopped"
                return
            if slot.recovering:
                return
            slot.recovering = True
            if not any(item.ready and item is not slot for item in self._slots):
                self._status = "recovering"
        self._launch_recovery(slot)

    def _launch_recovery(self, slot: _Slot) -> None:
        """Register and start one recovery thread while holding the lifecycle lock.

        Shutdown takes the same lock to publish `_closed` and to snapshot the
        set, so it cannot miss a thread that has already been started.
        """
        with self._lock:
            if self._closed:
                return
            thread = threading.Thread(
                target=self._recover_slot, args=(slot,), name="index-recovery", daemon=True
            )
            self._recovery.add(thread)
            hook = self._before_thread_start
            if hook is not None:
                hook(thread)
            try:
                thread.start()
            except Exception:
                self._recovery.discard(thread)
                raise
            self._recovery_started += 1
            self._thread_started_at = time.monotonic()

    def _recover_slot(self, slot: _Slot) -> None:
        try:
            try:
                hook = self._before_recover
                if hook is not None:
                    hook()
                if self._closed:
                    return
                identity = slot.identity
                leader_dead = slot.proc is None or not slot.proc.is_alive()
                if identity is not None and leader_dead:
                    if not self._kill_tree(identity, slot):
                        self._fail_slot(slot)
                        return
                if self._closed:
                    return
                self._spawn_into(slot, self.recovery_timeout_s)
                with self._lock:
                    slot.recovering = False
                    if self._closed:
                        return
                    if any(item.recovering for item in self._slots):
                        self._status = "recovering"
                    else:
                        self._status = "ready"
            except Exception:
                self._fail_slot(slot)
        finally:
            with self._lock:
                self._recovery.discard(threading.current_thread())

    def _fail_slot(self, slot: _Slot) -> None:
        with self._lock:
            slot.recovering = False
            slot.ready = False
            if not self._closed:
                self._status = "failed"

    def _spawn_into(self, slot: _Slot, timeout_s: float) -> None:
        if self._closed:
            raise IndexFailure("worker_failed")
        if slot.proc is not None and slot.proc.is_alive():
            if not self._kill_tree(slot.identity, slot):
                slot.ready = False
                self._status = "failed"
                raise IndexFailure("worker_failed")
        if slot.state is None and self.kind == "pipeline":
            slot.state = self._ctx.Value("i", 0)
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=slot_main, args=(child, self.kind, slot.state), daemon=False)
        self._commit_spawn(slot, proc, parent, child)
        slot.identity = capture(proc.pid, timeout_s=min(1.0, max(0.05, float(timeout_s))))
        if slot.identity is None or self._closed:
            cleaned = self._kill_tree(slot.identity, slot)
            if not cleaned:
                self._status = "failed"
            raise IndexFailure("worker_failed")
        if not parent.poll(timeout_s):
            cleaned = self._kill_tree(slot.identity, slot)
            if not cleaned:
                self._status = "failed"
            raise IndexFailure("timeout")
        try:
            msg = parent.recv()
        except (EOFError, OSError) as exc:
            cleaned = self._kill_tree(slot.identity, slot)
            if not cleaned:
                self._status = "failed"
            raise IndexFailure("worker_failed") from exc
        if not isinstance(msg, dict) or msg.get("event") != "ready" or not proc.is_alive() or slot.identity is None:
            cleaned = self._kill_tree(slot.identity, slot)
            if not cleaned:
                self._status = "failed"
            raise IndexFailure("worker_failed")
        slot.ready = True
        if isinstance(msg.get("report"), dict):
            self.report = msg["report"]

    def _commit_spawn(self, slot: _Slot, proc: mp.Process, parent, child) -> None:
        """Start a process only if shutdown has not published `_closed`.

        The lifecycle lock covers the closed check, slot registration, and
        `proc.start()`. It is not held while the caller captures identity or
        waits for the worker to become ready.
        """
        hook = self._before_proc_start
        if hook is not None:
            hook()
        start_failed = False
        with self._lock:
            if not self._closed:
                slot.proc = proc
                slot.conn = parent
                slot.ready = False
                slot.identity = None
                try:
                    proc.start()
                except Exception:
                    slot.proc = None
                    slot.conn = None
                    slot.identity = None
                    slot.ready = False
                    start_failed = True
                else:
                    child.close()
                    self.boot_count += 1
                    self._proc_started_at = time.monotonic()
                    return
        self._discard_pipes(parent, child)
        if start_failed and proc.pid:
            try:
                proc.kill()
            except OSError:
                pass
            proc.join(0.2)
        raise IndexFailure("worker_failed")

    def _discard_pipes(self, parent, child) -> None:
        for pipe in (parent, child):
            try:
                pipe.close()
            except OSError:
                continue

    def _alive_slots(self) -> int:
        return sum(1 for slot in self._slots if slot.proc is not None and slot.proc.is_alive() and slot.ready)
