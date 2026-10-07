"""One PaddleOCR process per slot. A hung call is killed before the slot is released."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from multiprocessing.connection import Connection

from app.indexing.errors import IndexFailure

_STATUSES = ("initializing", "ready", "recovering", "failed", "timed_out")
_STATE_CODE = {"ready": 1, "recovering": 2, "initializing": 2, "failed": 3, "timed_out": 3, "stopped": 3}


def _worker(conn: Connection, spec: dict) -> None:
    """Stay in the pipeline process group. Do not arm parent-death.

    Recovery runs on a short-lived thread. Linux sends PR_SET_PDEATHSIG when
    that thread exits, which would kill the replacement worker.
    """
    kind = spec.get("kind")
    try:
        if kind == "fake":
            behavior = spec.get("behavior", "ok")
            if behavior == "init_hang":
                time.sleep(3600)
            if behavior == "init_crash":
                os._exit(2)
            conn.send({"event": "ready", "pid": os.getpid()})
            while True:
                msg = conn.recv()
                if not isinstance(msg, dict) or msg.get("op") == "shutdown":
                    conn.send({"event": "bye"})
                    return
                if behavior == "hang":
                    time.sleep(3600)
                if behavior == "crash":
                    os._exit(3)
                if behavior == "slow":
                    time.sleep(float(spec.get("delay_s", 0.3)))
                conn.send({"event": "result", "text": "ok", "pid": os.getpid(), "page": msg.get("page")})
            return
        from app.indexing.ocr import PaddleOCR

        engine = PaddleOCR(str(spec.get("model_dir") or ""))
        conn.send({"event": "ready", "pid": os.getpid()})
        while True:
            msg = conn.recv()
            if not isinstance(msg, dict) or msg.get("op") == "shutdown":
                conn.send({"event": "bye"})
                return
            text = engine.read_page(msg.get("image"), int(msg.get("page") or 0))
            conn.send({"event": "result", "text": text, "pid": os.getpid(), "page": msg.get("page")})
    except Exception:
        try:
            conn.send({"event": "failed"})
        except Exception:
            pass
        os._exit(4)


class OcrSupervisor:
    """Bounded process pool. Each process loads one engine and serves one call at a time."""

    version = "paddleocr-2.9.1-dbnet-crnn-cpu"
    hard_timeout = True

    def __init__(self, slots: int, spec: dict, replace_timeout_s: float | None = None) -> None:
        self.slots = max(1, slots)
        self.spec = spec
        self.status = "initializing"
        configured = spec.get("replace_timeout_s", spec.get("init_timeout_s", 30))
        self.replace_timeout_s = float(replace_timeout_s if replace_timeout_s is not None else configured)
        self.spawn_count = 0
        self._ctx = mp.get_context("spawn")
        self._idle: list[tuple[mp.Process, Connection]] = []
        self._active: list[tuple[mp.Process, Connection]] = []
        self._sem = threading.BoundedSemaphore(self.slots)
        self._guard = threading.Lock()
        self._spawn_lock = threading.Lock()
        self._closed = False
        self._booting: list = []
        self._recovery_thread = None
        self._state = None
        self.max_recoveries = 1

    def start(self, timeout_s: float) -> None:
        self.status = "initializing"
        self._closed = False
        try:
            for _ in range(self.slots):
                self._spawn(timeout_s)
        except IndexFailure as exc:
            self.shutdown()
            if exc.code == "ocr_timeout":
                self.status = "timed_out"
            else:
                self.status = "failed"
            raise
        self.status = "ready"
        self._publish()

    def bind_state(self, state) -> None:
        self._state = state
        self._publish()

    def accepting(self) -> bool:
        return self.status == "ready" and not self._closed

    def wait_settled(self, timeout_s: float = 2.0) -> str:
        thread = self._recovery_thread
        if thread is not None:
            thread.join(timeout_s)
        return self.status

    def _publish(self) -> None:
        if self._state is None:
            return
        self._state.value = _STATE_CODE.get(self.status, 3)

    def _set_status(self, status: str) -> None:
        self.status = status
        self._publish()

    def read_page(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> str:
        if deadline is not None:
            timeout_s = deadline.timeout_for_io(timeout_s)
        if not self._await_ready(timeout_s):
            raise IndexFailure("ocr_model_missing")
        if not self._sem.acquire(blocking=False):
            raise IndexFailure("pipeline_busy")
        worker: tuple[mp.Process, Connection] | None = None
        checked_out = False
        try:
            worker = self._checkout()
            checked_out = True
            proc, conn = worker
            conn.send({"op": "ocr", "image": image, "page": page_number})
            end = time.monotonic() + timeout_s
            ready = False
            deadline_hit = False
            while True:
                if self._closed:
                    break
                if deadline is not None:
                    try:
                        deadline.check()
                    except IndexFailure:
                        deadline_hit = True
                        break
                left = end - time.monotonic()
                if left <= 0:
                    break
                if conn.poll(min(0.05, left)):
                    ready = True
                    break
            if not ready:
                checked_out = False
                self._retire(worker)
                raise IndexFailure("timeout" if deadline_hit else "ocr_timeout")
            try:
                msg = conn.recv()
            except (EOFError, OSError) as exc:
                checked_out = False
                self._retire(worker)
                raise IndexFailure("ocr_failed") from exc
            if proc.exitcode not in (None, 0) or not isinstance(msg, dict) or msg.get("event") != "result":
                checked_out = False
                self._retire(worker)
                raise IndexFailure("ocr_failed")
            if msg.get("page") not in (None, page_number):
                checked_out = False
                self._retire(worker)
                raise IndexFailure("ocr_failed")
            if deadline is not None:
                deadline.check()
            return str(msg.get("text") or "")
        finally:
            if checked_out and worker is not None:
                self._checkin(worker)
            self._sem.release()

    def shutdown(self) -> None:
        with self._guard:
            self._closed = True
        with self._spawn_lock:
            pass
        with self._guard:
            workers = list(self._idle) + list(self._active)
            booting = list(self._booting)
            self._idle = []
            self._active = []
            self._booting = []
        for proc in booting:
            if proc.is_alive():
                proc.kill()
            proc.join(1)
        for proc, conn in workers:
            self._kill_joined(proc, conn)
        thread = self._recovery_thread
        if thread is not None:
            thread.join(2)
        if self.status == "initializing":
            self._set_status("failed")
        else:
            self._set_status("stopped")

    def worker_count(self) -> int:
        return self.alive_workers()

    def alive_workers(self) -> int:
        with self._guard:
            return sum(1 for proc, _conn in self._idle + self._active if proc.is_alive())

    def idle_count(self) -> int:
        with self._guard:
            return len(self._idle)

    def active_count(self) -> int:
        with self._guard:
            return len(self._active)

    def _spawn(self, timeout_s: float) -> None:
        with self._spawn_lock:
            with self._guard:
                if self._closed:
                    raise IndexFailure("ocr_failed")
                parent, child = self._ctx.Pipe(duplex=True)
                proc = self._ctx.Process(target=_worker, args=(child, dict(self.spec)), daemon=False)
                self.spawn_count += 1
                self._booting.append(proc)
            delay = float(self.spec.get("start_delay_s") or 0)
            if delay:
                time.sleep(delay)
            with self._guard:
                if self._closed:
                    self._booting = [item for item in self._booting if item is not proc]
                    child.close()
                    parent.close()
                    raise IndexFailure("ocr_failed")
            proc.start()
            child.close()
            with self._guard:
                if self._closed:
                    self._abort_started(proc, parent)
                    raise IndexFailure("ocr_failed")
        if not parent.poll(timeout_s):
            self._forget_booting(proc)
            self._kill_joined(proc, parent)
            raise IndexFailure("ocr_timeout")
        try:
            msg = parent.recv()
        except (EOFError, OSError) as exc:
            self._forget_booting(proc)
            self._kill_joined(proc, parent)
            raise IndexFailure("ocr_failed") from exc
        if not isinstance(msg, dict) or msg.get("event") != "ready" or not proc.is_alive():
            self._forget_booting(proc)
            self._kill_joined(proc, parent)
            raise IndexFailure("ocr_failed")
        with self._guard:
            self._booting = [item for item in self._booting if item is not proc]
            self._idle.append((proc, parent))

    def _abort_started(self, proc: mp.Process, conn: Connection) -> None:
        with self._guard:
            self._booting = [item for item in self._booting if item is not proc]
        self._kill_joined(proc, conn)

    def _forget_booting(self, proc: mp.Process) -> None:
        with self._guard:
            self._booting = [item for item in self._booting if item is not proc]

    def _checkout(self) -> tuple[mp.Process, Connection]:
        with self._guard:
            if not self._idle:
                raise IndexFailure("ocr_model_missing")
            worker = self._idle.pop()
            self._active.append(worker)
            return worker

    def _checkin(self, worker: tuple[mp.Process, Connection]) -> None:
        with self._guard:
            self._active = [item for item in self._active if item[0] is not worker[0]]
            if not self._closed and worker[0].is_alive():
                self._idle.append(worker)

    def _retire(self, worker: tuple[mp.Process, Connection], replace_budget: float | None = None) -> None:
        del replace_budget
        with self._guard:
            self._active = [item for item in self._active if item[0] is not worker[0]]
            self._idle = [item for item in self._idle if item[0] is not worker[0]]
        self._kill_joined(worker[0], worker[1])
        if self._closed:
            self._set_status("stopped")
            return
        recover = self.spec.get("recover")
        if recover:
            self.spec = {**self.spec, "behavior": recover}
        self._schedule_recovery()

    def _await_ready(self, timeout_s: float) -> bool:
        if self.status == "ready" and not self._closed:
            return True
        thread = self._recovery_thread
        if thread is not None:
            thread.join(max(0.0, timeout_s))
        return self.status == "ready" and not self._closed

    def _schedule_recovery(self) -> None:
        with self._guard:
            if self._closed:
                return
            if self._recovery_thread is not None and self._recovery_thread.is_alive():
                return
            self._set_status("recovering")
            thread = threading.Thread(target=self._recover, name="ocr-recovery")
            self._recovery_thread = thread
        thread.start()

    def _recover(self) -> None:
        delay = 0.02
        for _attempt in range(self.max_recoveries):
            if self._closed:
                return
            time.sleep(delay)
            if self._closed:
                return
            try:
                self._spawn(self.replace_timeout_s)
            except IndexFailure:
                delay = min(delay * 2, 0.5)
                continue
            if self.alive_workers() >= 1:
                self._set_status("ready")
                return
        if self.alive_workers() < 1:
            self._set_status("failed")

    def _kill_joined(self, proc: mp.Process, conn: Connection) -> None:
        try:
            conn.close()
        except OSError:
            pass
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
