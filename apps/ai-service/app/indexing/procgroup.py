"""Own one job's process group without shelling out to ps.

Identity is the kernel tuple (pid, start time, verified pgid). Linux reads
/proc/<pid>/stat. macOS reads proc_pidinfo. A recycled PID has a different
start time and is not signaled.
"""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import sys
import threading
import time


class IdentityError(RuntimeError):
    """A process-identity read failed. Cleanup must be retried, not skipped."""


class ProcInfo:
    __slots__ = ("pid", "ppid", "pgid", "start", "state")

    def __init__(self, pid: int, ppid: int, pgid: int, start: int, state: str) -> None:
        self.pid = pid
        self.ppid = ppid
        self.pgid = pgid
        self.start = start
        self.state = state

    @property
    def zombie(self) -> bool:
        return self.state in ("Z", "X")


def arm_parent_death() -> None:
    """On Linux, ask the kernel to kill this process when its parent dies."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:
            return
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        return


def become_group_leader() -> int:
    """Move this process into a new session so killpg cannot hit the parent.

    Do not arm parent-death here. A replacement worker is spawned by a
    short-lived recovery thread, and Linux delivers PR_SET_PDEATHSIG when
    that thread exits.
    """
    try:
        os.setsid()
    except OSError:
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
    return os.getpgrp()


def _parse_linux_stat(text: str) -> ProcInfo:
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        raise IdentityError("truncated /proc stat")
    try:
        pid = int(text.split(" ", 1)[0])
        fields = text[end + 2 :].split()
        state = fields[0]
        ppid = int(fields[1])
        pgid = int(fields[2])
        start = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise IdentityError("unreadable /proc stat") from exc
    return ProcInfo(pid, ppid, pgid, start, state)


def _read_linux(pid: int) -> ProcInfo | None:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise IdentityError("cannot read /proc stat") from exc
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        raise IdentityError("cannot read pgid") from exc
    info = _parse_linux_stat(text)
    if info.pid != pid or info.pgid != pgid:
        raise IdentityError("process identity changed while reading")
    return info


_DARWIN_STATUS_ZOMBIE = 5
_libproc = None


def _darwin_lib():
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _libproc.proc_pidinfo.restype = ctypes.c_int
        _libproc.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _libproc.proc_listpids.restype = ctypes.c_int
    return _libproc


class _DarwinBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _read_darwin(pid: int) -> ProcInfo | None:
    info = _DarwinBsdInfo()
    size = ctypes.sizeof(info)
    read = _darwin_lib().proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
    if read == 0:
        err = ctypes.get_errno()
        if err in (0, errno.ESRCH, errno.ENOENT):
            return None
        raise IdentityError("proc_pidinfo failed")
    if read != size or int(info.pbi_pid) != pid:
        raise IdentityError("proc_pidinfo returned a different process")
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        raise IdentityError("cannot read pgid") from exc
    if int(info.pbi_pgid) != pgid:
        raise IdentityError("process group changed while reading")
    state = "Z" if int(info.pbi_status) == _DARWIN_STATUS_ZOMBIE else "R"
    start = int(info.pbi_start_tvsec) * 1_000_000 + int(info.pbi_start_tvusec)
    return ProcInfo(pid, int(info.pbi_ppid), pgid, start, state)


def read_process(pid: int) -> ProcInfo | None:
    """Return kernel identity, or None when the pid is gone.

    A transient read failure raises IdentityError. Callers must not treat that
    as proof that the process exited.
    """
    if pid <= 0:
        return None
    try:
        if sys.platform.startswith("linux"):
            info = _read_linux(pid)
        elif sys.platform == "darwin":
            info = _read_darwin(pid)
        else:
            raise IdentityError("kernel process identity is unavailable")
    except IdentityError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError as exc:
            raise IdentityError("cannot probe process") from exc
        raise
    return info


def process_alive(pid: int) -> bool:
    """True when the pid exists and is not a zombie. Does not call ps."""
    info = read_process(pid)
    return info is not None and not info.zombie


def list_processes_in_group(pgid: int) -> list[ProcInfo]:
    """Every process whose kernel process group is pgid, including zombies."""
    if pgid <= 1:
        return []
    found: list[ProcInfo] = []
    if sys.platform.startswith("linux"):
        try:
            names = os.listdir("/proc")
        except OSError as exc:
            raise IdentityError("cannot list /proc") from exc
        pids = [int(name) for name in names if name.isdigit()]
    elif sys.platform == "darwin":
        lib = _darwin_lib()
        needed = lib.proc_listpids(1, 0, None, 0)
        if needed < 0:
            raise IdentityError("proc_listpids failed")
        count = max(needed // 4, 1) + 128
        buf = (ctypes.c_int * count)()
        size = lib.proc_listpids(1, 0, ctypes.byref(buf), ctypes.sizeof(buf))
        if size < 0:
            raise IdentityError("proc_listpids failed")
        pids = [int(buf[i]) for i in range(size // ctypes.sizeof(ctypes.c_int)) if buf[i] > 0]
    else:
        raise IdentityError("kernel process identity is unavailable")
    for pid in pids:
        try:
            info = read_process(pid)
        except IdentityError:
            # Unrelated processes can be unreadable. Our own group is readable,
            # and a failed scan must not look like an empty group for a pid we
            # were able to read on a previous call. Skip only this pid.
            continue
        if info is not None and info.pgid == pgid:
            found.append(info)
    return found


def capture(pid: int, timeout_s: float = 1.0):
    """Return (pid, starttime, pgid) once the process leads its own group.

    None means the process exited or never became a dedicated group leader
    before the deadline. Callers must not mark that process ready.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        info = read_process(pid)
        if info is None:
            return None
        if not info.zombie and info.pgid == pid and info.pgid > 1:
            return (info.pid, info.start, info.pgid)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)


def reap_spawned(proc) -> bool:
    """SIGKILL and reap a process this parent just spawned. True only if it is gone."""
    if proc is None:
        return True
    pid = getattr(proc, "pid", None)
    if not pid:
        return True
    info = read_process(pid)
    own = os.getpgrp()
    if (
        info is not None
        and not info.zombie
        and info.pgid == pid
        and info.pgid > 1
        and info.pgid != own
        and info.ppid == os.getpid()
    ):
        _signal_group(info.pgid, signal.SIGKILL)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise IdentityError("cannot stop spawned process") from exc
    proc.join(1.0)
    if proc.is_alive():
        return False
    leftover = read_process(pid)
    if leftover is not None:
        return False
    if info is not None and info.pgid == pid and info.pgid > 1 and info.pgid != own:
        if list_processes_in_group(info.pgid):
            return False
    return True


def _signal_group(pgid: int, sig: int) -> None:
    if pgid <= 1 or pgid == os.getpgrp():
        raise IdentityError("refusing to signal the parent process group")
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise IdentityError("cannot signal process group") from exc


class GroupKiller:
    """SIGTERM, then SIGKILL the verified group. Done only after the group is empty.

    The bookkeeping lock is held only to read or update shared state. Signaling,
    the grace wait, process scans, and reaping run under a per-identity lock so
    different groups are cleaned up together.
    """

    def __init__(self) -> None:
        self._done: set[tuple] = set()
        self._gates: dict[tuple, threading.Lock] = {}
        self._book = threading.Lock()
        self.on_grace = None

    def kill(self, identity, grace_s: float) -> bool:
        """Return True only when this identity's group is confirmed gone.

        A second call for the same identity waits for the first and then returns
        without signaling again. A failed read or a surviving member leaves the
        identity out of the completed set.
        """
        if not identity or len(identity) != 3:
            return False
        key = (int(identity[0]), int(identity[1]), int(identity[2]))
        with self._book:
            if key in self._done:
                return True
            gate = self._gates.get(key)
            if gate is None:
                gate = threading.Lock()
                self._gates[key] = gate
        with gate:
            with self._book:
                if key in self._done:
                    return True
            try:
                confirmed = self._cleanup(key, grace_s)
            except IdentityError:
                return False
            if not confirmed:
                return False
            with self._book:
                self._done.add(key)
            return True

    def _cleanup(self, key, grace_s: float) -> bool:
        if self._clear(key):
            return True
        self._signal(key, signal.SIGTERM)
        hook = self.on_grace
        if hook is not None:
            hook(key)
        deadline = time.monotonic() + max(0.0, float(grace_s))
        while time.monotonic() < deadline:
            if self._clear(key):
                break
            time.sleep(0.01)
        if not self._clear(key):
            self._signal(key, signal.SIGKILL)
        self._reap_zombies(key)
        for _ in range(30):
            if self._clear(key):
                return True
            self._reap_zombies(key)
            time.sleep(0.01)
        return self._clear(key)

    def _leader(self, identity):
        return read_process(identity[0])

    def _owned(self, identity) -> tuple[list[ProcInfo], list[ProcInfo]]:
        pid, start, pgid = identity
        leader = self._leader(identity)
        reused_at = leader.start if leader is not None and leader.start != start else None
        alive: list[ProcInfo] = []
        zombies: list[ProcInfo] = []
        for member in list_processes_in_group(pgid):
            if member.start < start:
                continue
            if member.pid == pid and (leader is None or leader.start != start):
                continue
            if reused_at is not None and member.pid != pid and member.start >= reused_at:
                continue
            if member.zombie:
                zombies.append(member)
            else:
                alive.append(member)
        return alive, zombies

    def _clear(self, identity) -> bool:
        """True when no live member remains. The leader zombie is reaped by join()."""
        alive, zombies = self._owned(identity)
        if alive:
            return False
        leader_pid = identity[0]
        return all(item.pid == leader_pid for item in zombies)

    def _signal(self, identity, sig: int) -> None:
        pid, start, pgid = identity
        leader = self._leader(identity)
        if (
            leader is not None
            and leader.start == start
            and leader.pgid == pgid
            and not leader.zombie
        ):
            _signal_group(pgid, sig)
            return
        for member in self._owned(identity)[0]:
            current = read_process(member.pid)
            if current is None or current.start != member.start or current.pgid != pgid or current.zombie:
                continue
            try:
                os.kill(current.pid, sig)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                raise IdentityError("cannot signal group member") from exc

    def _reap_zombies(self, identity) -> None:
        leader_pid = identity[0]
        _, zombies = self._owned(identity)
        for item in zombies:
            if item.pid == leader_pid:
                continue
            current = read_process(item.pid)
            if current is None or current.ppid != os.getpid():
                continue
            try:
                os.waitpid(item.pid, os.WNOHANG)
            except ChildProcessError:
                continue
