#!/usr/bin/env python3
"""Filesystem probe v2 for the RunPod network volume (storage decision, Phase 8.1-B).

Measures, one property at a time, what the data services would need from the
filesystem under /workspace/cas: ownership and modes, atomic mkdir and rename,
flock and fcntl locks, O_DIRECT, shared mmap, directory fsync, fsync latency
(alone and concurrent), throughput, plus the container's capabilities and the
container disk. Every check reports PASS, FAIL or SKIP with its evidence; a failing
check never stops the others.

Safety:
  - writes ONLY random test data inside a new directory <root>/.probe-<utc>-<hex>
    (default root /workspace/cas), created with mkdir (never reused);
  - refuses a root that is a symlink or resolves elsewhere, and never follows
    symlinks inside the probe directory;
  - never touches anything else (preserve/, models/, the research tree);
  - plans every write against --max-bytes (default and maximum 600 MB in total) and
    --footprint-cap-mb (default and maximum 300 MB on disk), and also MEASURES the
    probe directory after every benchmark: over the cap, every remaining check stops;
  - every forked check process is tracked and always ends in os._exit; on SIGINT or
    SIGTERM all of them are killed and reaped BEFORE the probe directory is removed;
  - removes the probe directory on exit, then verifies that it is gone, that no
    check process is left, and that every other entry under <root> is unchanged;
  - prints the report to stdout only (pipe it to a file on the container disk).

Not measured here (by design): whether an fsync that returns success is durable
across a host or power failure, and locking between two pods on two hosts. Both
tests run on one host and one kernel. The S3 API is a separate, not yet approved step.

  python3 probe_v2.py [--root /workspace/cas] [--uid 10010] [--json]
Run as root (the uid checks fork and drop to --uid). Exit 0 when the probe and its
cleanup completed (whatever the PASS/FAIL results), 1 when the cleanup could not be
verified, 2 on a refused start.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import mmap
import os
import platform
import secrets
import shutil
import signal
import stat
import statistics
import sys
import time

KIB, MIB = 1024, 1024 * 1024
CAP_NAMES = {0: "CAP_CHOWN", 1: "CAP_DAC_OVERRIDE", 3: "CAP_FOWNER", 21: "CAP_SYS_ADMIN"}

results: list[dict] = []
written = {"bytes": 0, "peak_on_disk": 0}
state = {"probe_dir": None, "on_disk": 0, "max_bytes": 600 * MIB, "footprint_cap": 300 * MIB,
         "main_pid": os.getpid(), "measured_peak": 0}
children: set[int] = set()  # forked check processes still to be reaped (main process only)
STOP_SIGNALS = {signal.SIGINT, signal.SIGTERM}


def fork_tracked() -> int:
    """os.fork() whose child the main process always knows about.

    SIGINT/SIGTERM are blocked across the fork so that the handler can never run
    between the fork and the registration of the new pid. The child gets the
    default handlers (a signal simply ends it) and an empty pid set."""
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        pid = os.fork()
        if pid == 0:
            for sig in STOP_SIGNALS:
                signal.signal(sig, signal.SIG_DFL)
            children.clear()
        else:
            children.add(pid)
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
    return pid


def spawn(body) -> int:
    """Fork a check process that runs body() and always ends in os._exit.

    The child never returns into the caller's code, whatever body() raises. Without
    this, an error in a child (ENOSPC, EIO...) would unwind into the main loop and the
    child would run the remaining checks and the cleanup a second time. Exit code: the
    int body() returns, 0 for None, 1 on any exception."""
    pid = fork_tracked()
    if pid == 0:
        code = 1
        try:
            r = body()
            code = r if isinstance(r, int) and not isinstance(r, bool) else 0
        except BaseException:  # noqa: BLE001 - see docstring
            code = 1
        finally:
            os._exit(code)
    return pid


def reap(pid: int) -> int:
    _, status = os.waitpid(pid, 0)
    children.discard(pid)
    return os.waitstatus_to_exitcode(status)


def stop_children() -> int:
    """SIGKILL every tracked child, then wait for each; returns how many were signalled."""
    signalled = 0
    for pid in list(children):
        try:
            os.kill(pid, signal.SIGKILL)
            signalled += 1
        except ProcessLookupError:
            pass
    for pid in list(children):
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        children.discard(pid)
    return signalled


class HardStop(BaseException):
    """Measured footprint over the cap: stop every remaining check (not caught by guarded)."""


def probe_dir_usage() -> int:
    """Bytes really used by the probe directory now (max of apparent size and allocated blocks)."""
    size = blocks = 0
    for dirpath, _dirnames, filenames in os.walk(state["probe_dir"], followlinks=False):
        for name in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except FileNotFoundError:
                continue
            size += st.st_size
            blocks += st.st_blocks * 512
    return max(size, blocks)


def check_disk(where: str) -> None:
    used = probe_dir_usage()
    state["measured_peak"] = max(state["measured_peak"], used)
    if used > state["footprint_cap"]:
        raise HardStop(f"{where}: probe directory uses {used} bytes > cap {state['footprint_cap']}")
    if written["bytes"] > state["max_bytes"]:
        raise HardStop(f"{where}: {written['bytes']} bytes written > budget {state['max_bytes']}")


def report(check: str, status: str, detail: str, **data) -> None:
    results.append({"check": check, "status": status, "detail": detail, **data})
    print(f"{status:<5} {check:<28} {detail}", flush=True)


def guarded(check):
    """Run one check; an unexpected error is a FAIL of that check only."""
    def wrap(fn):
        def run(*a, **kw):
            try:
                fn(*a, **kw)
            except BudgetExceeded as exc:
                report(check, "SKIP", f"byte budget reached: {exc}")
            except Exception as exc:  # noqa: BLE001 - the probe must keep going
                report(check, "FAIL", f"unexpected {type(exc).__name__}: {exc}")
        return run
    return wrap


class BudgetExceeded(RuntimeError):
    pass


def account(nbytes: int, on_disk_delta: int) -> None:
    if written["bytes"] + nbytes > state["max_bytes"]:
        raise BudgetExceeded(f"{written['bytes'] + nbytes} > {state['max_bytes']} bytes")
    if state["on_disk"] + on_disk_delta > state["footprint_cap"]:
        raise BudgetExceeded(f"planned footprint would exceed {state['footprint_cap']} bytes")
    written["bytes"] += nbytes
    state["on_disk"] += on_disk_delta
    written["peak_on_disk"] = max(written["peak_on_disk"], state["on_disk"])


def freed(nbytes: int) -> None:
    state["on_disk"] = max(0, state["on_disk"] - nbytes)


def p(path: str) -> str:
    """Path inside the probe directory; refuses anything that would leave it."""
    full = os.path.join(state["probe_dir"], path)
    if os.path.commonpath([full, state["probe_dir"]]) != state["probe_dir"]:
        raise RuntimeError(f"path leaves the probe directory: {path}")
    return full


def write_file(path: str, data: bytes, *, fsync: bool = True) -> None:
    account(len(data), len(data))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, data)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)


def remove(path: str) -> None:
    try:
        size = os.lstat(path).st_size
    except FileNotFoundError:
        return
    os.unlink(path)
    freed(size)


def child(fn, uid: int | None = None, timeout: float = 60.0):
    """Run fn() in a forked child (optionally as uid); returns (exit code, json result)."""
    r, w = os.pipe()

    def body():
        os.close(r)
        code, out = 0, None
        try:
            signal.alarm(int(timeout))
            if uid is not None:
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)
            out = fn()
        except Exception as exc:  # noqa: BLE001
            code, out = 1, {"error": f"{type(exc).__name__}: {exc}", "errno": getattr(exc, "errno", None)}
        os.write(w, json.dumps(out).encode())
        os.close(w)
        return code

    pid = spawn(body)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    code = reap(pid)
    try:
        return code, json.loads(b"".join(chunks) or b"null")
    except json.JSONDecodeError:
        return code, None


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def lat_summary(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(pct(ms, 50), 2), "p95_ms": round(pct(ms, 95), 2),
            "p99_ms": round(pct(ms, 99), 2), "max_ms": round(max(ms), 2), "mean_ms": round(statistics.mean(ms), 2)}


# ---- environment -------------------------------------------------------------------
def mount_of(path: str):
    best = None
    with open("/proc/mounts") as f:
        for line in f:
            src, mnt, fstype = line.split()[:3]
            mnt = mnt.replace("\\040", " ")
            if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and (best is None or len(mnt) > len(best[1])):
                best = (src, mnt, fstype)
    return best


@guarded("env")
def check_environment(root: str) -> None:
    vol = os.path.dirname(root)
    m = mount_of(root)
    report("env.kernel", "INFO", f"{platform.release()} {platform.machine()}, python {platform.python_version()}")
    report("env.volume-mount", "PASS" if os.path.ismount(vol) else "FAIL",
           f"{vol} mount point={os.path.ismount(vol)}; {root} is on {m[2]} from {m[0]} at {m[1]}" if m else f"{vol}: no mount entry",
           fstype=m[2] if m else None, source=m[0] if m else None)
    for label, path in (("env.volume-statvfs", vol), ("env.container-disk", "/")):
        st = os.statvfs(path)
        mm = mount_of(path)
        report(label, "INFO", f"{path}: {mm[2] if mm else '?'} size {st.f_blocks * st.f_frsize / 1e9:.1f} GB, "
               f"free {st.f_bavail * st.f_frsize / 1e9:.1f} GB (on a network volume this is the backend, not the quota)",
               size_bytes=st.f_blocks * st.f_frsize, free_bytes=st.f_bavail * st.f_frsize)
    cap = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("CapEff:"):
                cap = int(line.split()[1], 16)
    have = [name for bit, name in CAP_NAMES.items() if cap >> bit & 1]
    report("env.capabilities", "INFO", f"CapEff={cap:#x}; of interest: {', '.join(have) or 'none'}", cap_eff=hex(cap), caps=have)
    devs = {d: os.path.exists(d) for d in ("/dev/loop-control", "/dev/loop0", "/dev/fuse")}
    loop_ok = (cap >> 21 & 1) and devs["/dev/loop-control"]
    report("env.loop-mount-possible", "PASS" if loop_ok else "FAIL",
           f"CAP_SYS_ADMIN={'yes' if cap >> 21 & 1 else 'no'}, " + ", ".join(f"{d}={'yes' if v else 'no'}" for d, v in devs.items())
           + " (no mount is attempted)", devices=devs)


# ---- ownership and modes -----------------------------------------------------------
@guarded("mode/owner.root")
def check_ownership_root(uid: int) -> None:
    d = p("mode-dir")
    os.mkdir(d)
    os.chmod(d, 0o700)
    mode = stat.S_IMODE(os.lstat(d).st_mode)
    report("mode.chmod-0700-dir", "PASS" if mode == 0o700 else "FAIL", f"after chmod 0700 the directory shows {mode:04o}", mode=f"{mode:04o}")
    f = os.path.join(d, "file")
    write_file(f, secrets.token_bytes(512))
    os.chmod(f, 0o600)
    mode = stat.S_IMODE(os.lstat(f).st_mode)
    report("mode.chmod-0600-file", "PASS" if mode == 0o600 else "FAIL", f"after chmod 0600 the file shows {mode:04o}", mode=f"{mode:04o}")
    if os.geteuid() != 0:
        report("owner.chown-by-root", "SKIP", "not running as root")
        remove(f)
        os.rmdir(d)
        return
    try:
        os.chown(d, uid, uid)
        st = os.lstat(d)
        report("owner.chown-by-root", "PASS" if st.st_uid == uid else "FAIL",
               f"chown {uid}:{uid} returned success; directory now {st.st_uid}:{st.st_gid}", uid=st.st_uid)
    except OSError as exc:
        report("owner.chown-by-root", "FAIL", f"chown {uid}:{uid} -> {errno.errorcode.get(exc.errno, exc.errno)} ({exc.strerror})",
               errno=errno.errorcode.get(exc.errno))
    remove(f)
    os.rmdir(d)


@guarded("owner.created-by-uid")
def check_ownership_uid(uid: int) -> None:
    if os.geteuid() != 0:
        report("owner.created-by-uid", "SKIP", "not running as root: cannot switch uid")
        return
    d = p("uid-dir")
    os.mkdir(d)
    os.chmod(d, 0o777)  # only our own probe directory; on a regular filesystem uid must be able to write here
    f = os.path.join(d, "by-uid")
    account(4096, 4096)

    def make():
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.write(fd, secrets.token_bytes(4096))
        os.fsync(fd)
        os.close(fd)
        os.chmod(f, 0o600)
        return {"self_uid": os.getuid(), "stat_uid": os.lstat(f).st_uid, "mode": f"{stat.S_IMODE(os.lstat(f).st_mode):04o}"}

    code, out = child(make, uid=uid)
    if code != 0:
        report("owner.created-by-uid", "FAIL", f"uid {uid} could not create a file: {out}")
    else:
        st = os.lstat(f)
        same = st.st_uid == uid
        report("owner.created-by-uid", "PASS" if same else "FAIL",
               f"file created by uid {uid}: root sees owner {st.st_uid}:{st.st_gid}, mode {stat.S_IMODE(st.st_mode):04o}; "
               f"the uid itself sees owner {out['stat_uid']}, mode {out['mode']} after chmod 0600",
               owner_seen_by_root=st.st_uid, mode_seen_by_uid=out["mode"])

    def read_other():
        try:
            with open(os.path.join(state["probe_dir"], "root-secret"), "rb") as fh:
                fh.read(1)
            return {"readable": True}
        except PermissionError:
            return {"readable": False}

    secret = p("root-secret")
    write_file(secret, secrets.token_bytes(64))
    os.chmod(secret, 0o600)
    code, out = child(read_other, uid=uid)
    if code == 0 and out:
        report("isolation.uid-reads-root-0600", "FAIL" if out["readable"] else "PASS",
               f"uid {uid} {'CAN' if out['readable'] else 'cannot'} read a root file chmod 0600 (service isolation between uids)")
    else:
        report("isolation.uid-reads-root-0600", "FAIL", f"child failed: {out}")
    remove(secret)
    remove(f)
    os.rmdir(d)


# ---- atomic mkdir and rename -------------------------------------------------------
@guarded("mkdir")
def check_mkdir(uid: int) -> None:
    d = p("lease")
    os.mkdir(d)
    try:
        os.mkdir(d)
        report("mkdir.eexist", "FAIL", "a second mkdir of the same name succeeded")
    except FileExistsError:
        report("mkdir.eexist", "PASS", "a second mkdir of the same name raised EEXIST")
    os.rmdir(d)
    # Two processes race 100 times for the same name: exactly one must win each round.
    rounds, bad = 100, 0
    for i in range(rounds):
        name = p(f"race-{i}")
        rfd, wfd = os.pipe()
        pids = []
        def racer():
            os.close(wfd)
            os.read(rfd, 1)  # start together
            try:
                os.mkdir(name)
                return 0
            except FileExistsError:
                return 3
            except Exception:  # noqa: BLE001
                return 4

        for _ in range(2):
            pids.append(spawn(racer))
        os.close(rfd)
        os.write(wfd, b"go")
        os.close(wfd)
        codes = sorted(reap(x) for x in pids)
        if codes != [0, 3]:
            bad += 1
        if os.path.isdir(name):
            os.rmdir(name)
    report("mkdir.race-2-procs", "PASS" if bad == 0 else "FAIL",
           f"{rounds - bad}/{rounds} rounds had exactly one winner (one host; says nothing about two pods)", bad_rounds=bad)


@guarded("rename")
def check_rename(writes: int = 300) -> None:
    target = p("heartbeat")
    write_file(target, b"A" * 4096)
    tmp = p("heartbeat.tmp")
    write_file(tmp, b"B" * 4096)
    os.replace(tmp, target)
    with open(target, "rb") as f:
        got = f.read()
    report("rename.replace-existing", "PASS" if got == b"B" * 4096 and not os.path.lexists(tmp) else "FAIL",
           "os.replace over an existing file: target has the new content and the source is gone")
    freed(4096)
    # A writer replaces the target `writes` times while a reader reads it 10 x as often:
    # the reader must only ever see one complete version, never a missing or partial file.
    payloads = [bytes([65 + i]) * 8192 for i in range(4)]
    write_file(target, payloads[0])
    account(writes * 8192, 8192)

    def writer():
        for i in range(writes):
            t = os.path.join(state["probe_dir"], f"w.{os.getpid()}.{i}")
            fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.write(fd, payloads[i % 4])
            os.fsync(fd)
            os.close(fd)
            os.replace(t, target)

    def reader():
        missing = partial = 0
        for _ in range(10 * writes):
            try:
                with open(target, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                missing += 1
                continue
            if data not in payloads:
                partial += 1
        return {"missing": missing, "partial": partial}

    pid_w = spawn(writer)
    code, out = child(reader, timeout=300)
    wcode = reap(pid_w)
    check_disk("rename.atomic-under-reads")
    if code == 0 and wcode == 0 and out:
        ok = out["missing"] == 0 and out["partial"] == 0
        report("rename.atomic-under-reads", "PASS" if ok else "FAIL",
               f"{10 * writes} reads during {writes} replaces: {out['missing']} missing, {out['partial']} partial (one host)", **out)
    else:
        report("rename.atomic-under-reads", "FAIL", f"writer exit {wcode}, reader exit {code}: {out}")
    remove(target)
    d1, d2 = p("dir-a"), p("dir-b")
    os.mkdir(d1)
    os.rename(d1, d2)
    report("rename.directory", "PASS" if os.path.isdir(d2) and not os.path.lexists(d1) else "FAIL", "rename of a directory")
    os.rmdir(d2)


# ---- locks -------------------------------------------------------------------------
def lock_check(name: str, kind: str) -> None:
    path = p(f"{kind}.lock")
    write_file(path, b"")
    holder_r, holder_w = os.pipe()
    release_r, release_w = os.pipe()
    def holder():
        os.close(holder_r)
        os.close(release_w)
        fd = os.open(path, os.O_RDWR)
        if kind == "flock":
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fcntl.lockf(fd, fcntl.LOCK_EX)
        os.write(holder_w, b"held")
        os.read(release_r, 1)

    pid = spawn(holder)
    os.close(holder_w)
    os.close(release_r)
    if os.read(holder_r, 4) != b"held":  # the holder died before taking the lock
        os.close(holder_r)
        os.close(release_w)
        reap(pid)
        report(name, "FAIL", f"the first process could not take the {kind} lock")
        remove(path)
        return

    def try_lock():
        fd = os.open(path, os.O_RDWR)
        try:
            if kind == "flock":
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return {"acquired": True}
        except OSError as exc:
            return {"acquired": False, "errno": errno.errorcode.get(exc.errno)}

    _, while_held = child(try_lock)
    os.write(release_w, b"x")
    os.close(release_w)
    reap(pid)
    os.close(holder_r)
    _, after = child(try_lock)
    ok = while_held and not while_held["acquired"] and after and after["acquired"]
    report(name, "PASS" if ok else "FAIL",
           f"second process while held: {'acquired (NOT exclusive)' if while_held and while_held.get('acquired') else 'refused ' + str((while_held or {}).get('errno'))}; "
           f"after release: {'acquired' if after and after.get('acquired') else 'refused'} (one host only)")
    remove(path)


@guarded("lock")
def check_locks() -> None:
    lock_check("lock.flock-exclusive", "flock")
    lock_check("lock.fcntl-exclusive", "fcntl")


# ---- O_DIRECT, mmap, directory fsync, hard link -----------------------------------
@guarded("io.o_direct")
def check_odirect() -> None:
    if not hasattr(os, "O_DIRECT"):
        report("io.o_direct", "SKIP", "O_DIRECT not available on this platform")
        return
    path = p("odirect")
    buf = mmap.mmap(-1, 4096)  # page aligned
    data = secrets.token_bytes(4096)
    buf.write(data)
    account(4096, 4096)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o600)
    except OSError as exc:
        report("io.o_direct", "FAIL", f"open(O_DIRECT) -> {errno.errorcode.get(exc.errno)}")
        return
    try:
        n = os.write(fd, buf)
        os.fsync(fd)
    except OSError as exc:
        report("io.o_direct", "FAIL", f"open(O_DIRECT) succeeded but write -> {errno.errorcode.get(exc.errno)} ({exc.strerror})")
        return
    finally:
        os.close(fd)
    rbuf = mmap.mmap(-1, 4096)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            m = os.readv(fd, [rbuf])
        finally:
            os.close(fd)
    except OSError as exc:
        report("io.o_direct", "FAIL", f"write with O_DIRECT worked, read -> {errno.errorcode.get(exc.errno)} ({exc.strerror})")
        remove(path)
        return
    ok = n == 4096 and m == 4096 and rbuf[:] == data
    report("io.o_direct", "PASS" if ok else "FAIL", f"aligned 4096-byte write/read with O_DIRECT: wrote {n}, read {m}, match={rbuf[:] == data}")
    remove(path)


@guarded("io.mmap-shared-write")
def check_mmap() -> None:
    path = p("mmap")
    size = MIB
    write_file(path, b"\0" * size)
    pattern = secrets.token_bytes(size)
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        mm[:] = pattern
        mm.flush()
        mm.close()
    account(size, 0)

    def reread():
        with open(path, "rb") as f:
            return {"match": f.read() == pattern}

    code, out = child(reread)
    report("io.mmap-shared-write", "PASS" if code == 0 and out and out["match"] else "FAIL",
           "MAP_SHARED write + msync, then read() from another process sees the same bytes")
    remove(path)


@guarded("io.directory-fsync")
def check_dir_fsync() -> None:
    d = p("dsync")
    os.mkdir(d)
    write_file(os.path.join(d, "f"), b"x" * 128)
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
        report("io.directory-fsync", "PASS", "fsync on a directory fd succeeded (Postgres does this after create/rename)")
    except OSError as exc:
        report("io.directory-fsync", "FAIL", f"fsync(directory) -> {errno.errorcode.get(exc.errno)}")
    finally:
        os.close(fd)
    remove(os.path.join(d, "f"))
    os.rmdir(d)


@guarded("io.hardlink")
def check_hardlink() -> None:
    a, b = p("link-a"), p("link-b")
    write_file(a, b"x" * 128)
    try:
        os.link(a, b)
        report("io.hardlink", "PASS", "link() succeeded")
        os.unlink(b)
    except OSError as exc:
        report("io.hardlink", "FAIL", f"link() -> {errno.errorcode.get(exc.errno)} (informational: the services here do not need it)")
    remove(a)


# ---- fsync latency and throughput ---------------------------------------------------
def fsync_series(path: str, n: int, size: int) -> list[float]:
    block = secrets.token_bytes(size)
    account(n * size, n * size)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    ms = []
    try:
        for i in range(n):
            t0 = time.perf_counter()
            os.pwrite(fd, block, i * size)
            os.fsync(fd)
            ms.append((time.perf_counter() - t0) * 1000)
    finally:
        os.close(fd)
    check_disk(f"fsync series {n} x {size}")
    return ms


@guarded("fsync")
def check_fsync(n_small: int) -> None:
    path = p("fsync-8k")
    ms = fsync_series(path, n_small, 8 * KIB)
    s = lat_summary(ms)
    report("fsync.8k-sequential", "INFO", f"{s['n']} x (8 KB write + fsync): p50 {s['p50_ms']} ms, p95 {s['p95_ms']} ms, "
           f"p99 {s['p99_ms']} ms, max {s['max_ms']} ms", **s)
    remove(path)
    path = p("fsync-16m")
    ms = fsync_series(path, 8, 16 * MIB)
    s = lat_summary(ms)
    report("fsync.16m-segment", "INFO", f"8 x (16 MB write + fsync, one WAL segment): p50 {s['p50_ms']} ms, p99 {s['p99_ms']} ms, "
           f"~{16 / (s['mean_ms'] / 1000):.0f} MB/s", **s)
    remove(path)


@guarded("fsync.8k-concurrent")
def check_fsync_concurrent(procs: int, n: int) -> None:
    account(procs * n * 8 * KIB, procs * n * 8 * KIB)
    workers = []  # (pid, read end of its own result pipe)
    rfd, wfd = os.pipe()
    for k in range(procs):
        out_r, out_w = os.pipe()

        def worker(k=k, out_r=out_r, out_w=out_w):
            os.close(wfd)
            os.close(out_r)
            os.read(rfd, 1)
            block = secrets.token_bytes(8 * KIB)
            fd = os.open(os.path.join(state["probe_dir"], f"conc-{k}"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            ms = []
            for i in range(n):
                t0 = time.perf_counter()
                os.pwrite(fd, block, i * 8 * KIB)
                os.fsync(fd)
                ms.append((time.perf_counter() - t0) * 1000)
            os.close(fd)
            data = json.dumps(ms).encode()
            while data:
                data = data[os.write(out_w, data):]

        workers.append((spawn(worker), out_r))
        os.close(out_w)
    os.close(rfd)
    t0 = time.perf_counter()
    os.write(wfd, b"g" * procs)
    os.close(wfd)
    ms, failed = [], 0
    for pid, out_r in workers:
        data = b""
        while True:
            b = os.read(out_r, 65536)
            if not b:
                break
            data += b
        os.close(out_r)
        code = reap(pid)
        if code != 0 or not data:
            failed += 1
        else:
            ms.extend(json.loads(data))
    wall = time.perf_counter() - t0
    check_disk("fsync.8k-concurrent")
    if failed:
        report("fsync.8k-concurrent", "FAIL", f"{failed} of {procs} worker processes failed")
        for k in range(procs):
            remove(p(f"conc-{k}"))
        return
    s = lat_summary(ms)
    report("fsync.8k-concurrent", "INFO", f"{procs} processes x {n} x (8 KB + fsync): p50 {s['p50_ms']} ms, p95 {s['p95_ms']} ms, "
           f"p99 {s['p99_ms']} ms; {len(ms) / wall:.0f} fsync/s in total", **s, fsync_per_s=round(len(ms) / wall, 1))
    for k in range(procs):
        remove(p(f"conc-{k}"))


@guarded("throughput.sequential")
def check_throughput(mb: int) -> None:
    path = p("seq")
    chunk = secrets.token_bytes(4 * MIB)
    total = mb * MIB
    account(total, total)
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        for i in range(total // len(chunk)):
            os.write(fd, chunk)
        os.fsync(fd)
    finally:
        os.close(fd)
    w = time.perf_counter() - t0
    check_disk("throughput.sequential")
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        while f.read(4 * MIB):
            pass
    r = time.perf_counter() - t0
    report("throughput.sequential", "INFO", f"{mb} MB: write+fsync {mb / w:.0f} MB/s, read {mb / r:.0f} MB/s (read may be cached)",
           write_mb_s=round(mb / w, 1), read_mb_s=round(mb / r, 1))
    remove(path)


# ---- safety, cleanup ----------------------------------------------------------------
def listing(root: str, skip: str) -> list[tuple]:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) != skip]
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            if full == skip:
                continue
            st = os.lstat(full)
            rows.append((os.path.relpath(full, root), stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns))
    return sorted(rows)


def cleanup() -> bool:
    d = state["probe_dir"]
    if not d or not os.path.lexists(d):
        return True
    if os.path.islink(d):
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not os.path.lexists(d)


GROUPS = ("env", "mode-owner", "owner-uid", "mkdir", "rename", "lock", "odirect", "mmap", "dirfsync", "hardlink",
          "fsync", "fsync-concurrent", "throughput")


def main() -> int:
    if sys.version_info < (3, 9) or not sys.platform.startswith("linux"):
        print(f"REFUSED: needs Linux and Python >= 3.9 (got {sys.platform}, {platform.python_version()}); nothing written", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/workspace/cas")
    ap.add_argument("--uid", type=int, default=10010)
    ap.add_argument("--fsync-n", type=int, default=200)
    ap.add_argument("--concurrent-procs", type=int, default=4)
    ap.add_argument("--concurrent-n", type=int, default=50)
    ap.add_argument("--throughput-mb", type=int, default=256)
    ap.add_argument("--rename-writes", type=int, default=300)
    ap.add_argument("--max-bytes", type=int, default=600 * MIB, help="total bytes written, at most 600 MB")
    ap.add_argument("--footprint-cap-mb", type=int, default=300, help="measured size of the probe directory, at most 300")
    ap.add_argument("--only", default="", help="comma-separated check groups (tests): " + ",".join(GROUPS))
    ap.add_argument("--json", action="store_true", help="also print the full result as JSON at the end")
    a = ap.parse_args()
    only = [g for g in a.only.split(",") if g]
    if any(g not in GROUPS for g in only):
        print(f"REFUSED: unknown --only group; known: {','.join(GROUPS)}", file=sys.stderr)
        return 2
    if not (0 < a.max_bytes <= 600 * MIB) or not (0 < a.footprint_cap_mb <= 300):
        print("REFUSED: --max-bytes must be in (0, 600 MB] and --footprint-cap-mb in (0, 300]", file=sys.stderr)
        return 2
    root = a.root.rstrip("/") or "/"
    if os.path.islink(root) or os.path.realpath(root) != root or not os.path.isdir(root):
        print(f"REFUSED: {root} must be an existing real directory (not a symlink, no symlinked parent)", file=sys.stderr)
        return 2
    if a.throughput_mb * MIB > 256 * MIB:
        print("REFUSED: --throughput-mb is limited to 256 (footprint cap)", file=sys.stderr)
        return 2
    state["max_bytes"] = a.max_bytes
    state["footprint_cap"] = a.footprint_cap_mb * MIB
    before = listing(root, skip="")
    name = f".probe-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(4)}"
    state["probe_dir"] = os.path.join(root, name)
    # 0711: other uids can traverse to the subdirectories the uid checks open up, but
    # cannot list or read the probe directory itself.
    try:
        os.mkdir(state["probe_dir"])
    except OSError as exc:
        print(f"REFUSED: cannot create {state['probe_dir']}: {exc.strerror}; nothing written", file=sys.stderr)
        return 2
    os.chmod(state["probe_dir"], 0o711)

    def on_signal(signum, _frame):
        if os.getpid() != state["main_pid"]:  # a forked check process: just stop
            os._exit(130)
        for sig in STOP_SIGNALS:  # no re-entry while stopping
            signal.signal(sig, signal.SIG_IGN)
        # Children first: none may still write when the probe directory is removed.
        killed = stop_children()
        ok = cleanup()
        print(f"INTERRUPTED by signal {signum}: {killed} check process(es) killed and reaped, {len(children)} left; "
              f"cleanup {'done, probe directory removed' if ok else 'FAILED: remove ' + state['probe_dir']}", file=sys.stderr, flush=True)
        os._exit(130 if ok and not children else 1)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"== probe v2 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} root={root} probe_dir={state['probe_dir']} "
          f"euid={os.geteuid()} uid_under_test={a.uid}", flush=True)
    t_start = time.time()
    runs = {
        "env": lambda: check_environment(root),
        "mode-owner": lambda: check_ownership_root(a.uid),
        "owner-uid": lambda: check_ownership_uid(a.uid),
        "mkdir": lambda: check_mkdir(a.uid),
        "rename": lambda: check_rename(a.rename_writes),
        "lock": check_locks,
        "odirect": check_odirect,
        "mmap": check_mmap,
        "dirfsync": check_dir_fsync,
        "hardlink": check_hardlink,
        "fsync": lambda: check_fsync(a.fsync_n),
        "fsync-concurrent": lambda: check_fsync_concurrent(a.concurrent_procs, a.concurrent_n),
        "throughput": lambda: check_throughput(a.throughput_mb),
    }
    try:
        for group in GROUPS:
            if only and group not in only:
                continue
            runs[group]()
            leaked = stop_children()
            if leaked:
                report("safety.children", "FAIL", f"{leaked} check process(es) still running after '{group}': killed and reaped")
            check_disk(group)
    except HardStop as exc:
        report("budget.hard-stop", "FAIL", f"{exc}; remaining checks not run")
    finally:
        left = stop_children()
        removed = cleanup()
        after = listing(root, skip="")
        unchanged = before == after
        leftovers = [r[0] for r in after if r[0].startswith(".probe-")]
        report("cleanup.children", "PASS" if not children else "FAIL",
               f"no check process left ({left} had to be killed at the end)" if not children else f"{len(children)} still present")
        report("cleanup.probe-dir-removed", "PASS" if removed else "FAIL", f"{state['probe_dir']} {'removed' if removed else 'STILL PRESENT'}")
        report("cleanup.root-unchanged", "PASS" if unchanged else "FAIL",
               f"every entry under {root} other than the probe directory is unchanged ({len(before)} entries)"
               if unchanged else f"entries under {root} changed: before {len(before)}, after {len(after)}; leftovers {leftovers}")
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "FAIL", "SKIP", "INFO")}
    print(f"== {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP, {counts['INFO']} INFO; "
          f"bytes written {written['bytes']} ({written['bytes'] / MIB:.1f} MB), peak on disk {written['peak_on_disk'] / MIB:.1f} MB planned, "
          f"{state['measured_peak'] / MIB:.1f} MB measured, "
          f"{time.time() - t_start:.0f} s", flush=True)
    print("   No result here proves durability across a host or power failure, nor locking between two pods.")
    if a.json:
        print(json.dumps({"results": results, "bytes_written": written["bytes"], "peak_on_disk_planned": written["peak_on_disk"],
                          "peak_on_disk_measured": state["measured_peak"]}, indent=1))
    return 0 if removed and unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
