#!/usr/bin/env python3
"""GPU and listener gates of verify_after_switch.sh (stdlib only, run as root in cas-aio).

  verify_gates.py gpu --speech-pid <pid> --speech-device <dev> [--speech-uid 10013]
  verify_gates.py listeners --program nginx=<pid> [--program <name>=<pid> ...]
  verify_gates.py fd-inodes          (internal: run under another uid by `listeners`)

Prints PASS/FAIL lines in the format of verify_after_switch.sh; exit 0 when every check
passed, 1 when a check failed (any other exit is a crash; the caller fails closed).

Why not root's view of /proc/<pid>/fd: Docker drops CAP_SYS_PTRACE, so root cannot read the
fds of a process owned by another uid (speech 10013, nginx 10014). The GPU gate therefore
uses the NVIDIA driver's own per-process accounting (nvidia-smi --query-compute-apps). The
listener gate reads every listening socket from /proc/net/{tcp,tcp6,udp,udp6} and proves
its owner only by the socket inode in a process's fds: the fds root cannot read are read by
this script itself running under that process's uid and gid (setpriv), which the kernel
allows for a process of the same credentials. A socket whose inode no process is shown to
hold has no owner and fails; the socket's uid alone is never taken as ownership.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

CAS_PORT = 8088
# A listener on these is reachable only from inside this network namespace.
LOOPBACK_V4 = ipaddress.ip_network("127.0.0.0/8")
DOCKER_DNS = ipaddress.ip_address("127.0.0.11")
TCP_LISTEN, UDP_UNCONNECTED = "0A", "07"


def line(status: str, name: str, msg: str) -> str:
    return f"{status:<5} {name:<18} {msg}"


# ---------------------------------------------------------------- processes

@dataclass
class Proc:
    pid: int
    ppid: int
    uid: int
    gid: int
    comm: str
    cmdline: str
    inodes: set[int] = field(default_factory=set)
    fds_readable: bool = False
    unobserved: str = ""   # set when the process exists but its metadata could not be read

    def describe(self) -> str:
        if self.unobserved:
            return f"pid {self.pid} ({self.unobserved})"
        return f"pid {self.pid} {self.comm} uid {self.uid}"


def socket_inodes(fd_dir: Path) -> set[int]:
    """Socket inodes among a process's fds (OSError when they cannot be read)."""
    out = set()
    for fd in fd_dir.iterdir():
        try:
            t = os.readlink(fd)
        except FileNotFoundError:
            continue  # closed meanwhile
        if t.startswith("socket:["):
            out.add(int(t[8:-1]))
    return out


def read_procs(proc: Path) -> dict[int, Proc]:
    procs: dict[int, Proc] = {}
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            status = dict(row.split(":", 1) for row in (d / "status").read_text().splitlines() if ":" in row)
            cmd = " ".join((d / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ").split())
            p = Proc(int(d.name), int(status["PPid"].split()[0]), int(status["Uid"].split()[0]),
                     int(status["Gid"].split()[0]), status["Name"].strip(), cmd)
        except (OSError, KeyError, IndexError, ValueError) as e:
            if not d.exists():
                continue  # exited meanwhile: it holds nothing now
            # Still there but not observable (permission, malformed status): kept, with no
            # uid to scan under, so the listener gate cannot prove any socket's holders.
            procs[int(d.name)] = Proc(int(d.name), -1, -1, -1, "?", "",
                                      unobserved=f"metadata unreadable: {type(e).__name__}: {e}"[:160])
            continue
        if status.get("State", "").split()[:1] == ["Z"]:
            p.fds_readable = True   # a zombie holds no fds (its fd dir cannot be listed)
        else:
            try:
                p.inodes = socket_inodes(d / "fd")
                p.fds_readable = True
            except OSError:
                pass  # another uid without CAP_SYS_PTRACE, or exited
        procs[p.pid] = p
    return procs


def tree(procs: dict[int, Proc], root: int) -> set[int]:
    """root and all its descendants."""
    out, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p in procs and p not in out:
            out.add(p)
            todo.extend(c.pid for c in procs.values() if c.ppid == p)
    return out


# ---------------------------------------------------------------- GPU gate

def parse_compute_apps(text: str) -> list[tuple[int, str, int]]:
    """nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"""
    rows = []
    for raw in text.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        mem = int(parts[-1]) if parts[-1].isdigit() else 0  # "[N/A]" counts as 0
        rows.append((int(parts[0]), ",".join(parts[1:-1]), mem))
    return rows


def gpu_gate(speech_pid: str, speech_device: str, speech_uid: int, procs: dict[int, Proc],
             apps_text: str | None, apps_error: str = "") -> list[str]:
    name = "speech-on-gpu"
    if speech_device != "cuda":
        return [line("FAIL", name, f"SPEECH_DEVICE={speech_device or '?'}, expected cuda")]
    if not speech_pid.isdigit() or int(speech_pid) not in procs:
        return [line("FAIL", name, f"speech is not running (supervisor pid {speech_pid or '?'})")]
    sp = procs[int(speech_pid)]
    if sp.uid != speech_uid or "src.c0_speech" not in sp.cmdline:
        return [line("FAIL", name, f"pid {sp.pid} is not Speech C0 (uid {sp.uid}, {sp.cmdline[:80]!r})")]
    if apps_text is None:
        return [line("FAIL", name, f"nvidia-smi --query-compute-apps failed: {apps_error[:200]}")]
    rows = parse_compute_apps(apps_text)
    mine = tree(procs, sp.pid)
    hits = [r for r in rows if r[0] in mine]
    if not rows:
        return [line("FAIL", name, "the NVIDIA driver reports no compute process at all "
                     "(Speech holds no CUDA context, or this container cannot see GPU PIDs)")]
    if not hits:
        return [line("FAIL", name, f"no GPU compute process is Speech (pids {sorted(mine)}); the driver reports "
                     + "; ".join(f"pid {p} {n} {m} MiB" for p, n, m in rows)
                     + " (pids outside this PID namespace cannot be attributed)")]
    used = sum(m for _, _, m in hits)
    if used <= 0:
        return [line("FAIL", name, f"Speech pid {hits[0][0]} is a GPU process but uses 0 MiB")]
    return [line("PASS", name, "SPEECH_DEVICE=cuda; NVIDIA driver: "
                 + ", ".join(f"pid {p} ({n}) {m} MiB" for p, n, m in hits)
                 + f" = Speech C0 (supervisor pid {sp.pid}, uid {sp.uid})")]


# ---------------------------------------------------------------- listener gate

@dataclass
class Sock:
    proto: str
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address
    port: int
    uid: int
    inode: int

    @property
    def where(self) -> str:
        a = f"[{self.addr}]" if self.addr.version == 6 else str(self.addr)
        return f"{self.proto} {a}:{self.port}"


def _hex_addr(h: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    # /proc/net prints each 32-bit word in host (little-endian) byte order.
    raw = b"".join(bytes.fromhex(h[i:i + 8])[::-1] for i in range(0, len(h), 8))
    return ipaddress.ip_address(raw)


def parse_proc_net(text: str, proto: str) -> list[Sock]:
    """Listening TCP sockets, and UDP sockets bound but not connected (a UDP server).
    ValueError on anything that is not a /proc/net table: no socket may go unread."""
    want = TCP_LISTEN if proto.startswith("tcp") else UDP_UNCONNECTED
    width = 32 if proto.endswith("6") else 8
    rows = text.splitlines()
    # the kernel names the third column rem_address in tcp/udp, remote_address in tcp6/udp6
    if not rows or rows[0].split()[:3] != ["sl", "local_address", "remote_address" if width == 32 else "rem_address"]:
        raise ValueError("no /proc/net header")
    out = []
    for n, raw in enumerate(rows[1:], 2):
        f = raw.split()
        try:
            local, port = f[1].split(":")
            remote = f[2].split(":")[0]
            if len(f) < 10 or len(local) != width or len(remote) != width or len(f[3]) != 2:
                raise ValueError("short line")
            sock = Sock(proto.rstrip("6"), _hex_addr(local), int(port, 16), int(f[7]), int(f[9]))
            int(f[3], 16)
        except (IndexError, ValueError) as e:
            raise ValueError(f"line {n} unparsable ({e}): {raw.strip()[:80]!r}") from None
        if f[3] != want:
            continue
        if proto.startswith("udp") and set(remote) != {"0"}:
            continue  # connected UDP (a client)
        out.append(sock)
    return out


def is_loopback(a) -> bool:
    if a.version == 6:
        if a.ipv4_mapped is not None:
            return a.ipv4_mapped in LOOPBACK_V4
        return a.is_loopback
    return a in LOOPBACK_V4


def scan_as_owners(procs: dict[int, Proc], proc: Path) -> list[str]:
    """Read the fds root could not, by running `fd-inodes` under each process's uid/gid.
    Returns problems (a uid/gid whose processes stay unread)."""
    problems = []
    for uid, gid in sorted({(p.uid, p.gid) for p in procs.values() if not p.fds_readable and not p.unobserved}):
        cmd = ["setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups", "--",
               sys.executable, "-I", os.path.abspath(__file__), "fd-inodes", "--proc", str(proc)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd="/", env={"PATH": "/usr/bin:/bin"})
            found = json.loads(r.stdout) if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            r, found = None, None
            problems.append(f"uid {uid}/gid {gid}: fd scan failed: {e}")
            continue
        if not isinstance(found, dict):
            problems.append(f"uid {uid}/gid {gid}: fd scan failed (exit {r.returncode}): {(r.stderr or r.stdout)[:200]}")
            continue
        for pid, inodes in found.items():
            p = procs.get(int(pid))
            if p and p.uid == uid and not p.fds_readable:
                p.inodes, p.fds_readable = set(inodes), True
    return problems


def owners(s: Sock, procs: dict[int, Proc]) -> tuple[list[Proc], str]:
    """(processes holding the socket inode, explanation when none is shown to)."""
    by_inode = [p for p in procs.values() if s.inode in p.inodes]
    if by_inode:
        return by_inode, "socket inode in the process's fds"
    unread = sorted({p.uid for p in procs.values() if not p.fds_readable})
    if unread:
        return [], (f"socket uid {s.uid}: inode {s.inode} in no readable fds; fds of uid(s) {unread} could not "
                    "be read, so the owner is not proven")
    return [], (f"socket uid {s.uid}: inode {s.inode} is held by no process in this PID namespace "
                "(all fds read): it is held outside this container")


def classify(s: Sock, procs: dict[int, Proc], programs: dict[str, set[int]]) -> tuple[str, str, list[str]]:
    """(PASS|FAIL, message, follow-up lines)."""
    if is_loopback(s.addr):
        kind = "Docker embedded DNS" if s.addr == DOCKER_DNS else "loopback"
        return "PASS", f"{s.where} {kind}: reachable only inside this container", []
    found, how = owners(s, procs)
    names = sorted({p.comm for p in found})
    who = ", ".join(f"pid {p.pid} {p.comm} uid {p.uid}" for p in found[:4]) or "no process shown to hold it"
    progs = [n for n, pids in programs.items() if found and all(p.pid in pids for p in found)]
    unread = sorted(p.pid for p in procs.values() if not p.fds_readable)
    if s.proto == "tcp" and s.port == CAS_PORT and progs == ["nginx"] and names == ["nginx"]:
        if unread:   # nginx holds it, but whether it is the only holder is not proven
            return "FAIL", (f"{s.where} held by CAS nginx ({who}), but pids {unread[:8]} were not fully read: "
                            "other holders (e.g. passed by SCM_RIGHTS) cannot be excluded"), []
        return "PASS", f"{s.where} CAS nginx, sole holder (supervisor program nginx: {who}; {how}; all fds read)", []
    if "gotty" in names:
        g = next(p for p in found if p.comm == "gotty")
        return "FAIL", (f"{s.where} RunPod Web Terminal (gotty pid {g.pid} uid {g.uid}: {g.cmdline[:160]}): "
                        "a potentially exposed interactive terminal"), [
            "seen here: -w accepts input, --ws-origin '.*' accepts any origin, it listens beyond loopback;",
            "not verified: whether the RunPod proxy adds authentication, and who can reach it from the Internet",
            "to clear it: first confirm basic SSH works (ssh <pod>@ssh.runpod.io -i <key>, Console > Connect),",
            "then stop the Web Terminal in Console > Connect and re-run this script from that SSH session"]
    detail = [f"owner: {who} ({how})"]
    if not found:
        detail += [
            "identify it before accepting the switch; this script never allowlists an unknown listener:",
            f"  grep -i ':{s.port:04X} ' /proc/net/{s.proto}* ; cat /etc/resolv.conf",
            "  ls /proc/[0-9]*/ns/net -l | awk '{print $NF}' | sort | uniq -c   (one netns = same container)",
            f"  RunPod Console > Pod > Edit: is {s.port} among the exposed TCP/UDP ports? from outside the pod: "
            f"nc -vz <pod public IP> {s.port} (and -u for UDP)",
            "  ask RunPod support which component listens there and whether it is reachable from outside"]
    return "FAIL", f"{s.where} not a CAS service and not identified as safe", detail


def read_tables(proc: Path) -> tuple[list[Sock], list[str], list[str]]:
    """(sockets, FAIL reasons, INFO notes). Every table must be read and parsed; tcp6/udp6
    may be absent only when this network namespace has no IPv6 at all (no if_inet6)."""
    socks, failed, notes = [], [], []
    no_ipv6 = not (proc / "net" / "if_inet6").exists()
    for proto in ("tcp", "tcp6", "udp", "udp6"):
        f = proc / "net" / proto
        try:
            socks += parse_proc_net(f.read_text(), proto)
        except FileNotFoundError:
            if proto.endswith("6") and no_ipv6:
                notes.append(f"/proc/net/{proto} absent: no IPv6 in this network namespace (no /proc/net/if_inet6)")
            else:
                failed.append(f"/proc/net/{proto} missing")
        except OSError as e:
            failed.append(f"/proc/net/{proto} unreadable: {e}")
        except ValueError as e:
            failed.append(f"/proc/net/{proto} {e}")
    return socks, failed, notes


def listener_gate(proc: Path, programs: dict[str, int], scan=scan_as_owners) -> list[str]:
    socks, failed, notes = read_tables(proc)
    procs = read_procs(proc)
    problems = scan(procs, proc)
    for pid in [pid for pid, p in procs.items() if not p.fds_readable and not (proc / str(pid)).exists()]:
        del procs[pid]   # exited since it was listed: it holds no socket now
    # A program counts only if the pid supervisord reports is that program itself: a wrong
    # pid (e.g. an ancestor, whose tree holds every process) attributes nothing.
    trees = {n: tree(procs, pid) for n, pid in programs.items() if pid in procs and procs[pid].comm == n}
    out, local = [], []
    for s in sorted(socks, key=lambda s: (s.proto, s.port, str(s.addr))):
        verdict, msg, more = classify(s, procs, trees)
        if verdict == "PASS" and is_loopback(s.addr):
            local.append(s.where)
            continue
        out.append(line(verdict, "listener", msg))
        out += [f"      {m}" for m in more]
    if not any(ln.startswith("PASS") and f":{CAS_PORT} CAS nginx" in ln for ln in out):
        out.append(line("FAIL", "listener", f"CAS nginx is not listening on :{CAS_PORT}"))
    unread = [p for p in procs.values() if not p.fds_readable]
    head = [line("INFO", "listener", f"{len(local)} loopback-only sockets: {' '.join(local)}")]
    head += [line("INFO", "listener", n) for n in notes]
    head += [line("FAIL", "listener", f"{r}: listening sockets not fully observed") for r in failed]
    if unread:
        head.append(line("FAIL", "listener", "socket ownership not fully observed: "
                         + ", ".join(p.describe() for p in unread[:8]) + " not read"))
        head += [f"      {pr}" for pr in problems]
    return head + out


def fd_inodes(proc: Path) -> dict[str, list[int]]:
    """{pid: socket inodes} of every process whose fds this uid can read."""
    out = {}
    for d in proc.iterdir():
        if d.name.isdigit():
            try:
                out[d.name] = sorted(socket_inodes(d / "fd"))
            except OSError:
                pass
    return out


# ---------------------------------------------------------------- CLI

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpu")
    g.add_argument("--speech-pid", default="")
    g.add_argument("--speech-device", default="")
    g.add_argument("--speech-uid", type=int, default=10013)
    ls = sub.add_parser("listeners")
    ls.add_argument("--program", action="append", default=[], help="name=pid of a supervisor program")
    fi = sub.add_parser("fd-inodes")
    for p in (g, ls, fi):
        p.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    proc = Path(a.proc)
    if a.cmd == "fd-inodes":
        print(json.dumps(fd_inodes(proc)))
        return 0
    if a.cmd == "gpu":
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
            text, err = (r.stdout, "") if r.returncode == 0 else (None, (r.stderr or r.stdout).strip())
        except (OSError, subprocess.TimeoutExpired) as e:
            text, err = None, str(e)
        out = gpu_gate(a.speech_pid, a.speech_device, a.speech_uid, read_procs(proc), text, err)
    else:
        progs = {}
        for item in a.program:
            n, _, pid = item.partition("=")
            if pid.isdigit():
                progs[n] = int(pid)
        out = listener_gate(proc, progs)
    print("\n".join(out))
    return 1 if any(ln.startswith("FAIL") for ln in out) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
