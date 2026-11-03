"""Unit tests of verify_gates.py on a fake /proc (stdlib unittest, no root, no GPU):
  python3 -m unittest discover -s deploy/runpod/staging/tests -p 'test_*.py'
"""
import ipaddress
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_gates as vg  # noqa: E402

SPEECH_CMD = "/opt/cas/venv-c0/bin/python -m src.c0_speech"
NGINX_MASTER = "nginx: master process /usr/sbin/nginx -e stderr -c /run/cas-nginx.conf -g daemon off;"
GOTTY = "gotty -w --random-url-length 32 -r --ws-origin .* -p 19123 /bin/bash"
# the kernel's own headers (Linux 6.x): the third column differs between IPv4 and IPv6
HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"
HEADER6 = "  sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"


def hexaddr(a: str) -> str:
    raw = ipaddress.ip_address(a).packed
    return "".join(raw[i:i + 4][::-1].hex().upper() for i in range(0, len(raw), 4))


class FakeProc:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "net").mkdir()
        (self.root / "net" / "if_inet6").write_text("")   # IPv6 present, as on the pod
        self.net = {p: [] for p in ("tcp", "tcp6", "udp", "udp6")}

    def proc(self, pid, uid, comm, cmd, ppid=1, sockets=(), readable=True, meta=None):
        """meta="status"|"cmdline": that file cannot be read (a directory: read raises an OSError,
        like EACCES); meta="malformed": status lacks its Uid line."""
        d = self.root / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\t{comm}\nPPid:\t{ppid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
                                  f"Gid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
        if meta in ("status", "cmdline"):
            (d / meta).unlink()
            (d / meta).mkdir()
        elif meta == "malformed":
            (d / "status").write_text(f"Name:\t{comm}\nPPid:\t{ppid}\n")
        if readable:  # unreadable = another uid without CAP_SYS_PTRACE: the fd dir cannot be listed
            (d / "fd").mkdir()
            for i, ino in enumerate(sockets):
                os.symlink(f"socket:[{ino}]", d / "fd" / str(i + 3))

    def sock(self, proto, addr, port, uid, inode, state=None, remote=None):
        state = state or ("0A" if proto.startswith("tcp") else "07")
        zero = "0" * len(hexaddr(addr))
        rem = remote or f"{zero}:0000"
        self.net[proto].append(f"   0: {hexaddr(addr)}:{port:04X} {rem} {state} 00000000:00000000 00:00000000 "
                               f"00000000 {uid:>5} 0 {inode} 1 0000000000000000 100 0 0 10 0")

    def run_listeners(self, programs, as_owner=None, problems=()):
        """as_owner: {pid: socket inodes} that the same-uid fd scan finds (fds root cannot read)."""
        for p, rows in self.net.items():
            if rows == "unreadable":
                (self.root / "net" / p).mkdir()   # read_text() raises an OSError, like EACCES
            elif rows is not None:                # None: the table does not exist
                (self.root / "net" / p).write_text("\n".join([HEADER6 if p.endswith("6") else HEADER] + rows) + "\n")

        def scan(procs, _proc):
            for pid, inodes in (as_owner or {}).items():
                procs[pid].inodes, procs[pid].fds_readable = set(inodes), True
            return list(problems)
        return vg.listener_gate(self.root, programs, scan=scan)

    def procs(self):
        return vg.read_procs(self.root)

    def close(self):
        self.tmp.cleanup()


def fails(out):
    return [ln for ln in out if ln.startswith("FAIL")]


class GpuGate(unittest.TestCase):
    def setUp(self):
        self.f = FakeProc()
        self.f.proc(1938, 10013, "python", SPEECH_CMD, readable=False)
        self.f.proc(1, 0, "supervisord", "supervisord")

    def tearDown(self):
        self.f.close()

    def gate(self, apps, pid="1938", dev="cuda", err=""):
        return vg.gpu_gate(pid, dev, 10013, self.f.procs(), apps, err)

    def test_speech_really_on_gpu(self):
        out = self.gate("1938, /opt/cas/venv-c0/bin/python, 3920\n")
        self.assertEqual(fails(out), [])
        self.assertIn("3920 MiB", out[0])

    def test_child_of_speech_holds_the_context(self):
        self.f.proc(2001, 10013, "python", SPEECH_CMD, ppid=1938, readable=False)
        self.assertEqual(fails(self.gate("2001, python, 1200\n")), [])

    def test_no_compute_process(self):
        self.assertIn("no compute process", fails(self.gate(""))[0])

    def test_gpu_used_by_another_process_only(self):
        self.f.proc(3000, 0, "python", "python train.py")
        out = fails(self.gate("3000, python, 8000\n"))
        self.assertIn("no GPU compute process is Speech", out[0])

    def test_zero_memory(self):
        self.assertIn("0 MiB", fails(self.gate("1938, python, 0\n"))[0])

    def test_memory_not_available_counts_as_zero(self):
        self.assertTrue(fails(self.gate("1938, python, [N/A]\n")))

    def test_device_not_cuda(self):
        self.assertIn("SPEECH_DEVICE=cpu", fails(self.gate("1938, python, 3920\n", dev="cpu"))[0])

    def test_speech_not_running(self):
        self.assertIn("not running", fails(self.gate("1938, python, 3920\n", pid=""))[0])

    def test_pid_is_not_speech(self):
        self.f.proc(4000, 0, "python", "python other.py")
        self.assertIn("is not Speech C0", fails(self.gate("4000, python, 3920\n", pid="4000"))[0])

    def test_nvidia_smi_failed(self):
        self.assertIn("nvidia-smi", fails(self.gate(None, err="NVIDIA-SMI has failed"))[0])


class ListenerGate(unittest.TestCase):
    def setUp(self):
        self.f = FakeProc()
        f = self.f
        f.proc(1, 0, "supervisord", "supervisord", ppid=0)
        f.proc(500, 10014, "nginx", NGINX_MASTER, readable=False)        # uid web: fds unreadable
        f.proc(501, 10014, "nginx", "nginx: worker process", ppid=500, readable=False)
        f.proc(600, 10010, "postgres", "postgres -D /var/lib/cas/pgdata", readable=False)
        f.sock("tcp", "0.0.0.0", 8088, 10014, 111)
        f.sock("tcp", "127.0.0.1", 5432, 10010, 112)
        f.sock("tcp", "127.0.0.11", 41234, 0, 113)                      # Docker embedded DNS
        f.sock("udp", "127.0.0.11", 53000, 0, 114)
        f.sock("tcp6", "::1", 6333, 10012, 115)

    def tearDown(self):
        self.f.close()

    # the same-uid scan reads every fd root cannot: nginx master and worker hold the
    # listening socket, postgres its loopback one
    NGINX = {500: [111], 501: [111], 600: [112]}

    def test_cas_only_is_pass(self):
        out = self.f.run_listeners({"nginx": 500}, self.NGINX)
        self.assertEqual(fails(out), [], out)
        self.assertTrue(any("tcp 0.0.0.0:8088 CAS nginx" in ln for ln in out))
        self.assertIn("127.0.0.11:41234", out[0])

    def test_ipv4_mapped_loopback_is_local(self):
        self.f.sock("tcp6", "::ffff:127.0.0.1", 7000, 0, 116)
        self.assertEqual(fails(self.f.run_listeners({"nginx": 500}, self.NGINX)), [])

    def test_runpod_web_terminal_fails(self):
        self.f.proc(4138, 0, "gotty", GOTTY, sockets=[200])
        self.f.sock("tcp6", "::", 19123, 0, 200)
        out = fails(self.f.run_listeners({"nginx": 500}, self.NGINX))
        self.assertEqual(len(out), 1)
        self.assertIn("RunPod Web Terminal", out[0])
        self.assertIn("potentially exposed", out[0])
        self.assertIn("19123", out[0])

    def test_unowned_dns_fails_with_evidence(self):
        self.f.sock("tcp", "0.0.0.0", 53, 0, 300)
        self.f.sock("tcp6", "::", 53, 0, 301)
        self.f.sock("udp", "0.0.0.0", 53, 0, 302)
        out = self.f.run_listeners({"nginx": 500}, self.NGINX)   # every fd read
        self.assertEqual(len(fails(out)), 3)
        self.assertTrue(any("held outside this container" in ln for ln in out), out)
        self.assertTrue(any("nc -vz" in ln for ln in out))

    def test_unknown_listener_owned_by_root_process(self):
        self.f.proc(700, 0, "python3", "python3 -m http.server 9000", sockets=[400])
        self.f.sock("tcp", "0.0.0.0", 9000, 0, 400)
        out = fails(self.f.run_listeners({"nginx": 500}, self.NGINX))
        self.assertIn("not a CAS service", out[0])

    def test_other_service_on_cas_port_fails(self):
        f = FakeProc()
        try:
            f.proc(1, 0, "supervisord", "supervisord", ppid=0)
            f.proc(500, 10014, "nginx", NGINX_MASTER, readable=False)
            f.proc(800, 0, "python3", "python3 -m http.server 8088", sockets=[111])
            f.sock("tcp", "0.0.0.0", 8088, 0, 111)
            out = fails(f.run_listeners({"nginx": 500}))
            self.assertTrue(any("not a CAS service" in ln for ln in out))
            self.assertTrue(any("CAS nginx is not listening" in ln for ln in out))
        finally:
            f.close()

    def test_same_uid_as_nginx_without_inode_proof_fails(self):
        # socket uid 10014 like nginx, but no process is shown to hold inode 111
        out = self.f.run_listeners({"nginx": 500}, problems=["uid 10014/gid 10014: fd scan failed: setpriv missing"])
        f = fails(out)
        self.assertTrue(any("8088 not a CAS service" in ln for ln in f), out)
        self.assertTrue(any("CAS nginx is not listening" in ln for ln in f))
        self.assertTrue(any("owner is not proven" in ln for ln in out), out)
        self.assertTrue(any("ownership not fully observed" in ln for ln in fails(out)), out)
        self.assertTrue(any("uid 10014/gid 10014: fd scan failed: setpriv missing" in ln for ln in out), out)

    def test_same_uid_socket_held_by_another_uid_fails(self):
        # created by uid 10014, passed (SCM_RIGHTS) to a uid 10001 process: nginx does not hold it
        self.f.proc(900, 10001, "python3", "python3 holder.py", readable=False)
        out = self.f.run_listeners({"nginx": 500}, {500: [], 501: [], 600: [112], 900: [111]})
        self.assertTrue(any("8088 not a CAS service" in ln for ln in fails(out)), out)
        self.assertTrue(any("pid 900 python3 uid 10001" in ln for ln in out), out)

    def test_socket_shared_by_nginx_and_another_process_fails(self):
        self.f.proc(900, 10014, "python3", "python3 sneaky.py", readable=False)
        out = self.f.run_listeners({"nginx": 500}, {**self.NGINX, 900: [111]})
        self.assertTrue(fails(out))

    def test_wrong_supervisor_pid_fails(self):
        out = fails(self.f.run_listeners({"nginx": 600}, self.NGINX))
        self.assertTrue(any("CAS nginx is not listening" in ln for ln in out))

    def test_ancestor_pid_as_program_attributes_nothing(self):
        out = fails(self.f.run_listeners({"nginx": 1}, self.NGINX))   # supervisord's tree holds every process
        self.assertTrue(any("CAS nginx is not listening" in ln for ln in out))

    def test_multiline_cmdline_stays_one_line(self):
        self.f.proc(4138, 0, "gotty", "gotty -p 19123\nbash", sockets=[200])
        self.f.sock("tcp", "0.0.0.0", 19123, 0, 200)
        out = self.f.run_listeners({"nginx": 500}, self.NGINX)
        self.assertTrue(any("gotty -p 19123 bash" in ln for ln in out), out)

    # -- incomplete fd scan (review v2 HIGH): nginx holds the socket, but another process's
    #    fds were not read, so nginx being the only holder is not proven
    def test_nginx_proven_but_another_uid_unread_fails(self):
        self.f.proc(900, 10001, "python3", "python3 worker.py", readable=False)
        out = self.f.run_listeners({"nginx": 500}, self.NGINX,
                                   problems=["uid 10001/gid 10001: fd scan failed: Operation not permitted"])
        f = fails(out)
        self.assertFalse(any(ln.startswith("PASS") and "8088" in ln for ln in out), out)
        self.assertTrue(any("held by CAS nginx" in ln and "cannot be excluded" in ln for ln in f), out)
        self.assertTrue(any("ownership not fully observed" in ln and "pid 900" in ln for ln in f), out)
        self.assertTrue(any("fd scan failed" in ln for ln in out), out)

    # -- review v3 HIGH: a process that still exists but whose metadata cannot be read is
    #    not dropped as "exited": it may share nginx's socket, so nginx is not proven sole holder
    def test_shared_socket_holder_with_unreadable_metadata_fails(self):
        for meta in ("status", "cmdline", "malformed"):
            with self.subTest(meta=meta):
                f = FakeProc()
                try:
                    f.proc(1, 0, "supervisord", "supervisord", ppid=0)
                    f.proc(500, 10014, "nginx", NGINX_MASTER, readable=False)
                    f.proc(902, 10001, "python3", "python3 holder.py", sockets=[111], meta=meta)  # shares 8088
                    f.sock("tcp", "0.0.0.0", 8088, 10014, 111)
                    out = f.run_listeners({"nginx": 500}, {500: [111]})
                    self.assertFalse(any(ln.startswith("PASS") and "8088" in ln for ln in out), out)
                    self.assertTrue(any("ownership not fully observed" in ln and "pid 902 (metadata unreadable" in ln
                                        for ln in fails(out)), out)
                    self.assertTrue(any("8088 held by CAS nginx" in ln and "cannot be excluded" in ln
                                        for ln in fails(out)), out)
                finally:
                    f.close()

    def test_unreadable_metadata_is_never_scanned_as_some_uid(self):
        procs = {}
        f = FakeProc()
        try:
            f.proc(902, 10001, "python3", "python3 x", meta="status")
            procs = vg.read_procs(f.root)
        finally:
            f.close()
        self.assertEqual(procs[902].uid, -1)
        self.assertFalse(procs[902].fds_readable)
        self.assertEqual(vg.scan_as_owners({902: procs[902]}, Path("/nonexistent")), [])   # nothing to run as

    def test_gpu_gate_fails_when_speech_metadata_unreadable(self):
        f = FakeProc()
        try:
            f.proc(1938, 10013, "python", SPEECH_CMD, meta="cmdline")
            out = vg.gpu_gate("1938", "cuda", 10013, vg.read_procs(f.root), "1938, python, 3920\n")
            self.assertTrue(fails(out), out)
        finally:
            f.close()

    def test_unreadable_metadata_process_that_exits_is_dropped(self):
        self.f.proc(903, 10001, "python3", "python3 gone.py", meta="status")

        def scan(procs, proc):
            import shutil
            shutil.rmtree(proc / "903")   # exits after the listing
            for pid, inodes in self.NGINX.items():
                procs[pid].inodes, procs[pid].fds_readable = set(inodes), True
            return []
        for p, rows in self.f.net.items():
            (self.f.root / "net" / p).write_text("\n".join([HEADER6 if p.endswith("6") else HEADER] + rows) + "\n")
        self.assertEqual(fails(vg.listener_gate(self.f.root, {"nginx": 500}, scan=scan)), [])

    def test_unread_process_that_exited_is_dropped(self):
        self.f.proc(900, 10001, "python3", "python3 short.py", readable=False)

        def scan(procs, proc):
            import shutil
            shutil.rmtree(proc / "900")   # exited between the listing and the scan
            for pid, inodes in self.NGINX.items():
                procs[pid].inodes, procs[pid].fds_readable = set(inodes), True
            return []
        for p, rows in self.f.net.items():
            (self.f.root / "net" / p).write_text("\n".join([HEADER6 if p.endswith("6") else HEADER] + rows) + "\n")
        self.assertEqual(fails(vg.listener_gate(self.f.root, {"nginx": 500}, scan=scan)), [])

    def test_zombie_holds_no_fds(self):
        self.f.proc(901, 10001, "python3", "python3 <defunct>", readable=False)
        st = self.f.root / "901" / "status"
        st.write_text(st.read_text() + "State:\tZ (zombie)\n")
        self.assertEqual(fails(self.f.run_listeners({"nginx": 500}, self.NGINX)), [])

    # -- /proc/net tables (review v2 MEDIUM): every table read and parsed, or FAIL
    def test_missing_udp_tables_fail(self):
        self.f.net["udp"] = None
        self.f.net["udp6"] = None
        f = fails(self.f.run_listeners({"nginx": 500}, self.NGINX))
        self.assertEqual(sorted(ln.split()[2] for ln in f), ["/proc/net/udp", "/proc/net/udp6"])
        self.assertTrue(all("missing" in ln and "not fully observed" in ln for ln in f))

    def test_missing_ipv6_tables_fail_when_ipv6_exists(self):
        self.f.net["tcp6"] = None
        self.assertTrue(any("/proc/net/tcp6 missing" in ln for ln in fails(self.f.run_listeners({"nginx": 500}, self.NGINX))))

    def test_ipv6_tables_absent_without_ipv6_is_verified_exception(self):
        (self.f.root / "net" / "if_inet6").unlink()
        self.f.net["tcp6"] = None
        self.f.net["udp6"] = None
        out = self.f.run_listeners({"nginx": 500}, self.NGINX)
        self.assertEqual(fails(out), [], out)
        self.assertTrue(any(ln.startswith("INFO") and "no IPv6 in this network namespace" in ln for ln in out))

    def test_ipv4_tables_never_optional(self):
        (self.f.root / "net" / "if_inet6").unlink()
        self.f.net["tcp"] = None
        self.assertTrue(any("/proc/net/tcp missing" in ln for ln in fails(self.f.run_listeners({"nginx": 500}, self.NGINX))))

    def test_unreadable_table_fails(self):
        self.f.net["udp"] = "unreadable"
        out = self.f.run_listeners({"nginx": 500}, self.NGINX)
        self.assertTrue(any("/proc/net/udp unreadable" in ln for ln in fails(out)), out)

    def test_malformed_tables_fail(self):
        for bad, why in ((HEADER + "\n   0: zz\n", "unparsable"), ("garbage\n", "no /proc/net header"),
                         ("", "no /proc/net header"),
                         (HEADER + "\n   0: 00000000:0035 00000000:0000 0A 00000000:00000000\n", "unparsable")):
            with self.subTest(content=bad[-40:]):
                with self.assertRaisesRegex(ValueError, why):
                    vg.parse_proc_net(bad, "tcp")

    def test_header_must_match_the_table_family(self):
        with self.assertRaisesRegex(ValueError, "no /proc/net header"):
            vg.parse_proc_net(HEADER + "\n", "tcp6")
        with self.assertRaisesRegex(ValueError, "no /proc/net header"):
            vg.parse_proc_net(HEADER6 + "\n", "udp")
        self.assertEqual(vg.parse_proc_net(HEADER6 + "\n", "udp6"), [])

    def test_malformed_table_fails_the_gate(self):
        self.f.net["udp"] = ["this is not a socket line"]
        self.assertTrue(any("/proc/net/udp line 2 unparsable" in ln for ln in fails(self.f.run_listeners({"nginx": 500}, self.NGINX))))

    def test_nginx_not_running(self):
        f = FakeProc()
        try:
            f.proc(1, 0, "supervisord", "supervisord", ppid=0)
            out = fails(f.run_listeners({}))
            self.assertIn("CAS nginx is not listening", out[0])
        finally:
            f.close()

    def test_udp_server_on_public_address_fails(self):
        self.f.proc(950, 0, "dnsmasq", "dnsmasq -k", sockets=[500])
        self.f.sock("udp", "0.0.0.0", 5353, 0, 500)
        self.assertEqual(len(fails(self.f.run_listeners({"nginx": 500}, self.NGINX))), 1)

    def test_connected_udp_client_is_ignored(self):
        self.f.sock("udp", "10.0.0.5", 40000, 10001, 600, state="01", remote="0800080A:0035")
        self.assertEqual(fails(self.f.run_listeners({"nginx": 500}, self.NGINX)), [])


class Cli(unittest.TestCase):
    """main(): exit 0 only when every check passed, 1 on a FAIL; dependencies missing fail closed."""

    def setUp(self):
        self.f = FakeProc()
        self.f.proc(1938, 10013, "python", SPEECH_CMD, readable=False)
        self.path = os.environ.get("PATH", "")

    def tearDown(self):
        os.environ["PATH"] = self.path
        self.f.close()

    def test_gpu_without_nvidia_smi_fails_closed(self):
        import contextlib
        import io
        os.environ["PATH"] = self.f.tmp.name   # no nvidia-smi on PATH
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vg.main(["gpu", "--speech-pid", "1938", "--speech-device", "cuda", "--proc", str(self.f.root)])
        self.assertEqual(rc, 1)
        self.assertIn("nvidia-smi --query-compute-apps failed", buf.getvalue())

    def test_fd_inodes_prints_readable_processes(self):
        self.f.proc(77, 0, "python3", "python3 x", sockets=[123])
        import contextlib
        import io
        import json
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(vg.main(["fd-inodes", "--proc", str(self.f.root)]), 0)
        got = json.loads(buf.getvalue())
        self.assertEqual(got.get("77"), [123])
        self.assertNotIn("1938", got)   # unreadable fds are not reported as empty


if __name__ == "__main__":
    unittest.main()
