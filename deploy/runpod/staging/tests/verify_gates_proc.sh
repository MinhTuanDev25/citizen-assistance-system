#!/usr/bin/env bash
# verify_gates.py listeners on a REAL /proc, in a Docker container as root without
# CAP_SYS_PTRACE (as on the pod): root cannot read nginx's fds (uid 10014); the gate reads
# them under uid 10014 (setpriv) and proves ownership by the socket inode only.
# Run by run_offline_tests.sh: docker run ... python:3.11-bookworm bash /t/verify_gates_proc.sh
# Prints "GATES-PROC: N passed, M failed"; exit 1 on any failure.
set -uo pipefail
pass=0 failn=0
ok() { echo "  ok   $1"; pass=$((pass + 1)); }
no() { echo "  FAIL $1"; failn=$((failn + 1)); }
G=/staging/verify_gates.py
useradd -u 10014 -M web 2>/dev/null; useradd -u 10001 -M cas 2>/dev/null
for b in nginx gotty python3; do cp "$(command -v python3)" /tmp/$b; done
serve() {  # serve <bin> <uid> <addr> <port> [udp]: a process named <bin> listening there
  local code="import socket,sys,time
fam=socket.AF_INET6 if ':' in sys.argv[1] else socket.AF_INET
s=socket.socket(fam, socket.SOCK_DGRAM if len(sys.argv)>3 else socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((sys.argv[1], int(sys.argv[2])))
len(sys.argv)>3 or s.listen(); time.sleep(600)"
  setpriv --reuid "$2" --regid "$2" --clear-groups "/tmp/$1" -c "$code" "$3" "$4" ${5:+udp} &
}
serve nginx 10014 0.0.0.0 8088; NGX=$!
serve nginx 0 127.0.0.11 41234
serve nginx 0 127.0.0.1 5432
sleep 1
if ! readlink /proc/$NGX/fd/0 >/dev/null 2>&1; then ok "fds of uid 10014 unreadable as root (as on the pod)"; else no "fds of uid 10014 readable: test does not model the pod"; fi

out="$(python3 $G listeners --program nginx=$NGX)"; rc=$?; echo "$out" | sed 's/^/      /'
[[ $rc == 0 ]] && ok "CAS only: rc 0" || no "CAS only: rc $rc"
grep -q "^PASS  listener .*tcp 0.0.0.0:8088 CAS nginx, sole holder .*pid $NGX nginx uid 10014.*socket inode in the process's fds; all fds read" <<<"$out" \
  && ok "8088 owned by nginx: inode read under uid 10014" || no "8088 not attributed by inode"

serve gotty 0 :: 19123; GT=$!
serve python3 0 0.0.0.0 53; serve python3 0 0.0.0.0 53 udp
sleep 1
out="$(python3 $G listeners --program nginx=$NGX)"; rc=$?; echo "$out" | sed 's/^/      /'
[[ $rc == 1 ]] && ok "abnormal listeners: rc 1" || no "abnormal listeners: rc $rc"
grep -q "^FAIL  listener .*\[::\]:19123 RunPod Web Terminal (gotty pid $GT.*potentially exposed" <<<"$out" && ok "gotty classified as RunPod Web Terminal" || no "gotty"
[[ $(grep -c '^FAIL  listener .*:53 not a CAS service' <<<"$out") == 2 ]] && ok "tcp+udp :53 FAIL" || no ":53 tcp+udp"
grep -q '^PASS  listener .*8088 CAS nginx, sole holder' <<<"$out" && ok "nginx still PASS" || no "nginx lost"

# review v2 HIGH: a process whose fds cannot be read even under its own uid (non-dumpable:
# the kernel then requires CAP_SYS_PTRACE). nginx holds :8088, but being its only holder
# is no longer proven: the gate must FAIL, not PASS with an INFO.
setpriv --reuid 10001 --regid 10001 --clear-groups /tmp/python3 -c "import ctypes,time
ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); time.sleep(600)" & ND=$!
sleep 1
out="$(python3 $G listeners --program nginx=$NGX)"; rc=$?; echo "$out" | sed 's/^/      /'
[[ $rc == 1 ]] && ok "unscannable process: rc 1" || no "unscannable process: rc $rc"
grep -q "^FAIL  listener .*ownership not fully observed.*pid $ND python3 uid 10001" <<<"$out" && ok "incomplete fd scan is a FAIL" || no "incomplete scan not FAIL"
grep -q "^FAIL  listener .*8088 held by CAS nginx.*cannot be excluded" <<<"$out" && ok "nginx not PASS while a holder is unread" || no "nginx PASS with an unread process (review v2 HIGH)"
kill $ND; wait $ND 2>/dev/null

out="$(python3 $G listeners --program nginx=1)"; rc=$?
[[ $rc == 1 ]] && grep -q 'CAS nginx is not listening' <<<"$out" && ok "wrong nginx pid: FAIL" || no "wrong nginx pid (rc $rc)"
kill $(jobs -p) 2>/dev/null; wait 2>/dev/null

# HIGH 2 regression: a socket of uid 10014 on 0.0.0.0:8088 that nginx does not hold. A uid
# 10014 process creates it, passes it (SCM_RIGHTS) to a uid 10001 process and exits; the
# real nginx (uid 10014) runs on loopback. Same uid as nginx, but not nginx's socket.
setpriv --reuid 10001 --regid 10001 --clear-groups /tmp/python3 -c "import socket,os,time
u=socket.socket(socket.AF_UNIX); u.bind('/tmp/pass.sock'); os.chmod('/tmp/pass.sock',0o777); u.listen()
c,_=u.accept(); _,fds,_,_=socket.recv_fds(c,16,1); time.sleep(600)" & HOLD=$!
sleep 1
setpriv --reuid 10014 --regid 10014 --clear-groups /tmp/nginx -c "import socket
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0',8088)); s.listen()
u=socket.socket(socket.AF_UNIX); u.connect('/tmp/pass.sock'); socket.send_fds(u,[b'x'],[s.fileno()])"
serve nginx 10014 127.0.0.1 8089; NGX=$!
sleep 1
grep -qi ':1F98 .* 0A .* 10014 ' /proc/net/tcp && ok "setup: 0.0.0.0:8088 listening with socket uid 10014" || no "setup: no uid-10014 socket on 8088"
out="$(python3 $G listeners --program nginx=$NGX)"; rc=$?; echo "$out" | sed 's/^/      /'
[[ $rc == 1 ]] && ok "same-uid socket not held by nginx: rc 1" || no "same-uid socket: rc $rc"
grep -q "^FAIL  listener .*tcp 0.0.0.0:8088 not a CAS service" <<<"$out" && ok "8088 not attributed to nginx" || no "8088 attributed to nginx (HIGH 2)"
grep -q "owner: pid $HOLD python3 uid 10001" <<<"$out" && ok "real holder found by inode under uid 10001" || no "holder not shown"
grep -q 'CAS nginx is not listening on :8088' <<<"$out" && ok "CAS nginx reported missing" || no "missing nginx not reported"
kill $(jobs -p) 2>/dev/null
echo "GATES-PROC: $pass passed, $failn failed"
(( failn == 0 ))
