#!/usr/bin/env bash
# Tests of probe_v2.py, run inside a throwaway container by run_probe_tests.sh.
#   probe_tests.sh ext4      /workspace is a Docker volume (regular POSIX filesystem)
#   probe_tests.sh bindfs    /workspace is bindfs --perms=a+rwx --chmod-ignore --chown-deny,
#                            which mimics the pod's network volume (0777, chown EPERM)
# Every expectation is asserted; last line "PROBE-TESTS <fs>: N passed, M failed"; exit 1 on failure.
set -uo pipefail
FS="${1:?ext4|bindfs}"
P=/probe/probe_v2.py
R=/workspace/cas
pass=0 failn=0
ok()  { echo "  ok   $1"; pass=$((pass + 1)); }
no()  { echo "  FAIL $1"; failn=$((failn + 1)); }
t()   { echo; echo "######## [$FS] $*"; }

if [[ $FS == bindfs ]]; then
  apt-get -qq update >/dev/null && apt-get -qq install -y bindfs fuse3 procps >/dev/null 2>&1 || { echo "cannot install bindfs"; exit 1; }
  mkdir -p /backing /workspace && bindfs -o allow_other --perms=a+rwx --chmod-ignore --chown-deny /backing /workspace || exit 1
else
  apt-get -qq update >/dev/null && apt-get -qq install -y procps >/dev/null 2>&1
fi
mkdir -p $R/preserve/x && echo keep > $R/preserve/x/f
before="$(cd $R && find . -mindepth 1 -printf '%p %s %T@\n' | sort | sha256sum)"
same_root() { [[ "$(cd $R && find . -mindepth 1 -printf '%p %s %T@\n' | sort | sha256sum)" == "$before" ]]; }
no_probe_dir() { [[ -z "$(find $R -maxdepth 1 -name '.probe-*' -print -quit)" ]]; }
no_probe_proc() { ! pgrep -f "python3 $P" >/dev/null; }
fails_of() { grep -oE '^FAIL  [a-z0-9._-]+' "$1" | awk '{print $2}' | sort -u | tr '\n' ' '; }

t "regression: full run"
python3 $P > /tmp/full 2>&1; rc=$?
[[ $rc == 0 ]] && ok "exit 0" || no "exit $rc"
got="$(fails_of /tmp/full)"
if [[ $FS == ext4 ]]; then want="env.loop-mount-possible "
else want="env.loop-mount-possible io.o_direct isolation.uid-reads-root-0600 mode.chmod-0600-file mode.chmod-0700-dir owner.chown-by-root "; fi
# rename.atomic-under-reads on bindfs depends on timing (2-10 of 3000 reads in earlier runs): allowed either way there.
got_cmp="$got"; [[ $FS == bindfs ]] && got_cmp="$(tr ' ' '\n' <<<"$got" | grep -v '^rename.atomic-under-reads$' | grep . | tr '\n' ' ')"
[[ "$got_cmp" == "$want" ]] && ok "FAIL set as expected: ${got:-none}" || no "FAIL set: got '${got}', want '${want}' (+ rename.atomic-under-reads allowed on bindfs)"
grep -q '^PASS  cleanup.children' /tmp/full && grep -q '^PASS  cleanup.probe-dir-removed' /tmp/full && grep -q '^PASS  cleanup.root-unchanged' /tmp/full \
  && ok "cleanup PASS (children, probe dir, root unchanged)" || no "cleanup lines"
grep -qE 'measured' /tmp/full && ok "measured footprint reported: $(grep -oE '[0-9.]+ MB measured' /tmp/full)" || no "no measured footprint"
no_probe_dir && same_root && [[ "$(cat $R/preserve/x/f)" == keep ]] && ok "root and preserve unchanged" || no "root changed"

interrupt() {  # interrupt <label> <signal> <file pattern that shows the phase is running> <probe args...>
  local label="$1" sig="$2" pat="$3"; shift 3
  t "interrupt: $label ($sig)"
  python3 $P "$@" > /tmp/int.out 2> /tmp/int.err & local pid=$!
  local i=0
  until find $R -path "$R/.probe-*/$pat" -print -quit 2>/dev/null | grep -q .; do
    sleep 0.05; i=$((i + 1)); (( i < 400 )) || { no "$label: phase never started"; kill -9 $pid 2>/dev/null; return; }
  done
  sleep 0.3
  local kids; kids="$(pgrep -P $pid | wc -l)"
  kill -"$sig" $pid; wait $pid; local rc=$?
  [[ $rc == 130 ]] && ok "$label: exit 130" || no "$label: exit $rc"
  (( kids > 0 )) && ok "$label: $kids child process(es) were running when the signal came" || no "$label: no child running at the signal (test did not hit the phase)"
  grep -q "INTERRUPTED by signal" /tmp/int.err && grep -q ", 0 left; cleanup done" /tmp/int.err \
    && ok "$label: $(grep -oE '[0-9]+ check process\(es\) killed and reaped, 0 left' /tmp/int.err)" || no "$label: stderr: $(cat /tmp/int.err)"
  sleep 0.2
  no_probe_proc && ok "$label: no probe process left" || { no "$label: processes left: $(pgrep -af "python3 $P")"; pkill -9 -f "python3 $P"; }
  no_probe_dir && ok "$label: no .probe-* left" || no "$label: .probe-* left: $(ls -a $R)"
  same_root && ok "$label: rest of $R unchanged" || no "$label: $R changed"
}
interrupt "concurrent fsync" TERM 'conc-*' --only fsync-concurrent --concurrent-procs 4 --concurrent-n 7000
interrupt "rename under reads" TERM 'heartbeat' --only rename --rename-writes 15000
interrupt "concurrent fsync" INT 'conc-*' --only fsync-concurrent --concurrent-procs 4 --concurrent-n 7000

t "budget (planned): writes larger than --footprint-cap-mb are refused before they happen"
python3 $P --only fsync,throughput --footprint-cap-mb 1 > /tmp/b1 2>&1; rc=$?
[[ $rc == 0 ]] && ok "exit 0" || no "exit $rc"
grep -q '^SKIP  fsync .*planned footprint would exceed 1048576' /tmp/b1 && grep -q '^SKIP  throughput.sequential .*planned footprint' /tmp/b1 \
  && ok "fsync and throughput SKIP on the planned footprint" || no "$(grep -E 'fsync|throughput' /tmp/b1)"
w="$(grep -oE 'bytes written [0-9]+' /tmp/b1 | awk '{print $3}')"; [[ $w == 0 ]] && ok "0 bytes written" || no "bytes written $w"
no_probe_dir && same_root && ok "cleanup" || no "left over"

t "budget (measured): unexpected data in the probe directory over the cap stops every remaining check"
python3 $P --only fsync,throughput --fsync-n 5000 --footprint-cap-mb 60 > /tmp/b1m 2>&1 & pid=$!
i=0; until d="$(find $R -maxdepth 1 -name '.probe-*' -print -quit)"; [[ -n $d && -e $d/fsync-8k ]]; do sleep 0.02; i=$((i+1)); (( i < 500 )) || break; done
head -c $((30 * 1024 * 1024)) /dev/zero > "$d/planted" 2>/dev/null
wait $pid; rc=$?
[[ $rc == 0 ]] && ok "exit 0 (cleanup verified)" || no "exit $rc"
grep -q '^FAIL  budget.hard-stop .*> cap 62914560' /tmp/b1m && ok "hard stop: $(grep -oE 'fsync series [^;]*' /tmp/b1m | head -1)" || no "no hard stop: $(grep -E 'budget|fsync|SKIP' /tmp/b1m)"
! grep -qE '^INFO  (fsync.16m-segment|throughput.sequential)' /tmp/b1m && ok "no check ran after the hard stop" || no "a check ran after the hard stop"
no_probe_dir && same_root && ok "cleanup after hard stop (planted file removed with the probe directory)" || no "left over after hard stop"

t "budget: --max-bytes 10 MB -> heavy checks SKIP, total written <= 10 MB"
python3 $P --max-bytes 10485760 > /tmp/b2 2>&1; rc=$?
[[ $rc == 0 ]] && ok "exit 0" || no "exit $rc"
w="$(grep -oE 'bytes written [0-9]+' /tmp/b2 | awk '{print $3}')"
(( ${w:-999999999} <= 10485760 )) && ok "bytes written $w <= 10485760" || no "bytes written $w"
grep -q '^SKIP  .*byte budget reached' /tmp/b2 && ok "SKIP lines: $(grep -c '^SKIP  .*byte budget' /tmp/b2)" || no "no budget SKIP"
no_probe_dir && same_root && ok "cleanup" || no "left over"

t "a failing check process never runs the parent's code"
python3 - > /tmp/sp 2>&1 <<'PY'
import os, sys
sys.path.insert(0, "/probe")
import probe_v2 as pv
codes = [pv.reap(pv.spawn(lambda: 1 / 0)), pv.reap(pv.spawn(lambda: (_ for _ in ()).throw(OSError(28, "No space left on device")))),
         pv.reap(pv.spawn(lambda: 3)), pv.reap(pv.spawn(lambda: None))]
print("after-spawn", os.getpid(), codes, len(pv.children))
PY
[[ "$(grep -c '^after-spawn' /tmp/sp)" == 1 ]] && ok "only the parent reached the code after spawn ($(cat /tmp/sp))" || no "children escaped: $(cat /tmp/sp)"
grep -q '\[1, 1, 3, 0\] 0$' /tmp/sp && ok "exit codes ZeroDivision=1, ENOSPC=1, int=3, None=0; no child left tracked" || no "codes: $(cat /tmp/sp)"

t "concurrent fsync with long results (one pipe per worker, no interleaving)"
python3 $P --only fsync-concurrent --concurrent-procs 4 --concurrent-n 3000 > /tmp/cc 2>&1; rc=$?
[[ $rc == 0 ]] && grep -q '^INFO  fsync.8k-concurrent .*4 processes x 3000' /tmp/cc && ok "4 x 3000 results parsed: $(grep -oE 'p50 [0-9.]+ ms' /tmp/cc)" || no "rc $rc: $(grep -E 'fsync|FAIL' /tmp/cc)"
no_probe_dir && same_root && ok "cleanup" || no "left over"

t "refused arguments write nothing"
for args in "--max-bytes 734003200" "--footprint-cap-mb 400" "--only bogus" "--throughput-mb 512"; do
  # shellcheck disable=SC2086
  python3 $P $args > /tmp/r 2>&1; rc=$?
  [[ $rc == 2 ]] && grep -q REFUSED /tmp/r && no_probe_dir && ok "$args: REFUSED, exit 2, nothing created" || no "$args: exit $rc"
done
same_root && [[ "$(cat $R/preserve/x/f)" == keep ]] && ok "preserve intact at the end" || no "preserve changed"

echo
echo "PROBE-TESTS $FS: $pass passed, $failn failed"
exit $(( failn > 0 ))
