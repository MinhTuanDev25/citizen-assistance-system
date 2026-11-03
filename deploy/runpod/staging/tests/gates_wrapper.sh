#!/usr/bin/env bash
# Integration tests of gates() in verify_after_switch.sh (the Bash side of verify_gates.py):
# the function is taken verbatim from the script and run against stub gates that crash,
# exit inconsistently, print nothing or malformed lines, or miss a dependency.
#   bash deploy/runpod/staging/tests/gates_wrapper.sh [staging dir]
# Prints "GATES-WRAPPER: N passed, M failed"; exit 1 on any failure.
set -uo pipefail
STAGING="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
V="$STAGING/verify_after_switch.sh"
eval "$(sed -n '/^bad()/p; /^gates() {/,/^}/p' "$V")"
declare -F gates >/dev/null && declare -F bad >/dev/null || { echo "gates()/bad() not found in $V"; exit 1; }
pass=0 failn=0
STUBS="$(mktemp -d)"; trap '/bin/rm -rf "$STUBS"' EXIT

# case <name> <expected fails> <expected warns> <python body of the stub verify_gates.py>
case_() {
  local name="$1" ef="$2" ew="$3"
  mkdir -p "$STUBS/$name"; printf '%s\n' "$4" > "$STUBS/$name/verify_gates.py"
  fails=0 warns=0 HERE="$STUBS/$name"
  gates gpu > "$STUBS/$name.out" 2>&1
  if [[ $fails == "$ef" && $warns == "$ew" ]]; then echo "  ok   $name: fails=$fails warns=$warns"; pass=$((pass + 1))
  else echo "  FAIL $name: fails=$fails warns=$warns, expected $ef/$ew"; sed 's/^/       > /' "$STUBS/$name.out"; failn=$((failn + 1)); fi
}
P='print("PASS  speech-on-gpu      ok")'
case_ valid-pass        0 0 "print('INFO  listener           2 loopback-only sockets'); $P"
case_ valid-fail-warn   2 1 'import sys
print("FAIL  listener           tcp 0.0.0.0:53 not a CAS service"); print("      owner: none")
print("WARN  listener           something"); print("FAIL  listener           tcp [::]:53 not a CAS service"); sys.exit(1)'
case_ crash             1 0 'raise RuntimeError("cannot read /proc")'
case_ missing-module    1 0 'import cas_module_that_does_not_exist'
case_ exit1-without-fail 1 0 "import sys; $P; sys.exit(1)"
case_ exit0-with-fail   2 0 'print("FAIL  speech-on-gpu      x")'
case_ exit2             1 0 "import sys; $P; sys.exit(2)"
case_ empty-output      1 0 'pass'
case_ info-only         1 0 'print("INFO  listener           nothing checked")'
case_ malformed-line    1 0 "$P; print('Traceback-like garbage')"
case_ fail-in-stderr    1 0 'import sys; sys.stderr.write("FAIL but not a gate line\n"); sys.exit(1)'

# python3 missing on PATH: command not found (127)
mkdir -p "$STUBS/nopy"; printf '%s\n' "$P" > "$STUBS/nopy/verify_gates.py"
fails=0 warns=0 HERE="$STUBS/nopy"; PATH_SAVE="$PATH"; PATH="$STUBS/empty-path"; gates gpu >/dev/null 2>&1; PATH="$PATH_SAVE"
if [[ $fails == 1 ]]; then echo "  ok   python3-missing: fails=1"; pass=$((pass + 1)); else echo "  FAIL python3-missing: fails=$fails"; failn=$((failn + 1)); fi

# the real verify_gates.py with arguments it refuses (argparse exit 2)
fails=0 warns=0 HERE="$STAGING"; gates listeners --no-such-option >/dev/null 2>&1
if [[ $fails == 1 ]]; then echo "  ok   real-module-bad-args: fails=1"; pass=$((pass + 1)); else echo "  FAIL real-module-bad-args: fails=$fails"; failn=$((failn + 1)); fi

echo "GATES-WRAPPER: $pass passed, $failn failed"
(( failn == 0 ))
