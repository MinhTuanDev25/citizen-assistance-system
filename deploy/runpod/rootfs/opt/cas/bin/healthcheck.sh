#!/usr/bin/env bash
# Operator health report for the all-in-one pod. Run as root: docker exec <c> healthcheck.sh
#   --quiet   only the exit code (used by the image HEALTHCHECK)
# Exit 0 only when every required check passes. Speech is required in gpu mode only.
set -uo pipefail

QUIET=0; [[ "${1:-}" == --quiet ]] && QUIET=1
W="${CAS_WORKSPACE:-/workspace/cas}"
CA="$W/tls/ca.crt"
fail=0
row() { [[ $QUIET == 1 ]] || printf '%-14s %-5s %s\n' "$1" "$2" "$3"; }
check() {  # check <name> <required 1|0> <command...>
  local name="$1" req="$2"; shift 2
  local out
  if out="$("$@" 2>&1)"; then row "$name" OK "${out:0:90}"
  else row "$name" FAIL "${out:0:90}"; [[ $req == 1 ]] && fail=1; fi
}

boot_ok() {
  [[ -f /run/cas/bootstrap.ok ]] || { echo "no bootstrap.ok"; return 1; }
  [[ "$(cat /run/cas/bootstrap.ok)" == "$(cat /run/cas/boot_id)" ]] || { echo "stale marker"; return 1; }
  echo "boot $(cat /run/cas/boot_id)"
}
mode="${CAS_SPEECH_MODE:-unknown}"

if [[ -f /run/cas-maintenance ]]; then
  row maintenance FAIL "$(cat /run/cas-maintenance)"
  exit 1
fi
check bootstrap   1 boot_ok
check postgres    1 pg_isready -h 127.0.0.1 -p 5432 -U cas -d citizen_assistance
check qdrant      1 curl -fsS --max-time 3 http://127.0.0.1:6333/readyz
check minio       1 curl -fsS --max-time 3 --cacert "$CA" https://127.0.0.1:9000/minio/health/ready
check ai-service  1 curl -fsS --max-time 3 http://127.0.0.1:8001/ready
check api-ready   1 curl -fsS --max-time 3 http://127.0.0.1:8080/ready
check nginx       1 curl -fsS --max-time 3 http://127.0.0.1:8088/readyz
if [[ "$mode" == gpu ]]; then
  check speech    1 curl -fsS --max-time 3 http://127.0.0.1:8090/readyz
else
  row speech SKIP "disabled (CAS_SPEECH_MODE=$mode)"
fi
if [[ $QUIET == 0 && -S /run/supervisor.sock ]]; then
  echo "--- supervisor"; supervisorctl -c /etc/cas/supervisord.conf status
fi
exit $fail
