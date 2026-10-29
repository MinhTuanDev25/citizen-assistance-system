#!/usr/bin/env bash
# Block until startup dependencies are ready, then exec the real program
# (so supervisord signals reach it directly).
#   --marker        wait for /run/cas/bootstrap.ok written in THIS boot
#                   (its content must equal /run/cas/boot_id)
#   --http URL      wait for an HTTP 2xx (repeatable)
# Only affects startup. A dependency that dies later is handled by the
# program itself (readiness endpoints, reconnects), not by this script.
set -euo pipefail

MARKER=0
URLS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --marker) MARKER=1; shift ;;
    --http) URLS+=("$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "wait-for: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "wait-for: no command" >&2; exit 2; }

name="$(basename "$1")"
start=$SECONDS
next_log=0

ready() {
  if [[ $MARKER == 1 ]]; then
    [[ -f /run/cas/bootstrap.ok && -f /run/cas/boot_id ]] || return 1
    [[ "$(cat /run/cas/bootstrap.ok)" == "$(cat /run/cas/boot_id)" ]] || return 1
  fi
  local u
  for u in ${URLS[@]+"${URLS[@]}"}; do
    curl -fs -o /dev/null --max-time 3 "$u" || return 1
  done
  return 0
}

until ready; do
  if (( SECONDS >= next_log )); then
    echo "wait-for[$name]: waiting $((SECONDS - start))s (marker=$MARKER urls=${URLS[*]-})" >&2
    next_log=$((SECONDS + 15))
  fi
  sleep 1
done
echo "wait-for[$name]: ready after $((SECONDS - start))s, exec" >&2
exec "$@"
