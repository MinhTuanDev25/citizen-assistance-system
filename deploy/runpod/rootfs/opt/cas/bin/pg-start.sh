#!/usr/bin/env bash
# Wrapper for the postgres / qdrant / minio programs: exec the data service only
# while this container holds the workspace lease with a fresh heartbeat.
#   pg-start.sh <command> [args...]
set -euo pipefail
W="${CAS_WORKSPACE:-/workspace/cas}"
LEASE="$W/locks/workspace.lease"
MAX_AGE="${CAS_LEASE_STALE_S:-60}"
ME="$(cat /run/cas-lease.owner)"
owner="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("owner_id",""))' "$LEASE/owner.json" 2>/dev/null || true)"
hb="$(cat "$LEASE/heartbeat" 2>/dev/null || echo 0)"
age=$(( $(date +%s) - hb ))
if [[ "$owner" != "$ME" ]]; then
  echo "[pg-start] refusing $(basename "$1"): lease owner '$owner' is not this container '$ME'" >&2; exit 1
fi
if (( age > MAX_AGE )); then
  echo "[pg-start] refusing $(basename "$1"): lease heartbeat is ${age}s old (> ${MAX_AGE}s)" >&2; exit 1
fi
exec "$@"
