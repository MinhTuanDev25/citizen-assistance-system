#!/usr/bin/env bash
# supervisord program (root, lowest priority number: started first, stopped last).
# Keeps the workspace lease taken by entrypoint.sh alive:
#   - every CAS_LEASE_HEARTBEAT_S writes the current epoch to lease/heartbeat
#   - fencing: if owner.json no longer names this owner, or the heartbeat cannot
#     be written, stops postgres/qdrant/minio at once and exits 1
#   - on SIGTERM (container shutdown): releases the lease only after Postgres,
#     Qdrant and MinIO have stopped; otherwise keeps it (safe default)
set -uo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
LEASE="$W/locks/workspace.lease"
EVERY="${CAS_LEASE_HEARTBEAT_S:-10}"
ME="$(cat /run/cas-lease.owner 2>/dev/null || true)"
log() { echo "[pg-lease] $(date -u +%FT%TZ) $*" >&2; }
owner() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("owner_id",""))' "$LEASE/owner.json" 2>/dev/null; }
data_running() { pgrep -u postgres -x postgres >/dev/null || pgrep -u qdrant -x qdrant >/dev/null || pgrep -u minio -x minio >/dev/null; }
fence() {
  log "FENCING: $*; stopping data services"
  supervisorctl -c /etc/cas/supervisord.conf stop postgres qdrant minio >&2 || true
  exit 1
}

release() {

  for _ in $(seq 1 90); do data_running || break; sleep 1; done
  if data_running; then
    log "data services still running at shutdown; lease KEPT (next start needs the same owner or cas-pg-recover)"
    exit 0
  fi
  if [[ "$(owner)" == "$ME" ]]; then
    mv "$LEASE" "$LEASE.released-$(date -u +%Y%m%dT%H%M%SZ)-$ME" && log "lease released by $ME (clean shutdown)"
    # Keep only the 5 newest released records for audit.
    ls -1dt "$LEASE".released-* 2>/dev/null | tail -n +6 | xargs -r rm -rf
  fi
  exit 0
}
trap release TERM INT

[[ -n "$ME" ]] || fence "no owner id for this boot (/run/cas-lease.owner)"
log "holding workspace lease as $ME (heartbeat every ${EVERY}s)"
# One failed heartbeat write is tolerated (transient network-volume hiccup); two in
# a row fence the data services. 2 x EVERY (20 s) stays below the recovery wait of
# cas-pg-recover (3 x EVERY + 5 = 35 s), so another owner can never take over a
# workspace whose Postgres is still running here.
misses=0
while true; do
  [[ "$(owner)" == "$ME" ]] || fence "lease owner is now '$(owner)', not '$ME'"
  if { date +%s > "$LEASE/heartbeat.tmp" && mv -f "$LEASE/heartbeat.tmp" "$LEASE/heartbeat"; } 2>/dev/null; then
    misses=0
  else
    misses=$((misses + 1))
    log "heartbeat write failed ($misses in a row)"
    (( misses < 2 )) || fence "cannot write the lease heartbeat"
  fi
  sleep "$EVERY" & wait $!
done
