#!/usr/bin/env bash
# supervisord program (user postgres, started through pg-start.sh under the lease).
# Streams WAL to the network volume with pg_receivewal --synchronous: every write is
# fsynced on the volume before the server is told it is flushed. postgresql.cas.conf
# names this connection (application_name cas_vol) in synchronous_standby_names with
# synchronous_commit=on, so a COMMIT returns only after its WAL is on the volume, and
# while this receiver is down commits wait (fail-closed) instead of completing locally.
set -euo pipefail
W="${CAS_WORKSPACE:-/workspace/cas}"
DIR="$W/pg/wal"
CONN="host=/run/postgresql user=postgres application_name=cas_vol"
log() { echo "[pg-receivewal] $(date -u +%FT%TZ) $*" >&2; }

for _ in $(seq 1 120); do
  pg_isready -q -h /run/postgresql && break
  sleep 1
done
pg_isready -q -h /run/postgresql || { log "postgres not ready after 120 s"; exit 1; }
[[ -d "$DIR" && -w "$DIR" ]] || { log "FATAL: $DIR missing or not writable"; exit 1; }
# The slot keeps every WAL segment the receiver has not confirmed (bounded by
# max_slot_wal_keep_size on the container disk). A restored cluster has no slot yet.
pg_receivewal --create-slot --if-not-exists --slot=cas_vol -d "$CONN" --no-password
log "streaming to $DIR (slot cas_vol, synchronous)"
exec pg_receivewal -D "$DIR" --slot=cas_vol --synchronous --no-password -d "$CONN"
