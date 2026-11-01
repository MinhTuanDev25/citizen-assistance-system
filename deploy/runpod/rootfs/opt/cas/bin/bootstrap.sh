#!/usr/bin/env bash
# One-shot bootstrap, run by supervisord as user "cas" after the data services start.
#   1. wait until Postgres (TLS query) and Qdrant answer, and the WAL receiver streams
#      synchronously to the volume (no commit before WAL can reach the volume)
#   2. refuse a dirty migration state; pg_dump before applying pending migrations
#   3. migrate up (never down, never force)
#   4. object store: the bucket directory on the volume, and every document's object
#      present with its size and SHA-256 (cas-storage objects-verify)
#   5. Qdrant: restore an empty collection from the newest snapshot on the volume and
#      verify it (fingerprint + Postgres READY generations), or verify the existing
#      one; provision an empty collection only when nothing is indexed
#   6. wait for the verified base backup of this cluster (cas-keeper) when one is due
#   7. write /run/cas/bootstrap.ok = this boot id, exit 0  (supervisor state EXITED)
# Any failure exits non-zero: no marker, app programs keep waiting, healthcheck reports it.
set -Eeuo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
TIMEOUT="${CAS_BOOTSTRAP_TIMEOUT_S:-300}"
log() { echo "[bootstrap] $(date -u +%FT%TZ) $*" >&2; }
die() { log "FATAL: $*"; exit 1; }
trap 'die "failed at line $LINENO"' ERR

wait_until() {
  local what="$1"; shift
  local start=$SECONDS err=""
  until err="$("$@" 2>&1 >/dev/null)"; do
    (( SECONDS - start < TIMEOUT )) || die "$what not ready after ${TIMEOUT}s; last error: ${err:-none}"
    sleep 1
  done
  log "$what ready after $((SECONDS - start))s"
}

[[ -f /run/cas/boot_id ]] || die "/run/cas/boot_id missing (entrypoint did not run)"
BOOT_ID="$(cat /run/cas/boot_id)"

wait_until "postgres" psql "$DATABASE_URL" -tAc "select 1"
wait_until "qdrant" curl -fsS --max-time 3 http://127.0.0.1:6333/readyz
cas-storage receiver-wait "$TIMEOUT" || die "the WAL receiver does not stream synchronously to the volume; no write is allowed"

# ---- migrations ---------------------------------------------------------------
MIG_DIR=/opt/cas/migrations
latest="$(ls "$MIG_DIR" | sed -n 's/^\([0-9]\{1,\}\)_.*\.up\.sql$/\1/p' | sort -n | tail -1 | sed 's/^0*//')"
[[ "$latest" =~ ^[0-9]+$ ]] || die "no migrations found in $MIG_DIR"
# Parse strictly: a connection error must not be read as a version number.
rc=0; ver_out="$(migrate -path "$MIG_DIR" -database "$DATABASE_URL" version 2>&1)" || rc=$?
ver_line="$(tail -1 <<<"$ver_out")"
if [[ $rc -eq 0 && "$ver_line" =~ ^([0-9]+)$ ]]; then
  current="${BASH_REMATCH[1]}"
elif [[ $rc -eq 0 && "$ver_line" =~ ^[0-9]+\ \(dirty\)$ ]]; then
  die "migration state is dirty ($ver_line); fix manually, bootstrap never forces"
elif [[ $rc -ne 0 && "$ver_line" == *"no migration"* ]]; then
  current=0
else
  die "cannot read the migration version (exit $rc): $ver_out"
fi
log "migrations: current=$current latest=$latest"

if (( current > latest )); then
  die "database version $current is newer than this image ($latest); refusing to start an older image"
fi
if (( current < latest )); then
  if (( current > 0 )); then
    mkdir -p "$W/backups/pre-migrate"
    dump="$W/backups/pre-migrate/v${current}-to-v${latest}-$(date -u +%Y%m%dT%H%M%SZ).dump"
    log "pg_dump before migrating -> $dump"
    pg_dump -Fc -f "$dump" "$DATABASE_URL"
  fi
  migrate -path "$MIG_DIR" -database "$DATABASE_URL" up
  log "migrations applied up to $latest"
fi

# ---- object store (files on the volume) -----------------------------------------
d="$OBJECT_STORAGE_FILE_ROOT/$OBJECT_STORAGE_BUCKET"
[[ -d "$d" && ! -L "$d" && -w "$d" ]] || die "object bucket directory $d missing, a symlink or not writable"
cas-storage objects-verify || die "object store verification failed"

# ---- Qdrant: restore + verify, or provision ------------------------------------------
cas-storage qdrant-restore-verify || die "qdrant restore/verification failed"
# Creates the collection when absent (nothing indexed yet); an existing collection
# with another size/distance fails.
if [[ "${EMBEDDING_PROVIDER:-onnx}" == onnx ]]; then
  dim="$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["dimension"]))' "$EMBEDDING_MODEL_DIR/manifest.json")" \
    || die "cannot read the embedding dimension from $EMBEDDING_MODEL_DIR/manifest.json"
else
  dim=384   # fake provider (tests only); same default as provision_qdrant_collection
fi
( cd /opt/cas/ai && /opt/cas/venv-ai/bin/python -c "
import os
from app.indexing.runtime import provision_qdrant_collection
provision_qdrant_collection(os.environ['QDRANT_URL'], os.environ['QDRANT_COLLECTION'], $dim)
print('[bootstrap] qdrant collection', os.environ['QDRANT_COLLECTION'], 'ready (dim $dim)', flush=True)
" )

# ---- base backup ------------------------------------------------------------------------
# A fresh or restored cluster has no base backup for its current timeline yet; the
# services open only once cas-keeper has taken and verified one.
start=$SECONDS
until [[ -f /run/cas/base-backup.ok ]]; do
  (( SECONDS - start < TIMEOUT )) || die "no verified base backup after ${TIMEOUT}s (see cas-keeper lines)"
  sleep 2
done
log "base backup on the volume verified (cas-keeper, pg_verifybackup)"
# The durable LSN record (cas-keeper, ~1 s) must be current for this boot: a later
# restore proves it recovered at least that far.
# (cas-keeper writes /run/cas/durable-lsn after each record on the volume; /run/cas is
# emptied at every boot.)
until [[ -s /run/cas/durable-lsn ]]; do
  (( SECONDS - start < TIMEOUT )) || die "durable LSN not recorded on the volume after ${TIMEOUT}s (see cas-keeper lines)"
  sleep 1
done
log "durable LSN recorded on the volume ($(cat /run/cas/durable-lsn))"

# ---- marker -------------------------------------------------------------------
tmp="$(mktemp /run/cas/.bootstrap.XXXXXX)"
printf '%s' "$BOOT_ID" > "$tmp"
chmod 0644 "$tmp"
mv -f "$tmp" /run/cas/bootstrap.ok
log "bootstrap complete (boot $BOOT_ID)"
