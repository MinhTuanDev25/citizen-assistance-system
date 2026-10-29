#!/usr/bin/env bash
# One-shot bootstrap, run by supervisord as user "cas" after the data services start.
#   1. wait until Postgres (TLS query), Qdrant and MinIO (HTTPS, internal CA) answer
#   2. refuse a dirty migration state; pg_dump before applying pending migrations
#   3. migrate up (never down, never force)
#   4. create the MinIO bucket if missing (TLS verified, no --insecure)
#   5. provision the Qdrant collection if missing (dimension from the embedding manifest)
#   6. write /run/cas/bootstrap.ok = this boot id, exit 0  (supervisor state EXITED)
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
wait_until "minio" curl -fsS --max-time 3 --cacert "$CAS_CA_CERT" https://127.0.0.1:9000/minio/health/ready

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

# ---- object storage bucket ----------------------------------------------------
/opt/cas/venv-ai/bin/python - <<'PY'
import os
from minio import Minio
c = Minio(os.environ["OBJECT_STORAGE_ENDPOINT"], access_key=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
          secret_key=os.environ["OBJECT_STORAGE_SECRET_KEY"], secure=True,
          region=os.environ.get("OBJECT_STORAGE_REGION") or None)
b = os.environ["OBJECT_STORAGE_BUCKET"]
if c.bucket_exists(b):
    print(f"[bootstrap] bucket {b} exists", flush=True)
else:
    c.make_bucket(b)
    print(f"[bootstrap] bucket {b} created", flush=True)
PY

# ---- Qdrant collection (same call as scripts/provision-qdrant.py) -----------------
# Creates it when absent; an existing collection with another size/distance fails.
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

# ---- marker -------------------------------------------------------------------
tmp="$(mktemp /run/cas/.bootstrap.XXXXXX)"
printf '%s' "$BOOT_ID" > "$tmp"
chmod 0644 "$tmp"
mv -f "$tmp" /run/cas/bootstrap.ok
log "bootstrap complete (boot $BOOT_ID)"
