#!/usr/bin/env bash
# Entry point of the all-in-one image. Runs as root under tini, before supervisord.
# Does not start any long-running service. Steps:
#   1. validate CAS_SPEECH_MODE, LLM_PROVIDER and secrets (fail fast, no implicit defaults)
#   2. model and Speech mode checks: C0 code manifest always; checkpoints and GPU
#      fail-closed in gpu mode (before any init work)
#   3. single-writer lease on the workspace (flock + atomic lease dir); a lease of
#      another owner puts the container in maintenance mode (see cas-pg-recover)
#   4. directories + ownership on /workspace/cas (re-applied every start: a
#      network volume may come from another pod)
#   5. internal CA + leaf certs: create only when absent, never regenerate or
#      overwrite existing keys; refuse a half-present or invalid set
#   6. Postgres: first-time initdb + role/db/extensions (unix socket only),
#      pg_hba/conf from the image on every start
#   7. reset /run/cas (new boot id, remove stale bootstrap.ok); write one root-only
#      env file per service (/run/cas-env, used by svc-run) so that each program
#      sees only its own variables; exec supervisord (keeps the flock fd)
set -Eeuo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
export CAS_WORKSPACE="$W"
log() { echo "[entrypoint] $(date -u +%FT%TZ) $*" >&2; }
die() { log "FATAL: $*"; exit 64; }
trap 'die "failed at line $LINENO"' ERR

# ---- 1. mode and secrets ------------------------------------------------------
MODE="${CAS_SPEECH_MODE:-}"
case "$MODE" in
  off|gpu) ;;
  *) die "CAS_SPEECH_MODE must be 'off' or 'gpu' (got '${MODE}'); there is no default" ;;
esac
log "speech_mode=$MODE"

need_secret() {
  local name="$1" min="${2:-16}" v="${!1:-}"
  [[ -n "$v" ]] || die "$name is required"
  (( ${#v} >= min )) || die "$name must be at least $min characters"
  case "$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')" in
    *local-dev*|*change-me*|*changeme*|*minioadmin*|*placeholder*) die "$name looks like a placeholder" ;;
  esac
}
need_secret JWT_SECRET 32
need_secret AI_SERVICE_TOKEN 16
need_secret POSTGRES_PASSWORD 16
need_secret MINIO_ROOT_USER 8
need_secret MINIO_ROOT_PASSWORD 16

if [[ "$MODE" == off ]]; then
  if [[ -n "${SPEECH_ENABLED:-}" && "${SPEECH_ENABLED,,}" != false ]]; then
    die "CAS_SPEECH_MODE=off but SPEECH_ENABLED=${SPEECH_ENABLED}; conflicting configuration"
  fi
  export SPEECH_ENABLED=false CAS_SPEECH_AUTOSTART=false
else
  if [[ -n "${SPEECH_ENABLED:-}" && "${SPEECH_ENABLED,,}" != true ]]; then
    die "CAS_SPEECH_MODE=gpu but SPEECH_ENABLED=${SPEECH_ENABLED}; conflicting configuration"
  fi
  need_secret SPEECH_SERVICE_TOKEN 16
  export SPEECH_ENABLED=true CAS_SPEECH_AUTOSTART=true
fi

# No default provider under APP_ENV=prod: a silent mock would answer citizens with
# canned extraction. mock is accepted only for tests, behind an explicit flag.
LLM_PROVIDER="$(printf '%s' "${LLM_PROVIDER:-}" | tr '[:upper:]' '[:lower:]')"
case "$LLM_PROVIDER" in
  gemini) need_secret GEMINI_API_KEY 20 ;;
  openai) need_secret OPENAI_API_KEY 20 ;;
  mock) [[ "${CAS_ALLOW_MOCK_LLM:-}" == true ]] || die "LLM_PROVIDER=mock is for tests only; set CAS_ALLOW_MOCK_LLM=true to allow it" ;;
  "") die "LLM_PROVIDER is required (gemini | openai | mock with CAS_ALLOW_MOCK_LLM=true)" ;;
  *) die "LLM_PROVIDER must be gemini, openai or mock (got '$LLM_PROVIDER')" ;;
esac
export LLM_PROVIDER

# ---- derived runtime environment -------------------------------------------------
# Computed here for the whole boot, but NOT inherited as-is by the programs: step 7
# writes a per-service allowlist (/run/cas-env/<service>.env) and svc-run starts
# every program with env -i plus that list only.
urlenc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
export APP_ENV=prod
export CAS_CA_CERT="$W/tls/ca.crt"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# verify-full: the client checks the server certificate chain (internal CA) and that
# it was issued for 127.0.0.1 (SAN), not only that the channel is encrypted.
pg_pw_enc="$(urlenc "$POSTGRES_PASSWORD")" || die "cannot URL-encode POSTGRES_PASSWORD"
[[ -n "$pg_pw_enc" ]] || die "URL-encoded POSTGRES_PASSWORD is empty"
export DATABASE_URL="postgres://cas:${pg_pw_enc}@127.0.0.1:5432/citizen_assistance?sslmode=verify-full&sslrootcert=$W/tls/ca.crt"
export API_ADDR=127.0.0.1:8080
export CONFIG_FILE=/opt/cas/api/configs/prod/config.yaml
export XA_ID="${XA_ID:-xa_chu_se}"
export AI_SERVICE_URL=http://127.0.0.1:8001
export AI_EXTRACT_ENABLED="${AI_EXTRACT_ENABLED:-true}"
export AI_RETRIEVE_ENABLED="${AI_RETRIEVE_ENABLED:-true}"
export ADMIN_INGESTION_ENABLED="${ADMIN_INGESTION_ENABLED:-true}"
export ADMIN_INDEXING_ENABLED="${ADMIN_INDEXING_ENABLED:-true}"
export INDEX_MODE=pipeline
# Synchronous indexing would outlive the 100 s RunPod proxy limit.
export INDEX_ASYNC=true
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-onnx}"
export OCR_PROVIDER="${OCR_PROVIDER:-tesseract}"
export EMBEDDING_MODEL_DIR="${EMBEDDING_MODEL_DIR:-$W/models/embedding}"
export OCR_MODEL_DIR="${OCR_MODEL_DIR:-$W/models/ocr}"
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-knowledge_chunks}"
export OBJECT_STORAGE_ENDPOINT=127.0.0.1:9000
export OBJECT_STORAGE_ACCESS_KEY="$MINIO_ROOT_USER"
export OBJECT_STORAGE_SECRET_KEY="$MINIO_ROOT_PASSWORD"
export OBJECT_STORAGE_BUCKET="${OBJECT_STORAGE_BUCKET:-cas-documents}"
export OBJECT_STORAGE_REGION="${OBJECT_STORAGE_REGION:-us-east-1}"
export OBJECT_STORAGE_USE_SSL=true
export OBJECT_STORAGE_AUTO_CREATE_BUCKET=false
export MINIO_BROWSER=off
export QDRANT__SERVICE__HOST=127.0.0.1
export QDRANT__STORAGE__STORAGE_PATH="$W/qdrant/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$W/qdrant/snapshots"
export QDRANT__TELEMETRY_DISABLED=true
# index-reaper runs once and exits unless an interval is set (same values as
# docker-compose.production.yml).
export INDEX_REAPER_INTERVAL_SECONDS="${INDEX_REAPER_INTERVAL_SECONDS:-60}"
export INDEX_ORPHAN_GRACE_SECONDS="${INDEX_ORPHAN_GRACE_SECONDS:-120}"
export INDEX_CLEANUP_BATCH_SIZE="${INDEX_CLEANUP_BATCH_SIZE:-8}"
export INDEX_REAPER_SHUTDOWN_SECONDS="${INDEX_REAPER_SHUTDOWN_SECONDS:-30}"
export QDRANT_INIT_FILE_PATH="$W/qdrant/.qdrant-initialized"
export SPEECH_SERVICE_URL=http://127.0.0.1:8090
export SPEECH_TIMEOUT_MS="${SPEECH_TIMEOUT_MS:-90000}"
[[ "$SPEECH_TIMEOUT_MS" =~ ^[0-9]+$ ]] || die "SPEECH_TIMEOUT_MS must be an integer (ms), got '$SPEECH_TIMEOUT_MS'"
(( SPEECH_TIMEOUT_MS > 0 && SPEECH_TIMEOUT_MS <= 95000 )) || die "SPEECH_TIMEOUT_MS=$SPEECH_TIMEOUT_MS must be in (0, 95000]: the RunPod proxy cuts at 100 s"
export SPEECH_DEVICE=cuda SPEECH_AUTH=on SPEECH_FAIL_FAST=true
export HF_HOME="$W/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export C0_INFERENCE_CONFIG="${C0_INFERENCE_CONFIG:-$W/config/c0_inference.yaml}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/bahnar-s2tt-thesis}"
# Tuning defaults (same values as the local stack); overridable from the template.
# AI drafting runs as background jobs (not behind the 100 s proxy), so it keeps
# the local values; only the legacy synchronous /ai-drafts call can hit 524.
: "${AI_EXTRACT_TIMEOUT_MS:=3000}" "${AI_RETRIEVE_TIMEOUT_MS:=10000}" "${AI_DRAFT_TIMEOUT_MS:=260000}"
: "${LLM_DRAFT_TIMEOUT_SECONDS:=240}" "${INDEX_CLAIM_TTL_SECONDS:=120}" "${INDEX_TIMEOUT_MS:=120000}"
: "${INDEX_PIPELINE_TIMEOUT_SECONDS:=3600}" "${INDEX_PUBLISH_TIMEOUT_MS:=10000}" "${INDEX_HEARTBEAT_SECONDS:=15}"
: "${INDEX_MAX_CONCURRENT:=1}" "${INDEX_MAX_PDF_PAGES:=1000}" "${INDEX_MAX_OCR_PAGES:=1000}" "${EMBED_BATCH_SIZE:=16}"
: "${OCR_DPI:=200}" "${DOCUMENT_MAX_BYTES:=20971520}"
export AI_EXTRACT_TIMEOUT_MS AI_RETRIEVE_TIMEOUT_MS AI_DRAFT_TIMEOUT_MS LLM_DRAFT_TIMEOUT_SECONDS \
  INDEX_CLAIM_TTL_SECONDS INDEX_TIMEOUT_MS INDEX_PIPELINE_TIMEOUT_SECONDS INDEX_PUBLISH_TIMEOUT_MS \
  INDEX_HEARTBEAT_SECONDS INDEX_MAX_CONCURRENT INDEX_MAX_PDF_PAGES INDEX_MAX_OCR_PAGES EMBED_BATCH_SIZE \
  OCR_DPI DOCUMENT_MAX_BYTES

# ---- 2. models and Speech mode checks (before any init work: fail fast) ----------
if [[ "$EMBEDDING_PROVIDER" == onnx ]]; then
  [[ -f "$EMBEDDING_MODEL_DIR/manifest.json" ]] || die "embedding model missing: $EMBEDDING_MODEL_DIR/manifest.json"
  gosu cas test -r "$EMBEDDING_MODEL_DIR/model.onnx" || die "user cas cannot read $EMBEDDING_MODEL_DIR/model.onnx"
  # Validate with the ai-service's own rules (single source of truth): the variant
  # this CPU needs (int8 on x86 with AVX-512 VNNI, fp32 on arm64, refused otherwise)
  # and the manifest + SHA-256 of every model file. A model fetched for another CPU,
  # a partial or a corrupted model stops the boot here, with a clear message,
  # instead of crash-looping the ai-service later.
  emb_out="$(cd /opt/cas/ai && gosu cas /opt/cas/venv-ai/bin/python - "$EMBEDDING_MODEL_DIR" 2>&1 <<'PY'
import json, os, sys
from app.indexing.embed import load_manifest, select_embedding_variant
from app.indexing.errors import IndexFailure
d = sys.argv[1]
try:
    variant = select_embedding_variant()
except IndexFailure as exc:
    sys.exit(f"this CPU cannot run the ai-service embedding ({exc}); x86 needs avx512f/bw/dq/vl + avx512vnni, arm64 uses fp32")
try:
    load_manifest(d, variant)
except IndexFailure as exc:
    try:
        have = json.load(open(os.path.join(d, "manifest.json"))).get("quantization_method")
    except Exception:
        have = "unreadable"
    sys.exit(f"{exc}: this CPU needs the '{variant}' model, the installed manifest is '{have}'"
             " (fetch it on this host: deploy/runpod/fetch_models.sh --variant auto)")
print(variant)
PY
)" || die "embedding model in $EMBEDDING_MODEL_DIR is not usable by the ai-service on this host: $(tail -1 <<<"$emb_out")"
  log "embedding model verified by the ai-service rules (variant $(tail -1 <<<"$emb_out"), SHA-256 of every file)"
fi
if [[ "$OCR_PROVIDER" == paddle ]]; then
  [[ -f "$OCR_MODEL_DIR/manifest.json" ]] || die "OCR model missing: $OCR_MODEL_DIR/manifest.json"
fi

( cd /opt/cas/c0 && sha256sum --strict --quiet -c c0_code.sha256 ) || die "Speech C0 code does not match c0_code.sha256"
log "speech C0 code verified at revision $(cat /opt/cas/c0/SOURCE_REVISION)"
if [[ "${CAS_C0_IMPORT_CHECK:-true}" == true ]]; then
  ( cd /opt/cas/c0 && gosu speech env PYTHONPATH=/opt/cas/c0 /opt/cas/venv-c0/bin/python -c "import src.c0_speech.app" ) \
    || die "Speech C0 code does not import"
  log "speech C0 import check ok"
fi

if [[ "$MODE" == gpu ]]; then
  [[ -f "$C0_INFERENCE_CONFIG" ]] || die "gpu mode: C0_INFERENCE_CONFIG not found: $C0_INFERENCE_CONFIG"
  M="$W/manifests/c0_checkpoints.sha256"
  [[ -s "$M" ]] || die "gpu mode: checkpoint manifest missing or empty: $M"
  log "gpu mode: verifying checkpoint manifest ($(wc -l < "$M") files, may take a while)"
  ( cd / && sha256sum --strict --quiet -c "$M" ) || die "gpu mode: checkpoint files do not match $M"
  # Verified as root above; the Speech process runs as user speech and must be able
  # to read every checkpoint file and the offline tokenizer cache.
  while read -r _ path; do
    path="${path#\*}"   # sha256sum binary-mode marker
    [[ -n "$path" ]] || continue
    gosu speech test -r "/${path#/}" || die "gpu mode: user speech cannot read /${path#/}"
  done < "$M"
  [[ -d "$HF_HOME" ]] || die "gpu mode: tokenizer cache missing: $HF_HOME (offline mode)"
  gosu speech test -r "$HF_HOME" -a -x "$HF_HOME" || die "gpu mode: user speech cannot read $HF_HOME"
  command -v nvidia-smi >/dev/null || die "gpu mode: nvidia-smi not available (no GPU runtime)"
  nvidia-smi -L >/dev/null || die "gpu mode: no GPU visible"
  gosu speech test -r "$C0_INFERENCE_CONFIG" || die "gpu mode: user speech cannot read $C0_INFERENCE_CONFIG"
  log "gpu mode checks passed: $(nvidia-smi -L | head -1)"
else
  log "speech disabled (mode=off): checkpoints and GPU not checked, program speech not started"
fi

# ---- 3. single-writer lease on the workspace (pgdata, qdrant, minio) -------------
# Two writers on one volume would corrupt Postgres/Qdrant/MinIO data. Layers:
#   a. flock on locks/workspace.flock, held by supervisord (inherited fd 9) for the
#      container's lifetime: blocks a second container on the same host/kernel.
#   b. lease dir locks/workspace.lease created with mkdir (atomic): owner.json +
#      heartbeat (pg-lease program, every CAS_LEASE_HEARTBEAT_S). Works even where
#      flock is not shared between machines (network volume: UNVERIFIED on RunPod).
# A lease held by another owner is never broken automatically: the container
# enters maintenance mode (no service starts) until an operator runs the
# verified procedure `cas-pg-recover <old owner>`. The same owner (container
# restarted, or the same RunPod pod id) reclaims its own lease.
LOCKS="$W/locks"; LEASE="$LOCKS/workspace.lease"
export CAS_LEASE_HEARTBEAT_S="${CAS_LEASE_HEARTBEAT_S:-10}" CAS_LEASE_STALE_S="${CAS_LEASE_STALE_S:-60}"
[[ "$CAS_LEASE_HEARTBEAT_S" =~ ^[0-9]+$ ]] && (( CAS_LEASE_HEARTBEAT_S >= 2 && CAS_LEASE_HEARTBEAT_S <= 60 )) \
  || die "CAS_LEASE_HEARTBEAT_S must be an integer in [2, 60]"
[[ "$CAS_LEASE_STALE_S" =~ ^[0-9]+$ ]] && (( CAS_LEASE_STALE_S > 2 * CAS_LEASE_HEARTBEAT_S )) \
  || die "CAS_LEASE_STALE_S must be an integer greater than 2 x CAS_LEASE_HEARTBEAT_S"
install -d -o root -g root -m 0755 "$W" "$LOCKS"
# Owner identity: the RunPod pod id, else the Docker container id (stable across
# restarts of the same container, different for every new container even with
# the same --hostname), else the hostname.
cid="$(grep -oE 'containers/[0-9a-f]{64}' /proc/self/mountinfo 2>/dev/null | head -1 | cut -d/ -f2 || true)"
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then OWNER="runpod-$RUNPOD_POD_ID"
elif [[ -n "$cid" ]]; then OWNER="docker-${cid:0:16}"
else OWNER="host-$(cat /proc/sys/kernel/hostname)"; fi
printf '%s' "$OWNER" > /run/cas-lease.owner; chmod 0644 /run/cas-lease.owner
lease_owner() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("owner_id",""))' "$LEASE/owner.json" 2>/dev/null; }
hb_age() { local hb; hb="$(cat "$LEASE/heartbeat" 2>/dev/null || echo 0)"; echo $(( $(date +%s) - ${hb:-0} )); }
maintenance() {
  log "MAINTENANCE: $*"
  log "MAINTENANCE: no service started. Verify the other owner is stopped, then run as root in this container:"
  log "MAINTENANCE:   cas-pg-recover <old owner id>      (see deploy/runpod/README.md)"
  printf '%s\n' "$*" > /run/cas-maintenance
  exec 9>&-
  trap - ERR
  exec sleep infinity
}
rm -f /run/cas-maintenance
exec 9>>"$LOCKS/workspace.flock"
flock -n 9 || maintenance "workspace flock is held by another process on this host (another container uses this volume)"
if [[ -d "$LEASE" ]]; then
  prev="$(lease_owner)"
  [[ -n "$prev" && "$prev" == "$OWNER" ]] \
    || maintenance "workspace lease is held by owner '${prev:-<unreadable>}' (last heartbeat $(hb_age)s ago); this container is '$OWNER'"
  LEASE_STATE=reclaimed
  log "workspace lease reclaimed by the same owner ($OWNER) after a restart"
else
  # A postmaster.pid without any lease means an unknown writer (or a pre-lease
  # image) may own pgdata: do not guess, require the verified procedure.
  [[ ! -f "$W/pgdata/postmaster.pid" ]] \
    || maintenance "pgdata/postmaster.pid exists but no workspace lease: cannot prove the previous Postgres stopped"
  mkdir "$LEASE" 2>/dev/null || maintenance "another container created the workspace lease first"
  LEASE_STATE=new
fi
# heartbeat_s is recorded so that cas-pg-recover, run elsewhere, waits long enough
# for THIS owner's interval (not for its own default).
python3 - "$LEASE/owner.json" "$OWNER" "$CAS_LEASE_HEARTBEAT_S" <<'PY'
import json, os, socket, sys, time
path, owner, hb = sys.argv[1], sys.argv[2], int(sys.argv[3])
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"owner_id": owner, "hostname": socket.gethostname(), "acquired_at": int(time.time()), "heartbeat_s": hb}, f)
os.replace(tmp, path)
PY
date +%s > "$LEASE/heartbeat.tmp" && mv -f "$LEASE/heartbeat.tmp" "$LEASE/heartbeat"
chmod 0644 "$LEASE/owner.json" "$LEASE/heartbeat"
log "workspace lease $LEASE_STATE: owner=$OWNER"

# ---- 4. directories and ownership (only after the lease is held) ----------------------------------------------
dir() { install -d -o "$2" -g "$2" -m "$3" "$1"; }
own_tree() {  # chown -R only when something is owned by another user (fast on reboot)
  if [[ -n "$(find "$1" ! -user "$2" -print -quit 2>/dev/null)" ]]; then
    log "fixing ownership of $1 -> $2"
    chown -R "$2:$2" "$1"
  fi
}
dir "$W" root 0755
dir "$W/pgdata" postgres 0700;  own_tree "$W/pgdata" postgres
dir "$W/qdrant" qdrant 0700;    own_tree "$W/qdrant" qdrant
dir "$W/minio" minio 0700;      own_tree "$W/minio" minio
dir "$W/backups" cas 0700;      own_tree "$W/backups" cas
dir "$W/tls" root 0711
dir "$W/tls/ca" root 0700
dir "$W/tls/postgres" postgres 0700
dir "$W/tls/minio" minio 0700
dir "$W/config" root 0755
dir "$W/manifests" root 0755
dir /run/postgresql postgres 2775
dir /run/nginx web 0700
dir /run/cas-log root 0755

# ---- 5. internal CA and leaf certificates ---------------------------------------
CA_KEY="$W/tls/ca/ca.key"; CA_CRT="$W/tls/ca.crt"
same_pubkey() { [[ "$(openssl pkey -in "$1" -pubout 2>/dev/null)" == "$(openssl x509 -in "$2" -noout -pubkey 2>/dev/null)" ]]; }

if [[ -f "$CA_KEY" && -f "$CA_CRT" ]]; then
  same_pubkey "$CA_KEY" "$CA_CRT" || die "internal CA key and certificate do not match; not regenerating"
  openssl x509 -in "$CA_CRT" -noout -checkend 0 >/dev/null || die "internal CA certificate expired; rotate it manually"
  openssl x509 -in "$CA_CRT" -noout -checkend 7776000 >/dev/null || log "WARNING: internal CA expires within 90 days"
  log "internal CA reused ($(openssl x509 -in "$CA_CRT" -noout -fingerprint -sha256 | cut -d= -f2))"
elif [[ -e "$CA_KEY" || -e "$CA_CRT" ]]; then
  die "only one of $CA_KEY / $CA_CRT exists; refusing to overwrite"
else
  ( umask 077; openssl ecparam -name prime256v1 -genkey -noout -out "$CA_KEY" )
  openssl req -x509 -new -key "$CA_KEY" -sha256 -days 3650 -subj "/CN=CAS internal CA" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" -out "$CA_CRT"
  chmod 0600 "$CA_KEY"; chmod 0644 "$CA_CRT"
  log "internal CA created ($(openssl x509 -in "$CA_CRT" -noout -fingerprint -sha256 | cut -d= -f2))"
fi

leaf() {  # leaf <dir> <key name> <cert name> <owner>
  local d="$1" key="$1/$2" crt="$1/$3" owner="$4"
  if [[ -f "$key" && -f "$crt" ]]; then
    same_pubkey "$key" "$crt" || die "$crt does not match $key; not regenerating"
    openssl verify -CAfile "$CA_CRT" "$crt" >/dev/null || die "$crt is not signed by the current internal CA"
    openssl x509 -in "$crt" -noout -ext subjectAltName | grep -q "IP Address:127.0.0.1" || die "$crt lacks SAN IP:127.0.0.1"
    openssl x509 -in "$crt" -noout -checkend 0 >/dev/null || die "$crt expired; rotate it manually"
    openssl x509 -in "$crt" -noout -checkend 2592000 >/dev/null \
      || log "WARNING: $crt expires within 30 days ($(openssl x509 -in "$crt" -noout -enddate | cut -d= -f2)); plan a rotation"
    log "certificate reused: $crt"
  elif [[ -e "$key" || -e "$crt" ]]; then
    die "only one of $key / $crt exists; refusing to overwrite"
  else
    ( umask 077; openssl ecparam -name prime256v1 -genkey -noout -out "$key" )
    openssl req -new -key "$key" -subj "/CN=127.0.0.1" -out "$d/req.csr"
    printf 'subjectAltName=IP:127.0.0.1,DNS:localhost\nextendedKeyUsage=serverAuth\nkeyUsage=critical,digitalSignature\n' > "$d/ext.cnf"
    openssl x509 -req -in "$d/req.csr" -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
      -CAserial "$W/tls/ca/ca.srl" -days 825 -sha256 -extfile "$d/ext.cnf" -out "$crt"
    rm -f "$d/req.csr" "$d/ext.cnf"
    log "certificate issued: $crt"
  fi
  chown "$owner:$owner" "$key" "$crt"; chmod 0600 "$key"; chmod 0644 "$crt"
}
leaf "$W/tls/postgres" server.key server.crt postgres
leaf "$W/tls/minio" private.key public.crt minio

cp "$CA_CRT" /usr/local/share/ca-certificates/cas-internal-ca.crt
update-ca-certificates >/dev/null 2>&1 || die "update-ca-certificates failed"

# ---- 6. Postgres ---------------------------------------------------------------
PGBIN=/usr/lib/postgresql/16/bin
PGDATA="$W/pgdata"
as_pg() { gosu postgres "$@"; }
if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
  [[ -z "$(ls -A "$PGDATA")" ]] || die "$PGDATA is not empty but has no PG_VERSION; refusing to initdb over it"
  log "initdb (first start on this volume)"
  as_pg "$PGBIN/initdb" -D "$PGDATA" --username=postgres --auth-local=peer --auth-host=reject \
    --encoding=UTF8 --locale=C.UTF-8 >/dev/null
  echo "include_if_exists = '/etc/cas/postgresql.cas.conf'" >> "$PGDATA/postgresql.conf"
  install -o postgres -g postgres -m 0600 /etc/cas/pg_hba.conf "$PGDATA/pg_hba.conf"
  # Temporary server: unix socket only, no TCP. Creates the app role, DB and the
  # extensions the migrations need (CREATE EXTENSION requires superuser).
  as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -w -t 60 -o "-c listen_addresses='' -c ssl=off" start >/dev/null
  as_pg psql -v ON_ERROR_STOP=1 -h /run/postgresql -d postgres -q \
    -v pw="$POSTGRES_PASSWORD" <<'SQL'
CREATE ROLE cas LOGIN PASSWORD :'pw';
CREATE DATABASE citizen_assistance OWNER cas ENCODING 'UTF8';
\connect citizen_assistance
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
SQL
  as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -w -t 60 -m fast stop >/dev/null
  # Written last: a first init interrupted after initdb (role/db/extensions
  # missing) is detected on the next start instead of failing later in bootstrap.
  as_pg touch "$PGDATA/.cas-init-complete"
  log "postgres initialised (role cas, database citizen_assistance)"
else
  [[ "$(cat "$PGDATA/PG_VERSION")" == 16 ]] || die "pgdata is PostgreSQL $(cat "$PGDATA/PG_VERSION"), image runs 16"
  [[ -f "$PGDATA/.cas-init-complete" ]] \
    || die "pgdata exists but the first initialisation did not complete (no .cas-init-complete); inspect it, then move $PGDATA aside to re-initialise"
  grep -q "/etc/cas/postgresql.cas.conf" "$PGDATA/postgresql.conf" \
    || echo "include_if_exists = '/etc/cas/postgresql.cas.conf'" >> "$PGDATA/postgresql.conf"
  install -o postgres -g postgres -m 0600 /etc/cas/pg_hba.conf "$PGDATA/pg_hba.conf"
  # Only when this owner reclaimed its own lease: the previous Postgres was this
  # same container/pod, which cannot run twice, so its postmaster.pid is stale and
  # its PID may collide with a process of this boot. Any other case reached
  # maintenance mode above.
  if [[ -f "$PGDATA/postmaster.pid" ]]; then
    [[ "$LEASE_STATE" == reclaimed ]] || die "postmaster.pid present on a new lease (should be unreachable)"
    log "removing stale postmaster.pid of this owner's previous boot"
    rm -f "$PGDATA/postmaster.pid"
  fi
fi

# ---- 7. boot state and per-service environments ------------------------------------
# One allowlist per program. Only variables that are set are written. Values with
# a newline are refused (one KEY=VALUE per line).
envfile() {
  local svc="$1" f v; shift
  f="/run/cas-env/$svc.env"
  ( umask 077; : > "$f" )
  for v in "$@"; do
    [[ -n "${!v+x}" ]] || continue
    [[ "${!v}" != *$'\n'* ]] || die "$v contains a newline"
    printf '%s=%s\n' "$v" "${!v}" >> "$f"
  done
}
install -d -o root -g root -m 0700 /run/cas-env
rm -f /run/cas-env/*.env
LEASE_VARS=(CAS_WORKSPACE CAS_LEASE_HEARTBEAT_S CAS_LEASE_STALE_S)
TLS_VARS=(SSL_CERT_FILE CAS_CA_CERT)
OBJ_VARS=(OBJECT_STORAGE_ENDPOINT OBJECT_STORAGE_ACCESS_KEY OBJECT_STORAGE_SECRET_KEY OBJECT_STORAGE_BUCKET
          OBJECT_STORAGE_REGION OBJECT_STORAGE_USE_SSL OBJECT_STORAGE_AUTO_CREATE_BUCKET)
envfile pg-lease "${LEASE_VARS[@]}"
envfile postgres "${LEASE_VARS[@]}"
envfile qdrant "${LEASE_VARS[@]}" QDRANT__SERVICE__HOST QDRANT__STORAGE__STORAGE_PATH QDRANT__STORAGE__SNAPSHOTS_PATH \
  QDRANT__TELEMETRY_DISABLED QDRANT_INIT_FILE_PATH
envfile minio "${LEASE_VARS[@]}" MINIO_ROOT_USER MINIO_ROOT_PASSWORD MINIO_BROWSER
envfile bootstrap CAS_WORKSPACE CAS_BOOTSTRAP_TIMEOUT_S DATABASE_URL "${TLS_VARS[@]}" "${OBJ_VARS[@]}" \
  QDRANT_URL QDRANT_COLLECTION EMBEDDING_PROVIDER EMBEDDING_MODEL_DIR
envfile ai-service APP_ENV AI_SERVICE_TOKEN DATABASE_URL "${TLS_VARS[@]}" "${OBJ_VARS[@]}" QDRANT_URL QDRANT_COLLECTION \
  LLM_PROVIDER LLM_MODEL LLM_MODEL_EXTRACTION LLM_MODEL_DEFINITION LLM_MAX_OUTPUT_TOKENS LLM_DRAFT_MAX_OUTPUT_TOKENS \
  LLM_REQUEST_TIMEOUT_S LLM_TIMEOUT_SECONDS LLM_DRAFT_TIMEOUT_SECONDS GEMINI_API_KEY GEMINI_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL \
  INDEX_MODE INDEX_MAX_INFLIGHT INDEX_MAX_PDF_PAGES INDEX_MAX_OCR_PAGES INDEX_MAX_REJECTED_PAGES INDEX_PIPELINE_TIMEOUT_SECONDS \
  INDEX_TERMINATION_GRACE_SECONDS EMBEDDING_PROVIDER EMBEDDING_MODEL_DIR EMBEDDING_CPU_FLAGS EMBED_BATCH_SIZE \
  OCR_PROVIDER OCR_MODEL_DIR OCR_DPI OCR_MIN_MEAN_CONFIDENCE OCR_MAX_LOW_CONFIDENCE_RATIO OCR_PAGE_TIMEOUT_SECONDS \
  EXTRACT_RATE_PER_MINUTE EXTRACT_MAX_INFLIGHT EXTRACT_RETRY_AFTER_SECONDS
envfile api APP_ENV CONFIG_FILE API_ADDR XA_ID CITIZEN_DOMAIN_IDS LOG_LEVEL DATABASE_URL DB_MAX_CONNS DB_MIN_CONNS \
  DB_CONN_TIMEOUT_SEC SHUTDOWN_TIMEOUT_SEC JWT_SECRET JWT_EXPIRE_HOURS "${TLS_VARS[@]}" "${OBJ_VARS[@]}" DOCUMENT_MAX_BYTES \
  AI_SERVICE_URL AI_SERVICE_TOKEN AI_EXTRACT_ENABLED AI_EXTRACT_TIMEOUT_MS AI_RETRIEVE_ENABLED AI_RETRIEVE_TIMEOUT_MS \
  AI_DRAFT_TIMEOUT_MS AI_INTENT_SELECT_MIN AI_INTENT_CONFIRM_MIN AI_SLOT_CONFIDENCE_MIN \
  ADMIN_INGESTION_ENABLED ADMIN_INDEXING_ENABLED INDEX_MODE INDEX_ASYNC INDEX_CLAIM_TTL_SECONDS INDEX_TIMEOUT_MS \
  INDEX_PIPELINE_TIMEOUT_SECONDS INDEX_PUBLISH_TIMEOUT_MS INDEX_HEARTBEAT_SECONDS INDEX_MAX_CONCURRENT \
  SPEECH_ENABLED SPEECH_SERVICE_URL SPEECH_SERVICE_TOKEN SPEECH_TIMEOUT_MS SPEECH_MAX_UPLOAD_BYTES SPEECH_MAX_INFLIGHT
envfile index-reaper DATABASE_URL "${TLS_VARS[@]}" QDRANT_URL QDRANT_COLLECTION INDEX_REAPER_INTERVAL_SECONDS \
  INDEX_ORPHAN_GRACE_SECONDS INDEX_CLEANUP_BATCH_SIZE INDEX_REAPER_SHUTDOWN_SECONDS
# shellcheck disable=SC2034  # read indirectly by envfile (${!v})
SPEECH_HOST=127.0.0.1 SPEECH_PORT=8090 SPEECH_WORKERS=1 PYTHONPATH=/opt/cas/c0
envfile speech PYTHONPATH SPEECH_HOST SPEECH_PORT SPEECH_WORKERS SPEECH_SERVICE_TOKEN SPEECH_DEVICE SPEECH_AUTH \
  SPEECH_FAIL_FAST SPEECH_MAX_INFLIGHT SPEECH_MAX_UPLOAD_BYTES SPEECH_MAX_AUDIO_SECONDS SPEECH_REQUEST_TIMEOUT_S \
  SPEECH_SHUTDOWN_TIMEOUT_S SPEECH_LOG_LEVEL C0_INFERENCE_CONFIG ARTIFACT_ROOT HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
envfile nginx
log "per-service environments written: $(cd /run/cas-env && ls | sed 's/\.env$//' | tr '\n' ' ')"

dir /run/cas cas 0755
rm -f /run/cas/bootstrap.ok /run/cas/.bootstrap.*
python3 -c 'import uuid;print(uuid.uuid4())' > /run/cas/boot_id
chmod 0644 /run/cas/boot_id
# Operator tools started with docker exec do not inherit these exports; cas-run
# sources this root-only file (it contains secrets).
( umask 077; export -p > /run/cas-runtime.env )
log "boot $(cat /run/cas/boot_id): handing over to supervisord"
trap - ERR
exec /usr/bin/supervisord -c /etc/cas/supervisord.conf
