#!/usr/bin/env bash
# Restore drill: prove that what is on the volume restores, without touching live data.
#   Postgres: the newest verified base backup + the WAL on the volume are recovered into a
#             scratch cluster on the container disk (cas-pg-restore.sh, same checks as a
#             real restore: pg_verifybackup, cluster identity, WAL continuity, end of
#             recovery). Every public table must have the same row count and the same
#             content hash as the live database.
#   Qdrant:   the newest snapshot is uploaded into a scratch collection; its fingerprint
#             (ids, payloads, float32 vectors per generation) must equal the one recorded
#             with the snapshot.
#   Objects:  every document's file exists with its recorded size and SHA-256.
# Scratch names are unique per run (cas_drill_<utc>_<random>); cleanup removes ONLY what
# this run created and is verified; anything left behind makes the drill FAIL. The last
# lines are always:
#   SCRATCH dir=<path> collection=<name> created_dir=0|1 created_collection=0|1
#   RESULT FULL|PARTIAL|FAIL ...
#   --postgres-only   explicit operator choice: Qdrant and objects not tested (PARTIAL).
# Run as root when nothing writes (live and restored are compared at the same moment):
#   docker exec <c> restore_drill.sh [--postgres-only]
set -Eeuo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
L="${CAS_LOCAL:-/var/lib/cas}"
PG_ONLY=0
case "${1:-}" in
  "") ;;
  --postgres-only) PG_ONLY=1 ;;
  *) echo "usage: restore_drill.sh [--postgres-only]" >&2; exit 2 ;;
esac
[[ $EUID -eq 0 ]] || { echo "restore_drill.sh: run as root" >&2; exit 1; }
# Keep the caller's PATH (the boot environment has its own).
caller_path="$PATH"
# shellcheck disable=SC1091
source /run/cas-runtime.env
PATH="$caller_path"
COLL="${QDRANT_COLLECTION:-knowledge_chunks}"
TAG="$(date -u +%Y%m%d%H%M%S)_$(od -An -N4 -tx4 /dev/urandom | tr -d ' ')"
DRILL_DIR="$L/drill/cas_drill_$TAG"
DRILL_SOCK="/run/cas_drill_$TAG"
DRILL_PORT=5544
DRILL_COLL="cas_drill_${TAG}_$COLL"
created_dir=0; created_coll=0
log() { echo "[restore-drill] $(date -u +%FT%TZ) $*" >&2; }
die() { log "FAIL: $*"; exit 1; }
live_sql() { gosu postgres psql -h /run/postgresql -d citizen_assistance -XAtq -v ON_ERROR_STOP=1 "$@"; }
drill_sql() { gosu postgres psql -h "$DRILL_SOCK" -p "$DRILL_PORT" -d citizen_assistance -XAtq -v ON_ERROR_STOP=1 "$@"; }

result=""
finish() {
  local rc=$? left=""
  if [[ $created_dir == 1 ]]; then
    gosu postgres /usr/lib/postgresql/16/bin/pg_ctl -D "$DRILL_DIR" -m immediate -w -t 60 stop >/dev/null 2>&1 || true
    rm -rf "$DRILL_DIR" "$DRILL_DIR.log" "$DRILL_DIR.restore" "$DRILL_DIR.wal-tail" "$DRILL_SOCK" || log "removing $DRILL_DIR failed"
    if [[ -e "$DRILL_DIR" || -e "$DRILL_DIR.wal-tail" || -e "$DRILL_SOCK" ]] || pgrep -f -- "-D $DRILL_DIR" >/dev/null; then left+=" dir=$DRILL_DIR"
    else log "removed scratch cluster $DRILL_DIR"; fi
  fi
  if [[ $created_coll == 1 ]]; then
    curl -fsS -X DELETE "http://127.0.0.1:6333/collections/$DRILL_COLL" >/dev/null 2>&1 || log "delete of scratch collection $DRILL_COLL failed"
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:6333/collections/$DRILL_COLL")" == 404 ]]; then
      log "removed scratch collection $DRILL_COLL"
    else left+=" collection=$DRILL_COLL"; fi
  fi
  echo "SCRATCH dir=$DRILL_DIR collection=$DRILL_COLL created_dir=$created_dir created_collection=$created_coll"
  if [[ -n "$left" ]]; then
    log "FAIL: scratch resources left behind:$left (remove them by hand)"
    echo "RESULT FAIL cleanup_left:${left}"
    exit 1
  fi
  if [[ $rc -eq 0 && -n "$result" ]]; then echo "$result"; exit 0; fi
  echo "RESULT FAIL exit=$rc"
  exit $(( rc == 0 ? 1 : rc ))
}
trap finish EXIT

# ---- Postgres: base backup + WAL into a scratch cluster ----------------------------------
[[ ! -e "$DRILL_DIR" ]] || die "scratch dir $DRILL_DIR already exists"
install -d -o root -g root -m 0755 "$L/drill"
created_dir=1
out="$(CAS_RESTORE_TARGET="$DRILL_DIR" CAS_RESTORE_SOCKET="$DRILL_SOCK" CAS_RESTORE_PORT="$DRILL_PORT" CAS_RESTORE_DRILL=1 \
       /opt/cas/bin/cas-pg-restore.sh)" || die "restore of the volume into $DRILL_DIR failed (see cas-pg-restore lines)"
restored="$(grep '^RESTORED ' <<<"$out")"
install -o postgres -g postgres -m 0600 /dev/null "$DRILL_DIR.log"
gosu postgres /usr/lib/postgresql/16/bin/pg_ctl -D "$DRILL_DIR" -w -t 300 -l "$DRILL_DIR.log" \
  -o "-c listen_addresses='' -c port=$DRILL_PORT -c unix_socket_directories=$DRILL_SOCK -c ssl=off -c synchronous_standby_names='' -c max_wal_senders=0 -c wal_level=minimal -c max_replication_slots=0" start >/dev/null \
  || die "scratch cluster did not start"

# Per public table: row count and an order-independent content hash.
tables_sql="select format('%s|%s|%s', c.relname,
  (xpath('/row/n/text()', query_to_xml(format('select count(*) as n from %I.%I', n.nspname, c.relname), false, true, '')))[1]::text,
  (xpath('/row/h/text()', query_to_xml(format('select coalesce(md5(string_agg(md5(t::text), %L order by md5(t::text))), %L) as h from %I.%I t', '', 'empty', n.nspname, c.relname), false, true, '')))[1]::text)
  from pg_class c join pg_namespace n on n.oid = c.relnamespace
  where c.relkind = 'r' and n.nspname = 'public' order by c.relname"
live="$(live_sql -c "$tables_sql")"; got="$(drill_sql -c "$tables_sql")"
tables="$(grep -c . <<<"$live")"
(( tables > 0 )) || die "live database has no public tables"
if [[ "$live" != "$got" ]]; then
  diff <(echo "$live") <(echo "$got") >&2 || true
  die "row counts or content differ between live and restored (was something writing during the drill?)"
fi
mig="$(drill_sql -c "select version || ' dirty=' || dirty from schema_migrations")"
log "postgres: OK, $tables tables with identical counts and content; $restored; schema_migrations=$mig"

if [[ $PG_ONLY == 1 ]]; then
  log "qdrant and objects: NOT tested (--postgres-only requested by the operator)"
  result="RESULT PARTIAL postgres=OK tables=$tables qdrant=NOT_TESTED objects=NOT_TESTED"; exit 0
fi

# ---- Qdrant: newest snapshot into a scratch collection, fingerprint must match ------------
snap="$(ls -1 "$W/qdrant/snapshots/$COLL"/*.snapshot 2>/dev/null | sort | tail -1 || true)"
[[ -n "$snap" ]] || die "no Qdrant snapshot in $W/qdrant/snapshots/$COLL (run backup.sh first, or use --postgres-only)"
meta="${snap%.snapshot}.json"
[[ -f "$meta" ]] || die "snapshot $snap has no metadata/fingerprint file"
if curl -fsS "http://127.0.0.1:6333/collections/$DRILL_COLL" >/dev/null 2>&1; then
  die "scratch collection $DRILL_COLL already exists; refusing to touch it"
fi
want_sha="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["snapshot_sha256"])' "$meta")"
[[ "$(sha256sum "$snap" | cut -d' ' -f1)" == "$want_sha" ]] || die "snapshot $snap does not match its recorded SHA-256"
created_coll=1
curl -fsS -X POST "http://127.0.0.1:6333/collections/$DRILL_COLL/snapshots/upload?priority=snapshot&wait=true" \
  -F "snapshot=@$snap" >/dev/null || die "qdrant snapshot upload failed"
fp="$(QDRANT_COLLECTION="$DRILL_COLL" cas-storage qdrant-fingerprint)" || die "cannot fingerprint $DRILL_COLL"
FP="$fp" python3 - "$meta" <<'PY' || die "scratch collection fingerprint differs from the snapshot's recorded fingerprint"
import json, os, sys
want = json.load(open(sys.argv[1]))["fingerprint"]
got = json.loads(os.environ["FP"])
sys.exit(0 if got["generations"] == want["generations"] and got["points"] == want["points"] else 1)
PY
# Every READY generation the snapshot holds equals its baseline (written once, at its first snapshot).
FP="$fp" python3 - "$meta" "$W/qdrant/baselines/$COLL" <<'PY' || die "a READY generation in the snapshot differs from its baseline"
import json, os, sys
from pathlib import Path
ready = json.load(open(sys.argv[1]))["pg_ready"]
got = json.loads(os.environ["FP"])["generations"]
bad = []
for gid, n in sorted(ready.items()):
    if (got.get(gid) or {}).get("count") != n:
        continue   # retired while the snapshot was taken: not recorded as READY
    f = Path(sys.argv[2]) / f"{gid}.json"
    if not f.is_file():
        bad.append(f"{gid}: no baseline")
    elif json.loads(f.read_text())["fingerprint"] != got[gid]:
        bad.append(f"{gid}: differs from its baseline")
if bad:
    print("restore_drill: " + "; ".join(bad), file=sys.stderr)
sys.exit(1 if bad else 0)
PY
pts="$(python3 -c 'import json,sys;print(json.load(sys.stdin)["points"])' <<<"$fp")"
log "qdrant: OK, $(basename "$snap") restored into a scratch collection, fingerprint identical ($pts points), READY generations equal their baselines"

# ---- objects ------------------------------------------------------------------------------
gosu cas env DATABASE_URL="$DATABASE_URL" SSL_CERT_FILE="$SSL_CERT_FILE" OBJECT_STORAGE_FILE_ROOT="$OBJECT_STORAGE_FILE_ROOT" \
  OBJECT_STORAGE_BUCKET="$OBJECT_STORAGE_BUCKET" cas-storage objects-verify || die "object verification failed"
docs="$(live_sql -c 'select count(*) from documents')"
result="RESULT FULL postgres=OK tables=$tables qdrant=OK points=$pts objects=OK documents=$docs"
exit 0
