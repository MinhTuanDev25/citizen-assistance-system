#!/usr/bin/env bash
# Restore drill: prove the newest backups can be restored, without touching live data.
#   Postgres: restore the newest pg_dump into a scratch DB, compare the row count of
#             every public table with the live DB.
#   Qdrant:   restore the newest snapshot into a scratch collection, compare the
#             point count with the live collection.
# Scratch names are unique per run (cas_drill_<utc>_<random>); creation fails if
# the name exists, and cleanup removes ONLY what this run created. Nothing else is
# dropped, ever. Cleanup is verified (the DB and collection must be gone); anything
# left behind makes the drill FAIL. The last lines are always:
#   SCRATCH db=<name> collection=<name> created_db=0|1 created_collection=0|1
#   RESULT FULL|PARTIAL|FAIL ...
# Full drill (default): Postgres AND Qdrant must both restore and match; a missing
# Qdrant snapshot is a FAIL (exit 1), not a skip.
#   --postgres-only   explicit operator choice: Qdrant is not tested; the result line
#                     says PARTIAL (exit 0) so it can never be read as a full pass.
# Run as root right after backup.sh: docker exec <c> restore_drill.sh [--postgres-only]
set -Eeuo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
PG_ONLY=0
case "${1:-}" in
  "") ;;
  --postgres-only) PG_ONLY=1 ;;
  *) echo "usage: restore_drill.sh [--postgres-only]" >&2; exit 2 ;;
esac
COLL="${QDRANT_COLLECTION:-knowledge_chunks}"
TAG="$(date -u +%Y%m%d%H%M%S)_$(od -An -N4 -tx4 /dev/urandom | tr -d ' ')"
DRILL_DB="cas_drill_$TAG"
DRILL_COLL="cas_drill_${TAG}_$COLL"
created_db=0; created_coll=0
log() { echo "[restore-drill] $(date -u +%FT%TZ) $*" >&2; }
die() { log "FAIL: $*"; exit 1; }
su_psql() { gosu postgres psql -h /run/postgresql -v ON_ERROR_STOP=1 -qtA "$@"; }
qcount() {
  curl -fsS -X POST "http://127.0.0.1:6333/collections/$1/points/count" -H 'Content-Type: application/json' -d '{"exact":true}' \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["count"])'
}

result=""   # set by the main flow on success; printed only after a verified cleanup

# Runs on every exit. Removes what this run created, then VERIFIES it is gone. A
# scratch database or collection left behind turns the whole drill into a FAIL
# (exit 1), whatever the restore result was.
finish() {
  local rc=$? left=""
  if [[ $created_db == 1 ]]; then
    su_psql -d postgres -c "DROP DATABASE \"$DRILL_DB\"" >/dev/null 2>&1 || log "drop of scratch database $DRILL_DB failed"
    if [[ "$(su_psql -d postgres -c "select count(*) from pg_database where datname = '$DRILL_DB'" 2>/dev/null)" == 0 ]]; then
      log "removed scratch database $DRILL_DB"
    else left+=" database=$DRILL_DB"; fi
  fi
  if [[ $created_coll == 1 ]]; then
    curl -fsS -X DELETE "http://127.0.0.1:6333/collections/$DRILL_COLL" >/dev/null 2>&1 || log "delete of scratch collection $DRILL_COLL failed"
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:6333/collections/$DRILL_COLL")" == 404 ]]; then
      log "removed scratch collection $DRILL_COLL"
    else left+=" collection=$DRILL_COLL"; fi
  fi
  echo "SCRATCH db=$DRILL_DB collection=$DRILL_COLL created_db=$created_db created_collection=$created_coll"
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

dump="$(ls -1t "$W"/backups/pg/*.dump 2>/dev/null | head -1 || true)"
[[ -n "$dump" ]] || die "no pg_dump archive in $W/backups/pg (run backup.sh first)"

# ---- Postgres ----------------------------------------------------------------------
# Plain CREATE DATABASE (no IF NOT EXISTS): an existing name is an error, never reused.
su_psql -d postgres -c "CREATE DATABASE \"$DRILL_DB\" ENCODING 'UTF8'" || die "could not create scratch database $DRILL_DB"
created_db=1
log "postgres: restoring $(basename "$dump") into $DRILL_DB"
# Restored as superuser over the local socket (the archive has CREATE/COMMENT ON
# EXTENSION). backups/ is cas-only (0700): root opens the archive, stdin feeds it.
gosu postgres pg_restore -h /run/postgresql --no-owner --exit-on-error -d "$DRILL_DB" < "$dump" || die "pg_restore failed"

counts() {  # every public table: name|rows, sorted
  su_psql -d "$1" -F'|' -c "
    SELECT format('%s|%s', c.relname,
           (xpath('/row/n/text()', query_to_xml(format('select count(*) as n from %I.%I', n.nspname, c.relname), false, true, '')))[1]::text)
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind = 'r' AND n.nspname = 'public' ORDER BY c.relname"
}
live="$(counts citizen_assistance)"; restored="$(counts "$DRILL_DB")"
tables="$(grep -c . <<<"$live")"
(( tables > 0 )) || die "live database has no public tables"
if [[ "$live" != "$restored" ]]; then
  diff <(echo "$live") <(echo "$restored") >&2 || true
  die "row counts differ between live and restored DB"
fi
mig="$(su_psql -d "$DRILL_DB" -c "select version || ' dirty=' || dirty from schema_migrations")"
log "postgres: OK, $tables tables with identical row counts; restored schema_migrations=$mig"

# ---- Qdrant ------------------------------------------------------------------------
if [[ $PG_ONLY == 1 ]]; then
  log "qdrant: NOT tested (--postgres-only requested by the operator)"
  result="RESULT PARTIAL postgres=OK tables=$tables qdrant=NOT_TESTED"; exit 0
fi
snap="$(ls -1t "$W"/backups/qdrant/*.snapshot 2>/dev/null | head -1 || true)"
[[ -n "$snap" ]] || die "no Qdrant snapshot in $W/backups/qdrant (run backup.sh first, or use --postgres-only to test Postgres alone)"
if curl -fsS "http://127.0.0.1:6333/collections/$DRILL_COLL" >/dev/null 2>&1; then
  die "scratch collection $DRILL_COLL already exists; refusing to touch it"
fi
live_pts="$(qcount "$COLL")" || die "cannot count points of $COLL"
# The name was checked absent just above, so anything under it from here on was
# created by this run (also a half-created collection if the upload fails).
created_coll=1
curl -fsS -X POST "http://127.0.0.1:6333/collections/$DRILL_COLL/snapshots/upload?priority=snapshot" \
  -F "snapshot=@$snap" >/dev/null || die "qdrant snapshot upload failed"
drill_pts="$(qcount "$DRILL_COLL")" || die "cannot count points of $DRILL_COLL"
[[ "$live_pts" == "$drill_pts" ]] || die "qdrant point count differs: live=$live_pts restored=$drill_pts"
log "qdrant: OK, $(basename "$snap") restored with $drill_pts points (live $live_pts)"
result="RESULT FULL postgres=OK tables=$tables qdrant=OK points=$drill_pts"
exit 0
