#!/usr/bin/env bash
# Backup on the volume: pg_dump (custom format, over TLS as role cas) and a Qdrant
# snapshot of the knowledge collection. Keeps the newest $CAS_BACKUP_KEEP of each.
# Run as root: docker exec <c> backup.sh   (PDF objects: mirror /workspace/cas/minio off-site)
# Off-site copy is NOT done here; copy /workspace/cas/backups out of RunPod.
set -Eeuo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
KEEP="${CAS_BACKUP_KEEP:-7}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
log() { echo "[backup] $(date -u +%FT%TZ) $*" >&2; }

# docker exec does not see variables exported by entrypoint.sh; rebuild the DSN.
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
pw="$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$POSTGRES_PASSWORD")"
DATABASE_URL="postgres://cas:${pw}@127.0.0.1:5432/citizen_assistance?sslmode=verify-full&sslrootcert=$W/tls/ca.crt"

install -d -o cas -g cas -m 0700 "$W/backups/pg" "$W/backups/qdrant"

dump="$W/backups/pg/citizen_assistance-$TS.dump"
gosu cas pg_dump -Fc -f "$dump" "$DATABASE_URL"
gosu cas pg_restore --list "$dump" > /dev/null   # readable archive, not just a file
log "postgres -> $dump ($(du -h "$dump" | cut -f1))"

coll="${QDRANT_COLLECTION:-knowledge_chunks}"
if curl -fsS "http://127.0.0.1:6333/collections/$coll" >/dev/null 2>&1; then
  name="$(curl -fsS -X POST "http://127.0.0.1:6333/collections/$coll/snapshots?wait=true" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["name"])')"
  src="$W/qdrant/snapshots/$coll/$name"
  install -o cas -g cas -m 0600 "$src" "$W/backups/qdrant/$coll-$TS.snapshot"
  curl -fsS -X DELETE "http://127.0.0.1:6333/collections/$coll/snapshots/$name" >/dev/null
  log "qdrant -> $W/backups/qdrant/$coll-$TS.snapshot"
else
  log "qdrant collection $coll not found; snapshot skipped"
fi

for d in pg qdrant; do
  ls -1t "$W/backups/$d" | tail -n +"$((KEEP + 1))" | while read -r old; do
    rm -f "$W/backups/$d/$old"; log "pruned $d/$old"
  done
done
