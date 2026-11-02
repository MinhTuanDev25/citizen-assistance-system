#!/usr/bin/env bash
# On-demand backup on the volume (cas-keeper also takes base backups and Qdrant
# snapshots on its own schedule). Run as root: docker exec <c> backup.sh
#   1. pg_dump (custom format, over TLS as role cas) -> backups/pg, newest $CAS_BACKUP_KEEP kept
#   2. a verified base backup now (pg/base; the continuous WAL is pg/wal)
#   3. a Qdrant snapshot + fingerprint now (qdrant/snapshots; refused when a READY generation
#      differs from its baseline in qdrant/baselines)
# Off-site copy is NOT done here: copy /workspace/cas/pg, backups, qdrant/snapshots,
# qdrant/baselines and objects/ out of RunPod.
set -Eeuo pipefail

W="${CAS_WORKSPACE:-/workspace/cas}"
KEEP="${CAS_BACKUP_KEEP:-7}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
log() { echo "[backup] $(date -u +%FT%TZ) $*" >&2; }
[[ $EUID -eq 0 ]] || { echo "backup.sh: run as root" >&2; exit 1; }
# docker exec does not see the variables exported by entrypoint.sh.
# Keep the caller's PATH (the boot environment has its own).
caller_path="$PATH"
# shellcheck disable=SC1091
source /run/cas-runtime.env
PATH="$caller_path"

dump="$W/backups/pg/citizen_assistance-$TS.dump"
gosu cas pg_dump -Fc -f "$dump" "$DATABASE_URL"
gosu cas pg_restore --list "$dump" > /dev/null   # readable archive, not just a file
log "postgres -> $dump ($(du -h "$dump" | cut -f1))"
ls -1t "$W/backups/pg" | tail -n +"$((KEEP + 1))" | while read -r old; do
  rm -f "$W/backups/pg/$old"; log "pruned pg/$old"
done

log "base backup -> $(cas-storage base-backup)"
if curl -fsS "http://127.0.0.1:6333/collections/${QDRANT_COLLECTION:-knowledge_chunks}" >/dev/null 2>&1; then
  log "qdrant -> $(cas-storage qdrant-snapshot)"
else
  log "qdrant collection not found; snapshot skipped"
fi
