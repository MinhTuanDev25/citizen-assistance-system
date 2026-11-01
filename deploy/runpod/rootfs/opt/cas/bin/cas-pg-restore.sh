#!/usr/bin/env bash
# Restore-on-boot of PostgreSQL from the network volume (run as root by entrypoint.sh
# when the container disk has no pgdata). Fail-closed: any check that does not pass
# stops the boot with exit 1 and leaves the volume untouched.
#   1. newest base backup with .cas-complete; pg_verifybackup again (manifest checksums)
#   2. its system identifier equals pg/state/system_identifier (same cluster)
#   3. WAL continuity on the volume: every segment number from the backup's start
#      segment to the newest file present exists (any timeline; .partial allowed last)
#   4. copy to the container disk, recover with restore_command from the volume
#      (temporary server, unix socket only), promote at the end of the WAL
#   5. where recovery ended (the new timeline's switch point, an exact LSN) is checked
#      three ways; any failure refuses the restore instead of opening a database that
#      misses acknowledged commits:
#      a. it is at or after the durable LSN recorded on the volume (pg/state/flushed_lsn.json,
#         written about every second by cas-keeper from the receiver's flush position, read
#         BEFORE recovery starts): every commit acknowledged before that record is replayed
#      b. it is in or after the newest WAL segment on the volume
#      c. in the copy of the last segment recovery read, nothing replayable follows it
#         (cas-storage wal-end-check): only zeros, or one record cut at a page boundary
#      Not provable from the volume: commits acknowledged after the newest durable-LSN
#      record whose WAL bytes were then lost on the volume (zeroed or never persisted).
#   6. pg_controldata system identifier still matches; restore recorded in pg/state/restore.log
# restore_drill.sh runs the same steps into a scratch directory:
#   CAS_RESTORE_TARGET=<dir> CAS_RESTORE_SOCKET=<dir> CAS_RESTORE_PORT=<n> CAS_RESTORE_DRILL=1
# (nothing is recorded on the volume in a drill).
set -Eeuo pipefail
W="${CAS_WORKSPACE:-/workspace/cas}"
L="${CAS_LOCAL:-/var/lib/cas}"
PGDATA="${CAS_RESTORE_TARGET:-$L/pgdata}"; SOCK="${CAS_RESTORE_SOCKET:-/run/postgresql}"; PORT="${CAS_RESTORE_PORT:-5432}"
DRILL="${CAS_RESTORE_DRILL:-0}"
WAL="$W/pg/wal"; BASE="$W/pg/base"; STATE="$W/pg/state"
PGBIN=/usr/lib/postgresql/16/bin
log() { echo "[cas-pg-restore] $(date -u +%FT%TZ) $*" >&2; }
die() { log "FATAL: $*"; exit 1; }
trap 'die "failed at line $LINENO"' ERR
as_pg() { gosu postgres "$@"; }
q() { as_pg psql -h "$SOCK" -p "$PORT" -d postgres -XAt "$@"; }

base="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -exec test -f '{}/.cas-complete' \; -print 2>/dev/null | sort | tail -1)"
[[ -n "$base" ]] || die "no complete base backup in $BASE"
log "base backup $(basename "$base")"
# .cas-complete is the keeper's own marker, written after the manifest.
as_pg "$PGBIN/pg_verifybackup" -q -i .cas-complete "$base" || die "pg_verifybackup failed for $base"

want_id="$(cat "$STATE/system_identifier" 2>/dev/null || true)"
[[ -n "$want_id" ]] || die "$STATE/system_identifier missing: cannot prove the base backup belongs to this volume's cluster"
got_id="$(as_pg "$PGBIN/pg_controldata" "$base" | awk -F': *' '/^Database system identifier/ {print $2}')"
[[ "$got_id" == "$want_id" ]] || die "base backup system identifier $got_id != volume cluster $want_id"

# ---- WAL continuity before replay -------------------------------------------------------
start_file="$(sed -n 's/^START WAL LOCATION: .*(file \([0-9A-F]\{24\}\)).*/\1/p' "$base/backup_label")"
[[ ${#start_file} == 24 ]] || die "cannot read the start WAL file from $base/backup_label"
read -r gap last_seg last_name < <(python3 - "$WAL" "$start_file" <<'PY'
import os, re, sys
wal, start = sys.argv[1], sys.argv[2]
segno = lambda name: int(name[8:16], 16) * 0x100 + int(name[16:24], 16)   # 16 MB segments
names = [n for n in os.listdir(wal) if re.fullmatch(r"[0-9A-F]{24}(\.partial)?", n)]
first = segno(start)
have = {segno(n) for n in names if segno(n) >= first}
if not have:
    print("none 0 -"); sys.exit()
top = max(have)
missing = [s for s in range(first, top + 1) if s not in have]
newest = max((n for n in names if segno(n) == top), key=lambda n: (n.endswith(".partial") is False, n))
print(("gap:%x" % missing[0]) if missing else "ok", top, newest)
PY
)
[[ "$gap" != none ]] || log "no WAL newer than the base backup on the volume (base backup only)"
[[ "$gap" == ok || "$gap" == none ]] || die "WAL continuity broken on the volume: segment $gap missing between $start_file and $last_name"
log "WAL continuity ok from $start_file to ${last_name}"

# ---- the durable LSN, read before recovery (the receiver may still write in a drill) ----
durable_lsn=""
if [[ -f "$STATE/flushed_lsn.json" ]]; then
  read -r d_id durable_lsn < <(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["system_identifier"], d["flushed_lsn"])' "$STATE/flushed_lsn.json") \
    || die "cannot read $STATE/flushed_lsn.json"
  [[ "$d_id" == "$want_id" ]] || die "$STATE/flushed_lsn.json belongs to cluster $d_id, not $want_id"
  log "durable LSN recorded on the volume: $durable_lsn"
elif [[ "${CAS_RESTORE_WITHOUT_DURABLE_LSN:-0}" == 1 ]]; then
  log "WARNING: no durable LSN record; restoring anyway (CAS_RESTORE_WITHOUT_DURABLE_LSN=1, operator choice): only segment-level and end-of-WAL checks"
else
  die "no durable LSN record ($STATE/flushed_lsn.json): cannot prove how far recovery must go. If the WAL on the volume is known complete, an operator may set CAS_RESTORE_WITHOUT_DURABLE_LSN=1 for this one boot"
fi

# ---- recover on the container disk --------------------------------------------------------
rm -rf "$PGDATA.restore" "$PGDATA.wal-tail"
install -d -o postgres -g postgres -m 0700 "$PGDATA.wal-tail"
cp -a "$base" "$PGDATA.restore"
rm -f "$PGDATA.restore/backup_manifest" "$PGDATA.restore/.cas-complete" "$PGDATA.restore/postmaster.pid"
chown -R postgres:postgres "$PGDATA.restore"; chmod 0700 "$PGDATA.restore"
cat >> "$PGDATA.restore/postgresql.auto.conf" <<EOF
# cas-pg-restore $(date -u +%FT%TZ): recovery from the volume; removed after promotion
restore_command = 'CAS_WORKSPACE=$W CAS_RESTORE_TAIL=$PGDATA.wal-tail /opt/cas/bin/pg-restore-wal.sh %f %p'
recovery_target_timeline = 'latest'
EOF
as_pg touch "$PGDATA.restore/recovery.signal"
rm -rf "$PGDATA"; mv "$PGDATA.restore" "$PGDATA"
install -o postgres -g postgres -m 0600 /etc/cas/pg_hba.conf "$PGDATA/pg_hba.conf"
[[ -d "$SOCK" ]] || install -d -o postgres -g postgres -m 0700 "$SOCK"
install -o postgres -g postgres -m 0600 /dev/null "$PGDATA.log"
log "recovering (temporary server, unix socket only)"
as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -w -t 3600 -l "$PGDATA.log" \
  -o "-c listen_addresses='' -c port=$PORT -c unix_socket_directories=$SOCK -c ssl=off -c synchronous_standby_names='' -c hot_standby=on" start >/dev/null \
  || { tail -20 "$PGDATA.log" >&2; die "recovery server did not start"; }
for _ in $(seq 1 3600); do
  [[ "$(q -c 'select pg_is_in_recovery()' 2>/dev/null)" == f ]] && break
  sleep 1
done
[[ "$(q -c 'select pg_is_in_recovery()')" == f ]] || die "recovery did not finish"
tli="$(q -c "select timeline_id from pg_control_checkpoint()")"
hist="$PGDATA/pg_wal/$(printf '%08X' "$tli").history"
switch_lsn=""; parent_tli=""
[[ -f "$hist" ]] && read -r parent_tli switch_lsn _ < <(tail -1 "$hist")
q -q -c "ALTER SYSTEM RESET restore_command" -c "ALTER SYSTEM RESET recovery_target_timeline" >/dev/null
as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -w -t 300 -m fast stop >/dev/null
# postgresql.auto.conf had our lines appended before ALTER SYSTEM rewrote it; make sure.
! grep -q "^restore_command" "$PGDATA/postgresql.auto.conf" || die "restore_command still set after promotion"

# ---- where recovery ended (exact LSN) -------------------------------------------------------
[[ -n "$switch_lsn" && -n "$parent_tli" ]] || die "no timeline history after promotion: cannot prove where recovery ended"
lsn() { python3 -c 'import sys;h,l=sys.argv[1].split("/");print((int(h,16)<<32)|int(l,16))' "$1"; }
end_lsn="$(lsn "$switch_lsn")"
# a. the durable LSN
if [[ -n "$durable_lsn" ]]; then
  (( end_lsn >= $(lsn "$durable_lsn") )) \
    || die "recovery ended at $switch_lsn, before the durable LSN $durable_lsn recorded on the volume: WAL on the volume is incomplete or corrupt"
fi
# b. the newest segment on the volume
end_seg=$(( end_lsn >> 24 ))
if [[ "$gap" == ok ]]; then
  (( end_seg >= last_seg )) || die "recovery ended at $switch_lsn (segment $end_seg) before the newest WAL on the volume (segment $last_seg): WAL is incomplete or corrupt"
fi
# c. nothing replayable after the end, in the bytes recovery read for that segment
end_name="$(printf '%08X%08X%08X' "$parent_tli" $(( end_seg >> 8 )) $(( end_seg & 255 )))"
if [[ -f "$PGDATA.wal-tail/$end_name" ]]; then
  /opt/cas/bin/cas-storage wal-end-check "$PGDATA.wal-tail/$end_name" "$switch_lsn" >/dev/null 2>"$PGDATA.wal-tail/check.err" \
    || die "end of recovery at $switch_lsn in $end_name ($(cat "$PGDATA.wal-tail/$end_name.source")): $(sed -n 's/^cas-storage: FAIL //p' "$PGDATA.wal-tail/check.err")"
elif compgen -G "$WAL/$end_name*" >/dev/null; then
  die "recovery ended in $end_name but never read it from the volume: cannot check the end of recovery"
fi
rm -rf "$PGDATA.wal-tail"
log "recovery ended at $switch_lsn, timeline $tli (durable LSN ${durable_lsn:-not recorded}; newest WAL on the volume: $last_name; nothing replayable after the end)"
id_now="$(as_pg "$PGBIN/pg_controldata" "$PGDATA" | awk -F': *' '/^Database system identifier/ {print $2}')"
[[ "$id_now" == "$want_id" ]] || die "restored cluster identifier $id_now != $want_id"
printf '%s' "$want_id" > "$PGDATA/.cas-cluster-id"; chown postgres:postgres "$PGDATA/.cas-cluster-id"
as_pg touch "$PGDATA/.cas-init-complete"
if [[ $DRILL == 1 ]]; then
  echo "RESTORED base=$(basename "$base") wal_to=$last_name end=${switch_lsn:-base-only} timeline=$tli"
  log "drill restore complete in $PGDATA"
  exit 0
fi
printf '%s restored base=%s wal_to=%s end=%s timeline=%s\n' "$(date -u +%FT%TZ)" "$(basename "$base")" "$last_name" "${switch_lsn:-base-only}" "$tli" >> "$STATE/restore.log"
log "restore complete; a new base backup is taken before the services open"
