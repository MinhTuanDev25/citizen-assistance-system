#!/usr/bin/env bash
# restore_command of a recovery from the volume (cas-pg-restore.sh):
#   [CAS_RESTORE_TAIL=<dir>] pg-restore-wal.sh <wal file name> <destination path>
# A complete segment wins; otherwise the receiver's in-progress <name>.partial (it is
# preallocated to the full segment size; recovery stops at its last valid record).
# With CAS_RESTORE_TAIL, a copy of the bytes served for the newest segment name is kept
# there (the first time each name is served), so that cas-pg-restore.sh checks the end of
# recovery against exactly what recovery read, even while the receiver keeps writing.
set -u
src="${CAS_WORKSPACE:-/workspace/cas}/pg/wal/$1"
# Present but unreadable looks like "absent" to recovery (it ends there and the restore
# is refused as incomplete); say why in the Postgres log.
for f in "$src" "$src.partial"; do
  [[ ! -e "$f" || -r "$f" ]] || echo "pg-restore-wal: $f exists but is not readable by $(id -un)" >&2
done
if [[ -f "$src" ]]; then from="$src"
elif [[ -f "$src.partial" ]]; then from="$src.partial"
else exit 1; fi
cp "$from" "$2" || exit 1
tail_dir="${CAS_RESTORE_TAIL:-}"
if [[ -n "$tail_dir" && "$1" =~ ^[0-9A-F]{24}$ && ! -e "$tail_dir/$1" ]]; then
  cp "$2" "$tail_dir/.$1.tmp" && mv "$tail_dir/.$1.tmp" "$tail_dir/$1" && echo "$from" > "$tail_dir/$1.source" || exit 1
  # Keep the copies of the two highest segment numbers (the end of recovery is in one of
  # them; at promotion Postgres re-reads the segment before a switch point that falls on a
  # segment boundary), drop older ones to bound the disk use.
  ls "$tail_dir" | grep -E '^[0-9A-F]{24}$' | awk '{print substr($0, 9, 16), $0}' | sort -r | tail -n +3 | cut -d' ' -f2 \
    | while read -r old; do rm -f "$tail_dir/$old" "$tail_dir/$old.source"; done
fi
exit 0
