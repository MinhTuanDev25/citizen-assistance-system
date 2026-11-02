#!/usr/bin/env bash
# Destructive persistence tests of the storage design (engines on the container disk,
# synchronous WAL + base backups + Qdrant snapshots + objects on the volume). Runs on
# the host with throwaway containers and volumes (prefix cas-aio-persist-<run>); never
# the dev stack. Results: deploy/runpod/tests/out/persist-<run>/results.tsv.
#   deploy/runpod/tests/persist_test.sh <image> [--keep]
#
# "Acknowledged" below means: the INSERT's COMMIT returned success to the client and
# the client recorded the id on the host. The claim tested is that every acknowledged
# commit survives; commits in flight when something is killed may or may not survive.
#   PT1  kill -9 of the whole container during writes, then a NEW container (empty
#        container disk) restores from the volume: every acknowledged id present
#   PT2  kill -9 of the postmaster during writes: every acknowledged id present
#   PT3  kill -9 of the WAL receiver during writes: no commit acknowledged while it is
#        down beyond those already flushed; every acknowledged id present afterwards
#   PT4  a missing WAL segment on the volume: the restore is refused (exit 64)
#   PT5  a corrupt WAL segment before the newest: recovery would stop early: refused
#   PT5c-f the NEWEST WAL segment (where a segment-number check sees nothing): a corrupt
#        record (refused by the durable LSN), the same with the durable LSN record behind
#        (refused by the end-of-recovery check), its records zeroed (lost writes: refused
#        by the durable LSN), the file truncated: all refused
#   PT6  a tampered Qdrant snapshot (bytes or recorded fingerprint): bootstrap refuses
#   PT7  a Qdrant point removed (count differs from Postgres vector_count): refused
#   PT7c a Qdrant vector changed, same point count: refused (recorded fingerprint)
#   PT7r/PT7s on the running system (no reboot): the keeper's periodic check makes /readyz 503
#        and reports the missing point / changed vector as a storage problem
#   PT7e the same changed vector is never re-recorded: a new snapshot is refused (the
#        baseline is per generation, written once), the snapshots and baselines unchanged
#   PT8  volume quota exhausted: database read-only, not ready, writes fail; normal
#        quota again: writable, data intact
#   PT10 three more cycles of kill -9 of the container during writes + restore on a new
#        container: no false refusal by the end-of-recovery checks, every ack present
#   PT9  the whole flow on a bindfs volume that behaves like the RunPod network volume
#        (every mode 0777, chown EPERM): boots, writes, restores on a new container
set -uo pipefail

IMG="${1:?usage: persist_test.sh <image> [--keep]}"
KEEP="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RUN="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HERE/out/persist-$RUN"; mkdir -p "$OUT"
P="cas-aio-persist-$RUN"; C="$P-c"; VOL="$P-ws"; BVOL="$P-bindfs"
PORT="${CAS_PERSIST_PORT:-18089}"; BASE="http://127.0.0.1:$PORT"
RES="$OUT/results.tsv"; printf 'id\tresult\tcheck\tevidence\n' > "$RES"
exec > >(tee -a "$OUT/persist.log") 2>&1

rnd() { openssl rand -hex "$1"; }
JWT="$(rnd 32)"; AIT="$(rnd 16)"; PGPW="$(rnd 16)"
ENVS=(-e CAS_SPEECH_MODE=off -e EMBEDDING_VARIANT="${SMOKE_EMBEDDING_VARIANT:-fp32}" -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT"
      -e POSTGRES_PASSWORD="$PGPW" -e LLM_PROVIDER=mock -e CAS_ALLOW_MOCK_LLM=true -e CAS_KEEPER_INTERVAL_SECONDS=3 -e CAS_QDRANT_VERIFY_SECONDS=15)
MODELS=(-v "$ROOT/models:/workspace/cas/models:ro")
DSN="postgresql://cas:$PGPW@127.0.0.1:5432/citizen_assistance?sslmode=verify-full&sslrootcert=/var/lib/cas/tls/ca.crt"

tsv() { printf '%s' "$1" | tr '\t\r\n' '   '; }
pass() { printf '%s\tPASS\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "PASS $1 $2"; }
fail() { printf '%s\tFAIL\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "FAIL $1 $2 :: ${3:-}"; }
dx() { docker exec "$C" "$@"; }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@"; }
sql() { dx gosu cas psql "$DSN" -XAtq -c "$1" 2>&1; }
su_sql() { dx gosu postgres psql -h /run/postgresql -d citizen_assistance -XAtq -c "$1" 2>&1; }

start() {  # start [extra docker run args...]: new container on $VOL
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --name "$C" -p "127.0.0.1:$PORT:8088" -v "$VOL:/workspace" "${MODELS[@]}" "${ENVS[@]}" "$@" "$IMG" >/dev/null
}
wait_healthy() {
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    if dx healthcheck.sh --quiet >/dev/null 2>&1; then echo "healthy after $((SECONDS - t0))s"; return 0; fi
    [[ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" == true ]] || { echo "container exited ($(docker inspect -f '{{.State.ExitCode}}' "$C"))"; return 1; }
    sleep 3
  done
  echo "not healthy after $1s"; return 1
}
wait_exit() {  # prints the exit code, or "running" if the container did not exit in time
  for _ in $(seq 1 ${1:-90}); do
    [[ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" == false ]] && { docker inspect -f '{{.State.ExitCode}}' "$C"; return; }
    sleep 2
  done
  echo running
}
cleanup() {
  docker logs "$C" > "$OUT/last-container.log" 2>&1 || true
  if [[ "$KEEP" != --keep ]]; then docker rm -f "$C" >/dev/null 2>&1; docker volume rm "$VOL" "$BVOL" >/dev/null 2>&1; fi
}
trap cleanup EXIT

# writer <label> <count>: one INSERT per transaction inside the container; each id is
# echoed only after its COMMIT returned success, and appended on the host as it arrives.
writer_start() {
  : > "$OUT/$1.acked"
  docker exec "$C" bash -c "for i in \$(seq 1 $2); do gosu cas psql '$DSN' -XAtq -c \"insert into cas_persist_test(label, n) values ('$1', \$i)\" >/dev/null 2>&1 && echo \$i; done" \
    >> "$OUT/$1.acked" 2>/dev/null &
  WPID=$!
}
acked_check() {  # acked_check <id> <name> <label>: every acknowledged id is in the table
  local n missing
  n="$(grep -c . "$OUT/$3.acked")"
  missing="$(sql "select count(*) from unnest(array[$(paste -sd, "$OUT/$3.acked" | sed 's/^$/0/')]::int[]) a(n) where not exists (select 1 from cas_persist_test t where t.label = '$3' and t.n = a.n)")"
  echo "acknowledged=$n missing=$missing" > "$OUT/$1.txt"
  if [[ "$n" -ge 1 && "$missing" == 0 ]]; then pass "$1" "$2" "acknowledged=$n, all present"; else fail "$1" "$2" "acknowledged=$n missing=$missing"; fi
}

echo "image=$IMG run=$RUN"
docker volume create "$VOL" >/dev/null
start; h="$(wait_healthy 900)"; echo "$h"
[[ "$h" == healthy* ]] || { fail PT0 "fresh start" "$h"; exit 1; }
su_sql "create table cas_persist_test (label text not null, n int not null, at timestamptz default now(), primary key (label, n)); grant all on cas_persist_test to cas" >/dev/null

# ---- index one native PDF so Qdrant has a READY generation (PT6, PT7) -----------------
ADMIN_PW="Persist-$(rnd 8)"
docker exec -e BOOTSTRAP_ADMIN_EMAIL=admin@chuse.vn -e BOOTSTRAP_ADMIN_PASSWORD="$ADMIN_PW" "$C" cas-run cas /opt/cas/api/bootstrap-admin >/dev/null 2>&1
TOKEN="$(curl -s "$BASE/api/v1/auth/login" -H 'Content-Type: application/json' -d "{\"email\":\"admin@chuse.vn\",\"password\":\"$ADMIN_PW\"}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("data") or {}).get("access_token") or (d.get("data") or {}).get("token") or "")')"
python3 - "$OUT/doc.pdf" "Persist $RUN: nguoi di dang ky khai sinh nop to khai va giay chung sinh" <<'PY'
import sys
body = b"BT /F1 14 Tf 72 720 Td (" + sys.argv[2].encode() + b") Tj ET"
objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
out, offs = bytearray(b"%PDF-1.4\n"), []
for i, o in enumerate(objs, 1):
    offs.append(len(out)); out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
x = len(out); out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
for o in offs: out += b"%010d 00000 n \n" % o
out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
open(sys.argv[1], "wb").write(out)
PY
DOC="$(curl -s "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/doc.pdf;type=application/pdf" -F title="Persist $RUN" -F domain_id=ho_tich_chung_thuc \
  | python3 -c 'import json,sys;print((json.load(sys.stdin).get("data") or {}).get("id",""))')"
VER="$(curl -s "$BASE/api/v1/admin/documents/link-targets" -H "Authorization: Bearer $TOKEN" | python3 -c 'import json,sys;it=json.load(sys.stdin)["data"]["items"];print(it[0]["procedure_version_id"] if it else "")')"
curl -s -o /dev/null "$BASE/api/v1/admin/documents/$DOC/links" -H "Authorization: Bearer $TOKEN" -H "X-Request-ID: $(uuidgen | tr A-Z a-z)" \
  -H 'Content-Type: application/json' -d "{\"procedure_version_id\":\"$VER\",\"relationship_type\":\"SOURCE\"}"
curl -s -o /dev/null -X POST "$BASE/api/v1/admin/documents/$DOC/links/$VER/index" -H "Authorization: Bearer $TOKEN" -H "X-Request-ID: $(uuidgen | tr A-Z a-z)"
st=""; for _ in $(seq 1 200); do
  st="$(curl -s "$BASE/api/v1/admin/documents/$DOC/links" -H "Authorization: Bearer $TOKEN" | python3 -c 'import json,sys;it=json.load(sys.stdin)["data"]["items"];print(it[0]["index_status"] if it else "")' 2>/dev/null)"
  [[ "$st" == READY || "$st" == FAILED ]] && break; sleep 3
done
for _ in $(seq 1 40); do dx bash -c 'ls /workspace/cas/qdrant/snapshots/knowledge_chunks/*.snapshot' >/dev/null 2>&1 && break; sleep 3; done
snaps="$(dx bash -c 'ls /workspace/cas/qdrant/snapshots/knowledge_chunks/ 2>/dev/null' | tr '\n' ' ')"
echo "doc=$DOC index=$st snapshots=$snaps" > "$OUT/PT0_index.txt"
[[ "$st" == READY && "$snaps" == *.snapshot* ]] && pass PT0 "indexed one PDF; cas-keeper took a Qdrant snapshot when the READY set changed" "$snaps" || fail PT0 "index + snapshot" "$(cat "$OUT/PT0_index.txt")"

# ---- PT2: kill -9 of the postmaster during writes ------------------------------------------
writer_start pt2 400; sleep 3
dx bash -c 'kill -9 "$(head -1 /var/lib/cas/pgdata/postmaster.pid)"'
wait "$WPID" 2>/dev/null
for _ in $(seq 1 60); do [[ "$(code "$BASE/readyz")" == 200 ]] && break; sleep 2; done
acked_check PT2 "kill -9 postmaster during writes: every acknowledged commit present after the restart" pt2

# ---- PT3: kill -9 of the WAL receiver during writes ------------------------------------------
writer_start pt3 100000; sleep 3
dx bash -c 'kill -9 "$(pgrep -x pg_receivewal)"'
n0="$(grep -c . "$OUT/pt3.acked")"; sleep 6; n1="$(grep -c . "$OUT/pt3.acked")"
dx supervisorctl -c /etc/cas/supervisord.conf stop pg-receivewal >/dev/null 2>&1   # keep it down for a clear window
sleep 6; n2="$(grep -c . "$OUT/pt3.acked")"; sleep 6; n3="$(grep -c . "$OUT/pt3.acked")"
dx supervisorctl -c /etc/cas/supervisord.conf start pg-receivewal >/dev/null 2>&1
sleep 8; n4="$(grep -c . "$OUT/pt3.acked")"
docker exec "$C" pkill -f "cas_persist_test(label, n) values ('pt3'" >/dev/null 2>&1; kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null
echo "acked: at kill=$n0 +6s=$n1 (supervisord restarts it) | stopped: +6s=$n2 +12s=$n3 (must not grow) | started again +8s=$n4 (must grow)" > "$OUT/PT3_window.txt"
[[ "$n2" == "$n3" && "$n4" -gt "$n3" ]] && pass PT3a "receiver down: no commit acknowledged; receiver back: commits acknowledged again" "$(cat "$OUT/PT3_window.txt")" || fail PT3a "receiver window" "$(cat "$OUT/PT3_window.txt")"
for _ in $(seq 1 30); do [[ "$(code "$BASE/readyz")" == 200 ]] && break; sleep 2; done
acked_check PT3 "kill -9 / stop of the WAL receiver during writes: every acknowledged commit present" pt3

# ---- PT1: kill -9 of the whole container during writes, restore on a new container ----------
writer_start pt1 100000; sleep 5
docker kill -s KILL "$C" >/dev/null; wait "$WPID" 2>/dev/null
OWNER="$(docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'python3 -c "import json;print(json.load(open(\"/w/cas/locks/workspace.lease/owner.json\"))[\"owner_id\"])"')"
docker rm -f "$C" >/dev/null
start; sleep 20
docker exec -e CAS_PG_RECOVER_WAIT_S=40 "$C" cas-pg-recover "$OWNER" > "$OUT/PT1_recover.txt" 2>&1
docker restart -t 60 "$C" >/dev/null
h="$(wait_healthy 900)"
docker logs "$C" 2>&1 | grep -E "restoring from the volume|recovery ended|restore complete|qdrant restored|objects verified" > "$OUT/PT1_restore.txt"
echo "$h" >> "$OUT/PT1_restore.txt"
[[ "$h" == healthy* ]] && grep -q "recovery ended at" "$OUT/PT1_restore.txt" && pass PT1a "container SIGKILL, new container disk: restored from base backup + WAL and verified" "PT1_restore.txt" || fail PT1a "restore after container kill" "PT1_restore.txt"
acked_check PT1 "container SIGKILL during writes: every acknowledged commit present after the restore" pt1

# Several WAL segments with acknowledged rows in between, for PT4/PT5.
for i in 1 2 3 4; do sql "insert into cas_persist_test(label, n) values ('seg', $i)" >/dev/null; su_sql "select pg_switch_wal()" >/dev/null; done
sql "insert into cas_persist_test(label, n) values ('seg', 99)" >/dev/null
sleep 3
docker stop -t 120 "$C" >/dev/null

# ---- PT4: a missing WAL segment -> restore refused ------------------------------------------
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'cd /w/cas/pg/wal && ls | grep -E "^[0-9A-F]{24}$" | sort | tail -3 | head -1 > /tmp/m && f=$(cat /tmp/m) && mv "$f" "/w/cas/pg/$f.pt4" && echo "$f" > /w/cas/pg/pt4.name'
start; ec="$(wait_exit 120)"
docker logs "$C" 2>&1 | grep -E "FATAL|WAL continuity|lease released" > "$OUT/PT4_gap.txt"
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'f=$(cat /w/cas/pg/pt4.name) && mv "/w/cas/pg/$f.pt4" "/w/cas/pg/wal/$f" && rm /w/cas/pg/pt4.name'
[[ "$ec" == 64 ]] && grep -q "WAL continuity broken" "$OUT/PT4_gap.txt" && grep -q "lease released" "$OUT/PT4_gap.txt" && pass PT4 "missing WAL segment: restore refused (exit 64), nothing opened, own lease released" "$(head -2 "$OUT/PT4_gap.txt" | tr '\n' ' ')" || fail PT4 "missing segment" "exit=$ec $(cat "$OUT/PT4_gap.txt")"

# ---- PT5: a corrupt WAL segment before the newest -> recovery stops early -> refused ---------
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'cd /w/cas/pg/wal && f=$(ls | grep -E "^[0-9A-F]{24}$" | sort | tail -2 | head -1) && cp -p "$f" "/w/cas/pg/$f.pt5" && echo "$f" > /w/cas/pg/pt5.name && dd if=/dev/urandom of="$f" bs=1 seek=48 count=64 conv=notrunc 2>/dev/null \
  && r=/w/cas/pg/state/flushed_lsn.json && cp -p $r $r.pt5 && sed -i "s#\"flushed_lsn\": \"[0-9A-F/]*\"#\"flushed_lsn\": \"0/1\"#" $r'
# the durable LSN record is rewound so that the segment-number check (b) is what refuses
start; ec="$(wait_exit 180)"
docker logs "$C" 2>&1 | grep -E "FATAL|recovery ended|lease released|MAINTENANCE" > "$OUT/PT5_corrupt.txt"
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'f=$(cat /w/cas/pg/pt5.name) && mv "/w/cas/pg/$f.pt5" "/w/cas/pg/wal/$f" && rm /w/cas/pg/pt5.name \
  && mv /w/cas/pg/state/flushed_lsn.json.pt5 /w/cas/pg/state/flushed_lsn.json'
[[ "$ec" == 64 ]] && grep -q "before the newest WAL" "$OUT/PT5_corrupt.txt" && pass PT5 "corrupt WAL before the newest segment: recovery would stop early; restore refused (exit 64)" "$(head -2 "$OUT/PT5_corrupt.txt" | tr '\n' ' ')" || fail PT5 "corrupt segment" "exit=$ec $(cat "$OUT/PT5_corrupt.txt")"

# ---- PT5c-f: damage inside the NEWEST segment -------------------------------------------------
# wal_tamper <action>: on the volume, save | corrupt | rewind | zero | truncate | restore the
# newest WAL file (highest segment, then timeline; complete before .partial) and the durable
# LSN record.
wal_tamper() {
  docker run --rm -i -v "$VOL:/w" --entrypoint python3 "$IMG" - "$1" <<'TAMPER'
import json, os, re, shutil, sys
wal, keep, rec = "/w/cas/pg/wal", "/w/cas/pg/.pt5-keep", "/w/cas/pg/state/flushed_lsn.json"
act = sys.argv[1]
def copy(a, b):
    shutil.copy2(a, b); st = os.stat(a); os.chown(b, st.st_uid, st.st_gid)
if act == "restore":
    for n in os.listdir(keep):
        copy(os.path.join(keep, n), rec if n == "flushed_lsn.json" else os.path.join(wal, n))
    shutil.rmtree(keep); print("restored"); sys.exit()
names = [n for n in os.listdir(wal) if re.fullmatch(r"[0-9A-F]{24}(\.partial)?", n)]
newest = max(names, key=lambda n: (n[8:24], n[:8], not n.endswith(".partial")))
path = os.path.join(wal, newest)
if act == "save":
    os.makedirs(keep); copy(path, os.path.join(keep, newest)); copy(rec, os.path.join(keep, "flushed_lsn.json"))
elif act == "corrupt":      # 64 random bytes inside the first record of the segment
    with open(path, "r+b") as f: f.seek(48); f.write(os.urandom(64))
elif act == "rewind":       # the durable LSN record behind the damage (as if it lagged)
    d = json.load(open(rec)); d["flushed_lsn"] = "0/1"; json.dump(d, open(rec, "w"))
elif act == "zero":         # every record of the segment lost (zeros), page header kept
    with open(path, "r+b") as f: f.seek(40); f.write(bytes(os.path.getsize(path) - 40))
elif act == "truncate":
    os.truncate(path, os.path.getsize(path) // 2)
print(act, newest, json.load(open(rec))["flushed_lsn"])
TAMPER
}
pt5x() {  # pt5x <id> <description> <expected log regex> <actions...>
  local id="$1" what="$2" want="$3" ec; shift 3
  wal_tamper save > "$OUT/${id}.txt"
  for a in "$@"; do wal_tamper "$a" >> "$OUT/${id}.txt"; done
  start; ec="$(wait_exit 180)"
  docker logs "$C" 2>&1 | grep -E "durable LSN|FATAL|recovery ended|lease released" >> "$OUT/${id}.txt"
  wal_tamper restore >> "$OUT/${id}.txt"
  [[ "$ec" == 64 ]] && grep -qE "$want" "$OUT/${id}.txt" && grep -q "lease released" "$OUT/${id}.txt" \
    && pass "$id" "$what: restore refused (exit 64)" "$(grep -E "$want" "$OUT/${id}.txt" | head -1)" || fail "$id" "$what" "exit=$ec $(tr '\n' ' ' < "$OUT/${id}.txt")"
}
pt5x PT5c "corrupt record in the newest WAL segment" "before the durable LSN" corrupt
pt5x PT5d "corrupt record in the newest segment, durable LSN record behind it" "end of recovery at .*(complete record|invalid record|does not continue|invalid page header)" corrupt rewind
pt5x PT5e "newest segment's records lost (zeroed), only the durable LSN shows it" "before the durable LSN" zero
pt5x PT5f "newest WAL file truncated" "wrong size|not 16777216|recovery server did not start" truncate

# After PT4/PT5 the volume is intact again: a new container restores everything.
start; h="$(wait_healthy 900)"
seg="$(sql "select count(*) from cas_persist_test where label = 'seg'")"
[[ "$h" == healthy* && "$seg" == 5 ]] && pass PT5b "volume repaired: restore succeeds and the rows written across 5 WAL segments are all present" "$h seg_rows=$seg" || fail PT5b "restore after repair" "$h seg_rows=$seg"

# ---- PT6: tampered Qdrant snapshot -> bootstrap refuses --------------------------------------
docker stop -t 120 "$C" >/dev/null
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'cd /w/cas/qdrant/snapshots/knowledge_chunks && s=$(ls *.json | sort | tail -1) && cp "$s" "$s.pt6" && python3 -c "
import json,sys
p=sys.argv[1]; m=json.load(open(p)); g=next(iter(m[\"fingerprint\"][\"generations\"].values())); g[\"payload_sha256\"]=\"0\"*64
json.dump(m,open(p,\"w\"))" "$s"'
start; sleep 60
docker logs "$C" 2>&1 | grep -E "\[bootstrap\].*FATAL|cas-storage: FAIL" > "$OUT/PT6_tamper.txt"
api6="$(dx pgrep -u cas -x api >/dev/null && echo running || echo not-started)"; r6="$(code "$BASE/readyz")"
docker stop -t 60 "$C" >/dev/null
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'cd /w/cas/qdrant/snapshots/knowledge_chunks && for f in *.pt6; do mv "$f" "${f%.pt6}"; done'
echo "api=$api6 readyz=$r6" >> "$OUT/PT6_tamper.txt"
[[ "$api6" == not-started && "$r6" != 200 ]] && grep -q "does not match the fingerprint" "$OUT/PT6_tamper.txt" \
  && pass PT6 "snapshot fingerprint tampered: bootstrap refuses, API never starts" "$(head -1 "$OUT/PT6_tamper.txt")" || fail PT6 "snapshot tamper" "$(cat "$OUT/PT6_tamper.txt")"

# running_problem <id> <what> <regex>: without a reboot, the keeper's periodic Qdrant check
# (CAS_QDRANT_VERIFY_SECONDS=15) turns /readyz to 503 with the problem in the storage status.
running_problem() {
  local r="" st="" i
  for i in $(seq 1 40); do
    r="$(code "$BASE/readyz")"; st="$(dx cas-storage status 2>/dev/null | tr -d '\n')"
    [[ "$r" == 503 ]] && grep -qE "$3" <<<"$st" && break; sleep 2
  done
  echo "readyz=$r status=$st" > "$OUT/${1}_running.txt"
  [[ "$r" == 503 ]] && grep -qE "$3" <<<"$st" && pass "$1" "$2: /readyz 503 from the keeper's check, no reboot needed" "readyz=$r" \
    || fail "$1" "$2: running-system detection" "$(cat "$OUT/${1}_running.txt")"
}

# ---- PT7: a Qdrant point missing on a running system -> refused at the next boot ------------
start; h="$(wait_healthy 900)"
pid="$(dx curl -fsS -X POST http://127.0.0.1:6333/collections/knowledge_chunks/points/scroll -H 'Content-Type: application/json' -d '{"limit":1}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["points"][0]["id"])')"
dx curl -fsS -X POST "http://127.0.0.1:6333/collections/knowledge_chunks/points/delete?wait=true" -H 'Content-Type: application/json' -d "{\"points\":[\"$pid\"]}" >/dev/null
running_problem PT7r "point missing on the running system" "has no points in Qdrant|Postgres vector_count"
docker restart -t 60 "$C" >/dev/null; sleep 60
docker logs --since 2m "$C" 2>&1 | grep -E "cas-storage: FAIL|\[bootstrap\].*FATAL" > "$OUT/PT7_missing_point.txt"
api7="$(dx pgrep -u cas -x api >/dev/null && echo running || echo not-started)"
echo "deleted point $pid; api=$api7" >> "$OUT/PT7_missing_point.txt"
[[ "$api7" == not-started ]] && grep -qE "Postgres vector_count|has no points in Qdrant" "$OUT/PT7_missing_point.txt" \
  && pass PT7 "Qdrant point count differs from Postgres vector_count: bootstrap refuses (fail-closed)" "$(head -1 "$OUT/PT7_missing_point.txt")" || fail PT7 "missing point" "$(cat "$OUT/PT7_missing_point.txt")"
# Repair: a new container restores Qdrant from the snapshot taken before the deletion.
docker stop -t 60 "$C" >/dev/null; start; h="$(wait_healthy 900)"
[[ "$h" == healthy* ]] && pass PT7b "new container restores Qdrant from the snapshot (taken before the deletion) and verifies it" "$h" || fail PT7b "restore after PT7" "$h"

# ---- PT7c: a Qdrant vector changed, same point count -> refused at the next boot -------------
cat > "$OUT/pt7c.py" <<'PT7C'
import json, sys
pt = json.load(sys.stdin)["result"]["points"][0]
v = pt["vector"]
(v[sorted(v)[0]] if isinstance(v, dict) else v)[0] += 0.5
print(json.dumps({"points": [{"id": pt["id"], "vector": v, "payload": pt["payload"]}]}))
PT7C
dx curl -fsS -X POST http://127.0.0.1:6333/collections/knowledge_chunks/points/scroll -H 'Content-Type: application/json' \
  -d '{"limit":1,"with_payload":true,"with_vector":true}' | python3 "$OUT/pt7c.py" > "$OUT/pt7c.json"
docker exec -i "$C" curl -fsS -X PUT "http://127.0.0.1:6333/collections/knowledge_chunks/points?wait=true" \
  -H 'Content-Type: application/json' --data-binary @- < "$OUT/pt7c.json" >/dev/null
running_problem PT7s "vector changed on the running system" "vector_sha256 differ from the fingerprint recorded in its baseline"
# PT7e: before the reboot, a new snapshot must not turn the change into the baseline.
ls7="dx bash -c 'ls /workspace/cas/qdrant/snapshots/knowledge_chunks; sha256sum /workspace/cas/qdrant/baselines/knowledge_chunks/*.json'"
before7e="$(eval "$ls7")"
docker exec "$C" cas-storage qdrant-snapshot > "$OUT/PT7e_snapshot.txt" 2>&1; rc7e=$?
after7e="$(eval "$ls7")"
{ echo "rc=$rc7e"; echo "--- before"; echo "$before7e"; echo "--- after"; echo "$after7e"; } >> "$OUT/PT7e_snapshot.txt"
[[ $rc7e -ne 0 && "$before7e" == "$after7e" && "$before7e" == *.json* ]] && grep -q "snapshot refused: READY generation .*vector_sha256 differ from the fingerprint recorded in its baseline" "$OUT/PT7e_snapshot.txt" \
  && pass PT7e "changed vector: a new snapshot is refused; snapshots and baselines unchanged" "$(grep -m1 FAIL "$OUT/PT7e_snapshot.txt")" || fail PT7e "snapshot after a changed vector" "$(tr '\n' ' ' < "$OUT/PT7e_snapshot.txt")"
n7c="$(dx curl -fsS -X POST http://127.0.0.1:6333/collections/knowledge_chunks/points/count -H 'Content-Type: application/json' -d '{"exact":true}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["count"])')"
docker restart -t 60 "$C" >/dev/null; sleep 60
docker logs --since 2m "$C" 2>&1 | grep -E "cas-storage: FAIL|\[bootstrap\].*FATAL" > "$OUT/PT7c_vector.txt"
api7c="$(dx pgrep -u cas -x api >/dev/null && echo running || echo not-started)"
echo "one vector changed, point count still ${n7c:-?}; api=$api7c" >> "$OUT/PT7c_vector.txt"
[[ "$api7c" == not-started ]] && grep -q "vector_sha256 differ from the fingerprint recorded" "$OUT/PT7c_vector.txt" \
  && pass PT7c "Qdrant vector changed with the same point count: bootstrap refuses (recorded fingerprint)" "$(head -1 "$OUT/PT7c_vector.txt")" || fail PT7c "changed vector" "$(cat "$OUT/PT7c_vector.txt")"
docker stop -t 60 "$C" >/dev/null; start; h="$(wait_healthy 900)"
[[ "$h" == healthy* ]] && pass PT7d "new container restores Qdrant from the snapshot and verifies it after PT7c" "$h" || fail PT7d "restore after PT7c" "$h"

# ---- PT8: quota exhausted -> read-only, not ready, writes fail; normal quota -> writable ---
# A new container needs a base backup before it opens, and the keeper takes none while
# the quota is exhausted: nginx never starts (readyz 000). On a running system it is 503.
docker stop -t 120 "$C" >/dev/null
start -e CAS_VOLUME_QUOTA_GB=0.001; sleep 75
docker logs "$C" 2>&1 | grep -E "READ-ONLY|volume free" | head -3 > "$OUT/PT8_quota.txt"
r8="$(code "$BASE/readyz")"
ro="$(dx gosu postgres psql -h /run/postgresql -d citizen_assistance -XAtc 'show default_transaction_read_only' 2>&1)"
w8="$(sql "insert into cas_persist_test(label, n) values ('pt8', 1)" | tail -1)"
echo "readyz=$r8 default_transaction_read_only=$ro insert: $w8" >> "$OUT/PT8_quota.txt"
docker stop -t 120 "$C" >/dev/null; start; h="$(wait_healthy 900)"
w8b="$(sql "insert into cas_persist_test(label, n) values ('pt8', 2)"; sql "select count(*) from cas_persist_test where label = 'pt8'")"
echo "normal quota: $h; insert + count: $w8b" >> "$OUT/PT8_quota.txt"
[[ "$r8" != 200 && "$ro" == on && "$w8" == *"read-only"* && "$h" == healthy* && "$w8b" == *1 ]] \
  && pass PT8 "volume quota exhausted: read-only + not ready (readyz $r8), write refused; normal quota: writable again" "PT8_quota.txt" || fail PT8 "quota guard" "$(tr '\n' ' ' < "$OUT/PT8_quota.txt")"
docker stop -t 120 "$C" >/dev/null

# ---- PT10: more container SIGKILLs during writes: the end-of-recovery checks never refuse a ----
# good volume, and every acknowledged commit survives each time.
start; h="$(wait_healthy 900)"
for i in 1 2 3; do
  writer_start "pt10_$i" 100000; sleep $(( 2 + 2 * i ))
  docker kill -s KILL "$C" >/dev/null; wait "$WPID" 2>/dev/null
  owner="$(docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'python3 -c "import json;print(json.load(open(\"/w/cas/locks/workspace.lease/owner.json\"))[\"owner_id\"])"')"
  docker rm -f "$C" >/dev/null; start; sleep 20
  docker exec -e CAS_PG_RECOVER_WAIT_S=40 "$C" cas-pg-recover "$owner" > "$OUT/PT10_$i.recover.txt" 2>&1
  docker restart -t 60 "$C" >/dev/null; h="$(wait_healthy 900)"
  docker logs "$C" 2>&1 | grep -E "durable LSN|recovery ended|FATAL" > "$OUT/PT10_$i.restore.txt"; echo "$h" >> "$OUT/PT10_$i.restore.txt"
  if [[ "$h" == healthy* ]] && grep -q "nothing replayable after the end" "$OUT/PT10_$i.restore.txt"; then
    acked_check "PT10_$i" "SIGKILL cycle $i: restore accepted by the end-of-recovery checks, every acknowledged commit present" "pt10_$i"
  else fail "PT10_$i" "SIGKILL cycle $i: restore" "$(tr '\n' ' ' < "$OUT/PT10_$i.restore.txt")"; fi
done
docker stop -t 120 "$C" >/dev/null

# ---- PT9: RunPod-like volume (bindfs: 0777 everywhere, chown EPERM) -----------------------------
docker volume create "$BVOL" >/dev/null
bind_start() {
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --name "$C" -p "127.0.0.1:$PORT:8088" --privileged -v "$BVOL:/backing" -v "$ROOT/models:/models:ro" "${ENVS[@]}" \
    --entrypoint bash "$IMG" -c 'set -e
      apt-get -qq update >/dev/null && apt-get -qq install -y bindfs fuse3 >/dev/null 2>&1
      mkdir -p /workspace && bindfs -o allow_other --perms=a+rwx --chmod-ignore --chown-deny /backing /workspace
      mkdir -p /workspace/cas/models && [ -e /workspace/cas/models/embedding ] || cp -r /models/embedding /workspace/cas/models/
      exec /usr/bin/tini -- /opt/cas/bin/entrypoint.sh' >/dev/null
}
bind_start; h="$(wait_healthy 1200)"
probe="$(dx bash -c 'd=/workspace/cas/pg/wal; m=$(stat -c %a $d); chown 10010 $d 2>/dev/null && c=ok || c=EPERM; echo "mode=$m chown=$c"')"
sql "create table if not exists cas_persist_test (label text not null, n int not null, at timestamptz default now(), primary key (label, n))" >/dev/null
su_sql "create table if not exists cas_persist_test (label text not null, n int not null, at timestamptz default now(), primary key (label, n)); grant all on cas_persist_test to cas" >/dev/null
writer_start pt9 300; wait "$WPID" 2>/dev/null
echo "$h $probe acked=$(grep -c . "$OUT/pt9.acked")" > "$OUT/PT9_bindfs.txt"
OWNER9="$(dx cat /run/cas-lease.owner)"
docker kill -s KILL "$C" >/dev/null
bind_start; sleep 20
docker exec -e CAS_PG_RECOVER_WAIT_S=40 "$C" cas-pg-recover "$OWNER9" >> "$OUT/PT9_bindfs.txt" 2>&1
docker restart -t 60 "$C" >/dev/null 2>&1
# The wrapper mounts bindfs again on restart (same command).
h2="$(wait_healthy 1200)"
echo "after SIGKILL + new container: $h2" >> "$OUT/PT9_bindfs.txt"
[[ "$h" == healthy* && "$probe" == "mode=777 chown=EPERM" && "$h2" == healthy* ]] \
  && pass PT9a "RunPod-like volume (0777, chown EPERM): boots, verifies, backs up; SIGKILL + new container restores" "$(tr '\n' ' ' < "$OUT/PT9_bindfs.txt" | cut -c1-200)" || fail PT9a "bindfs volume" "$(tr '\n' ' ' < "$OUT/PT9_bindfs.txt")"
acked_check PT9 "RunPod-like volume: every acknowledged commit present after SIGKILL and restore" pt9

echo; echo "=== results ($RES)"; column -t -s $'\t' "$RES"
! grep -q $'\tFAIL\t' "$RES"
