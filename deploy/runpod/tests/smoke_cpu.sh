#!/usr/bin/env bash
# CPU smoke test of the all-in-one image (CAS_SPEECH_MODE=off). Runs on the host.
# Uses only its own throwaway containers and volumes (prefix cas-aio-smoke-<run>),
# never the dev stack volumes. Writes logs + results.tsv to deploy/runpod/tests/out/<run>/.
#   deploy/runpod/tests/smoke_cpu.sh <image> [--keep]
# Each check records PASS or FAIL with evidence. A build is not an E2E pass:
# this script exercises the running container.
set -uo pipefail

IMG="${1:?usage: smoke_cpu.sh <image> [--keep]}"
KEEP="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RUN="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HERE/out/$RUN"
mkdir -p "$OUT"
P="cas-aio-smoke-$RUN"
C="$P-c"; C2="$P-c2"; VOL="$P-ws"; NEGVOL="$P-neg"
PORT="${CAS_SMOKE_PORT:-18088}"
BASE="http://127.0.0.1:$PORT"
RES="$OUT/results.tsv"
printf 'id\tresult\tcheck\tevidence\n' > "$RES"
exec > >(tee -a "$OUT/smoke.log") 2>&1

rnd() { openssl rand -hex "$1"; }
JWT="$(rnd 32)"; AIT="$(rnd 16)"; PGPW="$(rnd 16)"; MU="casminio$(rnd 4)"; MP="$(rnd 16)"
ADMIN_EMAIL=admin@chuse.vn; ADMIN_PW="Smoke-$(rnd 8)"
ENVS=(-e CAS_SPEECH_MODE=off -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT" -e POSTGRES_PASSWORD="$PGPW"
      -e MINIO_ROOT_USER="$MU" -e MINIO_ROOT_PASSWORD="$MP" -e LLM_PROVIDER=mock -e CAS_ALLOW_MOCK_LLM=true
      -e GEMINI_API_KEY="smoke-unused-$(rnd 12)")
MODELS=(-v "$ROOT/models:/workspace/cas/models:ro")

# Evidence is one TSV cell: tabs, CR and newlines become spaces.
tsv() { printf '%s' "$1" | tr '\t\r\n' '   '; }
pass() { printf '%s\tPASS\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "PASS $1 $2"; }
fail() { printf '%s\tFAIL\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "FAIL $1 $2 :: ${3:-}"; }
expect() { local id="$1" name="$2"; shift 2; if "$@"; then pass "$id" "$name"; else fail "$id" "$name" "command failed: $*"; fi; }
dx() { docker exec "$C" "$@"; }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@"; }
sup() { dx supervisorctl -c /etc/cas/supervisord.conf status "$@" 2>/dev/null; }
state() { sup "$1" | awk '{print $2}'; }

wait_healthy() {  # wait_healthy <timeout_s>
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    if dx healthcheck.sh --quiet >/dev/null 2>&1; then echo "healthy after $((SECONDS - t0))s"; return 0; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" != true ]]; then echo "container exited"; return 1; fi
    sleep 3
  done
  echo "not healthy after $1s"; return 1
}
cleanup() {
  docker logs "$C" > "$OUT/container.log" 2>&1 || true
  docker logs "$C2" > "$OUT/container2.log" 2>&1 || true
  if [[ "$KEEP" != --keep ]]; then
    docker rm -f "$C" "$C2" >/dev/null 2>&1 || true
    docker volume rm "$VOL" "$NEGVOL" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "image=$IMG run=$RUN arch=$(docker image inspect -f '{{.Architecture}}' "$IMG")"
docker image inspect -f '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMG" | tee "$OUT/image.txt"

# ---- N: configuration refusals (fail fast, no service started) -----------------
neg() {  # neg <id> <name> <expected log text> <docker run args...>
  local id="$1" name="$2" want="$3"; shift 3
  local out rc
  out="$(docker run --rm "$@" "$IMG" 2>&1)"; rc=$?
  echo "$out" > "$OUT/$id.log"
  if [[ $rc -eq 64 ]] && grep -q "$want" <<<"$out"; then pass "$id" "$name" "exit=$rc"; else fail "$id" "$name" "exit=$rc; see $id.log"; fi
}
neg N1 "missing CAS_SPEECH_MODE is refused" "CAS_SPEECH_MODE must be" \
  -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT" -e POSTGRES_PASSWORD="$PGPW" -e MINIO_ROOT_USER="$MU" -e MINIO_ROOT_PASSWORD="$MP"
neg N2 "mode=off with SPEECH_ENABLED=true is refused" "conflicting configuration" "${ENVS[@]}" -e SPEECH_ENABLED=true
neg N3 "placeholder JWT secret is refused" "JWT_SECRET looks like a placeholder" "${ENVS[@]}" -e JWT_SECRET=local-dev-jwt-secret-change-me-0123456789
neg N4 "missing POSTGRES_PASSWORD is refused" "POSTGRES_PASSWORD is required" \
  -e CAS_SPEECH_MODE=off -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT" -e MINIO_ROOT_USER="$MU" -e MINIO_ROOT_PASSWORD="$MP"
neg N6 "missing LLM_PROVIDER is refused (no silent default under APP_ENV=prod)" "LLM_PROVIDER is required" \
  -e CAS_SPEECH_MODE=off -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT" -e POSTGRES_PASSWORD="$PGPW" -e MINIO_ROOT_USER="$MU" -e MINIO_ROOT_PASSWORD="$MP"
neg N7 "LLM_PROVIDER=mock without CAS_ALLOW_MOCK_LLM is refused" "LLM_PROVIDER=mock is for tests only" \
  -e CAS_SPEECH_MODE=off -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT" -e POSTGRES_PASSWORD="$PGPW" -e MINIO_ROOT_USER="$MU" -e MINIO_ROOT_PASSWORD="$MP" -e LLM_PROVIDER=mock
docker volume create "$NEGVOL" >/dev/null
neg N5 "mode=gpu without checkpoints fails closed" "gpu mode: C0_INFERENCE_CONFIG not found" \
  -e CAS_SPEECH_MODE=gpu -e JWT_SECRET="$JWT" -e AI_SERVICE_TOKEN="$AIT" -e POSTGRES_PASSWORD="$PGPW" \
  -e MINIO_ROOT_USER="$MU" -e MINIO_ROOT_PASSWORD="$MP" -e SPEECH_SERVICE_TOKEN="$(rnd 16)" \
  -e LLM_PROVIDER=mock -e CAS_ALLOW_MOCK_LLM=true -v "$NEGVOL:/workspace" "${MODELS[@]}"
neg N8 "non-numeric SPEECH_TIMEOUT_MS is refused" "SPEECH_TIMEOUT_MS must be an integer" "${ENVS[@]}" -e SPEECH_TIMEOUT_MS=90s
# N9: a model that does not match its manifest stops the boot (full SHA-256 check).
BADM="$OUT/badmodel"; mkdir -p "$BADM"; cp "$ROOT/models/embedding/manifest.json" "$ROOT/models/embedding/tokenizer.json" "$BADM/"
printf 'not the model' > "$BADM/model.onnx"
neg N9 "embedding model with a wrong checksum is refused before any init" "does not match its manifest" \
  "${ENVS[@]}" -v "$BADM:/workspace/cas/models/embedding:ro"
rm -rf "$BADM"
# Prove N5 stopped before init work (no pgdata/CA written on that volume).
if docker run --rm -v "$NEGVOL:/w" --entrypoint bash "$IMG" -c '[ ! -e /w/cas/pgdata/PG_VERSION ] && [ ! -e /w/cas/tls/ca.crt ]'; then
  pass N5b "gpu refusal happened before initdb and CA creation"
else fail N5b "gpu refusal happened before initdb and CA creation" "pgdata or CA exists"; fi

# ---- S: first start on an empty volume -----------------------------------------
docker volume create "$VOL" >/dev/null
docker run -d --name "$C" -p "127.0.0.1:$PORT:8088" -v "$VOL:/workspace" "${MODELS[@]}" "${ENVS[@]}" "$IMG" >/dev/null
h="$(wait_healthy 900)"; echo "$h"
if [[ "$h" == healthy* ]]; then pass S1 "fresh volume reaches healthy" "$h"; else fail S1 "fresh volume reaches healthy" "$h"; docker logs "$C" > "$OUT/container.log" 2>&1; exit 1; fi
dx healthcheck.sh > "$OUT/S1_healthcheck.txt" 2>&1
docker logs "$C" > "$OUT/S1_boot.log" 2>&1

# Startup order from the log: postgres ready -> migrations -> bucket -> marker -> apps.
order_ok=1; prev=0
for pat in "\[bootstrap\].*postgres ready" "\[bootstrap\].*minio ready" "migrations applied up to" "bucket cas-documents created" "bootstrap complete" "wait-for\[api\]: ready" "wait-for\[nginx\]: ready"; do
  n="$(grep -n -m1 -E "$pat" "$OUT/S1_boot.log" | cut -d: -f1)"
  echo "order: '$pat' at line ${n:-MISSING}"
  if [[ -z "$n" || "$n" -le "$prev" ]]; then order_ok=0; fi; prev="${n:-$prev}"
done
grep -E "\[bootstrap\]|wait-for\[(api|nginx|ai-service|uvicorn|index-reaper)" "$OUT/S1_boot.log" > "$OUT/S2_order.txt"
# ai-service waits for the marker too (its wait-for line is named after uvicorn).
m="$(grep -n -m1 "bootstrap complete" "$OUT/S1_boot.log" | cut -d: -f1)"
a="$(grep -n -m1 "wait-for\[uvicorn\]: ready" "$OUT/S1_boot.log" | cut -d: -f1)"
[[ -n "$a" && -n "$m" && "$a" -gt "$m" ]] || order_ok=0
if [[ $order_ok == 1 ]]; then pass S2 "startup order: data ready -> migrate -> bucket -> marker -> apps" "S2_order.txt"; else fail S2 "startup order" "S2_order.txt"; fi

sup > "$OUT/S3_supervisor.txt"
s3=1
for p in pg-lease postgres qdrant minio ai-service api index-reaper nginx; do [[ "$(state "$p")" == RUNNING ]] || s3=0; done
# supervisorctl status does not print the exit code; supervisord logs it.
grep -q "exited: bootstrap (exit status 0; expected)" "$OUT/S1_boot.log" || s3=0
grep "exited: bootstrap" "$OUT/S1_boot.log" >> "$OUT/S3_supervisor.txt"
[[ "$(state bootstrap)" == EXITED ]] || s3=0
[[ "$(state speech)" == STOPPED ]] || s3=0
if [[ $s3 == 1 ]]; then pass S3 "long-running programs RUNNING, bootstrap EXITED, speech STOPPED (mode=off)" "S3_supervisor.txt"; else fail S3 "supervisor states" "S3_supervisor.txt"; fi

dx ss -ltnH > "$OUT/S4_listeners.txt"
public="$(awk '{print $4}' "$OUT/S4_listeners.txt" | grep -v -E '^(127\.0\.0\.1|\[::1\]):' | sort -u | tr '\n' ' ')"
if [[ "$public" == "0.0.0.0:8088 " ]]; then pass S4 "only nginx 8088 listens beyond loopback" "$public"; else fail S4 "only 8088 public" "non-loopback: $public"; fi

dx ps -eo user:10,comm --no-headers | sort | uniq -c > "$OUT/S5_process_users.txt"
dx ps -eo user:10,pid,ppid,args --no-headers > "$OUT/S5_process_args.txt"
s5=1
for pair in "postgres postgres" "qdrant qdrant" "minio minio" "cas api" "cas index-reaper" "cas uvicorn" "web nginx"; do
  grep -qE "^ *[0-9]+ ${pair% *} +${pair#* }" "$OUT/S5_process_users.txt" || { echo "missing: $pair"; s5=0; }
done
roots="$(awk '$2=="root"{print $3}' "$OUT/S5_process_users.txt" | sort -u | tr '\n' ' ')"
echo "root processes: $roots"
# Root by design: tini, supervisord, and pg-lease (bash pg-lease.sh and its sleep child).
lease_pid="$(awk '$1=="root" && $0 ~ /pg-lease\.sh/ {print $2}' "$OUT/S5_process_args.txt")"
while read -r u pid ppid args; do
  [[ "$u" == root ]] || continue
  case "$args" in
    /usr/bin/tini*|*supervisord*|"ps -eo"*) ;;
    *pg-lease.sh*) ;;
    sleep*) [[ -n "$lease_pid" && "$ppid" == "$lease_pid" ]] || { s5=0; echo "unexpected root sleep: $pid $args"; } ;;
    *) s5=0; echo "unexpected root process: $pid $args" ;;
  esac
done < "$OUT/S5_process_args.txt"
if [[ $s5 == 1 ]]; then pass S5 "each service runs as its own user; root only for tini, supervisord, pg-lease" "$roots"; else fail S5 "process users" "S5_process_args.txt"; fi

# ---- E: per-service environment (svc-run: env -i + allowlist) ----------------------
# Read /proc/<pid>/environ as the process owner (root in docker exec lacks CAP_SYS_PTRACE).
envnames() {  # envnames <user> <comm> -> sorted variable names of the oldest such process
  local pid; pid="$(dx pgrep -o -u "$1" -x "$2")" || { echo "NO_PROCESS"; return; }
  dx gosu "$1" cat "/proc/$pid/environ" | tr '\0' '\n' | cut -d= -f1 | sort | tr '\n' ' '
}
SECRETS="DATABASE_URL JWT_SECRET POSTGRES_PASSWORD MINIO_ROOT_PASSWORD OBJECT_STORAGE_SECRET_KEY AI_SERVICE_TOKEN GEMINI_API_KEY SPEECH_SERVICE_TOKEN"
e1=1; : > "$OUT/E1_env.txt"
check_env() {  # check_env <label> <user> <comm> <allowed secrets...>
  local label="$1" user="$2" comm="$3"; shift 3
  local names; names="$(envnames "$user" "$comm")"
  echo "$label ($user/$comm): $names" >> "$OUT/E1_env.txt"
  [[ "$names" != NO_PROCESS ]] || { e1=0; echo "  -> process not found" >> "$OUT/E1_env.txt"; return; }
  local sname
  for sname in $SECRETS; do
    if [[ " $names " == *" $sname "* ]] && [[ " $* " != *" $sname "* ]]; then e1=0; echo "  -> LEAK: $sname" >> "$OUT/E1_env.txt"; fi
    if [[ " $* " == *" $sname "* ]] && [[ " $names " != *" $sname "* ]]; then e1=0; echo "  -> MISSING: $sname" >> "$OUT/E1_env.txt"; fi
  done
}
check_env nginx web nginx
check_env postgres postgres postgres
check_env qdrant qdrant qdrant
check_env minio minio minio MINIO_ROOT_PASSWORD
check_env index-reaper cas index-reaper DATABASE_URL
check_env api cas api DATABASE_URL JWT_SECRET OBJECT_STORAGE_SECRET_KEY AI_SERVICE_TOKEN
check_env ai-service cas uvicorn DATABASE_URL OBJECT_STORAGE_SECRET_KEY AI_SERVICE_TOKEN GEMINI_API_KEY
lease_pid="$(dx pgrep -o -f pg-lease.sh)"; ln="$(dx cat "/proc/$lease_pid/environ" | tr '\0' '\n' | cut -d= -f1 | sort | tr '\n' ' ')"
echo "pg-lease (root): $ln" >> "$OUT/E1_env.txt"
for sname in $SECRETS; do [[ " $ln " != *" $sname "* ]] || { e1=0; echo "  -> LEAK in pg-lease: $sname" >> "$OUT/E1_env.txt"; }; done
# The allowlist files themselves (Postgres rewrites its environ area for the process
# title, so for it the file is the reliable evidence).
dx bash -c 'for f in /run/cas-env/*.env; do echo "$(basename "$f" .env): $(cut -d= -f1 "$f" | tr "\n" " ")"; done' > "$OUT/E1_envfiles.txt"
dx stat -c '%U %a %n' /run/cas-env /run/cas-env/api.env >> "$OUT/E1_envfiles.txt"
for svc in nginx postgres qdrant pg-lease; do
  line="$(grep "^$svc:" "$OUT/E1_envfiles.txt")"
  for sname in $SECRETS; do [[ " $line " != *" $sname "* ]] || { e1=0; echo "  -> LEAK in $svc.env: $sname" >> "$OUT/E1_env.txt"; }; done
done
grep -q "^root 700 /run/cas-env$" "$OUT/E1_envfiles.txt" && grep -q "^root 600 /run/cas-env/api.env$" "$OUT/E1_envfiles.txt" || { e1=0; echo "  -> env files not root-only" >> "$OUT/E1_env.txt"; }
[[ $e1 == 1 ]] && pass E1 "each program sees only its own variables (no secret of another service); env files root-only" "E1_env.txt, E1_envfiles.txt" || fail E1 "per-service environment" "E1_env.txt"

# ---- T: TLS ---------------------------------------------------------------------
CA=/workspace/cas/tls/ca.crt
DSN_V="postgresql://cas:$PGPW@127.0.0.1:5432/citizen_assistance?sslmode=verify-full&sslrootcert=$CA"
o="$(dx gosu cas psql "$DSN_V" -tAc "select ssl, version from pg_stat_ssl where pid = pg_backend_pid()" 2>&1)"; echo "$o" > "$OUT/T1.txt"
[[ "$o" == t\|TLS* ]] && pass T1 "Postgres verify-full against internal CA" "$o" || fail T1 "Postgres verify-full" "$o"
o="$(dx gosu cas psql "postgresql://cas:$PGPW@127.0.0.1:5432/citizen_assistance?sslmode=disable" -tAc "select 1" 2>&1)"; echo "$o" > "$OUT/T2.txt"
grep -q "no encryption" <<<"$o" && pass T2 "Postgres rejects non-TLS TCP connections" "${o:0:120}" || fail T2 "Postgres rejects non-TLS" "$o"
o="$(dx gosu cas psql "$DSN_V" -tAc "select count(*) filter (where s.ssl), count(*) from pg_stat_activity a join pg_stat_ssl s using (pid) where a.usename = 'cas' and a.pid <> pg_backend_pid()" 2>&1)"; echo "$o" > "$OUT/T3.txt"
if [[ "$o" =~ ^([0-9]+)\|([0-9]+)$ && "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" && "${BASH_REMATCH[2]}" -gt 0 ]]; then
  pass T3 "all app connections to Postgres use TLS" "ssl|total=$o"; else fail T3 "app connections use TLS" "$o"; fi
o1="$(dx curl -fsS --cacert "$CA" https://127.0.0.1:9000/minio/health/ready -o /dev/null -w '%{http_code}' 2>&1)"
CERTIFI="$(dx /opt/cas/venv-ai/bin/python -c 'import certifi;print(certifi.where())')"
# --capath /nonexistent: otherwise curl also searches /etc/ssl/certs, where the internal CA is installed.
o2="$(dx curl -sS --cacert "$CERTIFI" --capath /nonexistent https://127.0.0.1:9000/minio/health/ready -o /dev/null 2>&1)"
echo "with CA: $o1 | with public bundle only: $o2" > "$OUT/T4.txt"
[[ "$o1" == 200 ]] && grep -qi "certificate" <<<"$o2" && pass T4 "MinIO HTTPS verifies with internal CA, fails with public bundle" "$o1 / ${o2:0:80}" || fail T4 "MinIO TLS verification" "$(cat "$OUT/T4.txt")"
PYCHK='import os
from minio import Minio
c=Minio("127.0.0.1:9000",access_key=os.environ["OBJECT_STORAGE_ACCESS_KEY"],secret_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],secure=True,region="us-east-1")
print("bucket_exists", c.bucket_exists("cas-documents"))'
o_env="$(dx cas-run cas /opt/cas/venv-ai/bin/python -c "$PYCHK" 2>&1 | tail -1)"
o_noenv="$(dx cas-run cas env -u SSL_CERT_FILE /opt/cas/venv-ai/bin/python -c "$PYCHK" 2>&1 | tail -1)"
printf 'SSL_CERT_FILE set: %s\nSSL_CERT_FILE unset: %s\n' "$o_env" "$o_noenv" > "$OUT/T5_minio_py_trust.txt"
if [[ "$o_env" == "bucket_exists True" && "$o_noenv" == *CERTIFICATE_VERIFY_FAILED* ]]; then pass T5 "minio-py trusts the internal CA only through SSL_CERT_FILE (unset -> certifi -> verify failed)" "$(tr '\n' ';' < "$OUT/T5_minio_py_trust.txt")"; else fail T5 "minio-py TLS" "$(tr '\n' ';' < "$OUT/T5_minio_py_trust.txt")"; fi

# T6: apps use verify-full, and verify-full really checks the chain (wrong CA -> refused).
api_pid="$(dx pgrep -u cas -x api)"; ai_pid="$(dx pgrep -u cas -f 'uvicorn app.main' | head -1)"
# Read as user cas (owner of both processes): root in docker exec lacks CAP_SYS_PTRACE.
m1="$(dx gosu cas bash -c "tr '\\0' '\\n' < /proc/$api_pid/environ | grep -o 'sslmode=[a-z-]*'" 2>&1)"
m2="$(dx gosu cas bash -c "tr '\\0' '\\n' < /proc/$ai_pid/environ | grep -o 'sslmode=[a-z-]*'" 2>&1)"
wrong="$(dx gosu cas psql "postgresql://cas:$PGPW@127.0.0.1:5432/citizen_assistance?sslmode=verify-full&sslrootcert=$CERTIFI" -tAc 'select 1' 2>&1)"
printf 'api DATABASE_URL %s\nai-service DATABASE_URL %s\nverify-full with public CA bundle: %s\n' "$m1" "$m2" "$wrong" > "$OUT/T6_verify_full.txt"
if [[ "$m1" == sslmode=verify-full && "$m2" == sslmode=verify-full ]] && grep -qi "certificate verify failed" <<<"$wrong"; then
  pass T6 "Go API and ai-service connect with sslmode=verify-full; a wrong CA is refused" "T6_verify_full.txt"
else fail T6 "verify-full" "T6_verify_full.txt"; fi

# ---- K: key and directory permissions (read as the real service users) ----------
k() { dx gosu "$1" cat "$2" >/dev/null 2>&1; }
kp=1
k postgres /workspace/cas/tls/postgres/server.key || { kp=0; echo "postgres cannot read its key"; }
k minio /workspace/cas/tls/minio/private.key || { kp=0; echo "minio cannot read its key"; }
for u in cas postgres minio qdrant speech web; do k "$u" /workspace/cas/tls/ca/ca.key && { kp=0; echo "$u CAN read ca.key"; }; done
k cas /workspace/cas/tls/postgres/server.key && { kp=0; echo "cas can read postgres key"; }
k postgres /workspace/cas/tls/minio/private.key && { kp=0; echo "postgres can read minio key"; }
k cas /workspace/cas/tls/ca.crt || { kp=0; echo "cas cannot read ca.crt"; }
dx bash -c 'ls -ld /workspace/cas /workspace/cas/tls /workspace/cas/tls/* /workspace/cas/tls/*/* /run/cas /run/cas/*' > "$OUT/K1_perms.txt" 2>&1
if [[ $kp == 1 ]]; then pass K1 "TLS keys readable only by their service user; ca.key root-only" "K1_perms.txt"; else fail K1 "TLS key permissions" "K1_perms.txt"; fi
o="$(dx stat -c '%U %a' /run/cas /run/cas/bootstrap.ok | tr '\n' ' ')"
[[ "$o" == "cas 755 cas 644 " ]] && pass K2 "/run/cas owned by cas; bootstrap.ok written by cas" "$o" || fail K2 "/run/cas ownership" "$o"

# ---- A: API through nginx ---------------------------------------------------------
o="$(code "$BASE/healthz") $(code "$BASE/readyz") $(code "$BASE/") $(code "$BASE/api/v1/admin/documents") $(code "$BASE/swagger/index.html")"
echo "$o" > "$OUT/A1.txt"
[[ "$o" == "200 200 200 401 404" ]] && pass A1 "healthz/readyz/web 200, admin without token 401, swagger 404" "$o" || fail A1 "public endpoints" "$o"
o="$(docker exec -e BOOTSTRAP_ADMIN_EMAIL="$ADMIN_EMAIL" -e BOOTSTRAP_ADMIN_PASSWORD="$ADMIN_PW" "$C" cas-run cas /opt/cas/api/bootstrap-admin 2>&1)"; echo "$o" > "$OUT/A2_bootstrap_admin.txt"
TOKEN="$(curl -s "$BASE/api/v1/auth/login" -H 'Content-Type: application/json' -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PW\"}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("data") or {}).get("access_token") or (d.get("data") or {}).get("token") or "")' 2>/dev/null)"
[[ -n "$TOKEN" ]] && pass A2 "bootstrap-admin + admin login through nginx" || fail A2 "admin login" "A2_bootstrap_admin.txt"

mkpdf() {  # mkpdf <file> <text>: one-page text PDF; distinct text -> distinct checksum
python3 - "$1" "$2" <<'PY'
import sys
body = b"BT /F1 18 Tf 72 720 Td (" + sys.argv[2].encode() + b") Tj ET"
objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
out, offs = bytearray(b"%PDF-1.4\n"), []
for i, o in enumerate(objs, 1):
    offs.append(len(out)); out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
x = len(out); out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
for o in offs: out += b"%010d 00000 n \n" % o
out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
open(sys.argv[1], "wb").write(out)
PY
}
mkpdf "$OUT/smoke.pdf" "Smoke test $RUN: dang ky khai sinh"
for n in 2 3 4; do mkpdf "$OUT/smoke$n.pdf" "Smoke test $RUN upload $n"; done
up="$(curl -s "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/smoke.pdf;type=application/pdf" \
  -F title="Smoke PDF" -F domain_id=ho_tich_chung_thuc)"; echo "$up" > "$OUT/A3_upload.json"
DOC="$(python3 -c 'import json,sys;print((json.load(sys.stdin).get("data") or {}).get("id",""))' <<<"$up" 2>/dev/null)"
if [[ -n "$DOC" ]]; then
  curl -s "$BASE/api/v1/admin/documents/$DOC/content" -H "Authorization: Bearer $TOKEN" -o "$OUT/A3_download.pdf"
  if cmp -s "$OUT/smoke.pdf" "$OUT/A3_download.pdf"; then pass A3 "Go upload to MinIO over TLS and download, bytes identical" "doc=$DOC"; else fail A3 "Go download bytes" "doc=$DOC"; fi
else fail A3 "Go upload" "A3_upload.json"; fi
URI="$(dx gosu cas psql "$DSN_V" -tAc "select storage_uri from documents where id = '$DOC'" 2>/dev/null | tr -d '[:space:]')"
KEY="${URI#s3://cas-documents/}"; KEY="${KEY#cas-documents/}"
o="$(dx cas-run cas /opt/cas/venv-ai/bin/python -c "
import os,hashlib
from minio import Minio
c=Minio('127.0.0.1:9000',access_key=os.environ['OBJECT_STORAGE_ACCESS_KEY'],secret_key=os.environ['OBJECT_STORAGE_SECRET_KEY'],secure=True,region='us-east-1')
print(hashlib.sha256(c.get_object('cas-documents','$KEY').read()).hexdigest())" 2>&1 | tail -1)"
want="$(shasum -a 256 "$OUT/smoke.pdf" | cut -d' ' -f1)"
[[ -n "$KEY" && "$o" == "$want" ]] && pass A4 "Python (minio-py, TLS) reads the object Go wrote" "uri=$URI" || fail A4 "Python read object" "key=$KEY got=$o want=$want"

sess="$(curl -s "$BASE/api/v1/sessions" -H 'Content-Type: application/json' -d '{"xa_id":"xa_chu_se"}')"; echo "$sess" > "$OUT/A5_session.json"
SID="$(python3 -c 'import json,sys;print((json.load(sys.stdin).get("data") or {}).get("id",""))' <<<"$sess" 2>/dev/null)"
GT="$(python3 -c 'import json,sys;print((json.load(sys.stdin).get("data") or {}).get("guest_token",""))' <<<"$sess" 2>/dev/null)"
turn() { curl -s -o "$OUT/$1" -w '%{http_code}' --max-time 30 "$BASE/api/v1/sessions/$SID/turns" -H 'Content-Type: application/json' \
  -H "X-Guest-Token: $GT" -H "X-Request-ID: $(uuidgen | tr '[:upper:]' '[:lower:]')" -d "{\"message\":\"$2\"}"; }
c5="$(turn A5_turn.json "toi muon dang ky khai sinh")"
[[ -n "$SID" && "$c5" == 200 ]] && pass A5 "guest session + chat turn through nginx (Go -> ai-service mock LLM -> DB)" "session=$SID" || fail A5 "chat turn" "http=$c5 session=$SID"

# A6/A7: upload limits. The backend limit (DOCUMENT_MAX_BYTES=20 MiB) answers with a JSON
# PAYLOAD_TOO_LARGE; nginx (65 MiB = 64 MiB hard max + multipart overhead) only cuts
# bodies the backend could never accept, even if DOCUMENT_MAX_BYTES is raised.
python3 -c 'import sys;f=open(sys.argv[1],"wb");f.write(b"%PDF-1.4\n");f.write(b"0"*(21*1024*1024));f.close()' "$OUT/big21.pdf"
python3 -c 'import sys;f=open(sys.argv[1],"wb");f.write(b"%PDF-1.4\n");f.write(b"0"*(66*1024*1024));f.close()' "$OUT/big66.pdf"
c6="$(curl -s -o "$OUT/A6_21mib.json" -w '%{http_code}' "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/big21.pdf;type=application/pdf" -F title="21 MiB" -F domain_id=ho_tich_chung_thuc)"
c7="$(curl -s -o "$OUT/A7_66mib.txt" -w '%{http_code}' "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/big66.pdf;type=application/pdf" -F title="66 MiB" -F domain_id=ho_tich_chung_thuc)"
rm -f "$OUT/big21.pdf" "$OUT/big66.pdf"
[[ "$c6" == 413 ]] && grep -q PAYLOAD_TOO_LARGE "$OUT/A6_21mib.json" && pass A6 "21 MiB PDF: 413 PAYLOAD_TOO_LARGE from the API (DOCUMENT_MAX_BYTES)" || fail A6 "21 MiB upload" "http=$c6 $(head -c 200 "$OUT/A6_21mib.json")"
[[ "$c7" == 413 ]] && grep -qi "nginx\|413 Request Entity Too Large" "$OUT/A7_66mib.txt" && pass A7 "66 MiB body: 413 from nginx (client_max_body_size 65m)" || fail A7 "66 MiB upload" "http=$c7"

# I1: scanned (image-only) PDF through the real pipeline: upload -> link -> async index
# (Tesseract vie OCR -> e5 ONNX embedding -> Qdrant) -> READY.
FONT=""; for f in "/System/Library/Fonts/Supplemental/Arial Unicode.ttf" /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf; do [[ -f "$f" ]] && { FONT="$f"; break; }; done
if [[ -z "$FONT" ]]; then fail I1 "scanned PDF index" "no Unicode TTF font on the host to render the scan"
else
  docker cp "$FONT" "$C:/tmp/scan-font.ttf"
  docker exec -i "$C" /opt/cas/venv-ai/bin/python - "$RUN" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont
run = sys.argv[1]
lines = ["THỦ TỤC ĐĂNG KÝ KHAI SINH", f"Mã kiểm thử {run}",
         "Người đi đăng ký khai sinh nộp tờ khai theo mẫu", "và giấy chứng sinh cho cơ quan đăng ký hộ tịch.",
         "Trường hợp không có giấy chứng sinh thì nộp văn bản", "của người làm chứng xác nhận về việc sinh.",
         "Thời hạn giải quyết: ngay trong ngày làm việc."]
img = Image.new("L", (1654, 2339), 255)
d = ImageDraw.Draw(img); f = ImageFont.truetype("/tmp/scan-font.ttf", 46); y = 180
for line in lines:
    d.text((130, y), line, font=f, fill=0); y += 90
img.save("/tmp/scan.pdf", "PDF", resolution=200.0)
PY
  docker cp "$C:/tmp/scan.pdf" "$OUT/scan.pdf"
  up="$(curl -s "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/scan.pdf;type=application/pdf" -F title="Scan khai sinh $RUN" -F domain_id=ho_tich_chung_thuc)"
  SDOC="$(python3 -c 'import json,sys;print((json.load(sys.stdin).get("data") or {}).get("id",""))' <<<"$up" 2>/dev/null)"
  echo "$up" > "$OUT/I1_upload.json"
  VER="$(curl -s "$BASE/api/v1/admin/documents/link-targets" -H "Authorization: Bearer $TOKEN" \
    | python3 -c 'import json,sys;it=json.load(sys.stdin)["data"]["items"];print(it[0]["procedure_version_id"] if it else "")')"
  lc="$(curl -s -o "$OUT/I1_link.json" -w '%{http_code}' "$BASE/api/v1/admin/documents/$SDOC/links" -H "Authorization: Bearer $TOKEN" \
    -H "X-Request-ID: $(uuidgen | tr '[:upper:]' '[:lower:]')" -H 'Content-Type: application/json' -d "{\"procedure_version_id\":\"$VER\",\"relationship_type\":\"SOURCE\"}")"
  ic="$(curl -s -o "$OUT/I1_index.json" -w '%{http_code}' -X POST "$BASE/api/v1/admin/documents/$SDOC/links/$VER/index" -H "Authorization: Bearer $TOKEN" \
    -H "X-Request-ID: $(uuidgen | tr '[:upper:]' '[:lower:]')")"
  st=""; t0=$SECONDS
  while [[ -n "$SDOC" ]] && (( SECONDS - t0 < 600 )); do
    curl -s "$BASE/api/v1/admin/documents/$SDOC/links" -H "Authorization: Bearer $TOKEN" > "$OUT/I1_links.json"
    st="$(python3 -c 'import json,sys;it=json.load(open(sys.argv[1]))["data"]["items"];print(it[0]["index_status"] if it else "")' "$OUT/I1_links.json" 2>/dev/null)"
    [[ "$st" == READY || "$st" == FAILED ]] && break; sleep 3
  done
  summ="$(python3 -c 'import json,sys;l=json.load(open(sys.argv[1]))["data"]["items"][0];print("status=%s pages=%s ocr_pages=%s chunks=%s model=%s err=%s" % (l.get("index_status"),l.get("page_count"),l.get("ocr_page_count"),l.get("chunk_count"),l.get("embedding_model_id"),l.get("last_error_code")))' "$OUT/I1_links.json" 2>/dev/null)"
  pts="$(dx curl -fsS -X POST http://127.0.0.1:6333/collections/knowledge_chunks/points/count -H 'Content-Type: application/json' -d '{"exact":true}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["count"])' 2>/dev/null)"
  echo "link=$lc index=$ic after $((SECONDS - t0))s: $summ qdrant_points=$pts" > "$OUT/I1_index_summary.txt"
  if [[ "$lc" =~ ^20 && "$ic" =~ ^20 && "$summ" == status=READY* && "$summ" != *"ocr_pages=0 "* && "$summ" != *"ocr_pages=None"* && "${pts:-0}" -gt 0 ]]; then
    pass I1 "scanned PDF indexed: Tesseract OCR + e5 embedding + Qdrant, link READY" "$(cat "$OUT/I1_index_summary.txt")"
  else fail I1 "scanned PDF index" "$(cat "$OUT/I1_index_summary.txt")"; fi
fi

# ---- R: dependency loss while running ---------------------------------------------
db_count() { dx gosu cas psql "$DSN_V" -tAc "select count(*) from conversation_sessions" 2>/dev/null | tr -d '[:space:]'; }
new_session() { code "$BASE/api/v1/sessions" -H 'Content-Type: application/json' -d '{"xa_id":"xa_chu_se"}'; }
before="$(db_count)"
# R1: crash (kill -9 postmaster) -> supervisord restarts it by itself.
pid1="$(dx cat /workspace/cas/pgdata/postmaster.pid | head -1)"
dx kill -9 "$pid1"
codes=""; for i in $(seq 1 40); do codes+="$(code "$BASE/readyz") "; sleep 0.5; done
sleep 5
pid2="$(dx cat /workspace/cas/pgdata/postmaster.pid 2>/dev/null | head -1)"
st="$(state postgres)"
rc="$(new_session)"; after="$(db_count)"
printf 'old_pid=%s new_pid=%s state=%s\nreadyz during crash: %s\ncreate session after recovery: %s\nsessions before=%s after=%s\nbootstrap: %s\n' \
  "$pid1" "$pid2" "$st" "$codes" "$rc" "$before" "$after" "$(sup bootstrap)" > "$OUT/R1_crash.txt"
if [[ "$st" == RUNNING && -n "$pid2" && "$pid2" != "$pid1" && "$rc" == 201 && "$after" == $((before + 1)) && "$(state bootstrap)" == EXITED ]]; then
  pass R1 "Postgres kill -9: supervisord restarts it, API recovers without restart" "R1_crash.txt"; else fail R1 "Postgres crash recovery" "R1_crash.txt"; fi
grep -q " 503 " <<<" $codes " && pass R1b "readyz reported 503 during the crash window" || fail R1b "readyz 503 during crash window (window may be shorter than 0.5 s polling)" "$codes"

# R2: manual stop -> stays down until started; readiness and writes fail clearly.
api_pid="$(dx pgrep -u cas -x api)"
dx supervisorctl -c /etc/cas/supervisord.conf stop postgres >/dev/null
sleep 20
st_down="$(state postgres)"; r_ready="$(code "$BASE/readyz")"; r_health="$(code "$BASE/healthz")"
r_write="$(curl -s -o "$OUT/R2_write.json" -w '%{http_code}' "$BASE/api/v1/sessions" -H 'Content-Type: application/json' -d '{"xa_id":"xa_chu_se"}')"
dx supervisorctl -c /etc/cas/supervisord.conf start postgres >/dev/null
ok=""; for i in $(seq 1 60); do [[ "$(code "$BASE/readyz")" == 200 ]] && { ok=$i; break; }; sleep 1; done
r_after="$(new_session)"; final="$(db_count)"; api_pid2="$(dx pgrep -u cas -x api)"
printf 'state after stop+20s=%s readyz=%s healthz=%s write=%s\nready again after %ss, write=%s\nsessions before stop=%s after=%s\napi pid %s -> %s\nbootstrap: %s\n' \
  "$st_down" "$r_ready" "$r_health" "$r_write" "$ok" "$r_after" "$after" "$final" "$api_pid" "$api_pid2" "$(sup bootstrap)" > "$OUT/R2_manual_stop.txt"
if [[ "$st_down" == STOPPED && "$r_ready" == 503 && "$r_health" == 200 && "$r_write" =~ ^5 && -n "$ok" && "$r_after" == 201 \
      && "$final" == $((after + 1)) && "$api_pid" == "$api_pid2" && "$(state bootstrap)" == EXITED ]]; then
  pass R2 "Postgres manual stop: stays STOPPED, readyz 503, write 5xx, no partial row, recovers on start without API restart" "R2_manual_stop.txt"
else fail R2 "Postgres manual stop" "R2_manual_stop.txt"; fi

# R3: MinIO crash and manual stop.
mpid="$(dx pgrep -u minio -x minio)"; dx kill -9 "$mpid"; sleep 8
st_m="$(state minio)"; mpid2="$(dx pgrep -u minio -x minio)"
up2="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/smoke2.pdf;type=application/pdf" -F title="after minio crash" -F domain_id=ho_tich_chung_thuc)"
dx supervisorctl -c /etc/cas/supervisord.conf stop minio >/dev/null; sleep 3
up3="$(curl -s -o "$OUT/R3_upload_down.json" -w '%{http_code}' "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/smoke3.pdf;type=application/pdf" -F title="minio down" -F domain_id=ho_tich_chung_thuc)"
dx supervisorctl -c /etc/cas/supervisord.conf start minio >/dev/null; sleep 5
up4="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/v1/admin/documents" -H "Authorization: Bearer $TOKEN" -F "file=@$OUT/smoke4.pdf;type=application/pdf" -F title="minio back" -F domain_id=ho_tich_chung_thuc)"
orphans="$(dx gosu cas psql "$DSN_V" -tAc "select count(*) from documents where title = 'minio down'" | tr -d '[:space:]')"
printf 'crash: pid %s -> %s state=%s upload_after=%s\nmanual stop upload=%s (rows titled minio down: %s)\nafter start upload=%s\n' \
  "$mpid" "$mpid2" "$st_m" "$up2" "$up3" "$orphans" "$up4" > "$OUT/R3_minio.txt"
[[ "$st_m" == RUNNING && "$mpid2" != "$mpid" && "$up2" == 201 && "$up3" =~ ^5 && "$up4" == 201 ]] \
  && pass R3 "MinIO crash auto-restart; manual stop gives 5xx upload; recovers" "R3_minio.txt" || fail R3 "MinIO loss" "R3_minio.txt"
[[ "$orphans" == 0 ]] && pass R3b "failed upload while MinIO down left no document row" || fail R3b "failed upload left a document row" "rows=$orphans"

# ---- B: backup ---------------------------------------------------------------------
dx backup.sh > "$OUT/B1_backup.txt" 2>&1 && dx bash -c 'ls -l /workspace/cas/backups/pg' >> "$OUT/B1_backup.txt" 2>&1
grep -q "postgres -> " "$OUT/B1_backup.txt" && pass B1 "backup.sh: pg_dump archive written and listable" "B1_backup.txt" || fail B1 "backup.sh" "B1_backup.txt"
dx restore_drill.sh > "$OUT/B2_restore_drill.txt" 2>&1; rc=$?
r="$(grep '^RESULT' "$OUT/B2_restore_drill.txt")"
[[ $rc -eq 0 && "$r" == "RESULT FULL postgres=OK"*"qdrant=OK"* ]] && pass B2 "restore drill: pg_dump restored into a scratch DB with identical row counts; Qdrant snapshot restored with identical point count; scratch data removed" "$r" || fail B2 "restore drill" "rc=$rc $r"
# B2b: the exact scratch DB/collection of THIS run (SCRATCH line) are gone, and no
# cas_drill_* database or collection exists at all.
scr="$(grep '^SCRATCH ' "$OUT/B2_restore_drill.txt")"
sdb="$(sed -n 's/.* db=\([^ ]*\).*/\1/p' <<<"$scr")"; scoll="$(sed -n 's/.* collection=\([^ ]*\).*/\1/p' <<<"$scr")"
db_left="$(dx gosu postgres psql -h /run/postgresql -tAc "select count(*) from pg_database where datname = '$sdb'" 2>&1 | tr -d '[:space:]')"
db_any="$(dx gosu postgres psql -h /run/postgresql -tAc "select count(*) from pg_database where datname like 'cas\\_drill\\_%'" 2>&1 | tr -d '[:space:]')"
coll_code="$(dx curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:6333/collections/$scoll")"
coll_any="$(dx curl -fsS http://127.0.0.1:6333/collections | python3 -c 'import json,sys;print(sum(c["name"].startswith("cas_drill_") for c in json.load(sys.stdin)["result"]["collections"]))')"
echo "$scr | db_left=$db_left any_drill_db=$db_any collection_http=$coll_code any_drill_collection=$coll_any" > "$OUT/B2b_cleanup.txt"
if [[ "$sdb" == cas_drill_* && "$scoll" == cas_drill_* && "$db_left" == 0 && "$db_any" == 0 && "$coll_code" == 404 && "$coll_any" == 0 ]]; then
  pass B2b "the run's own scratch DB and collection are gone (checked by their exact names), no cas_drill_* left" "B2b_cleanup.txt"
else fail B2b "scratch cleanup" "$(cat "$OUT/B2b_cleanup.txt")"; fi
# B4: a cleanup that cannot drop the scratch DB makes the drill FAIL and names it.
# Injected with a psql shim that refuses DROP DATABASE (only for this exec).
dx bash -c 'mkdir -p /tmp/b4shim && printf "%s\n" "#!/bin/bash" "case \"\$*\" in *\"DROP DATABASE\"*) echo injected-drop-failure >&2; exit 1 ;; esac" "exec /usr/bin/psql \"\$@\"" > /tmp/b4shim/psql && chmod 755 /tmp/b4shim /tmp/b4shim/psql'
docker exec -e PATH=/tmp/b4shim:/opt/cas/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$C" restore_drill.sh > "$OUT/B4_cleanup_failure.txt" 2>&1; rc4=$?
ldb="$(sed -n 's/^RESULT FAIL cleanup_left: database=\([^ ]*\).*/\1/p' "$OUT/B4_cleanup_failure.txt")"
still="$(dx gosu postgres psql -h /run/postgresql -tAc "select count(*) from pg_database where datname = '$ldb'" 2>&1 | tr -d '[:space:]')"
dx gosu postgres psql -h /run/postgresql -qc "DROP DATABASE \"$ldb\"" >/dev/null 2>&1; dx rm -rf /tmp/b4shim
after="$(dx gosu postgres psql -h /run/postgresql -tAc "select count(*) from pg_database where datname like 'cas\\_drill\\_%'" 2>&1 | tr -d '[:space:]')"
echo "rc=$rc4 reported=$ldb existed_before_manual_drop=$still drill_dbs_after_manual_drop=$after" >> "$OUT/B4_cleanup_failure.txt"
if [[ $rc4 -ne 0 && "$ldb" == cas_drill_* && "$still" == 1 && "$after" == 0 ]] && grep -q "scratch resources left behind" "$OUT/B4_cleanup_failure.txt"; then
  pass B4 "cleanup failure (injected) makes the drill FAIL (exit $rc4) and names the leftover DB; removed by hand afterwards" "B4_cleanup_failure.txt"
else fail B4 "cleanup failure must fail the drill" "$(tail -1 "$OUT/B4_cleanup_failure.txt")"; fi
# B3: without a Qdrant snapshot the full drill must FAIL; --postgres-only is an explicit
# operator choice and reports PARTIAL. The snapshots are moved aside and put back.
dx bash -c 'mv /workspace/cas/backups/qdrant /workspace/cas/backups/qdrant.b3-aside && install -d -o cas -g cas -m 0700 /workspace/cas/backups/qdrant'
dx restore_drill.sh > "$OUT/B3_no_snapshot.txt" 2>&1; rc_full=$?
dx restore_drill.sh --postgres-only > "$OUT/B3_postgres_only.txt" 2>&1; rc_pg=$?
dx bash -c 'rm -rf /workspace/cas/backups/qdrant && mv /workspace/cas/backups/qdrant.b3-aside /workspace/cas/backups/qdrant'
if [[ $rc_full -ne 0 ]] && grep -q "no Qdrant snapshot" "$OUT/B3_no_snapshot.txt" && [[ $rc_pg -eq 0 ]] && grep -q "^RESULT PARTIAL postgres=OK.*qdrant=NOT_TESTED" "$OUT/B3_postgres_only.txt"; then
  pass B3 "no Qdrant snapshot: full drill FAILS (exit $rc_full); --postgres-only reports PARTIAL, never a full pass"
else fail B3 "missing Qdrant snapshot handling" "full_rc=$rc_full pg_only_rc=$rc_pg"; fi

# ---- P: persistence and stale marker -------------------------------------------------
fp1="$(dx openssl x509 -in /workspace/cas/tls/ca.crt -noout -fingerprint -sha256)"
kh1="$(dx sha256sum /workspace/cas/tls/postgres/server.key /workspace/cas/tls/minio/private.key | awk '{print $1}' | tr '\n' ' ')"
boot1="$(dx cat /run/cas/boot_id)"; sess1="$(db_count)"
docker restart -t 120 "$C" >/dev/null
h="$(wait_healthy 600)"
fp2="$(dx openssl x509 -in /workspace/cas/tls/ca.crt -noout -fingerprint -sha256)"
kh2="$(dx sha256sum /workspace/cas/tls/postgres/server.key /workspace/cas/tls/minio/private.key | awk '{print $1}' | tr '\n' ' ')"
boot2="$(dx cat /run/cas/boot_id)"; mark2="$(dx cat /run/cas/bootstrap.ok)"; sess2="$(db_count)"
curl -s "$BASE/api/v1/admin/documents/$DOC/content" -H "Authorization: Bearer $TOKEN" -o "$OUT/P1_download.pdf"
docker logs --since 10m "$C" 2>&1 | grep -E "internal CA reused|certificate reused|migrations: current" > "$OUT/P1_reuse.txt"
printf '%s\nca %s -> %s\nkeys %s -> %s\nboot %s -> %s marker=%s\nsessions %s -> %s\n' "$h" "$fp1" "$fp2" "$kh1" "$kh2" "$boot1" "$boot2" "$mark2" "$sess1" "$sess2" >> "$OUT/P1_reuse.txt"
if [[ "$h" == healthy* && "$fp1" == "$fp2" && "$kh1" == "$kh2" && "$sess1" == "$sess2" ]] && cmp -s "$OUT/smoke.pdf" "$OUT/P1_download.pdf" \
   && grep -q "migrations: current=25 latest=25" "$OUT/P1_reuse.txt"; then
  pass P1 "docker restart: data, PDF, CA and keys unchanged; no migration rerun" "P1_reuse.txt"; else fail P1 "restart persistence" "P1_reuse.txt"; fi
[[ "$boot1" != "$boot2" && "$mark2" == "$boot2" ]] && pass P2 "stale bootstrap.ok of the previous boot not reused; new marker matches new boot id" "$boot1 -> $boot2" || fail P2 "stale marker" "$boot1 $boot2 $mark2"

# M1: a dirty migration state is refused by bootstrap (never forced); the app stays
# gated; after the operator fixes the state, starting bootstrap again opens the gate.
pgsu() { dx gosu postgres psql -h /run/postgresql -d citizen_assistance -qtAc "$1"; }
pgsu "update schema_migrations set dirty = true" >/dev/null
docker restart -t 120 "$C" >/dev/null; sleep 25
docker logs --since 1m "$C" 2>&1 | grep -E "\[bootstrap\]|exited: bootstrap" > "$OUT/M1_dirty.txt"
st_api="$(state api)"; hc="$(dx healthcheck.sh --quiet >/dev/null 2>&1 && echo healthy || echo unhealthy)"
api_up="$(dx pgrep -u cas -x api >/dev/null && echo yes || echo no)"
echo "api program: $st_api (api process running: $api_up) health: $hc" >> "$OUT/M1_dirty.txt"
pgsu "update schema_migrations set dirty = false" >/dev/null
dx supervisorctl -c /etc/cas/supervisord.conf start bootstrap >/dev/null 2>&1
h="$(wait_healthy 300)"; echo "after fix: $h" >> "$OUT/M1_dirty.txt"
if grep -q "migration state is dirty" "$OUT/M1_dirty.txt" && [[ "$api_up" == no && "$hc" == unhealthy && "$h" == healthy* ]]; then
  pass M1 "dirty migration state refused (no force), app gated, recovers after the operator fix" "M1_dirty.txt"
else fail M1 "dirty migration handling" "M1_dirty.txt"; fi

# P3: clean stop, then a NEW container (other owner id) on the same volume, with ownership
# scrambled (volume from another pod). The clean stop released the lease, so it may start.
docker stop -t 120 "$C" >/dev/null
docker logs --tail 50 "$C" 2>&1 | grep -E "lease released" > "$OUT/P3_release.txt"
docker rm "$C" >/dev/null
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'chown -R 4242:4242 /w/cas/minio /w/cas/qdrant && chown 4242 /w/cas/tls/postgres/server.key'
docker run -d --name "$C" -p "127.0.0.1:$PORT:8088" -v "$VOL:/workspace" "${MODELS[@]}" "${ENVS[@]}" "$IMG" >/dev/null
h="$(wait_healthy 600)"
fp3="$(dx openssl x509 -in /workspace/cas/tls/ca.crt -noout -fingerprint -sha256)"; sess3="$(db_count)"
curl -s "$BASE/api/v1/admin/documents/$DOC/content" -H "Authorization: Bearer $TOKEN" -o "$OUT/P3_download.pdf"
docker logs "$C" 2>&1 | grep -E "fixing ownership|internal CA reused|certificate reused|workspace lease" > "$OUT/P3_reattach.txt"
echo "$h ca=$fp3 sessions=$sess3" >> "$OUT/P3_reattach.txt"
if [[ "$h" == healthy* && "$fp3" == "$fp1" && "$sess3" == "$sess2" ]] && cmp -s "$OUT/smoke.pdf" "$OUT/P3_download.pdf" \
   && grep -q "fixing ownership" "$OUT/P3_reattach.txt" && grep -q "lease released" "$OUT/P3_release.txt" && grep -q "workspace lease new" "$OUT/P3_reattach.txt"; then
  pass P3 "clean stop releases the lease; new container on the volume takes it, repairs ownership, reuses CA, data intact" "P3_reattach.txt"; else fail P3 "volume re-attach" "P3_reattach.txt"; fi

# ---- L: single writer --------------------------------------------------------------------
OWNER1="$(dx cat /run/cas-lease.owner)"
# L1: a second container on the same volume while the first runs -> maintenance, nothing started.
docker run -d --name "$C2" -v "$VOL:/workspace" "${MODELS[@]}" "${ENVS[@]}" "$IMG" >/dev/null
sleep 25
docker logs "$C2" > "$OUT/L1_second_container.log" 2>&1
procs="$(docker exec "$C2" ps -eo comm --no-headers | sort -u | tr '\n' ' ')"
hc2="$(docker exec "$C2" healthcheck.sh 2>&1 | head -1)"
w1="$(new_session)"
echo "C2 processes: $procs | C2 health: $hc2 | first container write: $w1" >> "$OUT/L1_second_container.log"
if grep -q "MAINTENANCE: workspace flock is held" "$OUT/L1_second_container.log" && [[ "$procs" != *postgres* && "$procs" != *supervisord* && "$w1" == 201 ]]; then
  pass L1 "second container on a volume in use: refused (flock), maintenance mode, no service started; first container unaffected" "L1_second_container.log"
else fail L1 "second container must be refused" "L1_second_container.log"; fi
# L2: recovery while the owner is alive must be refused (heartbeat keeps changing).
docker exec -e CAS_PG_RECOVER_WAIT_S=5 "$C2" cas-pg-recover "$OWNER1" > "$OUT/L2b_recover_short.txt" 2>&1; rcs=$?
[[ $rcs -ne 0 ]] && grep -q "shorter than the minimum" "$OUT/L2b_recover_short.txt" && pass L2b "a recovery wait shorter than 3 x heartbeat + 5 s is refused" "$(tail -1 "$OUT/L2b_recover_short.txt")" || fail L2b "short recovery wait" "rc=$rcs"
docker exec "$C2" cas-pg-recover "$OWNER1" > "$OUT/L2_recover_alive.txt" 2>&1; rc=$?
[[ $rc -ne 0 ]] && grep -q "changed during the wait" "$OUT/L2_recover_alive.txt" && pass L2 "cas-pg-recover refuses while the owner's heartbeat is alive" "$(tail -1 "$OUT/L2_recover_alive.txt")" || fail L2 "recover while alive" "rc=$rc $(tail -1 "$OUT/L2_recover_alive.txt")"
# L3: unclean death of the owner (SIGKILL), wrong owner id refused, verified recovery accepted.
docker kill "$C" >/dev/null; docker rm "$C" >/dev/null
docker exec "$C2" cas-pg-recover not-the-owner > "$OUT/L3_recover_wrong.txt" 2>&1; rcw=$?
docker exec "$C2" cas-pg-recover "$OWNER1" > "$OUT/L3_recover.txt" 2>&1; rcr=$?
docker exec "$C2" cat /workspace/cas/locks/recovery.log >> "$OUT/L3_recover.txt" 2>&1
if [[ $rcw -ne 0 && $rcr -eq 0 ]] && grep -q "lease owner is" "$OUT/L3_recover_wrong.txt" && grep -q "recovered=$OWNER1" "$OUT/L3_recover.txt" && grep -q "postmaster.pid ->" "$OUT/L3_recover.txt"; then
  pass L3 "after SIGKILL of the owner: wrong id refused; verified recovery renames lease + stale postmaster.pid and audits it" "L3_recover.txt"
else fail L3 "verified recovery" "wrong_rc=$rcw recover_rc=$rcr"; fi
# L4: restart the maintenance container: it takes a new lease and serves the same data.
docker restart -t 60 "$C2" >/dev/null
C="$C2"
h="$(wait_healthy 600)"; sess4="$(db_count)"
docker logs --since 5m "$C" 2>&1 | grep -E "workspace lease|MAINTENANCE" > "$OUT/L4_after_recovery.txt"
echo "$h sessions=$sess4 (before L1: $sess3, +1 written in L1)" >> "$OUT/L4_after_recovery.txt"
[[ "$h" == healthy* && "$sess4" == $((sess3 + 1)) ]] && grep -q "workspace lease new" "$OUT/L4_after_recovery.txt" \
  && pass L4 "after recovery the new container takes the lease and serves the same data" "L4_after_recovery.txt" || fail L4 "post-recovery start" "L4_after_recovery.txt"

# P4: an interrupted first initialisation (no .cas-init-complete) is refused, then the
# volume works again once the marker is back (simulated on the test volume only).
docker stop -t 120 "$C" >/dev/null
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'mv /w/cas/pgdata/.cas-init-complete /w/cas/pgdata/.cas-init-complete.moved'
docker start "$C" >/dev/null
for _ in $(seq 1 60); do [[ "$(docker inspect -f '{{.State.Running}}' "$C")" == false ]] && break; sleep 2; done
st4="$(docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' "$C")"
docker logs --tail 30 "$C" 2>&1 | grep -E "FATAL|init" > "$OUT/P4_init_marker.txt"
docker run --rm -v "$VOL:/w" --entrypoint bash "$IMG" -c 'mv /w/cas/pgdata/.cas-init-complete.moved /w/cas/pgdata/.cas-init-complete'
docker start "$C" >/dev/null; h="$(wait_healthy 300)"
echo "refused run: running/exit=$st4; after restoring the marker: $h" >> "$OUT/P4_init_marker.txt"
if [[ "$st4" == "false 64" && "$h" == healthy* ]] && grep -q "first initialisation did not complete" "$OUT/P4_init_marker.txt"; then
  pass P4 "pgdata without .cas-init-complete is refused (exit 64), not silently used" "P4_init_marker.txt"
else fail P4 "init marker" "P4_init_marker.txt"; fi

# ---- G: graceful stop ------------------------------------------------------------------
t0=$SECONDS; docker stop -t 120 "$C" >/dev/null; dt=$((SECONDS - t0))
docker logs --tail 200 "$C" > "$OUT/G1_stop.log" 2>&1
ec="$(docker inspect -f '{{.State.ExitCode}}' "$C")"
grep -q "database system is shut down" "$OUT/G1_stop.log" && grep -q "lease released" "$OUT/G1_stop.log" && [[ $dt -lt 120 ]] \
  && pass G1 "docker stop: clean Postgres shutdown, lease released, in ${dt}s (exit $ec)" "G1_stop.log" || fail G1 "graceful stop" "dt=${dt}s exit=$ec"

echo; echo "=== results ($OUT/results.tsv)"; column -t -s $'\t' "$RES"
! grep -q $'\tFAIL\t' "$RES"
