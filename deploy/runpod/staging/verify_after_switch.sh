#!/usr/bin/env bash
# Acceptance checks inside the cas-aio container after the image switch. Run as root
# in the pod shell (pod_stage.sh apply copied this script to /workspace/cas/preflight):
#
#   bash /workspace/cas/preflight/verify_after_switch.sh --audio <16 kHz mono WAV>
#
# Read-only: HTTP calls on 127.0.0.1, one read-only SQL session, nvidia-smi, the RQ
# fingerprint comparison, one e5 embedding as user cas, Gemini models.get (metadata, no
# tokens; required, a FAIL blocks the features that use the LLM). --audio (required) sends
# one file to Speech C0 directly (127.0.0.1:8090); it must be a sample the research data
# already holds, and it is not stored.
# Exit 1 if a required check fails: then follow the rollback in README.md.
set -uo pipefail

W=/workspace/cas
P="$W/preflight"
AUDIO=""
[[ "${1:-}" == --audio ]] && AUDIO="${2:?--audio needs a WAV path}"
fails=0 warns=0
ok()   { printf 'PASS  %-18s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-18s %s\n' "$1" "$2"; warns=$((warns + 1)); }
bad()  { printf 'FAIL  %-18s %s\n' "$1" "$2"; fails=$((fails + 1)); }
info() { printf 'INFO  %-18s %s\n' "$1" "$2"; }
[[ $(id -u) == 0 ]] || { echo "run as root" >&2; exit 1; }
[[ -f /opt/cas/REVISION ]] || { echo "not a cas-aio container" >&2; exit 1; }

echo "== verify_after_switch $(date -u +%FT%TZ) image revision $(cat /opt/cas/REVISION), C0 $(cat /opt/cas/c0/SOURCE_REVISION)"

# 0. the switch was made from a complete apply, and the files it prepared are unchanged
mk() { grep -m1 "^$1=" "$P/READY_TO_SWITCH" | cut -d= -f2-; }
if [[ -f $P/READY_TO_SWITCH ]]; then
  rp=""
  sim="$(mk simulated_hardware)"
  [[ $sim == no ]] || rp+="simulated_hardware=${sim:-<missing>}: only a marker from an apply on the real hardware (no) is accepted; "
  [[ "$(mk image_revision)" == "$(cat /opt/cas/REVISION)" ]] || rp+="image_revision $(mk image_revision) != running $(cat /opt/cas/REVISION); "
  [[ "$(mk manifest_sha256)" == "$(sha256sum "$W/manifests/c0_checkpoints.sha256" 2>/dev/null | cut -d' ' -f1)" ]] || rp+="manifest changed since apply; "
  [[ "$(mk config_sha256)" == "$(sha256sum "$W/config/c0_inference.yaml" 2>/dev/null | cut -d' ' -f1)" ]] || rp+="c0_inference.yaml changed since apply; "
  [[ "$(grep -m1 '^C0_INFERENCE_CONFIG=' /run/cas-env/speech.env | cut -d= -f2-)" == "$W/config/c0_inference.yaml" ]] || rp+="Speech does not use $W/config/c0_inference.yaml; "
  if [[ -z $rp ]]; then ok ready-marker "apply $(mk ready_at), clone $(mk clone | cut -c1-7); image revision, manifest and config match"
  else bad ready-marker "$rp"; fi
else bad ready-marker "$P/READY_TO_SWITCH missing: pod_stage.sh apply did not finish"; fi

# 1. services (the image's own report; speech is required in gpu mode)
if [[ -f /run/cas-maintenance ]]; then bad maintenance "$(cat /run/cas-maintenance)"; fi
if out="$(CAS_SPEECH_MODE=gpu /opt/cas/bin/healthcheck.sh 2>&1)"; then ok healthcheck "all services OK"
else bad healthcheck "see below"; fi
sed 's/^/      /' <<<"$out"

# 2. GPU and torch in the C0 venv
if nvidia-smi -L >/dev/null 2>&1; then ok gpu "$(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | head -1)"
else bad gpu "nvidia-smi sees no GPU"; fi
if t="$(/opt/cas/venv-c0/bin/python -c 'import torch; a=torch.cuda.is_available(); print(torch.__version__, "cuda_build=%s" % torch.version.cuda, "available=%s" % a, torch.cuda.get_device_name(0) if a else "-")' 2>&1)" \
   && [[ $t == *available=True* ]]; then ok torch-cuda "$t"
else bad torch-cuda "$(tail -1 <<<"$t")"; fi

# 3. Speech C0: ready, serving gates, on the GPU
if r="$(curl -fsS --max-time 5 http://127.0.0.1:8090/readyz 2>&1)"; then
  if python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get("status")=="ready" and d.get("tokenizer_parity")=="VERIFIED" and d.get("generation_matches_rq1") is True else 1)' "$r"
  then ok speech-ready "$r"; else bad speech-ready "$r"; fi
else bad speech-ready "$r"; fi
# The models are on the GPU when all of these hold: Speech is ready while forced to
# SPEECH_DEVICE=cuda (resolve_device fails closed when CUDA is unavailable), the speech
# process itself holds /dev/nvidia* open (its own CUDA context; nvidia-smi often cannot
# map PIDs inside a container, and VRAM alone may belong to another process), and a
# real inference succeeds (section 8, required).
spid="$(supervisorctl -c /etc/cas/supervisord.conf pid speech 2>/dev/null)"
sdev="$(grep -m1 '^SPEECH_DEVICE=' /run/cas-env/speech.env | cut -d= -f2)"
nvfd=0
if [[ $spid =~ ^[1-9][0-9]*$ ]]; then
  for fd in /proc/"$spid"/fd/*; do [[ "$(readlink "$fd" 2>/dev/null)" == /dev/nvidia* ]] && nvfd=$((nvfd + 1)); done
fi
if [[ $sdev == cuda ]] && (( nvfd > 0 )); then ok speech-on-gpu "SPEECH_DEVICE=cuda, speech pid $spid holds $nvfd /dev/nvidia* descriptors"
else bad speech-on-gpu "SPEECH_DEVICE=${sdev:-?}, speech pid ${spid:-?} holds $nvfd /dev/nvidia* descriptors"; fi

# 4. E5: the configured variant must be fp32, and a real embedding must run on this CPU
ev="$(grep -m1 '^EMBEDDING_VARIANT=' /run/cas-env/ai-service.env | cut -d= -f2)"
md="$(grep -m1 '^EMBEDDING_MODEL_DIR=' /run/cas-env/ai-service.env | cut -d= -f2)"
vnni=no; grep -qw avx512_vnni /proc/cpuinfo && vnni=yes
cpu="$(awk -F': *' '/^model name/ {print $2; exit}' /proc/cpuinfo)"
if grep -q '^EMBEDDING_CPU_FLAGS=' /run/cas-env/ai-service.env; then bad embedding-flags "EMBEDDING_CPU_FLAGS is set (test-only, hides the real CPU)"; fi
if [[ $ev == fp32 ]]; then ok embedding-variant "EMBEDDING_VARIANT=fp32, cpu $cpu, avx512_vnni=$vnni"
else bad embedding-variant "EMBEDDING_VARIANT=${ev:-<unset>}, expected fp32"; fi
# Same class and rules as the ai-service (app.indexing.embed.OnnxEmbedder), as user cas,
# on the real CPU flags: 1 query + 16 passages, 384 dimensions, finite, unit norm.
if e="$(cd /opt/cas/ai && gosu cas env -i PATH=/usr/bin:/bin EMBEDDING_VARIANT="$ev" /opt/cas/venv-ai/bin/python - "$md" 2>&1 <<'PY'
import math, resource, sys, time
from app.indexing.embed import OnnxEmbedder
t0 = time.perf_counter()
emb = OnnxEmbedder(sys.argv[1], batch_size=16)
load_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
q = emb.embed_query("Thủ tục đăng ký khai sinh cần giấy tờ gì?")
q_ms = (time.perf_counter() - t0) * 1000
texts = ["Người đi đăng ký khai sinh nộp tờ khai theo mẫu và giấy chứng sinh."] * 16
t0 = time.perf_counter()
batch = emb.embed_batch(texts)
b_ms = (time.perf_counter() - t0) * 1000
problems = []
for v in [q] + batch:
    if len(v) != 384: problems.append(f"dimension {len(v)}")
    if not all(math.isfinite(x) for x in v): problems.append("non-finite value")
    if abs(math.sqrt(sum(x * x for x in v)) - 1.0) > 1e-3: problems.append("norm != 1")
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"variant={emb.variant} checksum={emb.checksum[:12]} load={load_ms:.0f}ms query={q_ms:.0f}ms "
      f"batch16={b_ms:.0f}ms dim={len(q)} peak_rss={rss_mb:.0f}MB")
if problems:
    print("problems: " + ", ".join(sorted(set(problems))))
    sys.exit(1)
PY
)"; then
  if [[ $e == *variant=fp32* ]]; then ok embedding-run "$(tail -1 <<<"$e")"; else bad embedding-run "$(tail -1 <<<"$e")"; fi
else bad embedding-run "$(tail -2 <<<"$e" | tr '\n' ' ')"; fi
info ai-service "$(curl -sS --max-time 5 http://127.0.0.1:8001/ready 2>&1 | cut -c1-300)"

# 5. only nginx :8088 listens beyond loopback
# A listener this image does not start (for example one RunPod injects) is a FAIL with
# its process name: identify it before accepting the switch.
bad_l=""
while read -r addr proc; do
  case "$addr" in 127.0.0.1:*|\[::1\]:*|::1:*) ;; *:8088) ;; *) bad_l+="$addr ${proc:-?}; " ;; esac
done < <(ss -ltnpH | awk '{print $4, $6}')
if [[ -z $bad_l ]]; then ok listeners "only :8088 is not loopback ($(ss -ltnH | awk '{print $4}' | tr '\n' ' '))"
else bad listeners "non-loopback listeners besides :8088: $bad_l"; fi
if c="$(curl -fsS --max-time 5 http://127.0.0.1:8088/readyz 2>&1)"; then ok nginx "/readyz via :8088: $c"
else bad nginx "$c"; fi

# 6. database state, read-only session (expected empty on a first boot)
sql() { gosu postgres env PGOPTIONS='-c default_transaction_read_only=on' psql -h /run/postgresql -d citizen_assistance -XAtq -c "$1" 2>&1; }
mig="$(sql 'select version || case when dirty then '"' DIRTY'"' else '"''"' end from schema_migrations')"
if [[ $mig =~ ^[0-9]+$ ]]; then ok migrations "schema version $mig, clean"; else bad migrations "$mig"; fi
info data "$(sql "select 'users=' || (select count(*) from users) || ' procedures=' || (select count(*) from procedures) || ' versions=' || (select count(*) from procedure_versions) || ' documents=' || (select count(*) from documents) || ' chunks=' || (select count(*) from knowledge_chunks)")"
info qdrant "$(curl -sS --max-time 5 http://127.0.0.1:6333/collections 2>&1 | cut -c1-200)"

# 7. research data untouched since pod_stage.sh apply
if [[ -f $P/rq_tree.listing.gz ]]; then
  if r="$(python3 "$P/rq_fingerprint.py" /workspace --exclude "$W" --compare "$P/rq_tree.listing.gz" 2>&1)"; then ok rq-untouched "$(tail -1 <<<"$r")"
  else bad rq-untouched "research tree changed:"; sed 's/^/      /' <<<"$r"; fi
else bad rq-untouched "$P/rq_tree.listing.gz missing (pod_stage.sh apply not run?)"; fi

# 7b. bytes of every inference file (weights, configs, contract, tokenizer) unchanged:
# the fingerprint above compares metadata only, this re-hashes content (about 3 GB).
if [[ -s $W/manifests/c0_checkpoints.sha256 ]] && ( cd / && sha256sum --strict --quiet -c "$W/manifests/c0_checkpoints.sha256" ) >/dev/null 2>&1
then ok manifest-intact "$(wc -l < "$W/manifests/c0_checkpoints.sha256") inference files match their SHA-256"
else bad manifest-intact "sha256sum -c $W/manifests/c0_checkpoints.sha256 failed"; fi

# 8. required: one real C0 inference, direct on loopback (proves the GPU path end to end).
# Secrets go to curl through a file descriptor, never on its command line (/proc/<pid>/cmdline).
if [[ -z $AUDIO ]]; then
  bad speech-infer "NOT_RUN: pass --audio <WAV> (a sample the research data already holds)"
elif [[ ! -r $AUDIO ]]; then
  bad speech-infer "cannot read $AUDIO"
else
  tok="$(grep -m1 '^SPEECH_SERVICE_TOKEN=' /run/cas-env/speech.env | cut -d= -f2-)"
  t0=$(date +%s%N)
  body="$(curl -sS --max-time 120 -w '\n%{http_code}' -H @<(printf 'Authorization: Bearer %s\n' "$tok") -F "audio=@$AUDIO;type=audio/wav" \
          http://127.0.0.1:8090/v1/speech/infer 2>&1)"
  unset tok
  ms=$(( ($(date +%s%N) - t0) / 1000000 )); code="$(tail -1 <<<"$body")"; body="$(sed '$d' <<<"$body")"
  if [[ $code == 200 ]] && msg="$(python3 - "$body" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
want = {"tokenizer_parity": "VERIFIED", "generation_matches_rq1": True, "checkpoint_identity": "configured_hash_match",
        "asr_checkpoint_step": 12500, "mt_checkpoint_step": 17500, "artifacts_present": True}
problems = [f"{k}={d.get(k)!r}" for k, v in want.items() if d.get(k) != v]
for k in ("asr_text_bahnar", "translated_text_vi"):
    if not str(d.get(k) or "").strip():
        problems.append(f"{k} empty")
print("; ".join(problems) if problems else
      f"latency_ms={d.get('latency_ms')} duration_ms={d.get('duration_ms')} "
      f"research_provenance={d.get('research_provenance')} bahnar={d['asr_text_bahnar'][:80]!r} vi={d['translated_text_vi'][:80]!r}")
sys.exit(1 if problems else 0)
PY
)"; then ok speech-infer "HTTP 200 in ${ms} ms: $msg"
  else bad speech-infer "HTTP $code in ${ms} ms: ${msg:-$(cut -c1-300 <<<"$body")}"; fi
fi

# 8b. Gemini (required for extraction and Definition drafts): the provider is gemini, a key is
# configured, and both effective model IDs (as the ai-service resolves them, /v1/status)
# exist for this key. models.get is a metadata call: no tokens billed. Any other answer is FAIL.
st="$(curl -sS --max-time 5 http://127.0.0.1:8001/v1/status 2>&1)"
ak="$(grep -m1 '^GEMINI_API_KEY=' /run/cas-env/ai-service.env | cut -d= -f2-)"
base="$(grep -m1 '^GEMINI_BASE_URL=' /run/cas-env/ai-service.env | cut -d= -f2-)"
base="${base:-https://generativelanguage.googleapis.com/v1beta}"
if ! models="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); p=d.get("provider")
if p != "gemini": sys.exit(f"provider={p}, expected gemini")
print(d.get("model_extraction") or "", d.get("model_definition") or "")' "$st" 2>&1)"; then
  bad gemini "cannot read the effective models from ai-service /v1/status: $(tail -1 <<<"$models" | cut -c1-200)"
elif [[ -z $ak ]]; then
  bad gemini "GEMINI_API_KEY is not configured for the ai-service"
else
  read -r m_ext m_def <<<"$models"
  for pair in "extraction:$m_ext" "definition:$m_def"; do
    role="${pair%%:*}" m="${pair#*:}"
    if [[ -z $m ]]; then bad gemini-model "$role model is not configured"; continue; fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 -H @<(printf 'x-goog-api-key: %s\n' "$ak") "$base/models/$m" 2>/dev/null)"
    case "$code" in
      200) ok gemini-model "$role model $m exists for this key" ;;
      404) bad gemini-model "$role model $m: not found (HTTP 404)" ;;
      400|401|403) bad gemini-model "$role model $m: key refused (HTTP $code)" ;;
      *) bad gemini-model "$role model $m: HTTP ${code:-none} (no answer from $base)" ;;
    esac
  done
fi
unset ak

# 9. resources
info memory "$(free -m | awk '/^Mem:/ {print $3 " MB used of " $2 " MB"}')"
info disk "container $(df -h / | awk 'NR==2 {print $3 "/" $2}'), volume $(df -h /workspace | awk 'NR==2 {print $3 "/" $2}')"
echo "== $fails FAIL, $warns WARN"
echo "Also check the RunPod Logs tab for: 'embedding model verified ... -> fp32 ... avx512_vnni=no' and 'gpu mode checks passed'."
exit $(( fails > 0 ))
