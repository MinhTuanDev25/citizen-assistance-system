#!/usr/bin/env bash
# Prepare the existing RTX 4090 pod (still on its old PyTorch image) for cas-aio.
# Run as root on the pod, from a clone of this repository on the container disk
# (under /tmp, readable by other users; see README.md):
#
#   pod_stage.sh check                 read-only: CPU/GPU/driver, volume, checkpoints + weight
#                                      pins, contract, read access for the image's speech user,
#                                      processes on /workspace, container-disk changes, source
#                                      revision. Writes nothing.
#   pod_stage.sh preserve [path...]    writes /workspace/cas/preserve/<pod>-<ts>/ only: pip freeze,
#                                      nvidia-smi, env var NAMES, processes, container-disk change
#                                      list, and a tar of the given container-disk paths
#   pod_stage.sh apply                 writes under /workspace/cas only: filesystem probe, E5 fp32
#                                      model, tokenizer cache (+ offline parity check), C0 config,
#                                      checkpoint manifest, RQ fingerprint, verify scripts
#
# Never writes to /workspace outside /workspace/cas. Never overwrites a file that
# differs from what it would write (it stops instead). Exit 1 on any FAIL.
# No errexit: every step reports PASS/WARN/FAIL itself; writes that must succeed die.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
W=/workspace/cas
VOL=/workspace
RQ_ROOT=/workspace/bahnar-s2tt-thesis
# Commit the image ghcr.io/minhtuandev25/citizen-assistance-system/cas-aio@sha256:40deae61...
# was built from; C0 code, fetch_models.sh and the embedding rules must equal it.
IMAGE_REVISION=b474debecb6be3e936bd149fc0a3dcbfa415a677
ASR_DIR=$RQ_ROOT/bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500
MT_DIR=$RQ_ROOT/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500
CONTRACT=$RQ_ROOT/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json
ASR_SHA=dacbe7f82d8ed0a33f7e9798860d0e1808a20b9dc84725355a2db9b4229a6a2a
MT_SHA=f9bee0d53399f76548eee297fc67ff83bad4e43cd8fda630c9cb6dab93284ead
TOK_ID=vinai/bartpho-syllable
TOK_REV=36eee8b4d648dd99da56462edcda3c5c97f7f3de
# Files the C0 loader requires (src/c0_cascaded.py ASR/MT_REQUIRED_FILES); the ASR
# processor also reads the optional ones when present.
ASR_FILES=(model.safetensors config.json vocab.json tokenizer_config.json preprocessor_config.json)
ASR_OPTIONAL=(special_tokens_map.json added_tokens.json)
MT_FILES=(model.safetensors config.json tokenizer_config.json sentencepiece.bpe.model dict.txt generation_config.json)
# UIDs fixed in deploy/runpod/Dockerfile
SPEECH_UID=10013 PG_UID=10010
# Network volume bahnar-thesis-storage: 200 GB quota. Room for e5 (0.5 GB), Postgres,
# Qdrant, MinIO and backups/ (pg_dump + Qdrant snapshots, CAS_BACKUP_KEEP=7).
VOL_QUOTA_GB="${VOL_QUOTA_GB:-200}" VOL_MIN_FREE_GB=20
READY="$W/preflight/READY_TO_SWITCH"
# Tests only: CAS_STAGE_TEST_HW=<dir> with a fake `cpuinfo` and `nvidia-smi` stands in
# for the CPU and GPU. Every run with it says so (WARN simulated-hw) and READY_TO_SWITCH
# records simulated_hardware=yes, which verify_after_switch.sh refuses.
SIM_HW="${CAS_STAGE_TEST_HW:-}"
CPUINFO=/proc/cpuinfo
if [[ -n $SIM_HW ]]; then CPUINFO="$SIM_HW/cpuinfo"; PATH="$SIM_HW:$PATH"; fi
# Code this script runs or compares with the image: must equal the clone's commit
# exactly (no modified, staged, untracked or ignored files). apply also requires
# CAS_STAGE_EXPECT_COMMIT=<full SHA of the reviewed commit> to equal HEAD.
EXPECT_COMMIT="${CAS_STAGE_EXPECT_COMMIT:-}"
SOURCE_PATHS=(bahnar-s2tt-thesis/src deploy/runpod/fetch_models.sh deploy/runpod/models deploy/runpod/staging
              apps/ai-service/app/indexing/embed.py)
TOK_VENV=/tmp/cas-c0-tok
TOK_SNAP=$W/hf-cache/hub/models--vinai--bartpho-syllable/snapshots/$TOK_REV

fails=0 warns=0
ok()   { printf 'PASS  %-22s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-22s %s\n' "$1" "$2"; warns=$((warns + 1)); }
bad()  { printf 'FAIL  %-22s %s\n' "$1" "$2"; fails=$((fails + 1)); }
info() { printf 'INFO  %-22s %s\n' "$1" "$2"; }
die()  { echo "pod_stage: $*" >&2; exit 1; }

[[ $(id -u) == 0 ]] || die "run as root"

# Read access as an unprivileged uid that is not the owner (the image's speech user).
can_read_as() {  # can_read_as <uid> <path>
  python3 - "$1" "$2" <<'PY'
import os, sys
uid, path = int(sys.argv[1]), sys.argv[2]
pid = os.fork()
if pid == 0:
    try:
        os.setgroups([]); os.setgid(uid); os.setuid(uid)
        if os.path.isdir(path):
            os.listdir(path)
        else:
            with open(path, "rb") as f:
                f.read(1)
        os._exit(0)
    except OSError:
        os._exit(1)
_, status = os.waitpid(pid, 0)
sys.exit(os.waitstatus_to_exitcode(status))
PY
}

as_uid() {  # as_uid <uid> <command...>: run as that uid/gid with no supplementary groups
  local u="$1"; shift
  if command -v setpriv >/dev/null; then setpriv --reuid "$u" --regid "$u" --clear-groups -- "$@"
  else python3 -c 'import os,sys;u=int(sys.argv[1]);os.setgroups([]);os.setgid(u);os.setuid(u);os.execvp(sys.argv[2],sys.argv[2:])' "$u" "$@"; fi
}

check_host() {
  local cpu vnni=no avx2=no drv cuda procs
  if [[ -n $SIM_HW ]]; then warn simulated-hw "CAS_STAGE_TEST_HW=$SIM_HW: CPU and GPU are SIMULATED (tests only)"; fi
  cpu="$(awk -F': *' '/^model name/ {print $2; exit}' "$CPUINFO")"
  if grep -qw avx2 "$CPUINFO"; then avx2=yes; fi
  if grep -qw avx512_vnni "$CPUINFO"; then vnni=yes; fi
  info pod "RUNPOD_POD_ID=${RUNPOD_POD_ID:-<unset>} host=$(hostname)"
  if [[ $avx2 == yes ]]; then ok cpu "$cpu, avx2=yes, avx512_vnni=$vnni (EMBEDDING_VARIANT=fp32)"
  else bad cpu "$cpu has no AVX2: the fp32 e5 graph cannot run"; fi
  info ram "$(free -g | awk '/^Mem:/ {print $2 " GB total, " $7 " GB available"}')"
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
    ok gpu "$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1); driver $drv"
    # The image has torch 2.8.0 from PyPI: the CUDA 12.8 build (its venv layer is 4.1 GB with the CUDA libs).
    if [[ -n "$cuda" ]] && python3 -c "import sys; sys.exit(0 if tuple(map(int, sys.argv[1].split('.'))) >= (12, 8) else 1)" "$cuda"; then
      ok gpu-cuda "driver supports CUDA $cuda (>= 12.8)"
    else
      warn gpu-cuda "driver reports CUDA ${cuda:-?} < 12.8: torch cu128 then relies on minor-version compatibility; verify_after_switch.sh decides"
    fi
    procs="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)"
    if [[ -z "$procs" ]]; then ok gpu-idle "no process on the GPU"
    else warn gpu-idle "GPU in use: $(tr '\n' ';' <<<"$procs")"; fi
  else
    bad gpu "nvidia-smi does not see a GPU"
  fi
}

# df on a RunPod network volume shows the shared backend (hundreds of TB), not this
# volume's quota. Free space is therefore quota - usage measured with du.
# Without the network volume, /workspace is a plain directory on the container disk:
# anything written there would be lost at the switch. Used by check, preserve and apply.
check_mount() {
  if python3 -c 'import os,sys; sys.exit(0 if os.path.ismount(sys.argv[1]) else 1)' "$VOL"; then
    ok volume-mount "$VOL is a mount point ($(awk -v m="$VOL" '$2 == m {print $3 " from " $1; exit}' /proc/mounts))"
  else
    bad volume-mount "$VOL is not a mount point: the network volume is not attached"
  fi
}

check_volume() {
  local used_kb free_gb e
  info volume-backend "$(df -hT "$VOL" | awk 'NR==2 {print $2 ", backend " $3 " (not the quota), mounted " $7}')"
  echo "      measuring usage of $VOL with du (may take a few minutes on the network volume)..."
  # pipefail: a du error (e.g. an unreadable subtree, which would undercount) fails the assignment.
  if used_kb="$(du -sxk "$VOL" 2>/dev/null | cut -f1)" && [[ $used_kb =~ ^[0-9]+$ ]]; then
    free_gb=$(( VOL_QUOTA_GB - used_kb / 1024 / 1024 ))
    if (( free_gb >= VOL_MIN_FREE_GB )); then
      ok volume-free "used $((used_kb / 1024 / 1024)) GB of the ${VOL_QUOTA_GB} GB quota, ${free_gb} GB left (needs >= ${VOL_MIN_FREE_GB} GB)"
      info volume-quota "${VOL_QUOTA_GB} GB is configured (VOL_QUOTA_GB), not read from RunPod: confirm size and usage in Console > Storage"
    else bad volume-free "used $((used_kb / 1024 / 1024)) GB of ${VOL_QUOTA_GB} GB, only ${free_gb} GB left (needs >= ${VOL_MIN_FREE_GB} GB)"; fi
  else bad volume-free "du could not measure $VOL"; fi
  if [[ -e $W ]]; then
    info cas-dir "$W exists: $(ls -A "$W" | tr '\n' ' ')"
    if [[ -d $W/locks/workspace.lease ]]; then warn cas-lease "$W/locks/workspace.lease exists (owner $(cat "$W/locks/workspace.lease/owner.json" 2>/dev/null))"; fi
    if [[ -s $W/pgdata/PG_VERSION ]]; then warn cas-pgdata "$W/pgdata already initialised: POSTGRES_PASSWORD must be the one used then"; fi
  else
    ok cas-dir "$W does not exist yet (first boot creates it)"
  fi
  for e in "$VOL"/* "$VOL"/.[!.]*; do
    [[ -e $e && $e != "$W" ]] && info volume-entry "$e (never written by these scripts or the image)"
  done
}

check_checkpoints() {
  local f got missing=0
  for f in "${ASR_FILES[@]}"; do [[ -f $ASR_DIR/$f ]] || { bad asr-files "missing $ASR_DIR/$f"; missing=1; }; done
  for f in "${MT_FILES[@]}"; do [[ -f $MT_DIR/$f ]] || { bad mt-files "missing $MT_DIR/$f"; missing=1; }; done
  if (( missing == 0 )); then ok checkpoint-files "ASR ${#ASR_FILES[@]}/${#ASR_FILES[@]}, MT ${#MT_FILES[@]}/${#MT_FILES[@]}"; fi
  for f in "${ASR_OPTIONAL[@]}"; do
    if [[ -f $ASR_DIR/$f ]]; then info asr-optional "$f present (added to the manifest)"; fi
  done
  echo "      hashing the two weight files (about 3 GB on the network volume)..."
  got="$(sha256sum "$ASR_DIR/model.safetensors" | cut -d' ' -f1)"
  if [[ $got == "$ASR_SHA" ]]; then ok asr-weights "sha256 $got"; else bad asr-weights "sha256 ${got:-<unreadable>} != pin $ASR_SHA"; fi
  got="$(sha256sum "$MT_DIR/model.safetensors" | cut -d' ' -f1)"
  if [[ $got == "$MT_SHA" ]]; then ok mt-weights "sha256 $got"; else bad mt-weights "sha256 ${got:-<unreadable>} != pin $MT_SHA"; fi
  local msg
  if msg="$(python3 - "$CONTRACT" "$TOK_ID" "$TOK_REV" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
fp, mid, rev = c.get("tokenizer_fingerprint"), c.get("model_id"), c.get("model_revision")
problems = []
if not fp: problems.append("no tokenizer_fingerprint")
if mid and mid != sys.argv[2]: problems.append(f"model_id {mid}")
if rev and rev != sys.argv[3]: problems.append(f"model_revision {rev}")
print("; ".join(problems) or f"tokenizer_fingerprint {fp[:16]}..., model {mid}@{(rev or '')[:8]}")
sys.exit(1 if problems else 0)
PY
)"; then ok contract "$msg"; else bad contract "$CONTRACT: ${msg:-unreadable}"; fi
}

# The image runs Speech as uid 10013. It must read every checkpoint file through the
# whole directory chain; the image never chmods the research tree.
check_access() {
  local p n=0 denied=()
  for p in "${ASR_FILES[@]/#/$ASR_DIR/}" "${MT_FILES[@]/#/$MT_DIR/}" "$CONTRACT"; do
    n=$((n + 1)); can_read_as "$SPEECH_UID" "$p" || denied+=("$p")
  done
  if (( ${#denied[@]} == 0 )); then ok speech-read "uid $SPEECH_UID can read all $n files"
  else
    bad speech-read "uid $SPEECH_UID cannot read ${#denied[@]}/$n files (first: ${denied[0]}); see README 'Blockers'"
    namei -l "${denied[0]}" 2>/dev/null | sed 's/^/      /'
  fi
}

check_activity() {
  local pids=() pid fd cwd jup
  for pid in /proc/[0-9]*; do
    pid="${pid#/proc/}"
    [[ $pid == "$$" || $pid == "$BASHPID" ]] && continue
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null)"
    if [[ $cwd == "$VOL" || $cwd == "$VOL"/* ]]; then pids+=("$pid"); continue; fi
    for fd in "/proc/$pid/fd/"*; do
      if [[ "$(readlink "$fd" 2>/dev/null)" == "$VOL"/* ]]; then pids+=("$pid"); break; fi
    done
  done
  if (( ${#pids[@]} == 0 )); then ok workspace-idle "no process uses /workspace"
  else
    warn workspace-idle "${#pids[@]} process(es) use /workspace; finish or stop RQ work before apply and the switch:"
    for pid in "${pids[@]}"; do printf '      %s %s\n' "$pid" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)"; done
  fi
  jup="$(pgrep -af 'jupyter|ipykernel|tensorboard|python .*train' | grep -v pod_stage)"
  if [[ -n "$jup" ]]; then info processes "$(wc -l <<<"$jup") jupyter/python job(s) running; the switch stops them"; fi
}

# Files written on the container disk since this container started are lost when the
# image changes (the container disk is recreated; /workspace stays).
container_changes() {
  find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /tmp -o -path /var/cache -o -path /var/log -o -path /run -o -path /workspace \) -prune \
    -o -type f -newer /etc/hostname -printf '%s\t%p\n' 2>/dev/null
}
check_container_disk() {
  local list n top
  list="$(container_changes)"; n="$(grep -c . <<<"$list")"
  info container-disk "$n files changed since container start (lost at the image switch); largest dirs:"
  awk -F'\t' 'NF == 2 {split($2,p,"/"); d="/"p[2]"/"p[3]"/"p[4]; s[d]+=$1; c[d]++} END {for (d in s) printf "%d %d %s\n", s[d], c[d], d}' <<<"$list" \
    | sort -rn | head -15 | awk '{printf "      %8.1f MB %6d files  %s\n", $1/1048576, $2, $3}'
  for top in /root/.ssh /root/.jupyter /root/.config /root/.cache/huggingface /root/.netrc /root/.git-credentials; do
    if [[ -e $top ]]; then info keep-candidate "$top ($(du -sh "$top" 2>/dev/null | cut -f1))"; fi
  done
}

check_source() {
  local head dirty
  if ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then bad source "$ROOT is not a git clone"; return; fi
  head="$(git -C "$ROOT" rev-parse --short HEAD)"
  # The working tree is what runs, not HEAD: refuse any local change to that code.
  if ! dirty="$(git -C "$ROOT" status --porcelain --untracked-files=all --ignored=matching -- "${SOURCE_PATHS[@]}")"; then
    bad source-clean "git status failed in $ROOT"
  elif [[ -n $dirty ]]; then
    bad source-clean "working tree differs from commit $head: $(tr '\n' ';' <<<"$dirty" | cut -c1-300) (clone again)"
  else
    ok source-clean "no modified, staged, untracked or ignored files in the code this script runs"
  fi
  if [[ -z $EXPECT_COMMIT ]]; then
    if [[ ${MODE:-} == apply ]]; then bad source-commit "CAS_STAGE_EXPECT_COMMIT is not set: give the full SHA of the reviewed commit"
    else warn source-commit "CAS_STAGE_EXPECT_COMMIT not set; clone is at $(git -C "$ROOT" rev-parse HEAD) (apply requires it)"; fi
  elif [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECT_COMMIT" ]]; then
    ok source-commit "clone is at the reviewed commit $EXPECT_COMMIT"
  else
    bad source-commit "clone is at $(git -C "$ROOT" rev-parse HEAD), not the reviewed commit $EXPECT_COMMIT"
  fi
  if ! git -C "$ROOT" cat-file -e "$IMAGE_REVISION^{commit}" 2>/dev/null; then
    bad source "image revision ${IMAGE_REVISION:0:7} not in this clone (use git clone --filter=blob:none, not --depth 1)"
  elif git -C "$ROOT" diff --quiet "$IMAGE_REVISION" HEAD -- bahnar-s2tt-thesis/src deploy/runpod/fetch_models.sh \
         deploy/runpod/models apps/ai-service/app/indexing/embed.py; then
    ok source "C0 src, fetch_models.sh and e5 manifests equal image revision ${IMAGE_REVISION:0:7} (clone at $head)"
  else
    bad source "clone at $head differs from image revision ${IMAGE_REVISION:0:7} in C0/e5 files"
  fi
  if ! can_read_as "$SPEECH_UID" "$ROOT/bahnar-s2tt-thesis/src/c0_tokenizer.py"; then
    bad source-read "uid $SPEECH_UID cannot read the clone (clone it under /tmp, not /root)"
  fi
}

# Every write stays inside /workspace/cas only if no path on the way is a symlink:
# /workspace and /workspace/cas must be real directories at their own path, no
# directory this script writes to may be a symlink, and no symlink under /workspace/cas
# may resolve outside it (the tokenizer cache's own relative links stay inside).
check_paths() {
  local d l t n=0
  if [[ -L $VOL || "$(realpath -m "$VOL")" != "$VOL" ]]; then bad paths "$VOL is a symlink or resolves elsewhere"; n=1; fi
  if [[ -L $W || "$(realpath -m "$W")" != "$W" ]]; then bad paths "$W is a symlink or resolves to $(realpath -m "$W")"; n=1; fi
  for d in "$W/models" "$W/models/embedding" "$W/hf-cache" "$W/config" "$W/manifests" "$W/preflight" "$W/preserve"; do
    if [[ -L $d ]]; then bad paths "$d is a symlink (-> $(readlink "$d"))"; n=1; fi
  done
  if [[ -d $W && ! -L $W ]]; then
    while IFS= read -r -d '' l; do
      t="$(realpath -m "$l")"
      if [[ $t != "$W"/* ]]; then bad paths "symlink $l -> $t points outside $W"; n=1; fi
    done < <(find "$W" -type l -print0 2>/dev/null)
  fi
  if (( n == 0 )); then ok paths "no symlink leads a write outside $W"; fi
}

cmd_check() {
  echo "== pod_stage.sh check (read-only) $(date -u +%FT%TZ)"
  check_host; check_mount; check_paths; check_volume; check_checkpoints; check_access; check_activity; check_container_disk; check_source
  echo "== $fails FAIL, $warns WARN"
  exit $(( fails > 0 ))
}

cmd_preserve() {
  local dest p
  dest="$W/preserve/${RUNPOD_POD_ID:-$(hostname)}-$(date -u +%Y%m%dT%H%M%SZ)"
  local paths=()
  check_mount; check_paths
  (( fails == 0 )) || die "volume not mounted or unsafe paths under $W; nothing written"
  for p in "$@"; do
    [[ -e $p ]] || die "no such path: $p"
    p="$(realpath "$p")"
    [[ $p != "$VOL" && $p != "$VOL"/* ]] || die "$p is on the volume already; it survives the switch"
    paths+=("${p#/}")
  done
  install -d -m 0700 "$dest" || die "cannot create $dest"
  python3 -m pip freeze > "$dest/pip-freeze.txt" 2>/dev/null
  if command -v conda >/dev/null; then conda list > "$dest/conda-list.txt" 2>/dev/null; fi
  nvidia-smi -q > "$dest/nvidia-smi.txt" 2>/dev/null
  env | cut -d= -f1 | sort > "$dest/env-names.txt"          # names only: values may be secrets
  ps -eo pid,user,etime,args --sort=pid > "$dest/processes.txt"
  crontab -l > "$dest/crontab.txt" 2>/dev/null
  container_changes | sort -t$'\t' -k2 > "$dest/container-disk-changes.tsv"
  if (( $# > 0 )); then
    tar -C / -czf "$dest/container-files.tar.gz" "${paths[@]}" || die "tar failed; $dest is incomplete"
    tar -tzf "$dest/container-files.tar.gz" > "$dest/container-files.list"
  fi
  ( cd "$dest" && sha256sum -- * > SHA256SUMS ) || die "cannot write $dest/SHA256SUMS"
  echo "preserved in $dest:"; ls -la "$dest"
  echo "NOTE: secrets in the tar (ssh keys, tokens) are now on the volume: keep $dest at 0700 or move it off RunPod."
}

fs_probe() {  # what the image does on first boot: chown to service uids, 0700, fsync, rename, flock, mkdir
  local d="$W/.fs-probe.$$"
  install -d -m 0755 "$d" || die "cannot create $d"
  if chown "$PG_UID:$PG_UID" "$d" && chmod 0700 "$d" && python3 - "$PG_UID" "$d" <<'PY'
import fcntl, os, sys
uid, d = int(sys.argv[1]), sys.argv[2]
os.setgroups([]); os.setgid(uid); os.setuid(uid)
p = os.path.join(d, "f")
with open(p + ".tmp", "wb") as f:
    f.write(b"x" * 8192); f.flush(); os.fsync(f.fileno())
os.replace(p + ".tmp", p)
with open(p, "rb") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
os.mkdir(os.path.join(d, "lease"))
PY
  then
    if [[ "$(stat -c '%u %a' "$d")" == "$PG_UID 700" ]]; then ok fs-probe "chown/0700/fsync/rename/flock/mkdir work on the volume"
    else bad fs-probe "ownership/mode not kept: $(stat -c '%u %a' "$d")"; fi
  else
    bad fs-probe "the volume refused chown/chmod/fsync/flock (Postgres/MinIO/Qdrant setup would fail)"
  fi
  if python3 -c "import os,sys; fd=os.open(sys.argv[1], os.O_CREAT|os.O_WRONLY|os.O_DIRECT, 0o600); os.close(fd)" "$d/odirect" 2>/dev/null
  then ok fs-odirect "O_DIRECT supported (MinIO)"
  else warn fs-odirect "O_DIRECT refused: MinIO may refuse the drive; check its log after the switch"; fi
  rm -rf "$d"
}

install_same() {  # install_same <src> <dest> <mode>: create, or accept an identical file; never overwrite
  if [[ -e $2 ]]; then
    if cmp -s "$1" "$2"; then ok install "$2 (already identical)"; return 0; fi
    bad install "$2 exists and differs; not overwritten (diff it, move it aside yourself)"; return 1
  fi
  install -D -o root -g root -m "$3" "$1" "$2" || die "cannot install $2"
  ok install "$2"
}

tokenizer_cache() {
  local py out rc=0 home=/tmp/cas-c0-home
  # A fresh venv on every run: nothing installed earlier by hand takes part in the check.
  rm -rf "$TOK_VENV"
  python3 -m venv "$TOK_VENV" 2>/dev/null || { rm -rf "$TOK_VENV"; python3 -m virtualenv -q "$TOK_VENV"; } \
    || die "cannot create a venv in $TOK_VENV (apt-get install python3-venv, or pip install virtualenv)"
  py="$TOK_VENV/bin/python"
  # Same pins as bahnar-s2tt-thesis/requirements-c0-inference.txt (no torch: tokenizer only).
  "$py" -m pip install -q --disable-pip-version-check transformers==4.57.6 'tokenizers>=0.15,<1' \
    'sentencepiece>=0.2,<1' 'numpy>=1.24,<2.3' 'PyYAML>=6.0,<7' || die "pip install into $TOK_VENV failed"
  install -d -m 0755 "$W/hf-cache" || die "cannot create $W/hf-cache"
  HF_HOME="$W/hf-cache" HF_HUB_DISABLE_PROGRESS_BARS=1 "$py" - "$TOK_ID" "$TOK_REV" <<'PY' || die "tokenizer download failed"
import sys
from huggingface_hub import snapshot_download
print("snapshot", snapshot_download(sys.argv[1], revision=sys.argv[2],
      ignore_patterns=["*.bin", "*.safetensors", "*.h5", "*.msgpack", "*.ot", "*.onnx"]))
PY
  chmod -R a+rX "$W/hf-cache"
  # The Speech serving gate, run as the image's speech uid against the read-only cache:
  # offline load of the pinned Hub tokenizer + RQ1 contract fingerprint.
  install -d -o "$SPEECH_UID" -g "$SPEECH_UID" -m 0700 "$home"
  out="$(cd "$ROOT/bahnar-s2tt-thesis" && as_uid "$SPEECH_UID" env HOME="$home" HF_HOME="$W/hf-cache" HF_HUB_OFFLINE=1 \
          TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/bahnar-s2tt-thesis" \
          "$py" -m src.c0_tokenizer --contract "$CONTRACT" 2>&1)" || rc=$?
  if (( rc == 0 )) && grep -q '"tokenizer_parity": "VERIFIED"' <<<"$out"; then
    ok tokenizer "uid $SPEECH_UID, offline: $TOK_ID@${TOK_REV:0:8} matches the contract fingerprint (VERIFIED)"
  else
    bad tokenizer "offline parity check failed (rc=$rc): $(grep -E '"(tokenizer_parity|error|error_code)"' <<<"$out" | tr -s ' ' | tr '\n' ' ' | cut -c1-400)"
  fi
}

write_manifest() {
  local tmp="$W/manifests/.c0_checkpoints.sha256.$$" dest="$W/manifests/c0_checkpoints.sha256" f files=() unreadable=0
  for f in "${ASR_FILES[@]}"; do files+=("$ASR_DIR/$f"); done
  for f in "${ASR_OPTIONAL[@]}"; do
    if [[ -f $ASR_DIR/$f ]]; then files+=("$ASR_DIR/$f"); fi
  done
  for f in "${MT_FILES[@]}"; do files+=("$MT_DIR/$f"); done
  files+=("$CONTRACT")
  [[ -d $TOK_SNAP ]] || { bad manifest "tokenizer snapshot missing: $TOK_SNAP"; return 1; }
  while IFS= read -r f; do files+=("$f"); done < <(find "$TOK_SNAP" -mindepth 1 \( -type f -o -type l \) | sort)
  install -d -o root -g root -m 0755 "$W/manifests" || die "cannot create $W/manifests"
  sha256sum -- "${files[@]}" > "$tmp" || { rm -f "$tmp"; bad manifest "cannot hash the inference files"; return 1; }
  grep -qxF "$ASR_SHA  $ASR_DIR/model.safetensors" "$tmp" || { rm -f "$tmp"; bad manifest "ASR weights do not match the pin"; return 1; }
  grep -qxF "$MT_SHA  $MT_DIR/model.safetensors" "$tmp" || { rm -f "$tmp"; bad manifest "MT weights do not match the pin"; return 1; }
  if [[ -e $dest ]] && ! cmp -s "$tmp" "$dest"; then
    mv "$tmp" "$dest.new"; bad manifest "$dest exists and differs; the new one is at $dest.new"; return 1
  fi
  mv -f "$tmp" "$dest" && chmod 0644 "$dest" || die "cannot write $dest"
  # Exactly what entrypoint.sh runs in gpu mode.
  if ( cd / && sha256sum --strict --quiet -c "$dest" ); then ok manifest "$dest: $(wc -l < "$dest") files verified (sha256sum -c, as entrypoint.sh)"
  else bad manifest "sha256sum -c $dest failed"; fi
  while read -r _ f; do
    can_read_as "$SPEECH_UID" "$f" || { unreadable=1; bad manifest-read "uid $SPEECH_UID cannot read $f"; }
  done < "$dest"
  if (( unreadable == 0 )); then ok manifest-read "uid $SPEECH_UID can read every manifest file"; fi
}

# Fail-fast: the first failed step stops apply, before any later write. Every step is
# idempotent, so fixing the cause and running apply again continues from there.
# READY_TO_SWITCH is written last, only when every step passed; apply removes it
# first, so a stale marker never survives a failed run.
gate() {  # gate <step>
  (( fails == 0 )) && return 0
  echo "== STOPPED at step '$1' ($fails FAIL). NOT ready to switch: do not change the image."
  echo "   $W is partially prepared (completed steps are kept and re-checked on the next run); fix the cause, then run apply again."
  exit 1
}

cmd_apply() {
  echo "== pod_stage.sh apply (writes under $W only) $(date -u +%FT%TZ)"
  check_mount; check_paths
  # A stale marker never survives a failed run. Removed even when a path check failed,
  # unless the volume is not mounted or the way to it is a symlink (then it is not on
  # the volume under $W; rm -f on a symlinked marker itself removes only the link).
  if python3 -c 'import os,sys; sys.exit(0 if os.path.ismount(sys.argv[1]) else 1)' "$VOL" \
     && [[ ! -L $W && ! -L $W/preflight && "$(realpath -m "$W")" == "$W" ]]; then rm -f "$READY"; fi
  gate "volume mount + path safety (nothing written)"
  check_host; check_source; check_volume; check_checkpoints; check_access; check_activity
  gate "pre-checks (nothing written)"
  install -d -o root -g root -m 0755 "$W" || die "cannot create $W"
  fs_probe;                                                                   gate "filesystem probe"
  if CAS_MODEL_VARIANT=fp32 "$ROOT/deploy/runpod/fetch_models.sh" --variant fp32 "$W/models/embedding"; then
    ok embedding "fp32 e5 model verified in $W/models/embedding"
  else bad embedding "fetch_models.sh failed"; fi;                           gate "e5 fp32 model"
  tokenizer_cache;                                                            gate "tokenizer cache + parity"
  install_same "$HERE/c0_inference.yaml" "$W/config/c0_inference.yaml" 0644; gate "c0_inference.yaml"
  write_manifest;                                                             gate "checkpoint manifest"
  install -d -o root -g root -m 0755 "$W/preflight" || die "cannot create $W/preflight"
  install -m 0755 "$HERE/verify_after_switch.sh" "$HERE/rq_fingerprint.py" "$W/preflight/" || die "cannot copy the verify scripts"
  if python3 "$HERE/rq_fingerprint.py" "$VOL" --exclude "$W" --listing-out "$W/preflight/rq_tree.listing.gz" \
       > "$W/preflight/rq_tree.fingerprint"; then
    ok rq-fingerprint "$(cat "$W/preflight/rq_tree.fingerprint")"
  else bad rq-fingerprint "could not fingerprint $VOL"; fi;                  gate "RQ fingerprint"
  {
    echo "ready_at=$(date -u +%FT%TZ)"
    echo "clone=$(git -C "$ROOT" rev-parse HEAD)"
    echo "image_revision=$IMAGE_REVISION"
    echo "manifest_sha256=$(sha256sum "$W/manifests/c0_checkpoints.sha256" | cut -d' ' -f1)"
    echo "config_sha256=$(sha256sum "$W/config/c0_inference.yaml" | cut -d' ' -f1)"
    echo "warnings=$warns"
    echo "simulated_hardware=$([[ -n $SIM_HW ]] && echo yes || echo no)"
  } > "$READY.tmp" && mv -f "$READY.tmp" "$READY" || die "cannot write $READY"
  echo "== 0 FAIL, $warns WARN: READY_TO_SWITCH written to $READY"
  (( warns == 0 )) || echo "   Read every WARN above before switching."
  exit 0
}

MODE="${1:-}"
case "$MODE" in
  check) cmd_check ;;
  preserve) shift; cmd_preserve "$@" ;;
  apply) cmd_apply ;;
  *) echo "usage: $0 check | preserve [container-disk path...] | apply" >&2; exit 2 ;;
esac
