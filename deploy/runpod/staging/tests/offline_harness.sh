#!/usr/bin/env bash
# Offline test of deploy/runpod/staging on a simulated pod layout; run it through
# run_offline_tests.sh. Every test asserts its expectation; the last line is
# "HARNESS: N passed, M failed" and the exit code is 1 on any failure.
#   - checkpoints are FAKE files; this throwaway clone gets a commit "TEST ONLY" that pins
#     their hashes (the real script pins the real SHA-256);
#   - the tokenizer (vinai/bartpho-syllable@36eee8b4) and the e5 fp32 model are real;
#   - the contract fingerprint is computed here with the same code (proves the mechanism,
#     not the real contract);
#   - CPU/GPU: the Mac container has neither AVX2 nor a GPU. Tests that need them use
#     CAS_STAGE_TEST_HW (a fake cpuinfo + nvidia-smi), which the script reports as
#     SIMULATED and records in READY_TO_SWITCH.
set -uo pipefail
pass=0 failn=0
t()      { echo; echo "######## $*"; }
check()  { if [[ $2 == "$3" ]]; then echo "  ok   $1"; pass=$((pass + 1)); else echo "  FAIL $1 (expected '$3', got '$2')"; failn=$((failn + 1)); fi; }
has()    { if grep -qE -- "$2" "$3"; then echo "  ok   $1"; pass=$((pass + 1)); else echo "  FAIL $1 (no line matching '$2')"; failn=$((failn + 1)); fi; }
hasnt()  { if grep -qE -- "$2" "$3"; then echo "  FAIL $1 (unexpected line matching '$2')"; failn=$((failn + 1)); else echo "  ok   $1"; pass=$((pass + 1)); fi; }
exists() { if [[ -e $2 ]]; then echo "  ok   $1"; pass=$((pass + 1)); else echo "  FAIL $1 ($2 missing)"; failn=$((failn + 1)); fi; }
absent() { if [[ -e $2 || -L $2 ]]; then echo "  FAIL $1 ($2 exists)"; failn=$((failn + 1)); else echo "  ok   $1"; pass=$((pass + 1)); fi; }
run()    { local out="$1"; shift; "$@" > "$out" 2>&1; echo $?; }

git config --global safe.directory '*'; git config --global user.email test@local; git config --global user.name test
git clone -q /src /tmp/cas-src && git -C /tmp/cas-src fetch -q /src refs/remotes/origin/main \
  && git -C /tmp/cas-src checkout -q b474debecb6be3e936bd149fc0a3dcbfa415a677
mkdir -p /tmp/cas-src/deploy/runpod/staging && cp -r /staging/. /tmp/cas-src/deploy/runpod/staging/
S=/tmp/cas-src/deploy/runpod/staging/pod_stage.sh
W=/workspace/cas
RQ=/workspace/bahnar-s2tt-thesis/bahnar_s2tt
ASR=$RQ/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500
MT=$RQ/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500
C=$RQ/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json
mkdir -p "$ASR" "$MT" /workspace/other-notebooks && echo nb > /workspace/other-notebooks/a.ipynb
for f in model.safetensors config.json vocab.json tokenizer_config.json preprocessor_config.json special_tokens_map.json; do head -c 4096 /dev/urandom > "$ASR/$f"; done
for f in model.safetensors config.json tokenizer_config.json sentencepiece.bpe.model dict.txt generation_config.json optimizer.pt; do head -c 4096 /dev/urandom > "$MT/$f"; done
python -m venv /tmp/ref && /tmp/ref/bin/pip install -q --disable-pip-version-check transformers==4.57.6 'sentencepiece>=0.2,<1' 'tokenizers>=0.15,<1' 'numpy<2.3' PyYAML
( cd /tmp/cas-src/bahnar-s2tt-thesis && HF_HOME=/tmp/hf-ref PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /tmp/ref/bin/python - "$C" <<'PY'
import json, sys
from src.mt_tokenize import load_mt_tokenizer, tokenizer_fingerprint
rev = "36eee8b4d648dd99da56462edcda3c5c97f7f3de"
t = load_mt_tokenizer("vinai/bartpho-syllable", rev)
json.dump({"model_id": "vinai/bartpho-syllable", "model_revision": rev, "tokenizer_fingerprint": tokenizer_fingerprint(t)}, open(sys.argv[1], "w"))
PY
) 2>/dev/null
A=$(sha256sum "$ASR/model.safetensors" | cut -c1-64); M=$(sha256sum "$MT/model.safetensors" | cut -c1-64)
sed -i "s/^ASR_SHA=.*/ASR_SHA=$A/; s/^MT_SHA=.*/MT_SHA=$M/" "$S"
sed -i "s/^asr_weights_sha256: .*/asr_weights_sha256: $A/; s/^mt_weights_sha256: .*/mt_weights_sha256: $M/" /tmp/cas-src/deploy/runpod/staging/c0_inference.yaml
git -C /tmp/cas-src add -A deploy/runpod/staging && git -C /tmp/cas-src commit -qm "TEST ONLY: staging kit + pins of the fake weights"
export CAS_STAGE_EXPECT_COMMIT="$(git -C /tmp/cas-src rev-parse HEAD)"
HW=/tmp/sim-hw; mkdir -p $HW
printf 'model name\t: SIMULATED AMD EPYC 7B13\nflags\t\t: fpu sse sse2 avx avx2 fma\n' > $HW/cpuinfo
cat > $HW/nvidia-smi <<'SH'
#!/bin/sh
case "$*" in
  -L) echo "GPU 0: SIMULATED NVIDIA GeForce RTX 4090 (UUID: GPU-sim)" ;;
  *driver_version*) echo "570.00" ;;
  *name,memory.total*) echo "SIMULATED NVIDIA GeForce RTX 4090, 24564 MiB, 1 MiB" ;;
  *query-compute-apps*) ;;
  "") echo "| NVIDIA-SMI 570.00   Driver Version: 570.00   CUDA Version: 12.8 |" ;;
esac
SH
chmod +x $HW/nvidia-smi
cd /tmp

t "T0 the reviewed commit: apply refuses an unset or a different CAS_STAGE_EXPECT_COMMIT"
rc=$(run /tmp/t0 env -u CAS_STAGE_EXPECT_COMMIT CAS_STAGE_TEST_HW=$HW bash "$S" apply); check "unset: rc" "$rc" 1
has "unset: FAIL source-commit" '^FAIL  source-commit .*not set' /tmp/t0; absent "unset: nothing written" $W
rc=$(run /tmp/t0b env CAS_STAGE_EXPECT_COMMIT=b474debecb6be3e936bd149fc0a3dcbfa415a677 CAS_STAGE_TEST_HW=$HW bash "$S" apply); check "other commit: rc" "$rc" 1
has "other commit: FAIL source-commit" '^FAIL  source-commit .*not the reviewed commit' /tmp/t0b
rc=$(run /tmp/t0c env -u CAS_STAGE_EXPECT_COMMIT CAS_STAGE_TEST_HW=$HW bash "$S" check); check "check without it only warns: rc" "$rc" 0
has "check: WARN source-commit" '^WARN  source-commit' /tmp/t0c
has "volume is a mount point" '^PASS  volume-mount' /tmp/t0c

t "T1 check on the real (Mac) hardware: FAIL cpu + gpu, rc 1, writes nothing"
rc=$(run /tmp/t1 bash "$S" check); check "rc" "$rc" 1
has "FAIL cpu" '^FAIL  cpu ' /tmp/t1; has "FAIL gpu" '^FAIL  gpu ' /tmp/t1; absent "nothing created" $W

t "T1b check with SIMULATED hardware: rc 0, reported as simulated"
rc=$(run /tmp/t1b env CAS_STAGE_TEST_HW=$HW bash "$S" check); check "rc" "$rc" 0
has "WARN simulated-hw" '^WARN  simulated-hw' /tmp/t1b; has "PASS source-clean" '^PASS  source-clean' /tmp/t1b
has "PASS paths" '^PASS  paths' /tmp/t1b; has "quota is configured, not read" 'not read from RunPod' /tmp/t1b

t "T2 unreadable checkpoint file -> FAIL speech-read"
chmod 0600 "$MT/dict.txt"; rc=$(run /tmp/t2 env CAS_STAGE_TEST_HW=$HW bash "$S" check); chmod 0644 "$MT/dict.txt"
check "rc" "$rc" 1; has "FAIL speech-read" '^FAIL  speech-read' /tmp/t2

t "T2b apply on the real (Mac) hardware -> STOPPED at pre-checks, nothing written (hardware gate)"
rc=$(run /tmp/t2b bash "$S" apply); check "rc" "$rc" 1
has "STOPPED at pre-checks" "STOPPED at step 'pre-checks" /tmp/t2b; absent "no $W" $W

t "T3a apply with a contract fingerprint mismatch -> STOPPED at tokenizer; no config, manifest, READY"
mkdir -p $W/models && cp -r /model $W/models/embedding
cp "$C" /tmp/contract.good
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d['tokenizer_fingerprint']='0'*64; json.dump(d,open(sys.argv[1],'w'))" "$C"
rc=$(run /tmp/t3a env CAS_STAGE_TEST_HW=$HW bash "$S" apply); cp /tmp/contract.good "$C"
check "rc" "$rc" 1; has "STOPPED at tokenizer" "STOPPED at step 'tokenizer" /tmp/t3a
has "reason is the mismatch" 'tokenizer_mismatch' /tmp/t3a
absent "no config" $W/config/c0_inference.yaml; absent "no manifest" $W/manifests/c0_checkpoints.sha256; absent "no READY" $W/preflight/READY_TO_SWITCH

t "T3 apply (simulated hardware) -> READY_TO_SWITCH with simulated_hardware=yes"
rc=$(run /tmp/t3 env CAS_STAGE_TEST_HW=$HW bash "$S" apply); check "rc" "$rc" 0
has "tokenizer VERIFIED as uid 10013, offline" '^PASS  tokenizer .*VERIFIED' /tmp/t3
has "e5 fp32 verified" '^PASS  embedding' /tmp/t3; has "fs-probe" '^PASS  fs-probe' /tmp/t3
exists "READY written" $W/preflight/READY_TO_SWITCH
has "READY says simulated" '^simulated_hardware=yes$' $W/preflight/READY_TO_SWITCH
has "READY records the reviewed commit" "^clone=$CAS_STAGE_EXPECT_COMMIT$" $W/preflight/READY_TO_SWITCH
has "READY has the manifest hash" "^manifest_sha256=$(sha256sum $W/manifests/c0_checkpoints.sha256 | cut -c1-64)$" $W/preflight/READY_TO_SWITCH

t "T4 apply again: idempotent"
rc=$(run /tmp/t4 env CAS_STAGE_TEST_HW=$HW bash "$S" apply); check "rc" "$rc" 0
has "config already identical" 'c0_inference.yaml \(already identical\)' /tmp/t4

t "T5 manifest: as entrypoint.sh checks it"
( cd / && sha256sum --strict --quiet -c $W/manifests/c0_checkpoints.sha256 ); check "sha256sum -c" "$?" 0
check "22 files" "$(wc -l < $W/manifests/c0_checkpoints.sha256)" 22
hasnt "no training state" 'optimizer.pt' $W/manifests/c0_checkpoints.sha256
has "real ASR pin line" "^$A  $ASR/model.safetensors$" $W/manifests/c0_checkpoints.sha256

t "T6 config changed after apply -> not overwritten, STOPPED, READY removed"
echo "# local edit" >> $W/config/c0_inference.yaml
rc=$(run /tmp/t6 env CAS_STAGE_TEST_HW=$HW bash "$S" apply); sed -i '$d' $W/config/c0_inference.yaml
check "rc" "$rc" 1; has "STOPPED at config" "STOPPED at step 'c0_inference.yaml'" /tmp/t6; absent "READY removed" $W/preflight/READY_TO_SWITCH

t "T7 RQ fingerprint: identical, ignores $W, detects a change"
F=$W/preflight
rc=$(run /tmp/t7 python3 $F/rq_fingerprint.py /workspace --exclude $W --compare $F/rq_tree.listing.gz); check "identical" "$rc" 0
touch $W/newfile; rc=$(run /tmp/t7 python3 $F/rq_fingerprint.py /workspace --exclude $W --compare $F/rq_tree.listing.gz); check "ignores $W" "$rc" 0; rm $W/newfile
cp "$MT/config.json" /tmp/cfg.good; echo x >> "$MT/config.json"
rc=$(run /tmp/t7 python3 $F/rq_fingerprint.py /workspace --exclude $W --compare $F/rq_tree.listing.gz); check "detects a change" "$rc" 1
has "names the changed file" 'checkpoint-17500/config.json' /tmp/t7; cp /tmp/cfg.good "$MT/config.json"

t "T8 preserve: absolute, relative, on-volume refused, /workspace2 is not the volume"
mkdir -p /root/.jupyter && echo cfg > /root/.jupyter/c.py
rc=$(run /tmp/t8 bash "$S" preserve /root/.jupyter); check "absolute" "$rc" 0
rc=$(cd /root && run /tmp/t8b bash "$S" preserve .jupyter); check "relative" "$rc" 0
has "relative path archived as root/.jupyter" '^root/.jupyter/' "$(ls -d $W/preserve/*/ | tail -1)container-files.list"
rc=$(run /tmp/t8c bash "$S" preserve /workspace/other-notebooks); check "on-volume refused" "$rc" 1
mkdir -p /workspace2 && echo x > /workspace2/f; rc=$(run /tmp/t8d bash "$S" preserve /workspace2); check "/workspace2 accepted" "$rc" 0

t "T9 clone working tree must equal its commit"
echo "# local edit" >> /tmp/cas-src/bahnar-s2tt-thesis/src/c0_tokenizer.py
rc=$(run /tmp/t9 env CAS_STAGE_TEST_HW=$HW bash "$S" apply); git -C /tmp/cas-src checkout -q -- bahnar-s2tt-thesis/src/c0_tokenizer.py
check "modified file: rc" "$rc" 1; has "modified file: FAIL source-clean" '^FAIL  source-clean .*c0_tokenizer.py' /tmp/t9
touch /tmp/cas-src/bahnar-s2tt-thesis/src/extra.py
rc=$(run /tmp/t9b env CAS_STAGE_TEST_HW=$HW bash "$S" check); rm /tmp/cas-src/bahnar-s2tt-thesis/src/extra.py
check "untracked file: rc" "$rc" 1; has "untracked file: FAIL source-clean" '^FAIL  source-clean .*extra.py' /tmp/t9b
echo "# edit" >> "$S"; rc=$(run /tmp/t9c env CAS_STAGE_TEST_HW=$HW bash "$S" check); git -C /tmp/cas-src checkout -q -- deploy/runpod/staging/pod_stage.sh
check "edited staging script: rc" "$rc" 1; has "edited staging script: FAIL source-clean" '^FAIL  source-clean .*pod_stage.sh' /tmp/t9c

t "T10 symlinks: no write may leave $W"
python3 $F/rq_fingerprint.py /workspace --exclude $W --listing-out /tmp/rq.before.gz >/dev/null
mv $F /tmp/preflight.real && ln -s /workspace/bahnar-s2tt-thesis $F
rc=$(run /tmp/t10 env CAS_STAGE_TEST_HW=$HW bash "$S" apply); rm $F && mv /tmp/preflight.real $F
check "symlinked preflight: rc" "$rc" 1; has "STOPPED at mount + path safety" "STOPPED at step 'volume mount . path safety" /tmp/t10
rc=$(run /tmp/t10r python3 $F/rq_fingerprint.py /workspace --exclude $W --compare /tmp/rq.before.gz); check "research tree untouched" "$rc" 0
ln -s /workspace/bahnar-s2tt-thesis $W/models/evil
rc=$(run /tmp/t10b env CAS_STAGE_TEST_HW=$HW bash "$S" check); rm $W/models/evil
check "link out of $W: rc" "$rc" 1; has "link out of $W: FAIL paths" '^FAIL  paths .*evil' /tmp/t10b
mv $W /workspace/cas-real && ln -s /workspace/cas-real $W
rc=$(run /tmp/t10c bash "$S" preserve); rm $W && mv /workspace/cas-real $W
check "symlinked $W: preserve refused" "$rc" 1; has "symlinked $W reported" "FAIL  paths .*$W is a symlink" /tmp/t10c
rc=$(run /tmp/t10d env CAS_STAGE_TEST_HW=$HW bash "$S" apply); check "after restoring: apply passes again" "$rc" 0
exists "READY present before the unsafe run" $W/preflight/READY_TO_SWITCH
ln -s /workspace/bahnar-s2tt-thesis $W/models/evil
rc=$(run /tmp/t10e env CAS_STAGE_TEST_HW=$HW bash "$S" apply); rm $W/models/evil
check "unsafe link: apply rc" "$rc" 1; absent "stale READY removed although the path check failed" $W/preflight/READY_TO_SWITCH

echo
echo "HARNESS: $pass passed, $failn failed"
exit $(( failn > 0 ))
