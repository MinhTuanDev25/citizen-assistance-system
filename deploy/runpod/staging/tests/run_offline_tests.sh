#!/usr/bin/env bash
# Offline tests of the staging kit, in throwaway Linux containers (Docker):
#   offline_harness.sh  simulated pod layout on a mounted /workspace (fake checkpoints,
#                       real tokenizer from the Hub, real e5 fp32 model)
#   nomount.sh          /workspace not mounted: check, preserve and apply write nothing
#   test_verify_gates.py + verify_gates_proc.sh + gates_wrapper.sh   GPU and listener gates of
#                       verify_after_switch.sh (fake /proc unit tests; real /proc as root without
#                       CAP_SYS_PTRACE; the Bash wrapper failing closed on crashes and bad output)
#   deploy/runpod/staging/tests/run_offline_tests.sh [e5 fp32 model dir, default models/embedding]
# Needs network (Hub, pip). Both suites always run. Exit 1 if any test fails, else 2 if a
# harness test was skipped for an infrastructure (network) failure, else 0.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
MODEL="$(cd "${1:-$ROOT/models/embedding}" && pwd)"
IMG=python:3.11-bookworm
docker run --rm -v /workspace -v "$ROOT":/src:ro -v "$HERE/..":/staging:ro -v "$MODEL":/model:ro -v "$HERE":/t:ro \
  "$IMG" bash /t/offline_harness.sh; h=$?
docker run --rm -v "$HERE/..":/staging:ro -v "$HERE":/t:ro "$IMG" bash /t/nomount.sh; n=$?
docker run --rm -v "$HERE/..":/staging:ro -v "$HERE":/t:ro "$IMG" bash -c \
  'python3 -m unittest discover -s /staging/tests -p "test_*.py" && bash /t/verify_gates_proc.sh && bash /t/gates_wrapper.sh /staging'; g=$?
if (( h == 1 || n != 0 || g != 0 )); then exit 1; fi
exit "$h"
