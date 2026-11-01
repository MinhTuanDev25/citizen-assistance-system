#!/usr/bin/env bash
# Offline tests of the staging kit, in throwaway Linux containers (Docker):
#   offline_harness.sh  simulated pod layout on a mounted /workspace (fake checkpoints,
#                       real tokenizer from the Hub, real e5 fp32 model)
#   nomount.sh          /workspace not mounted: check, preserve and apply write nothing
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
if (( h == 1 || n != 0 )); then exit 1; fi
exit "$h"
