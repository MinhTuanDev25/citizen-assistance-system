#!/usr/bin/env bash
# Canonical Phase 7.3 read-only inspection (CAS layout).
# CAS_ROOT = git checkout (this repo).
# THESIS_SOURCE_ROOT = CAS_ROOT/bahnar-s2tt-thesis (source, configs, venv).
# ARTIFACT_ROOT = trained model/data root (NOT assumed inside THESIS_SOURCE_ROOT).
# Documented RunPod example: ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis
# Does not install packages, load ASR/MT weights, download Hub files, or mutate artifacts.
set -euo pipefail

script_path=${BASH_SOURCE[0]}
while [ -L "$script_path" ]; do
  script_dir=$(cd "$(dirname "$script_path")" && pwd)
  script_path=$(readlink "$script_path")
  case "$script_path" in
    /*) ;;
    *) script_path="$script_dir/$script_path" ;;
  esac
done
script_dir=$(cd "$(dirname "$script_path")" && pwd)
cas_root=$(cd "$script_dir/../.." && pwd)
thesis_source="$cas_root/bahnar-s2tt-thesis"

if [ ! -f "$thesis_source/src/c0_cascaded.py" ]; then
  echo "layout mismatch: expected $thesis_source/src/c0_cascaded.py" >&2
  echo "CAS_ROOT resolved to: $cas_root" >&2
  exit 2
fi

if [ -n "${PHASE7_PYTHON:-}" ]; then
  py=$PHASE7_PYTHON
elif [ -x "$thesis_source/.venv/bin/python" ]; then
  py=$thesis_source/.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  py=python3
else
  echo "python3 not found" >&2
  exit 2
fi

out=${PHASE7_INSPECT_OUT:-/tmp/phase7_runpod_inspection.json}
lib="$script_dir/inspect_runpod_lib.py"
if [ -n "${PHASE7_ARTIFACT_ROOT:-}" ]; then
  exec "$py" "$lib" --cas-root "$cas_root" --artifact-root "$PHASE7_ARTIFACT_ROOT" --output "$out" "$@"
fi
if [ -d /workspace/bahnar-s2tt-thesis/bahnar_s2tt ]; then
  exec "$py" "$lib" --cas-root "$cas_root" --artifact-root /workspace/bahnar-s2tt-thesis --output "$out" "$@"
fi
exec "$py" "$lib" --cas-root "$cas_root" --output "$out" "$@"
