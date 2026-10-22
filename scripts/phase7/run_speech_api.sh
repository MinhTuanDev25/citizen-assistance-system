#!/usr/bin/env bash
# Start the internal C0 Speech Inference API with the verified C0 venv.
# One worker. Does not train, download fallback checkpoints, or deploy CAS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAS_ROOT="${CAS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
THESIS_SOURCE_ROOT="${THESIS_SOURCE_ROOT:-${CAS_ROOT}/bahnar-s2tt-thesis}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/bahnar-s2tt-thesis}"
VENV="${C0_INFERENCE_VENV:-/workspace/.venvs/c0-inference}"
export ARTIFACT_ROOT

if [[ -z "${C0_INFERENCE_CONFIG:-}" ]]; then
  echo "C0_INFERENCE_CONFIG is required (path to operator YAML, not a Git example)." >&2
  exit 1
fi
if [[ ! -f "${C0_INFERENCE_CONFIG}" ]]; then
  echo "C0_INFERENCE_CONFIG is not a file: ${C0_INFERENCE_CONFIG}" >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "C0 venv python not found: ${VENV}/bin/python" >&2
  exit 1
fi

export PYTHONPATH="${THESIS_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SPEECH_HOST="${SPEECH_HOST:-127.0.0.1}"
export SPEECH_PORT="${SPEECH_PORT:-8090}"
export SPEECH_WORKERS="${SPEECH_WORKERS:-1}"
export SPEECH_MAX_INFLIGHT="${SPEECH_MAX_INFLIGHT:-1}"

cd "${THESIS_SOURCE_ROOT}"
exec "${VENV}/bin/python" -m src.c0_speech
