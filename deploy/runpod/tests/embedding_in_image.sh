#!/usr/bin/env bash
# Real ONNX inference of the image's ai-service code on THIS host's CPU (no CPU flag
# override): apps/ai-service/tests/test_embedding_variant.py run with the image's
# venv-ai (pinned onnxruntime/tokenizers) against both pinned models.
#   deploy/runpod/tests/embedding_in_image.sh <image> <fp32 model dir> <int8 model dir|-> <out dir>
# On a CPU without AVX-512 VNNI pass "-" for the int8 dir: the int8 cases are skipped
# and auto must pick fp32. Exit 0 only when pytest passes.
set -euo pipefail
IMG="${1:?usage: embedding_in_image.sh <image> <fp32 dir> <int8 dir|-> <out dir>}"
FP32="$(cd "${2:?}" && pwd)"; INT8="${3:?}"; OUT="${4:?}"; mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"; TESTS="$(cd "$HERE/../../../apps/ai-service/tests" && pwd)"
mounts=(-v "$TESTS:/t/tests:ro" -v "$FP32:/m/fp32:ro"); envs=(-e E5_FP32_MODEL_DIR=/m/fp32)
if [[ "$INT8" != - ]]; then INT8="$(cd "$INT8" && pwd)"; mounts+=(-v "$INT8:/m/int8:ro"); envs+=(-e E5_INT8_MODEL_DIR=/m/int8); fi
# pytest goes into a throwaway copy of the venv-ai interpreter's site (container is --rm).
docker run --rm --entrypoint bash "${mounts[@]}" "${envs[@]}" "$IMG" -c '
  set -e
  echo "cpu: $(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "avx512_vnni: $(grep -qw avx512_vnni /proc/cpuinfo && echo yes || echo no)"
  /opt/cas/venv-ai/bin/pip install -q --disable-pip-version-check pytest==8.3.4 >/dev/null
  /opt/cas/venv-ai/bin/python -c "import onnxruntime, tokenizers; print(\"onnxruntime\", onnxruntime.__version__, \"tokenizers\", tokenizers.__version__)"
  cd /t && PYTHONPATH=/opt/cas/ai /opt/cas/venv-ai/bin/python -m pytest -p no:cacheprovider -rs -v tests/test_embedding_variant.py
' 2>&1 | tee "$OUT/embedding_in_image.log"
