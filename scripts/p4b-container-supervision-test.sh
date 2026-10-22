#!/usr/bin/env bash
# Build the production AI-service image and run the process-supervision
# regression inside it. The production image is not modified: it has no
# procps, no pytest, and no tests. pytest is installed only in the writable
# container layer of one ephemeral run, and the tests are mounted read-only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${P4B_SUPERVISION_IMAGE:-citizen-assistance-ai-service:supervision}"

BUILD=(docker build -f "$ROOT/apps/ai-service/Dockerfile" -t "$IMAGE" "$ROOT")
printf 'COMMAND:'
printf ' %q' "${BUILD[@]}"
printf '\n'
"${BUILD[@]}"

PS_CHECK=(docker run --rm --entrypoint sh "$IMAGE" -c 'command -v ps')
printf 'COMMAND:'
printf ' %q' "${PS_CHECK[@]}"
printf '\n'
if "${PS_CHECK[@]}"; then
  echo "FAIL: production image provides ps" >&2
  exit 1
fi
echo "ps=absent"

ABSENT=(docker run --rm --entrypoint python "$IMAGE" -c 'import importlib.util, os, sys; sys.exit(0 if importlib.util.find_spec("pytest") is None and not os.path.exists("/app/tests") else 1)')
printf 'COMMAND:'
printf ' %q' "${ABSENT[@]}"
printf '\n'
"${ABSENT[@]}"
echo "pytest=absent tests=absent"

RUN=(
  docker run --rm --init
  -v "$ROOT/apps/ai-service/tests:/opt/cas-tests:ro"
  -e AI_SERVICE_TOKEN=test-service-token
  -e LLM_PROVIDER=mock
  -e PYTHONPYCACHEPREFIX=/tmp/cas-pyc
  -e PYTHONDONTWRITEBYTECODE=1
  --entrypoint python
  "$IMAGE"
  -c 'import shutil,sys,subprocess
assert shutil.which("ps") is None, "ps leaked into the production image"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytest==8.3.4"])
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", "/opt/cas-tests/test_p4b_bundle10.py", "/opt/cas-tests/test_p4b_bundle8.py", "/opt/cas-tests/test_p4b_bundle7.py"]))'
)
printf 'COMMAND:'
printf ' %q' "${RUN[@]}"
printf '\n'
"${RUN[@]}"
