#!/usr/bin/env bash
# Speech C0 verification level (b): inside the image, CPU, no GPU, no checkpoints.
#   C1 c0_code.sha256 matches the staged code; SOURCE_REVISION == image label
#   C2 /opt/cas/c0 holds exactly the staged layout; no checkpoints/research data anywhere
#   C3 import src.c0_speech.app with the C0 venv as user speech; research pins YAML loads
#   C4 pinned ML versions present in the venv
#   C5 the 75 C0 tests (fake runtime) pass against a COPY of the packed code.
#      Test-only inputs are mounted read-only and added to the copy only:
#      tests/ and src/c0_manifest.py (imported by test_c0_cascaded, not by the
#      runtime). --noconftest skips tests/conftest.py, which imports training code.
# Every docker command's exit code is checked before its output is interpreted.
# Exit status: 0 only if all checks PASS; 1 otherwise (CI relies on this).
# This is not level (c): no checkpoint is loaded and no GPU inference runs.
#   deploy/runpod/tests/c0_in_image.sh <image> <out dir>
set -uo pipefail
IMG="${1:?usage: c0_in_image.sh <image> <out dir>}"; OUT="${2:?usage: c0_in_image.sh <image> <out dir>}"
mkdir -p "$OUT"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
T="$ROOT/bahnar-s2tt-thesis"
RES="$OUT/results.tsv"; printf 'id\tresult\tcheck\tevidence\n' > "$RES"
FAILS=0
# Evidence is one TSV cell: tabs, CR and newlines become spaces.
tsv() { printf '%s' "$1" | tr '\t\r\n' '   '; }
pass() { printf '%s\tPASS\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "PASS $1 $2"; }
fail() { printf '%s\tFAIL\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "FAIL $1 $2 :: ${3:-}"; FAILS=$((FAILS + 1)); }
if ! label="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMG" 2>&1)"; then
  for id in C1 C2 C3 C4 C5; do fail "$id" "image not inspectable" "$label"; done
  echo "c0_in_image: $FAILS check(s) failed"; exit 1
fi
# Explicit platform: otherwise docker prints a platform-mismatch WARNING into the
# captured output when an amd64 image runs on an arm64 host (emulation).
PLATFORM="$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$IMG")"
run() { docker run --rm --platform "$PLATFORM" --entrypoint bash "$@"; }

# C1
o="$(run "$IMG" -c 'cd /opt/cas/c0 && sha256sum --strict -c c0_code.sha256 && cat SOURCE_REVISION' 2>&1)"; rc=$?
echo "$o" > "$OUT/C1_manifest.txt"
ok="$(grep -c ': OK$' <<<"$o")"; bad="$(grep -vc -e ': OK$' -e '^[0-9a-f]\{40\}$' <<<"$o")"; rev="$(tail -1 <<<"$o")"
if [[ $rc -eq 0 && "$ok" == 23 && "$bad" == 0 && "$rev" == "$label" && -n "$label" ]]; then
  pass C1 "C0 code manifest 23/23 OK, SOURCE_REVISION == image label" "$rev"
else fail C1 "C0 manifest/revision" "rc=$rc ok=$ok other_lines=$bad rev=$rev label=$label"; fi

# C2: exact top-level layout, and no checkpoint-like files anywhere in the image.
want="SOURCE_FILES SOURCE_REVISION c0_code.sha256 configs requirements-c0-inference.txt requirements-c0-speech-api.txt src "
o="$(run "$IMG" -c 'set -e; find /opt/cas/c0 -maxdepth 1 -mindepth 1 -printf "%f\n" | sort | tr "\n" " "; echo; find / -xdev \( -path /proc -o -path /sys \) -prune -o \( -name checkpoints -o -name direct_full_state -o -name "*.safetensors" -o -name "checkpoint-*" \) -print' 2>&1)"; rc=$?
echo "$o" > "$OUT/C2_layout.txt"
layout="$(head -1 <<<"$o")"; found="$(tail -n +2 <<<"$o" | sed '/^$/d')"
if [[ $rc -eq 0 && "$layout" == "$want" && -z "$found" ]]; then
  pass C2 "/opt/cas/c0 layout is exactly the staged set; no checkpoints in image" "$layout"
else fail C2 "C0 layout / no research data" "rc=$rc layout='$layout' found='$found'"; fi

# C3
o="$(run "$IMG" -c 'cd /opt/cas/c0 && PYTHONPATH=/opt/cas/c0 gosu speech /opt/cas/venv-c0/bin/python -c "
import src.c0_speech.app, yaml
from src.c0_cascaded import DEFAULT_RESEARCH_PINS
pins = yaml.safe_load(open(DEFAULT_RESEARCH_PINS))
assert isinstance(pins, dict) and pins, \"empty research pins\"
print(\"import ok; pins:\", DEFAULT_RESEARCH_PINS, len(pins))
"' 2>&1)"; rc=$?
echo "$o" > "$OUT/C3_import.txt"
if [[ $rc -eq 0 ]] && grep -q "^import ok" <<<"$o"; then pass C3 "import src.c0_speech.app as user speech; c0_research_pins.yaml loads" "$(tail -1 <<<"$o")"
else fail C3 "C0 import" "rc=$rc $(tail -2 <<<"$o" | tr '\n' ' ')"; fi

# C4
o="$(run "$IMG" -c '/opt/cas/venv-c0/bin/python -c "import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__, \"cuda_build=\" + str(torch.version.cuda))"' 2>&1)"; rc=$?
echo "$o" > "$OUT/C4_versions.txt"; v="$(tail -1 <<<"$o")"
if [[ $rc -eq 0 && "$v" == 2.8.0* && "$v" == *" 4.57.6 1.10.1 "* ]]; then pass C4 "pinned torch 2.8.0 / transformers 4.57.6 / accelerate 1.10.1" "$v"
else fail C4 "pinned versions" "rc=$rc $v"; fi

# C5
o="$(run -v "$T/tests:/x/tests:ro" -v "$T/src/c0_manifest.py:/x/c0_manifest.py:ro" "$IMG" -c '
set -e
cp -r /opt/cas/c0 /tmp/c0 && cp /x/c0_manifest.py /tmp/c0/src/ && cp -r /x/tests /tmp/c0/tests
cd /tmp/c0 && PYTHONPATH=/tmp/c0 /opt/cas/venv-c0/bin/python -m pytest -q --noconftest -p no:cacheprovider \
  tests/test_c0_speech.py tests/test_c0_cascaded.py tests/test_c0_tokenizer.py' 2>&1)"; rc=$?
echo "$o" > "$OUT/C5_pytest.txt"
summary="$(grep -E "passed|failed|error" <<<"$o" | tail -1)"
if [[ $rc -eq 0 && "$summary" =~ (^|[^0-9])75\ passed && "$summary" != *failed* && "$summary" != *error* ]]; then
  pass C5 "75 C0 tests (fake runtime) pass on the packed code" "$summary"
else fail C5 "C0 tests in image" "rc=$rc $summary"; fi

if (( FAILS > 0 )); then echo "c0_in_image: $FAILS check(s) failed"; exit 1; fi
echo "c0_in_image: all checks passed"
