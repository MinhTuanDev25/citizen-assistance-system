#!/usr/bin/env bash
# Negative regression for c0_in_image.sh: the checker must FAIL and exit non-zero when
#   N-C0-1  the image does not exist (every check FAIL, none PASS)
#   N-C0-2  a derived image has tampered C0 code, an extra research dir and a broken
#           import (C1, C2, C3 and C5 must FAIL)
# The derived image is built locally from <image>, then removed.
#   deploy/runpod/tests/c0_in_image_negative.sh <image> <out dir>
set -uo pipefail
IMG="${1:?usage: c0_in_image_negative.sh <image> <out dir>}"; OUT="${2:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
RES="$OUT/results.tsv"; printf 'id\tresult\tcheck\tevidence\n' > "$RES"
FAILS=0
# Evidence is one TSV cell: tabs, CR and newlines become spaces.
tsv() { printf '%s' "$1" | tr '\t\r\n' '   '; }
pass() { printf '%s\tPASS\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "PASS $1 $2"; }
fail() { printf '%s\tFAIL\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "FAIL $1 $2 :: ${3:-}"; FAILS=$((FAILS + 1)); }
state() { awk -F'\t' -v id="$2" '$1==id{print $2}' "$1"; }

# N-C0-1: missing image
"$HERE/c0_in_image.sh" "cas-aio:does-not-exist-$$" "$OUT/missing" > "$OUT/missing.log" 2>&1; rc=$?
passes="$(grep -c $'\tPASS\t' "$OUT/missing/results.tsv")"
if [[ $rc -ne 0 && "$passes" == 0 ]]; then pass N-C0-1 "missing image: exit $rc, 0 PASS rows"
else fail N-C0-1 "missing image must fail" "rc=$rc pass_rows=$passes"; fi

# N-C0-2: tampered derived image
BROKEN="cas-aio:c0-negative-$$"
docker build -q -t "$BROKEN" --platform "$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$IMG")" - > "$OUT/broken_build.log" 2>&1 <<EOF
FROM $IMG
RUN echo "# tampered" >> /opt/cas/c0/src/c0_cascaded.py \
 && echo 'raise RuntimeError("negative test: broken import")' > /opt/cas/c0/src/c0_speech/app.py \
 && mkdir /opt/cas/c0/data
EOF
if [[ $? -ne 0 ]]; then fail N-C0-2 "could not build the tampered image" "broken_build.log"
else
  "$HERE/c0_in_image.sh" "$BROKEN" "$OUT/broken" > "$OUT/broken.log" 2>&1; rc=$?
  r="$OUT/broken/results.tsv"
  got="C1=$(state "$r" C1) C2=$(state "$r" C2) C3=$(state "$r" C3) C5=$(state "$r" C5)"
  if [[ $rc -ne 0 && "$got" == "C1=FAIL C2=FAIL C3=FAIL C5=FAIL" ]]; then pass N-C0-2 "tampered image: exit $rc, $got"
  else fail N-C0-2 "tampered image must fail C1/C2/C3/C5" "rc=$rc $got"; fi
  docker image rm -f "$BROKEN" >/dev/null 2>&1
fi

if (( FAILS > 0 )); then echo "c0_in_image_negative: $FAILS check(s) failed"; exit 1; fi
echo "c0_in_image_negative: all checks passed"
