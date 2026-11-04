#!/usr/bin/env bash
# Every C0_EQUIVALENT pair in verify_after_switch.sh must be two commits whose C0 sources,
# fetch_models.sh, e5 manifests and embed.py are equal (the files pod_stage.sh compares with
# the image revision). Needs the full git history (CI: fetch-depth 0). Also checks that a pair
# with a real difference is caught, and that pod_stage.sh still compares the same paths.
#   deploy/runpod/staging/tests/c0_equivalence.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PATHS=(bahnar-s2tt-thesis/src deploy/runpod/fetch_models.sh deploy/runpod/models apps/ai-service/app/indexing/embed.py)
pass=0 fail=0
ok()  { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
equal() { git -C "$ROOT" diff --quiet "$1" "$2" -- "${PATHS[@]}"; }

stage_paths="$(grep -A1 'diff --quiet "$IMAGE_REVISION" HEAD --' "$HERE/../pod_stage.sh" | tr -s ' \\\n' ' ')"
for p in "${PATHS[@]}"; do
  [[ $stage_paths == *" $p "* || $stage_paths == *" $p;"* ]] || bad "pod_stage.sh no longer compares $p"
done
(( fail == 0 )) && ok "pod_stage.sh compares the same paths"

pairs="$(sed -n '/^C0_EQUIVALENT=(/,/^)/p' "$HERE/../verify_after_switch.sh" | grep -oE '"[0-9a-f]{40} [0-9a-f]{40}"' | tr -d '"')"
[[ -n $pairs ]] || bad "no C0_EQUIVALENT pair parsed"
while read -r running applied; do
  [[ -n ${running:-} ]] || continue
  for c in "$running" "$applied"; do
    git -C "$ROOT" cat-file -e "$c^{commit}" 2>/dev/null || { bad "$c is not a commit here (full history needed)"; continue 2; }
  done
  if equal "$applied" "$running"; then ok "${running:0:7} keeps the C0/e5 files of ${applied:0:7}"
  else bad "${running:0:7} differs from ${applied:0:7} in C0/e5 files: not an upgrade without apply"; fi
done <<<"$pairs"

# The check itself must see a difference: the last commit that changed those files vs its parent.
changed="$(git -C "$ROOT" log -1 --format=%H -- "${PATHS[@]}")"
if [[ -n $changed ]] && ! equal "$changed^" "$changed"; then ok "a real C0/e5 change (${changed:0:7}) is detected"
else bad "the comparison did not detect the change in ${changed:-<none>}"; fi

echo "C0EQ: $pass PASS, $fail FAIL"
(( fail == 0 ))
