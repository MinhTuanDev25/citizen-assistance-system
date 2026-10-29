#!/usr/bin/env bash
# Offline regression for fetch_models.sh (all-or-nothing publish). A local HTTP server
# plays the model mirror; a two-file manifest is used.
#   FM1 fresh download of both files -> exit 0, dest has both files + manifest.json
#   FM2 dest holds a previous good model; file 2 is missing on the mirror (404)
#       -> exit != 0, dest byte-identical to before, no staging dir left
#   FM3 same, but file 2 has a wrong checksum -> exit != 0, dest unchanged
#   FM4 dest does not exist; file 2 fails -> exit != 0, dest still does not exist
#   FM5 dest already complete and correct -> exit 0, no request reaches the mirror,
#       and the directory itself is not replaced (same inode: bind mounts keep working)
# Publish failures, injected with an mv shim on PATH that fails chosen calls
# (call 1 = move current model aside, 2 = publish new, 3 = restore previous):
#   FM6 publish fails, restore works -> exit != 0, dest identical to before, nothing left over
#   FM7 publish AND restore fail -> exit != 0, previous model KEPT intact in .old, path printed
#   FM8 moving the current model aside fails -> exit != 0, dest identical, no .old
# Exit 0 only if all cases PASS.
#   deploy/runpod/tests/fetch_models_test.sh <out dir>
set -uo pipefail
OUT="${1:?usage: fetch_models_test.sh <out dir>}"; mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"; FETCH="${CAS_FETCH_SCRIPT:-$HERE/../fetch_models.sh}"
RES="$OUT/results.tsv"; printf 'id\tresult\tcheck\tevidence\n' > "$RES"
FAILS=0
# Evidence is one TSV cell: tabs, CR and newlines become spaces.
tsv() { printf '%s' "$1" | tr '\t\r\n' '   '; }
pass() { printf '%s\tPASS\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "PASS $1 $2"; }
fail() { printf '%s\tFAIL\t%s\t%s\n' "$1" "$2" "$(tsv "${3:-}")" >> "$RES"; echo "FAIL $1 $2 :: ${3:-}"; FAILS=$((FAILS + 1)); }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
tree_hash() { ( cd "$1" 2>/dev/null && find . -type f | sort | while read -r f; do echo "$f $(sha "$f")"; done ) | (sha256sum 2>/dev/null || shasum -a 256) | cut -d' ' -f1; }

T="$(mktemp -d)"; trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -rf "$T"' EXIT
REV=0123456789abcdef0123456789abcdef01234567
REPO=test-org/test-model
mkdir -p "$T/srv/$REPO/resolve/$REV/onnx"
printf 'model-bytes-v2' > "$T/srv/$REPO/resolve/$REV/onnx/model.onnx"
printf '{"tokenizer":"v2"}' > "$T/srv/$REPO/resolve/$REV/tokenizer.json"
M_SHA="$(sha "$T/srv/$REPO/resolve/$REV/onnx/model.onnx")"; K_SHA="$(sha "$T/srv/$REPO/resolve/$REV/tokenizer.json")"
manifest() {  # manifest <file> <tokenizer sha>
  python3 - "$1" "$REPO" "$REV" "$M_SHA" "$2" <<'PY'
import json, sys
p, repo, rev, m, k = sys.argv[1:]
json.dump({"source_repository": "https://huggingface.co/" + repo, "revision": rev, "source_file": "onnx/model.onnx",
           "dimension": 384, "files": {"model.onnx": m, "tokenizer.json": k}}, open(p, "w"))
PY
}
manifest "$T/good.json" "$K_SHA"
manifest "$T/badsha.json" "$(printf '0%.0s' $(seq 64))"

PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')"
( cd "$T/srv" && exec python3 -m http.server "$PORT" --bind 127.0.0.1 ) > "$T/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 50); do curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 0.1; done
run() {  # run <manifest> <dest> -> exit code
  CAS_MODEL_MANIFEST="$1" CAS_MODEL_BASE_URL="http://127.0.0.1:$PORT" "$FETCH" "$2" >> "$OUT/fetch.log" 2>&1
}
leftovers() { find "$(dirname "$1")" -maxdepth 1 -name ".$(basename "$1").*" | wc -l | tr -d ' '; }

# FM1
D="$T/m1/embedding"; run "$T/good.json" "$D"; rc=$?
[[ $rc -eq 0 && "$(sha "$D/model.onnx")" == "$M_SHA" && "$(sha "$D/tokenizer.json")" == "$K_SHA" && -f "$D/manifest.json" && "$(leftovers "$D")" == 0 ]] \
  && pass FM1 "fresh download: both files verified and published" || fail FM1 "fresh download" "rc=$rc"

# FM2: previous model in dest, tokenizer missing on the mirror
D="$T/m2/embedding"; mkdir -p "$D"; printf 'old-model' > "$D/model.onnx"; printf 'old-tok' > "$D/tokenizer.json"; echo '{}' > "$D/manifest.json"
before="$(tree_hash "$D")"
mv "$T/srv/$REPO/resolve/$REV/tokenizer.json" "$T/tok.bak"
run "$T/good.json" "$D"; rc=$?
mv "$T/tok.bak" "$T/srv/$REPO/resolve/$REV/tokenizer.json"
after="$(tree_hash "$D")"
[[ $rc -ne 0 && "$before" == "$after" && "$(leftovers "$D")" == 0 ]] \
  && pass FM2 "file 2 missing on the mirror: exit $rc, dest unchanged (model.onnx not published), no staging left" \
  || fail FM2 "partial download must not publish" "rc=$rc before=$before after=$after leftovers=$(leftovers "$D")"

# FM3: wrong checksum for file 2
D="$T/m3/embedding"; mkdir -p "$D"; printf 'old-model' > "$D/model.onnx"; printf 'old-tok' > "$D/tokenizer.json"
before="$(tree_hash "$D")"
run "$T/badsha.json" "$D"; rc=$?
after="$(tree_hash "$D")"
[[ $rc -ne 0 && "$before" == "$after" && "$(leftovers "$D")" == 0 ]] \
  && pass FM3 "checksum mismatch on file 2: exit $rc, dest unchanged" || fail FM3 "checksum mismatch" "rc=$rc"

# FM4: no dest yet, failure -> still no dest
D="$T/m4/embedding"
run "$T/badsha.json" "$D"; rc=$?
[[ $rc -ne 0 && ! -e "$D" && "$(leftovers "$D")" == 0 ]] && pass FM4 "failure with no previous model: dest not created" || fail FM4 "no dest" "rc=$rc exists=$([[ -e $D ]] && echo yes)"

# FM5: already complete -> no request to the mirror
D="$T/m1/embedding"; reqs_before="$(grep -c '"GET ' "$T/server.log")"
ino() { python3 -c 'import os,sys;print(os.stat(sys.argv[1]).st_ino)' "$1"; }
ino_before="$(ino "$D")"
run "$T/good.json" "$D"; rc=$?
reqs_after="$(grep -c '"GET ' "$T/server.log")"; ino_after="$(ino "$D")"
[[ $rc -eq 0 && "$reqs_before" == "$reqs_after" && "$ino_before" == "$ino_after" ]] \
  && pass FM5 "already present and valid: no download, directory not replaced (inode $ino_after)" \
  || fail FM5 "no-op on a valid model" "rc=$rc requests $reqs_before -> $reqs_after inode $ino_before -> $ino_after"

# ---- publish-failure injection ---------------------------------------------------
REAL_MV="$(command -v mv)"
mkdir -p "$T/shim"
cat > "$T/shim/mv" <<'SHIM'
#!/usr/bin/env bash
n=$(( $(cat "$MV_COUNTER" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$MV_COUNTER"
case ",$MV_FAIL_CALLS," in *",$n,"*) echo "INJECTED mv failure call $n" >&2; exit 1 ;; esac
exec "$REAL_MV" "$@"
SHIM
chmod +x "$T/shim/mv"
inject() {  # inject <fail calls> <dest> -> exit code of fetch_models.sh with the shim
  rm -f "$T/mvcount"
  PATH="$T/shim:$PATH" MV_COUNTER="$T/mvcount" MV_FAIL_CALLS="$1" REAL_MV="$REAL_MV" \
    CAS_MODEL_MANIFEST="$T/good.json" CAS_MODEL_BASE_URL="http://127.0.0.1:$PORT" "$FETCH" "$2" >> "$OUT/fetch_inject.log" 2>&1
}
oldmodel() { mkdir -p "$1"; printf 'old-model' > "$1/model.onnx"; printf 'old-tok' > "$1/tokenizer.json"; echo '{"old":1}' > "$1/manifest.json"; }
olds() { find "$(dirname "$1")" -maxdepth 1 -name ".$(basename "$1").old.*" | sort; }

# FM6
D="$T/m6/embedding"; oldmodel "$D"; before="$(tree_hash "$D")"
inject 2 "$D"; rc=$?
[[ $rc -ne 0 && "$(tree_hash "$D")" == "$before" && -z "$(olds "$D")" && "$(leftovers "$D")" == 0 ]] \
  && pass FM6 "publish fails, rollback works: exit $rc, previous model back in place, nothing left over" \
  || fail FM6 "publish failure with rollback" "rc=$rc same=$([[ "$(tree_hash "$D")" == "$before" ]] && echo y) olds=$(olds "$D")"

# FM7
D="$T/m7/embedding"; oldmodel "$D"; before="$(tree_hash "$D")"
echo "--- FM7" >> "$OUT/fetch_inject.log"
inject 2,3 "$D"; rc=$?
kept="$(olds "$D")"
if [[ $rc -ne 0 && ! -e "$D" && "$(grep -c . <<<"$kept")" == 1 && "$(tree_hash "$kept")" == "$before" ]] \
   && grep -q "previous model is kept intact at $kept" "$OUT/fetch_inject.log"; then
  pass FM7 "publish AND rollback fail: exit $rc, previous model kept intact at .old and its path printed (not deleted)"
else fail FM7 "double failure must keep the previous model" "rc=$rc dest_exists=$([[ -e "$D" ]] && echo y) kept='$kept'"; fi

# FM8
D="$T/m8/embedding"; oldmodel "$D"; before="$(tree_hash "$D")"
inject 1 "$D"; rc=$?
[[ $rc -ne 0 && "$(tree_hash "$D")" == "$before" && -z "$(olds "$D")" && "$(leftovers "$D")" == 0 ]] \
  && pass FM8 "moving the current model aside fails: exit $rc, dest unchanged" || fail FM8 "move-aside failure" "rc=$rc"

cp "$T/server.log" "$OUT/mirror.log"
if (( FAILS > 0 )); then echo "fetch_models_test: $FAILS case(s) failed"; exit 1; fi
echo "fetch_models_test: all cases passed"
