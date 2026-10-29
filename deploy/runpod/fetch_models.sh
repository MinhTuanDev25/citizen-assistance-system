#!/usr/bin/env bash
# Fetch the ONNX embedding model pinned in deploy/runpod/models/embedding.manifest.json
# (repository + revision + SHA-256 per file) into <dest> (default: models/embedding).
#
# All-or-nothing: every file is downloaded (or reused when its checksum already
# matches) into a staging directory next to <dest> and verified there. Only when
# ALL files match does the staging directory replace <dest> (two renames). On a
# download/checksum failure <dest> is left exactly as it was. If publishing fails,
# the previous model is put back; if even that fails, it is KEPT at
# <parent>/.<name>.old.<pid> (never deleted) and the script says where.
# A SIGKILL between the two renames cannot run any handler: <dest> is then missing
# and the previous model sits in .<name>.old.<pid> (see deploy/runpod/README.md).
#   deploy/runpod/fetch_models.sh [dest]
# Env (tests): CAS_MODEL_MANIFEST (other manifest), CAS_MODEL_BASE_URL (other mirror,
# default https://huggingface.co; files are fetched from <base>/<repo>/resolve/<rev>/<path>).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MANIFEST="${CAS_MODEL_MANIFEST:-$HERE/models/embedding.manifest.json}"
BASE="${CAS_MODEL_BASE_URL:-https://huggingface.co}"
DEST="${1:-$ROOT/models/embedding}"
DEST="${DEST%/}"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# repo, revision, then one "name source sha256" line per file
plan="$(python3 - "$MANIFEST" <<'PY'
import json, re, sys
m = json.load(open(sys.argv[1]))
repo = m["source_repository"].rstrip("/")
for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
    if repo.startswith(prefix):
        repo = repo[len(prefix):]
rev = m["revision"]
if not re.fullmatch(r"[0-9a-f]{40}", rev):
    sys.exit("revision must be a full 40-hex commit id")
files = m.get("files") or {}
if not files:
    sys.exit("manifest lists no files")
src = {"model.onnx": m.get("source_file", "onnx/model.onnx"), "tokenizer.json": m.get("tokenizer_source_file", "tokenizer.json")}
print(repo, rev)
for name, digest in files.items():
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or "/" in name:
        sys.exit(f"bad manifest entry {name}")
    print(name, src.get(name, name), digest)
PY
)" || { echo "fetch_models: invalid manifest $MANIFEST" >&2; exit 1; }
read -r REPO REV <<<"$(head -1 <<<"$plan")"

# Already complete and correct: touch nothing. Replacing the directory would cut it
# from under any process that bind-mounts it (e.g. a running ai-service container).
complete=1
while read -r name src want; do
  [[ -f "$DEST/$name" && "$(sha "$DEST/$name")" == "$want" ]] || { complete=0; break; }
done < <(tail -n +2 <<<"$plan")
if [[ $complete == 1 ]] && cmp -s "$MANIFEST" "$DEST/manifest.json"; then
  echo "fetch_models: $DEST already complete and verified; nothing changed"
  exit 0
fi
# A real update swaps the directory: restart consumers that bind-mount it afterwards.

parent="$(dirname "$DEST")"; mkdir -p "$parent"
stage="$(mktemp -d "$parent/.$(basename "$DEST").staging.XXXXXX")"
old=""; published=0
# The previous model (.old) is deleted ONLY once the new one is in place. If the
# publish and the rollback both fail, .old is kept for the operator.
cleanup() {
  [[ ! -d "$stage" ]] || rm -rf "$stage"
  if [[ -n "$old" && $published == 1 ]]; then rm -rf "$old"; fi
}
trap cleanup EXIT

while read -r name src want; do
  if [[ -f "$DEST/$name" && "$(sha "$DEST/$name")" == "$want" ]]; then
    cp "$DEST/$name" "$stage/$name"
    echo "fetch_models: $name already present, checksum OK"
    continue
  fi
  echo "fetch_models: downloading $name from $REPO@$REV ($src)"
  curl -fL --retry 3 --retry-delay 5 -o "$stage/$name" "$BASE/$REPO/resolve/$REV/$src" \
    || { echo "fetch_models: download of $name failed; $DEST left unchanged" >&2; exit 1; }
done < <(tail -n +2 <<<"$plan")

# Verify the complete set in staging before publishing anything.
while read -r name src want; do
  got="$(sha "$stage/$name")"
  [[ "$got" == "$want" ]] || { echo "fetch_models: checksum mismatch for $name: got $got, want $want; $DEST left unchanged" >&2; exit 1; }
done < <(tail -n +2 <<<"$plan")
cp "$MANIFEST" "$stage/manifest.json"
# mktemp -d is 0700: make the model readable by the service user before publishing.
chmod 0755 "$stage"; chmod 0644 "$stage"/*

# Publish: swap directories (no window where DEST holds a mix of old and new files).
# Every mv below is tested explicitly (not left to set -e), so a failure always
# reaches the rollback/keep logic.
if [[ -e "$DEST" ]]; then
  candidate="$parent/.$(basename "$DEST").old.$$"
  if ! mv "$DEST" "$candidate"; then
    echo "fetch_models: could not move the current model aside; $DEST left unchanged" >&2
    exit 1
  fi
  old="$candidate"
fi
if mv "$stage" "$DEST"; then
  published=1
else
  if [[ -z "$old" ]]; then
    echo "fetch_models: could not publish the new model to $DEST" >&2
  elif mv "$old" "$DEST"; then
    old=""
    echo "fetch_models: could not publish the new model; previous model restored in $DEST" >&2
  else
    echo "fetch_models: ERROR: could not publish the new model AND could not restore the previous one." >&2
    echo "fetch_models: the previous model is kept intact at $old ; move it back to $DEST by hand." >&2
  fi
  exit 1
fi
echo "fetch_models: embedding model ready in $DEST ($(tail -n +2 <<<"$plan" | wc -l | tr -d ' ') files verified)"
