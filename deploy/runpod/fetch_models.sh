#!/usr/bin/env bash
# Fetch the ONNX embedding model (e5-small, pinned repository + revision + SHA-256 per
# file) into <dest> (default: models/embedding).
#
# The ai-service picks the model VARIANT from EMBEDDING_VARIANT and the CPU it runs on
# (apps/ai-service/app/indexing/embed.py, select_embedding_variant):
#   fp32       -> fp32 on any CPU                    (models/embedding.manifest.fp32.json)
#   int8-vnni  -> int8, refused without AVX-512 VNNI (models/embedding.manifest.int8-vnni.json)
#   auto       -> int8 on x86 with avx512f/bw/dq/vl + avx512vnni, fp32 otherwise
# --variant must match the ai-service's EMBEDDING_VARIANT. auto applies the rule to the
# machine running this script, so with auto run it on the host that runs the image.
#
# All-or-nothing: every file is downloaded (or reused when its checksum already
# matches) into a staging directory next to <dest> and verified there. Only when
# ALL files match does the staging directory replace <dest> (two renames). On a
# download/checksum failure <dest> is left exactly as it was. If publishing fails,
# the previous model is put back; if even that fails, it is KEPT at
# <parent>/.<name>.old.<pid> (never deleted) and the script says where.
# A SIGKILL between the two renames cannot run any handler: <dest> is then missing
# and the previous model sits in .<name>.old.<pid> (see deploy/runpod/README.md).
#   deploy/runpod/fetch_models.sh [--variant auto|int8-vnni|fp32] [dest]
# Env: CAS_MODEL_VARIANT, else EMBEDDING_VARIANT (same as --variant); tests: CAS_MODEL_MANIFEST (explicit
# manifest, overrides the variant), CAS_CPUINFO_FILE (cpuinfo read by auto), CAS_MODEL_BASE_URL (other mirror, default
# https://huggingface.co; files are fetched from <base>/<repo>/resolve/<rev>/<path>).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VARIANT="${CAS_MODEL_VARIANT:-${EMBEDDING_VARIANT:-auto}}"
if [[ "${1:-}" == --variant ]]; then VARIANT="${2:?--variant needs a value}"; shift 2; fi

detect_variant() {  # same rule as ai-service select_embedding_variant
  local flags="" info="${CAS_CPUINFO_FILE:-/proc/cpuinfo}"   # CAS_CPUINFO_FILE: tests only
  if [[ -r "$info" ]]; then
    flags="$(grep -m1 -E '^(flags|Features)[[:space:]]*:' "$info" | cut -d: -f2 | tr 'A-Z' 'a-z' | sed 's/avx512_vnni/avx512vnni/g')"
  fi
  local f ok=1
  for f in avx512f avx512bw avx512dq avx512vl avx512vnni; do
    [[ " $flags " == *" $f "* ]] || { ok=0; break; }
  done
  if [[ $ok == 1 ]]; then echo int8-vnni; else echo fp32; fi
}
if [[ -n "${CAS_MODEL_MANIFEST:-}" ]]; then
  MANIFEST="$CAS_MODEL_MANIFEST"
else
  [[ "$VARIANT" != auto ]] || VARIANT="$(detect_variant)"
  case "$VARIANT" in
    int8-vnni|fp32) MANIFEST="$HERE/models/embedding.manifest.$VARIANT.json" ;;
    *) echo "fetch_models: unknown variant '$VARIANT' (auto | int8-vnni | fp32)" >&2; exit 2 ;;
  esac
  echo "fetch_models: variant $VARIANT"
fi
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
