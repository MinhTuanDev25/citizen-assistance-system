#!/usr/bin/env bash
# Stage the Speech C0 runtime source into deploy/runpod/build/c0 for the image build.
# Source of truth is this Git repository (bahnar-s2tt-thesis/ is a tracked folder,
# not a submodule). Refuses untracked or locally modified files, and refuses an
# import closure that differs from c0_files.txt. Writes SOURCE_REVISION,
# SOURCE_FILES (path + git blob) and c0_code.sha256.
# Usage: deploy/runpod/stage_c0.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
LIST="$HERE/c0_files.txt"
OUT="$HERE/build/c0"
THESIS="bahnar-s2tt-thesis"

cd "$ROOT"
# bash 3.2 (macOS) has no mapfile.
FILES=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  FILES+=("$line")
done < "$LIST"

if [[ -e "$THESIS/.git" || -f .gitmodules && -n "$(git config -f .gitmodules --get-regexp "submodule\..*\.path" | grep -F "$THESIS" || true)" ]]; then
  echo "stage_c0: $THESIS is a separate repository or submodule; this script expects a tracked folder" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "stage_c0: not tracked by git: $f" >&2; exit 1; }
done
if [[ -n "$(git status --porcelain -- "${FILES[@]}")" ]]; then
  echo "stage_c0: uncommitted changes in C0 runtime files:" >&2
  git status --short -- "${FILES[@]}" >&2
  exit 1
fi

# Import closure of src.c0_speech (src.* imports only), compared with the list.
python3 - "$THESIS" "${FILES[@]}" <<'PY'
import os, re, sys
thesis, listed = sys.argv[1], sys.argv[2:]
pat = re.compile(r'^\s*(?:from|import)\s+src\.([\w\.]+)', re.M)
seen, todo = set(), [f"c0_speech/{n[:-3]}" for n in os.listdir(f"{thesis}/src/c0_speech") if n.endswith(".py")]
while todo:
    mod = todo.pop()
    if mod in seen:
        continue
    seen.add(mod)
    for cand in (f"{thesis}/src/{mod}.py", f"{thesis}/src/{mod}/__init__.py"):
        if os.path.exists(cand):
            todo += [m.replace(".", "/") for m in pat.findall(open(cand, encoding="utf-8").read())]
            break
found = set()
for mod in seen:
    for cand in (f"{thesis}/src/{mod}.py", f"{thesis}/src/{mod}/__init__.py"):
        if os.path.exists(cand):
            found.add(cand)
found |= {f"{thesis}/src/__init__.py", f"{thesis}/src/c0_speech/__init__.py"}
want = {f for f in listed if f.startswith(f"{thesis}/src/")}
if found != want:
    print("stage_c0: import closure differs from c0_files.txt", file=sys.stderr)
    print("  missing from list:", sorted(found - want), file=sys.stderr)
    print("  listed but not imported:", sorted(want - found), file=sys.stderr)
    sys.exit(1)
PY

rm -rf "$OUT"
mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  rel="${f#"$THESIS"/}"
  mkdir -p "$OUT/$(dirname "$rel")"
  # git show writes the committed blob, not the working-tree file.
  git show "HEAD:$f" > "$OUT/$rel"
done

REV="$(git rev-parse HEAD)"
echo "$REV" > "$OUT/SOURCE_REVISION"
for f in "${FILES[@]}"; do
  printf '%s  %s\n' "$(git rev-parse "HEAD:$f")" "$f"
done > "$OUT/SOURCE_FILES"
(cd "$OUT" && find . -type f ! -name c0_code.sha256 ! -name SOURCE_REVISION ! -name SOURCE_FILES -print0 \
  | sort -z | xargs -0 shasum -a 256 > c0_code.sha256)

echo "stage_c0: staged ${#FILES[@]} files at revision $REV into ${OUT#"$ROOT"/}"
