#!/usr/bin/env bash
# Validate ZIP deliverable contents: fail if forbidden paths are present.
set -euo pipefail
ZIP="${1:?usage: check-zip.sh <zipfile>}"

# Compiled Go test binaries (foo.test), not *_test.go sources.
FORBIDDEN_REGEX='(^|/)\.git(/|$)|(^|/)\.venv(/|$)|(^|/)\.env$|(^|/)\.env\.local$|(^|/)node_modules/|(^|/)dist/|(^|/)\.pytest_cache/|(^|/)__pycache__/|(^|/)\.mypy_cache/|(^|/)\.ruff_cache/|(^|/)\.tools/|\.pyc$|\.exe$|\.pem$|\.onnx$|\.pdiparams$|\.pdmodel$|\.pdopt$|(^|/)[^/]+\.test$|(^|/)deploy/\.env$|(^|/)\.DS_Store$|(^|/)phase_1_3_audit\.md$'

if [[ ! -f "$ZIP" ]]; then
  echo "FAIL: zip missing: $ZIP" >&2
  exit 1
fi

if ! unzip -tqq "$ZIP"; then
  echo "FAIL: corrupt zip: $ZIP" >&2
  exit 1
fi

LIST_FILE="$(mktemp)"
trap 'rm -f "$LIST_FILE"' EXIT
unzip -Z1 "$ZIP" > "$LIST_FILE"

bad=0
while IFS= read -r entry; do
  if echo "$entry" | grep -Eq "$FORBIDDEN_REGEX"; then
    echo "FAIL: forbidden path in zip: $entry" >&2
    bad=1
  fi
  case "$entry" in
    /*|../*|*/../*|*/..)
      echo "FAIL: path traversal in zip: $entry" >&2
      bad=1
      ;;
  esac
done < "$LIST_FILE"

dups="$(sort "$LIST_FILE" | uniq -d || true)"
if [[ -n "$dups" ]]; then
  echo "FAIL: duplicate entry in zip: $dups" >&2
  bad=1
fi

if ! python3 - "$ZIP" <<'PY'
import stat
import sys
import zipfile

path = sys.argv[1]
try:
    archive = zipfile.ZipFile(path)
except zipfile.BadZipFile:
    print("FAIL: corrupt zip", file=sys.stderr)
    sys.exit(1)
names = archive.namelist()
if len(names) != len(set(names)):
    print("FAIL: duplicate entry", file=sys.stderr)
    sys.exit(1)
for info in archive.infolist():
    name = info.filename
    if name.startswith("/") or ".." in name.split("/"):
        print(f"FAIL: path traversal in zip: {name}", file=sys.stderr)
        sys.exit(1)
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        print(f"FAIL: symlink in zip: {name}", file=sys.stderr)
        sys.exit(1)
PY
then
  bad=1
fi

has_entry() {
  grep -Fxq -- "$1" "$LIST_FILE"
}

for need in \
  phase_1_3_final_report.md \
  phase_1_3_test_output.txt \
  phase_p2_llm_extract_report.md \
  phase_p2_test_output.txt \
  phase_p3_1_document_intake_report.md \
  phase_p3_1_test_output.txt \
  phase_p4a_document_indexing_report.md \
  phase_p4a_test_output.txt \
  deploy/migrations/000001_init_schema.up.sql \
  deploy/migrations/000001_init_schema.down.sql \
  deploy/migrations/000002_seed_master.up.sql \
  deploy/migrations/000002_seed_master.down.sql \
  deploy/migrations/000003_seed_procedures.up.sql \
  deploy/migrations/000003_seed_procedures.down.sql \
  deploy/migrations/000004_seed_auth_users.up.sql \
  deploy/migrations/000004_seed_auth_users.down.sql \
  deploy/migrations/000005_phase13_chat.up.sql \
  deploy/migrations/000005_phase13_chat.down.sql \
  deploy/migrations/000006_idempotency_envelope.up.sql \
  deploy/migrations/000006_idempotency_envelope.down.sql \
  deploy/migrations/000007_clear_demo_passwords.up.sql \
  deploy/migrations/000007_clear_demo_passwords.down.sql \
  deploy/migrations/000008_extract_claim.up.sql \
  deploy/migrations/000008_extract_claim.down.sql \
  deploy/migrations/000009_document_intake.up.sql \
  deploy/migrations/000009_document_intake.down.sql \
  deploy/migrations/000010_document_indexing.up.sql \
  deploy/migrations/000010_document_indexing.down.sql \
  deploy/migrations/000011_link_index_status.up.sql \
  deploy/migrations/000011_link_index_status.down.sql \
  deploy/migrations/000012_p4a_hardening.up.sql \
  deploy/migrations/000012_p4a_hardening.down.sql \
  deploy/migrations/000013_legacy_idempotency_replay.up.sql \
  deploy/migrations/000013_legacy_idempotency_replay.down.sql \
  deploy/migrations/000014_p4b_index_generations.up.sql \
  deploy/migrations/000014_p4b_index_generations.down.sql \
  deploy/migrations/000015_p4b_unlink_reindex.up.sql \
  deploy/migrations/000015_p4b_unlink_reindex.down.sql \
  docs/p4b-runbook.md \
  phase_p4b_document_content_report.md \
  phase_p4b_test_output.txt \
  SHA256SUMS.txt \
  .github/workflows/ci.yml \
  .gitignore \
  deploy/.env.example \
  packages/contracts/README.md \
  apps/api/go.mod \
  apps/web/package-lock.json \
  apps/api/docs/swagger.json \
  apps/api/docs/swagger.yaml \
  apps/api/docs/docs.go \
  apps/ai-service/app/main.py \
  apps/ai-service/app/config.py \
  apps/ai-service/app/models/__init__.py \
  apps/ai-service/app/models/extract.py \
  apps/ai-service/app/models/index.py \
  docs/models/embedding.manifest.example.json \
  docs/models/ocr.manifest.example.json \
  deploy/migrations/000016_p4b_cleanup_claim.up.sql \
  deploy/migrations/000016_p4b_cleanup_claim.down.sql \
  scripts/p4b_bundle.py
do
  if ! has_entry "$need"; then
    echo "FAIL: missing required path in zip: $need" >&2
    bad=1
  fi
done

# Stale historical audit must not ship as a current verdict.
if has_entry "phase_1_3_audit.md"; then
  echo "FAIL: phase_1_3_audit.md is superseded and must not be in the final ZIP" >&2
  bad=1
fi

if [[ "$bad" -ne 0 ]]; then
  exit 1
fi

EXTRACT="$(mktemp -d)"
trap 'rm -rf "$EXTRACT"; rm -f "$LIST_FILE"' EXIT
unzip -qq "$ZIP" -d "$EXTRACT"

python3 - "$EXTRACT" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
sums = (root / "SHA256SUMS.txt").read_text(encoding="utf-8")
seen = set()
for line in sums.splitlines():
    if not line.strip():
        continue
    if "  " not in line:
        print("FAIL: checksum line malformed", file=sys.stderr)
        sys.exit(1)
    digest, rel = line.split("  ", 1)
    seen.add(rel)
    path = root / rel
    if not path.is_file():
        print(f"FAIL: checksum missing file: {rel}", file=sys.stderr)
        sys.exit(1)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        print(f"FAIL: checksum mismatch: {rel}", file=sys.stderr)
        sys.exit(1)
for path in root.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    if rel != "SHA256SUMS.txt" and rel not in seen:
        print(f"FAIL: file missing from SHA256SUMS.txt: {rel}", file=sys.stderr)
        sys.exit(1)
PY

PYBIN="${PYTHON:-python3}"
(
  cd "$EXTRACT"
  CAS_ZIP_EXTRACT="$EXTRACT" AI_SERVICE_TOKEN=test-service-token PYTHONPATH="$EXTRACT/apps/ai-service" "$PYBIN" -c '
import inspect, os, app.config, app.main, app.models.extract, app.models.index
root = os.path.realpath(os.environ["CAS_ZIP_EXTRACT"])
for mod in (app.config, app.main, app.models.extract, app.models.index):
    loaded = os.path.realpath(inspect.getfile(mod))
    if not loaded.startswith(root + os.sep):
        raise SystemExit("imported from outside extract: " + loaded)
'
) || {
  echo "FAIL: clean-extract import" >&2
  exit 1
}

echo "ZIP hygiene: PASS ($ZIP)"
