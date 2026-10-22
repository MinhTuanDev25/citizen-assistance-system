"""Pack the P4B source bundle without dropping Python packages named models."""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipInfo

# Directory names that are caches or the unrelated thesis tree.
# "models" is not here: apps/ai-service/app/models and docs/models are source.
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".tools",
    "bahnar-s2tt-thesis",
}

SKIP_FILES = {".DS_Store", ".env", ".env.local", "phase_1_3_audit.md"}
WEIGHT_SUFFIXES = (".onnx", ".pdiparams", ".pdmodel", ".pdopt")


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS


def should_skip_file(name: str) -> bool:
    if name in SKIP_FILES or name.endswith((".pyc", ".pem", ".exe", ".sha256")):
        return True
    if name.endswith(".test") and not name.endswith("_test.go"):
        return True
    return name.endswith(WEIGHT_SUFFIXES)


def iter_files(root: Path):
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [item for item in dirnames if not should_skip_dir(item)]
        for name in filenames:
            if should_skip_file(name):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel == "deploy/.env" or rel.endswith("/.env"):
                continue
            if rel.endswith(".zip"):
                continue
            yield rel


def build_bundle(root: Path, dest: Path, sidecar: Path | None = None) -> str:
    """Write SHA256SUMS.txt, a deterministic ZIP, then the sidecar. The ZIP is not modified after hashing."""
    root = root.resolve()
    dest = dest.resolve()
    files = sorted(rel for rel in iter_files(root) if rel != "SHA256SUMS.txt")
    lines = []
    for rel in files:
        digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    sums = "\n".join(lines) + "\n"
    (root / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, "w") as archive:
        for rel in files + ["SHA256SUMS.txt"]:
            info = ZipInfo(rel)
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (root / rel).read_bytes())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    side = sidecar or dest.with_name(dest.name + ".sha256")
    side.write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack the P4B source bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--zip", dest="zip_path", default="phase_p4b_document_content_bundle.zip")
    parser.add_argument("--sidecar", default="")
    args = parser.parse_args()
    sidecar = Path(args.sidecar) if args.sidecar else None
    digest = build_bundle(Path(args.root), Path(args.zip_path), sidecar)
    print(digest)


if __name__ == "__main__":
    main()
