"""
Export helpers for Notebook 04 runs (SHA256 artifact manifests).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from src.data_utils import sha256_file


def write_artifact_manifest(export_dir: Union[str, Path], files: Iterable[Path]) -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for f in files:
        f = Path(f)
        if not f.is_file():
            continue
        try:
            rel = str(f.relative_to(export_dir))
        except ValueError:
            rel = f.name
        entries.append(
            {
                "path": rel,
                "sha256": sha256_file(f),
                "size_bytes": int(f.stat().st_size),
            }
        )
    entries.sort(key=lambda e: e["path"])
    path = export_dir / "artifact_manifest.json"
    path.write_text(json.dumps({"files": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _copy_if_exists(src: Path, dest: Path) -> Optional[Path]:
    if not src.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def collect_notebook04_export_sources(
    *,
    artifacts_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    full_state_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Map logical export names -> source paths that exist."""
    mapping: Dict[str, Path] = {}
    art = Path(artifacts_dir) if artifacts_dir else None
    res = Path(results_dir) if results_dir else None
    state = Path(full_state_dir) if full_state_dir else None

    if art and art.is_dir():
        for p in art.rglob("*"):
            if p.is_file():
                mapping[f"artifacts/{p.relative_to(art)}"] = p
    if res and res.is_dir():
        for p in res.rglob("*"):
            if p.is_file():
                mapping[f"results/{p.relative_to(res)}"] = p

    if state and state.is_dir():
        for name in (
            "mt_contract.json",
            "mt_prepare_state.json",
            "mt_data_summary.json",
            "mt_data_exclusions.csv",
            "mt_validation_monitor_manifest.json",
            "mt_resume_test_summary.json",
            "mt_resume_test_phase_a.json",
            "mt_resume_test_contract.json",
            "mt_resume_test_durable_commit.json",
            "mt_training_contract.json",
            "mt_train_summary.json",
            "mt_evaluate_summary.json",
        ):
            p = state / name
            if p.is_file():
                mapping[f"state/{name}"] = p
        # Optional checkpoint pointers under durable experiment dirs (no weights).
        for kind in ("full_train", "resume_test", "pilot"):
            for ptr in state.rglob("full_experiment_fingerprint.json"):
                if f"/{kind}/" in str(ptr) or ptr.parent.name == kind:
                    rel = f"state/checkpoints/{kind}/{ptr.parent.name}_fingerprint.json"
                    mapping[rel] = ptr
    return mapping


def export_notebook04_run(
    *,
    export_root: Union[str, Path],
    run_id: str,
    artifacts_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    full_state_dir: Optional[Union[str, Path]] = None,
    extra_files: Optional[Mapping[str, Union[str, Path]]] = None,
    pointer: Optional[Mapping[str, Any]] = None,
    run_config: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, Any]] = None,
) -> Path:
    dest = Path(export_root) / str(run_id)
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []

    sources = collect_notebook04_export_sources(
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        full_state_dir=full_state_dir,
    )
    for rel, src in sources.items():
        out = _copy_if_exists(Path(src), dest / rel)
        if out is not None:
            copied.append(out)

    if run_config is not None:
        p = dest / "run_config.json"
        p.write_text(json.dumps(dict(run_config), ensure_ascii=False, indent=2), encoding="utf-8")
        copied.append(p)
    if environment is not None:
        p = dest / "environment.json"
        p.write_text(json.dumps(dict(environment), ensure_ascii=False, indent=2), encoding="utf-8")
        copied.append(p)

    if extra_files:
        for name, src in extra_files.items():
            out = _copy_if_exists(Path(src), dest / name)
            if out is not None:
                copied.append(out)

    if pointer is not None:
        ptr = dest / "full_export_pointer.json"
        ptr.write_text(json.dumps(dict(pointer), ensure_ascii=False, indent=2), encoding="utf-8")
        copied.append(ptr)

    write_artifact_manifest(dest, copied)
    return dest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_notebook04_review_bundle_files(project_root: Union[str, Path]) -> List[Path]:
    """Payload files for the Notebook 04 review zip (no parquet / venv / caches)."""
    root = Path(project_root).resolve()
    include: List[Path] = []
    for f in (root / "src").rglob("*.py"):
        include.append(f)
    for f in (root / "tests").rglob("*.py"):
        include.append(f)
    for f in (root / "notebooks").glob("*.ipynb"):
        include.append(f)
    for rel in ("configs/mt.yaml", "requirements.txt", "README.md", "AGENTS.md"):
        p = root / rel
        if p.is_file():
            include.append(p)
    for base in ("data/manifests", "data/audit"):
        p = root / base
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
                    include.append(f)
    # Stable unique, skip caches.
    seen = set()
    out: List[Path] = []
    for path in sorted(include, key=lambda p: str(p.relative_to(root))):
        if not path.is_file():
            continue
        if any(x in path.parts for x in (".venv", "__pycache__", ".pycache", ".git")):
            continue
        rel = str(path.relative_to(root))
        if rel in seen:
            continue
        seen.add(rel)
        out.append(path)
    return out


def build_notebook04_review_bundle(
    project_root: Union[str, Path],
    *,
    out_dir: Optional[Union[str, Path]] = None,
    stamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a self-consistent Notebook 04 review zip.

    Checksum strategy (avoids self-referential ZIP hash loops):
      * Inside ZIP: ``PAYLOAD_SHA256SUMS.txt`` — sha256 of each packaged file
        (pre-zip payload only). Also ``BUNDLE_META.json`` describing that the
        *final ZIP* sha256 lives only in external sidecars.
      * Outside ZIP: ``SHA256SUMS.txt`` + ``LATEST_BUNDLE.txt`` — sha256 of the
        final ``.zip`` file itself. These are NOT embedded as claiming to be
        the zip's own hash inside the archive.
    """
    root = Path(project_root).resolve()
    art = Path(out_dir) if out_dir is not None else (root / "artifacts" / "notebook04")
    art.mkdir(parents=True, exist_ok=True)
    ts = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_name = f"notebook04_mt_review_bundle_{ts}.zip"
    zip_path = art / zip_name

    for old in art.glob("notebook04_mt_*.zip"):
        old.unlink()

    payload_files = collect_notebook04_review_bundle_files(root)
    payload_lines = [
        f"{_sha256_path(p)}  {p.relative_to(root).as_posix()}"
        for p in payload_files
    ]
    payload_body = "\n".join(payload_lines) + ("\n" if payload_lines else "")
    payload_digest = _sha256_bytes(payload_body.encode("utf-8"))

    readme = (
        "# Notebook 04 Review Bundle\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "## Checksums\n"
        "- **Inside ZIP** `artifacts/notebook04/PAYLOAD_SHA256SUMS.txt`: "
        "sha256 of each packaged payload file (pre-zip).\n"
        "- **Inside ZIP** `artifacts/notebook04/BUNDLE_META.json`: payload digest + zip name; "
        "does **not** claim to be the final ZIP sha256.\n"
        "- **Outside ZIP** `SHA256SUMS.txt` / `LATEST_BUNDLE.txt`: sha256 of the final `.zip`.\n"
    )
    meta = {
        "zip_name": zip_name,
        "checksum_scheme": "payload_inside_zip__zip_sha_external_only",
        "payload_sha256sums_sha256": payload_digest,
        "payload_file_count": len(payload_files),
        "note": (
            "Final ZIP sha256 is written only to external SHA256SUMS.txt / LATEST_BUNDLE.txt "
            "beside the archive. It is intentionally not self-embedded inside the ZIP."
        ),
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in payload_files:
            zf.write(path, path.relative_to(root).as_posix())
        zf.writestr("artifacts/notebook04/README_NOTEBOOK04_BUNDLE.md", readme)
        zf.writestr("artifacts/notebook04/PAYLOAD_SHA256SUMS.txt", payload_body)
        zf.writestr(
            "artifacts/notebook04/BUNDLE_META.json",
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        )

    zip_digest = _sha256_path(zip_path)
    latest_text = (
        f"{zip_name}\n"
        f"ZIP_SHA256={zip_digest}\n"
        f"PAYLOAD_SHA256SUMS_SHA256={payload_digest}\n"
        f"checksum_scheme=payload_inside_zip__zip_sha_external_only\n"
    )
    sums_text = (
        f"{zip_digest}  {zip_name}\n"
        f"# payload_sha256sums_sha256 (content of PAYLOAD_SHA256SUMS.txt inside zip) "
        f"= {payload_digest}\n"
    )
    (art / "LATEST_BUNDLE.txt").write_text(latest_text, encoding="utf-8")
    (art / "SHA256SUMS.txt").write_text(sums_text, encoding="utf-8")
    (art / "PAYLOAD_SHA256SUMS.txt").write_text(payload_body, encoding="utf-8")
    (art / "BUNDLE_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (art / "README_NOTEBOOK04_BUNDLE.md").write_text(readme, encoding="utf-8")

    # Verify: inside zip must NOT contain a stale ZIP_SHA256 claiming to be this archive.
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "artifacts/notebook04/PAYLOAD_SHA256SUMS.txt" in names
        assert "artifacts/notebook04/BUNDLE_META.json" in names
        inner_meta = json.loads(zf.read("artifacts/notebook04/BUNDLE_META.json"))
        assert inner_meta.get("zip_name") == zip_name
        assert inner_meta.get("payload_sha256sums_sha256") == payload_digest
        # No misleading self-hash of the zip inside.
        for banned in ("LATEST_BUNDLE.txt", "SHA256SUMS.txt"):
            inner = f"artifacts/notebook04/{banned}"
            if inner in names:
                text = zf.read(inner).decode("utf-8")
                if "ZIP_SHA256=" in text or (
                    zip_digest in text and banned == "SHA256SUMS.txt"
                ):
                    raise RuntimeError(
                        f"Bundle zip embeds misleading final-zip checksum in {inner}"
                    )

    return {
        "zip_path": str(zip_path),
        "zip_name": zip_name,
        "zip_sha256": zip_digest,
        "payload_sha256sums_sha256": payload_digest,
        "n_payload_files": len(payload_files),
        "latest_path": str(art / "LATEST_BUNDLE.txt"),
        "sha256sums_path": str(art / "SHA256SUMS.txt"),
    }
