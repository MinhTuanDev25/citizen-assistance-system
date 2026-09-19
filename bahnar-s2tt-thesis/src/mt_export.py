"""
Export helpers for Notebook 04 runs (SHA256 artifact manifests).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

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
