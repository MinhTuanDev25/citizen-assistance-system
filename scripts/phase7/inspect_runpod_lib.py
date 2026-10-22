"""Read-only Phase 7.3 RunPod inspection. Does not load ASR/MT weights or download Hub files."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ASR_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "vocab.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)
MT_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "dict.txt",
    "generation_config.json",
)
LOCKED_TOKENIZER_ID = "vinai/bartpho-syllable"
LOCKED_TOKENIZER_REVISION = "36eee8b4d648dd99da56462edcda3c5c97f7f3de"
ASR_REL = "bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500"
MT_REL = "bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500"
MT_CONTRACT_REL = "bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json"
EXAMPLE_CONFIG_REL = "configs/c0_inference.example.yaml"
RUNTIME_CONFIG_NAME = "c0_inference.runpod.yaml"
DOCUMENTED_RUNPOD_ARTIFACT_ROOT = "/workspace/bahnar-s2tt-thesis"
AUDIO_EXTS = {".wav", ".flac"}
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "hf_parquet_cache",
        "bahnar_full_audio_cache",
        "full_state",
        "mt_full_state",
        "direct_full_state",
        "checkpoints",
    }
)
PREFERRED_AUDIO_DIRS = ("tests", "fixtures")
PACKAGE_SPECS = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PyYAML", "yaml"),
    ("soundfile", "soundfile"),
    ("safetensors", "safetensors"),
    ("tokenizers", "tokenizers"),
    ("sentencepiece", "sentencepiece"),
)


def cas_root_from_script_file(script_file: Path) -> Path:
    return Path(script_file).resolve().parent.parent.parent


def resolve_layout(cas_root: Path) -> dict[str, Any]:
    cas = Path(cas_root).resolve()
    thesis = cas / "bahnar-s2tt-thesis"
    entry = thesis / "src" / "c0_cascaded.py"
    ok = entry.is_file()
    return {
        "cas_root": str(cas),
        "thesis_source_root": str(thesis),
        "thesis_root": str(thesis),
        "c0_entrypoint": str(entry),
        "layout_ok": ok,
        "mismatch": None if ok else f"expected thesis entrypoint at {entry}",
    }


def resolve_artifact_root(
    *,
    explicit: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    documented: str = DOCUMENTED_RUNPOD_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Trained checkpoints live on ARTIFACT_ROOT, not necessarily under CAS source."""
    environ = os.environ if env is None else env
    if explicit is not None and str(explicit).strip():
        root = Path(str(explicit)).expanduser()
        return {
            "path": str(root.resolve()) if root.exists() else str(root),
            "source": "cli",
            "exists": root.is_dir(),
            "note": "explicit --artifact-root; checkpoints are not assumed under THESIS_SOURCE_ROOT",
        }
    env_val = str(environ.get("PHASE7_ARTIFACT_ROOT") or "").strip()
    if env_val:
        root = Path(env_val).expanduser()
        return {
            "path": str(root.resolve()) if root.exists() else str(root),
            "source": "PHASE7_ARTIFACT_ROOT",
            "exists": root.is_dir(),
            "note": "environment ARTIFACT_ROOT; checkpoints are not assumed under THESIS_SOURCE_ROOT",
        }
    documented_path = Path(documented)
    marker = documented_path / "bahnar_s2tt"
    if documented_path.is_dir() and marker.is_dir():
        return {
            "path": str(documented_path),
            "source": "documented_runpod_default",
            "exists": True,
            "note": "documented RunPod model root; not the CAS git checkout",
        }
    return {
        "path": None,
        "source": "unset",
        "exists": False,
        "note": "ARTIFACT_ROOT unset; do not look for checkpoints inside THESIS_SOURCE_ROOT",
    }


def resolve_hf_hub_cache(env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Effective Hub cache, matching huggingface_hub precedence.

    1. HF_HUB_CACHE (already the hub dir — do not append /hub)
    2. HUGGINGFACE_HUB_CACHE (legacy alias — do not append /hub)
    3. HF_HOME/hub
    4. Default Hugging Face hub cache
    """
    environ = os.environ if env is None else env
    hf_hub = str(environ.get("HF_HUB_CACHE") or "").strip()
    if hf_hub:
        cache_root = Path(hf_hub).expanduser()
        return {"cache_root": str(cache_root), "source": "HF_HUB_CACHE", "appended_hub": False}
    legacy = str(environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    if legacy:
        cache_root = Path(legacy).expanduser()
        return {"cache_root": str(cache_root), "source": "HUGGINGFACE_HUB_CACHE", "appended_hub": False}
    hf_home = str(environ.get("HF_HOME") or "").strip()
    if hf_home:
        cache_root = Path(hf_home).expanduser() / "hub"
        return {"cache_root": str(cache_root), "source": "HF_HOME/hub", "appended_hub": True}
    if env is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            return {
                "cache_root": str(Path(HF_HUB_CACHE)),
                "source": "huggingface_hub.constants.HF_HUB_CACHE",
                "appended_hub": False,
            }
        except Exception:
            pass
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    return {"cache_root": str(cache_root), "source": "default", "appended_hub": False}


def bounded_discover_audio(
    roots: Iterable[Path],
    *,
    max_depth: int = 3,
    max_visits: int = 400,
    max_results: int = 8,
) -> dict[str, Any]:
    found: list[str] = []
    visits = 0
    truncated = False
    reasons: list[str] = []
    skipped_symlinks = 0
    permission_errors = 0

    def walk(base: Path, depth: int) -> None:
        nonlocal visits, truncated, skipped_symlinks, permission_errors
        if truncated or len(found) >= max_results:
            return
        if depth > max_depth:
            truncated = True
            reasons.append("depth_limit")
            return
        try:
            with os.scandir(base) as it:
                for entry in it:
                    if truncated or len(found) >= max_results:
                        if len(found) >= max_results and "result_limit" not in reasons:
                            truncated = True
                            reasons.append("result_limit")
                        return
                    visits += 1
                    if visits > max_visits:
                        truncated = True
                        reasons.append("visit_limit")
                        return
                    try:
                        if entry.is_symlink():
                            skipped_symlinks += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                                continue
                            walk(Path(entry.path), depth + 1)
                        elif entry.is_file(follow_symlinks=False):
                            suffix = Path(entry.name).suffix.lower()
                            if suffix in AUDIO_EXTS:
                                found.append(str(Path(entry.path)))
                                if len(found) >= max_results:
                                    truncated = True
                                    reasons.append("result_limit")
                                    return
                    except OSError:
                        permission_errors += 1
                        continue
        except FileNotFoundError:
            return
        except PermissionError:
            permission_errors += 1
            return
        except OSError:
            permission_errors += 1
            return

    for root in roots:
        path = Path(root)
        if not path.exists():
            reasons.append(f"missing:{path}")
            continue
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            found.append(str(path.resolve()))
            continue
        if path.is_dir():
            walk(path, 0)

    return {
        "candidates": found[:max_results],
        "count": len(found[:max_results]),
        "truncated": truncated,
        "truncation_reasons": sorted(set(reasons)),
        "visits": visits,
        "skipped_symlinks": skipped_symlinks,
        "permission_errors": permission_errors,
        "max_depth": max_depth,
        "max_visits": max_visits,
        "max_results": max_results,
        "note": "filenames only; audio content not read",
    }


def distribution_status(dist_name: str, import_name: str) -> dict[str, Any]:
    installed: Optional[str] = None
    dist_error: Optional[str] = None
    try:
        installed = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        dist_error = "MISSING"
    except Exception as exc:
        dist_error = type(exc).__name__
    imported = False
    import_error: Optional[str] = None
    try:
        importlib.import_module(import_name)
        imported = True
    except Exception as exc:
        import_error = type(exc).__name__
    if installed is not None and imported:
        status = "PASS"
    elif dist_error == "MISSING" and not imported:
        status = "MISSING"
    else:
        status = "NOT_VERIFIED"
    return {
        "distribution": dist_name,
        "import_name": import_name,
        "installed_version": installed,
        "imported": imported,
        "status": status,
        "distribution_error": dist_error,
        "import_error": import_error,
    }


def inspect_packages() -> dict[str, Any]:
    rows = [distribution_status(dist, imp) for dist, imp in PACKAGE_SPECS]
    cuda = None
    torch_cuda_version = None
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        cuda = None
    return {
        "packages": rows,
        "torch_cuda_available": cuda,
        "torch_cuda_version": torch_cuda_version,
    }


def run_cmd(cmd: list[str], timeout: int = 20) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "missing"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 1, "", type(exc).__name__


def inspect_gpu() -> dict[str, Any]:
    code, out, err = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader"]
    )
    drv_code, drv_out, _ = run_cmd(["nvidia-smi"])
    driver_line = None
    if drv_code == 0:
        for line in drv_out.splitlines():
            if "CUDA Version" in line or "Driver Version" in line:
                driver_line = line.strip()
                break
    if code == 127:
        status = "NOT_APPLICABLE"
    elif code != 0:
        status = "NOT_VERIFIED"
    else:
        status = "PASS"
    return {
        "status": status,
        "nvidia_smi_exit": code,
        "query": out if code == 0 else None,
        "driver_line": driver_line,
        "error": err[:200] if code not in {0, 127} else None,
        "model_load_verified": "NOT_VERIFIED",
        "real_inference_verified": "NOT_VERIFIED",
    }


def inspect_git(cas_root: Path) -> dict[str, Any]:
    if shutil.which("git") is None:
        return {"status": "NOT_VERIFIED", "present": False}
    code, _, _ = run_cmd(["git", "-C", str(cas_root), "rev-parse", "--is-inside-work-tree"])
    if code != 0:
        return {"status": "MISSING", "present": False, "cas_root": str(cas_root)}
    def git(args: list[str]) -> Optional[str]:
        c, o, _ = run_cmd(["git", "-C", str(cas_root), *args])
        return o if c == 0 else None
    porcelain = git(["status", "--porcelain"])
    return {
        "status": "PASS",
        "present": True,
        "cas_root": str(cas_root),
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": git(["rev-parse", "HEAD"]),
        "working_tree_clean": porcelain == "",
        "dirty_count": len(porcelain.splitlines()) if porcelain else 0,
        "remote_url": "redacted",
    }


def checkpoint_presence(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    files = {}
    missing = []
    if not path.is_dir():
        return {
            "path": str(path),
            "files_found": "MISSING",
            "checkpoint_contract_verified": "FAIL",
            "model_load_verified": "NOT_VERIFIED",
            "real_inference_verified": "NOT_VERIFIED",
            "dir_exists": False,
            "missing": list(required),
            "files": {},
            "size": None,
        }
    for name in required:
        present = (path / name).is_file()
        files[name] = present
        if not present:
            missing.append(name)
    size = None
    code, out, _ = run_cmd(["du", "-sh", str(path)], timeout=30)
    if code == 0 and out:
        size = out.split()[0]
    complete = not missing
    return {
        "path": str(path),
        "dir_exists": True,
        "files_found": "PASS" if complete else "FAIL",
        "checkpoint_contract_verified": "PASS" if complete else "FAIL",
        "model_load_verified": "NOT_VERIFIED",
        "real_inference_verified": "NOT_VERIFIED",
        "missing": missing,
        "files": files,
        "size": size,
        "note": "file presence only; weights not hashed or loaded",
    }


def tokenizer_cache_status(env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    resolved = resolve_hf_hub_cache(env)
    cache_root = Path(resolved["cache_root"])
    snap = cache_root / "models--vinai--bartpho-syllable" / "snapshots" / LOCKED_TOKENIZER_REVISION
    cache_exists = cache_root.is_dir()
    snapshot_present = snap.is_dir()
    return {
        "source": "hub_pinned_rq1",
        "tokenizer_model_id": LOCKED_TOKENIZER_ID,
        "tokenizer_revision": LOCKED_TOKENIZER_REVISION,
        "cache_root": str(cache_root),
        "cache_source": resolved["source"],
        "cache_dir_exists": "PASS" if cache_exists else "MISSING",
        "snapshot_dir": str(snap),
        "snapshot_revision_present": "PASS" if snapshot_present else "MISSING",
        "files_found": "PASS" if snapshot_present else "MISSING",
        "tokenizer_contract_validated": "NOT_VERIFIED",
        "tokenizer_loaded": "NOT_VERIFIED",
        "tokenizer_verified": "NOT_VERIFIED",
        "note": "serving tokenizer is Hub pin, not MT checkpoint files; no download and no from_pretrained in this inspection",
    }


def inspect_c0_import(thesis: Path, python_exe: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(thesis) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import src.c0_cascaded, src.c0_tokenizer, src.c0_manifest; print('import_ok')"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(thesis),
            env=env,
        )
        ok = proc.returncode == 0 and "import_ok" in (proc.stdout or "")
        return {
            "status": "PASS" if ok else "FAIL",
            "error": None if ok else (proc.stderr or proc.stdout or "")[:400],
        }
    except Exception as exc:
        return {"status": "NOT_VERIFIED", "error": type(exc).__name__}


def bounded_ckpts(ckpts_dir: Path, limit: int = 12) -> list[str]:
    if not ckpts_dir.is_dir():
        return []
    names = []
    try:
        with os.scandir(ckpts_dir) as it:
            for entry in it:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False) and entry.name.startswith("checkpoint-"):
                    names.append(entry.name)
                if len(names) >= limit:
                    break
    except OSError:
        return []
    return sorted(names)[:limit]


def inspect_direct(artifact_root: Optional[Path]) -> dict[str, Any]:
    if artifact_root is None:
        return {"files_found": "MISSING", "path": None, "candidates": [], "note": "ARTIFACT_ROOT unset"}
    root = Path(artifact_root) / "bahnar_s2tt" / "direct_full_state"
    if not root.is_dir():
        return {"files_found": "MISSING", "path": str(root), "candidates": []}
    candidates = []
    try:
        with os.scandir(root) as it:
            for i, entry in enumerate(it):
                if i >= 8:
                    break
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("contract_"):
                    continue
                inner = Path(entry.path) / "checkpoints" / "full_train"
                experiments = []
                if inner.is_dir():
                    with os.scandir(inner) as exp_it:
                        for j, exp in enumerate(exp_it):
                            if j >= 6:
                                break
                            if exp.is_dir(follow_symlinks=False) and not exp.is_symlink():
                                ck = Path(exp.path) / "ckpts"
                                experiments.append(
                                    {"experiment": exp.name, "checkpoint_dirs": bounded_ckpts(ck if ck.is_dir() else Path(exp.path))}
                                )
                candidates.append({"contract": entry.name, "experiments": experiments})
    except OSError as exc:
        return {"files_found": "NOT_VERIFIED", "path": str(root), "error": type(exc).__name__, "candidates": []}
    return {
        "files_found": "PASS" if candidates else "MISSING",
        "path": str(root),
        "candidates": candidates,
        "model_load_verified": "NOT_VERIFIED",
        "note": "presence is not C0-vs-D0 evaluation",
    }


def build_report(
    cas_root: Path,
    python_exe: str,
    *,
    artifact_root: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    documented_artifact_root: str = DOCUMENTED_RUNPOD_ARTIFACT_ROOT,
) -> dict[str, Any]:
    layout = resolve_layout(cas_root)
    thesis = Path(layout["thesis_source_root"])
    artifacts = resolve_artifact_root(
        explicit=artifact_root,
        env=env,
        documented=documented_artifact_root,
    )
    art_path = Path(artifacts["path"]) if artifacts.get("path") else None
    asr_path = (art_path / ASR_REL) if art_path is not None else Path("")
    mt_path = (art_path / MT_REL) if art_path is not None else Path("")
    contract_path = (art_path / MT_CONTRACT_REL) if art_path is not None else Path("")
    asr = checkpoint_presence(asr_path, ASR_REQUIRED_FILES) if art_path is not None else {
        "path": None,
        "dir_exists": False,
        "files_found": "MISSING",
        "checkpoint_contract_verified": "FAIL",
        "model_load_verified": "NOT_VERIFIED",
        "real_inference_verified": "NOT_VERIFIED",
        "missing": list(ASR_REQUIRED_FILES),
        "files": {},
        "size": None,
        "note": "ARTIFACT_ROOT unset",
    }
    mt = checkpoint_presence(mt_path, MT_REQUIRED_FILES) if art_path is not None else {
        "path": None,
        "dir_exists": False,
        "files_found": "MISSING",
        "checkpoint_contract_verified": "FAIL",
        "model_load_verified": "NOT_VERIFIED",
        "real_inference_verified": "NOT_VERIFIED",
        "missing": list(MT_REQUIRED_FILES),
        "files": {},
        "size": None,
        "note": "ARTIFACT_ROOT unset",
    }
    example_cfg = thesis / EXAMPLE_CONFIG_REL
    runtime_cfg = thesis / "configs" / RUNTIME_CONFIG_NAME
    audio_roots = [thesis / rel for rel in PREFERRED_AUDIO_DIRS]
    if art_path is not None:
        audio_roots.extend([art_path / rel for rel in PREFERRED_AUDIO_DIRS])
    audio = bounded_discover_audio(audio_roots)
    pkgs = inspect_packages()
    git = inspect_git(cas_root)
    gpu = inspect_gpu()
    tok = tokenizer_cache_status(env)
    c0_import = inspect_c0_import(thesis, python_exe) if layout["layout_ok"] else {"status": "FAIL", "error": layout["mismatch"]}
    missing = []
    if not layout["layout_ok"]:
        missing.append("c0_entrypoint")
    if artifacts.get("exists") is not True:
        missing.append("artifact_root")
    if asr["checkpoint_contract_verified"] != "PASS":
        missing.append("asr_checkpoint_files")
    if mt["checkpoint_contract_verified"] != "PASS":
        missing.append("mt_checkpoint_files")
    if tok["snapshot_revision_present"] != "PASS":
        missing.append("hub_tokenizer_snapshot")
    if not example_cfg.is_file():
        missing.append("c0_inference.example.yaml")
    readiness = "NOT_VERIFIED"
    if layout["layout_ok"] and c0_import.get("status") == "PASS":
        readiness = "CONDITIONAL"
    documented_asr = f"{DOCUMENTED_RUNPOD_ARTIFACT_ROOT}/{ASR_REL}"
    documented_mt = f"{DOCUMENTED_RUNPOD_ARTIFACT_ROOT}/{MT_REL}"
    report = {
        "inspection_timestamp": datetime.now(timezone.utc).isoformat(),
        "runpod_runtime": "NOT_VERIFIED",
        "readiness_status": readiness,
        "cas_root": layout["cas_root"],
        "thesis_source_root": layout["thesis_source_root"],
        "thesis_root": layout["thesis_source_root"],
        "artifact_root": artifacts,
        "layout_ok": layout["layout_ok"],
        "layout_mismatch": layout["mismatch"],
        "documented_runpod_fixtures": {
            "cas_git": "/workspace/citizen-assistance-system",
            "artifact_root": DOCUMENTED_RUNPOD_ARTIFACT_ROOT,
            "asr_checkpoint": documented_asr,
            "mt_checkpoint": documented_mt,
            "mt_training_contract": f"{DOCUMENTED_RUNPOD_ARTIFACT_ROOT}/{MT_CONTRACT_REL}",
            "note": "file presence observed by owner; MODEL_LOAD still NOT_VERIFIED",
        },
        "git": git,
        "gpu": gpu,
        "python": {"executable": python_exe, "version": sys.version.split()[0]},
        "packages": pkgs,
        "c0_entrypoint": {
            "module": "src.c0_cascaded",
            "cli": "python -m src.c0_cascaded --config <CONFIG_PATH> --audio <AUDIO_PATH>",
            "example_config": str(example_cfg),
            "example_config_present": example_cfg.is_file(),
            "runtime_config": str(runtime_cfg),
            "runtime_config_present": runtime_cfg.is_file(),
            "runtime_config_note": "copy bahnar-s2tt-thesis/configs/c0_inference.example.yaml to c0_inference.runpod.yaml after real ARTIFACT_ROOT paths are known; do not guess; do not commit secrets",
            "import_check": c0_import,
        },
        "asr_checkpoint": asr,
        "mt_checkpoint": mt,
        "mt_training_contract": {
            "path": str(contract_path) if art_path is not None else None,
            "files_found": "PASS" if art_path is not None and contract_path.is_file() else "MISSING",
        },
        "serving_tokenizer": tok,
        "rq2_direct": inspect_direct(art_path),
        "candidate_audio": audio,
        "states": {
            "FILES_FOUND": "PASS" if asr.get("dir_exists") and mt.get("dir_exists") else "MISSING",
            "CHECKPOINT_CONTRACT_VERIFIED": "PASS"
            if asr["checkpoint_contract_verified"] == "PASS" and mt["checkpoint_contract_verified"] == "PASS"
            else "FAIL",
            "TOKENIZER_VERIFIED": tok["tokenizer_verified"],
            "MODEL_LOAD_VERIFIED": "NOT_VERIFIED",
            "REAL_INFERENCE_VERIFIED": "NOT_VERIFIED",
        },
        "missing_requirements": missing,
        "unverified_assumptions": [
            "GPU/CUDA identity on this host",
            "checkpoint identity vs trainer_state best step",
            "Hub tokenizer fingerprint vs mt_training_contract",
            "real Bahnar 16 kHz utterance",
        ],
        "secrets": "not printed",
        "sha256_policy": "weights not hashed in this inspection",
    }
    return report


def human_summary(report: dict[str, Any]) -> str:
    states = report["states"]
    lines = [
        f"CAS_ROOT={report['cas_root']}",
        f"THESIS_SOURCE_ROOT={report.get('thesis_source_root') or report.get('thesis_root')}",
        f"ARTIFACT_ROOT={report.get('artifact_root', {}).get('path')}",
        f"ARTIFACT_ROOT_SOURCE={report.get('artifact_root', {}).get('source')}",
        f"git={report['git'].get('branch')} {report['git'].get('commit')}",
        f"FILES_FOUND={states['FILES_FOUND']}",
        f"CHECKPOINT_CONTRACT_VERIFIED={states['CHECKPOINT_CONTRACT_VERIFIED']}",
        f"TOKENIZER_VERIFIED={states['TOKENIZER_VERIFIED']}",
        f"MODEL_LOAD_VERIFIED={states['MODEL_LOAD_VERIFIED']}",
        f"REAL_INFERENCE_VERIFIED={states['REAL_INFERENCE_VERIFIED']}",
        f"RUNPOD_RUNTIME={report['runpod_runtime']}",
        f"readiness={report['readiness_status']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase 7.3 CAS/RunPod inspection")
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phase7_runpod_inspection.json"))
    args = parser.parse_args(argv)
    layout = resolve_layout(args.cas_root)
    if not layout["layout_ok"]:
        print(layout["mismatch"], file=sys.stderr)
        payload = {"error": layout["mismatch"], "cas_root": layout["cas_root"], "runpod_runtime": "NOT_VERIFIED"}
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    report = build_report(
        Path(layout["cas_root"]),
        sys.executable,
        artifact_root=args.artifact_root,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"could not write {args.output}: {exc}", file=sys.stderr)
    sys.stdout.write(human_summary(report))
    sys.stdout.write(f"json={args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
