"""
Runtime path layout for Notebook 03 — RunPod only.

Defaults (overridable via env):
  LOCAL_ROOT   = /tmp/bahnar-runtime     (Parquet, WAV, HF cache, temps)
  DURABLE_ROOT = /workspace/bahnar-s2tt-thesis  (source, state, ckpts, metrics)

No Colab / Google Drive profile exists. Offline unit tests may override both
roots explicitly or via ``BAHNAR_LOCAL_ROOT`` / ``BAHNAR_DURABLE_ROOT``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

ENV_LOCAL_ROOT = "BAHNAR_LOCAL_ROOT"
ENV_DURABLE_ROOT = "BAHNAR_DURABLE_ROOT"

DEFAULT_LOCAL_ROOT = Path("/tmp/bahnar-runtime")
DEFAULT_DURABLE_ROOT = Path("/workspace/bahnar-s2tt-thesis")

OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP = "pair_key_train_drop_vs_validation_v1"

EFFECTIVE_TRAIN_MANIFEST = "effective_rq1_train.csv"
EFFECTIVE_VAL_MANIFEST = "effective_rq1_validation.csv"
EFFECTIVE_EXCLUSIONS_CSV = "effective_manifest_exclusions.csv"
EFFECTIVE_MANIFEST_SUMMARY = "effective_manifest_summary.json"

# Forbidden substrings in Notebook 03 / related src (AST / string guards).
COLAB_DRIVE_FORBIDDEN_MARKERS = (
    "google.colab",
    "drive.mount",
    "/content/drive",
    "/content/",
    "MyDrive",
    "DRIVE_ROOT",
    "DRIVE_MOUNT",
    "sync_to_drive",
    "restore_from_drive",
    "sync_experiment_checkpoints_to_drive",
    "restore_experiment_checkpoints_from_drive",
    "resolve_best_checkpoint_from_drive",
    "make_drive_checkpoint_sync_callback",
    "drive_rt",
    "drive_checkpoint_dir",
)


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved local vs durable roots and the derived directories stages use."""

    local_root: Path
    durable_root: Path
    project_root: Path

    @property
    def profile(self) -> str:
        return "runpod"

    @property
    def audio_cache_dir(self) -> Path:
        return self.local_root / "bahnar_full_audio_cache"

    @property
    def local_ckpt_root(self) -> Path:
        return self.local_root / "bahnar_full_checkpoints"

    @property
    def hf_parquet_cache_dir(self) -> Path:
        return self.local_root / "hf_parquet_cache"

    @property
    def durable_state_root(self) -> Path:
        """Parent of per-contract prepare/train state directories."""
        return self.durable_root / "bahnar_s2tt" / "full_state"

    @property
    def export_root(self) -> Path:
        return self.durable_root / "exports" / "notebook03_runs"

    def prepare_state_dir(self, contract_hash: str) -> Path:
        """Isolate prepare artifacts by contract so MAX=30 state cannot resume under MAX=40."""
        h = str(contract_hash or "").strip()
        if len(h) < 12:
            raise ValueError(
                f"contract_hash too short for state isolation: {contract_hash!r}"
            )
        return self.durable_state_root / f"contract_{h[:16]}"

    def as_dict(self) -> Dict[str, str]:
        return {
            "profile": self.profile,
            "local_root": str(self.local_root),
            "durable_root": str(self.durable_root),
            "project_root": str(self.project_root),
            "audio_cache_dir": str(self.audio_cache_dir),
            "local_ckpt_root": str(self.local_ckpt_root),
            "hf_parquet_cache_dir": str(self.hf_parquet_cache_dir),
            "durable_state_root": str(self.durable_state_root),
            "export_root": str(self.export_root),
        }


def resolve_runtime_paths(
    *,
    project_root: Union[str, Path],
    env: Optional[Mapping[str, str]] = None,
    local_root: Optional[Union[str, Path]] = None,
    durable_root: Optional[Union[str, Path]] = None,
) -> RuntimePaths:
    """
    Resolve ``LOCAL_ROOT`` / ``DURABLE_ROOT``.

    Priority: explicit args → env → RunPod defaults.
    """
    env = env if env is not None else os.environ
    project = Path(project_root).resolve()
    loc = Path(
        local_root
        if local_root is not None
        else (env.get(ENV_LOCAL_ROOT) or DEFAULT_LOCAL_ROOT)
    )
    dur = Path(
        durable_root
        if durable_root is not None
        else (env.get(ENV_DURABLE_ROOT) or DEFAULT_DURABLE_ROOT)
    )
    if not str(loc).strip() or not str(dur).strip():
        raise ValueError("LOCAL_ROOT and DURABLE_ROOT must be non-empty paths")
    return RuntimePaths(local_root=loc, durable_root=dur, project_root=project)


def find_colab_drive_markers(text: str) -> list[str]:
    """Return forbidden Colab/Drive markers found in ``text``."""
    return [m for m in COLAB_DRIVE_FORBIDDEN_MARKERS if m in text]


def assert_no_colab_drive_markers(text: str, *, label: str = "source") -> None:
    hits = find_colab_drive_markers(text)
    if hits:
        raise AssertionError(f"{label} still contains Colab/Drive markers: {hits}")


# Back-compat alias used by earlier tests — same semantics, stronger check.
def assert_no_forbidden_hardcoded_roots(text: str) -> None:
    assert_no_colab_drive_markers(text, label="notebook/src")
