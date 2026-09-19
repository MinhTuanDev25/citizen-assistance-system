"""
Runtime path layout for Notebook 04 — MT baseline (RunPod-oriented).

Defaults mirror Notebook 03 local/durable split but under mt_* namespaces.
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

# Locked BARTPho syllable baseline (see mt_contract.LOCKED_*). Kept for path helpers.
DEFAULT_MODEL_ID_CANDIDATE = "vinai/bartpho-syllable"

MT_NORMALIZATION_VERSION = "mt_bahnar_vi_nfc_ws_v1"
SOURCE_FIELD = "text_bahnar"
TARGET_FIELD = "text_vi"

PILOT_MARKER = "pilot"
RESUME_TEST_MARKER = "resume_test"
FULL_TRAIN_MARKER = "full_train"


@dataclass(frozen=True)
class MtRuntimePaths:
    local_root: Path
    durable_root: Path
    project_root: Path

    @property
    def local_ckpt_root(self) -> Path:
        return self.local_root / "bahnar_mt_checkpoints"

    @property
    def durable_state_root(self) -> Path:
        return self.durable_root / "bahnar_s2tt" / "mt_full_state"

    @property
    def export_root(self) -> Path:
        return self.durable_root / "exports" / "notebook04_runs"

    def prepare_state_dir(self, contract_hash: str) -> Path:
        h = str(contract_hash or "").strip()
        if len(h) < 12:
            raise ValueError(f"contract_hash too short for MT state isolation: {contract_hash!r}")
        return self.durable_state_root / f"contract_{h[:16]}"

    def as_dict(self) -> Dict[str, str]:
        return {
            "local_root": str(self.local_root),
            "durable_root": str(self.durable_root),
            "project_root": str(self.project_root),
            "local_ckpt_root": str(self.local_ckpt_root),
            "durable_state_root": str(self.durable_state_root),
            "export_root": str(self.export_root),
        }


def resolve_mt_runtime_paths(
    *,
    project_root: Union[str, Path],
    env: Optional[Mapping[str, str]] = None,
    local_root: Optional[Union[str, Path]] = None,
    durable_root: Optional[Union[str, Path]] = None,
) -> MtRuntimePaths:
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
    return MtRuntimePaths(local_root=loc, durable_root=dur, project_root=project)
