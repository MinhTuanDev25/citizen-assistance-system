"""Runtime paths for Notebook 05 — Direct Bahnar speech -> Vietnamese text."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

ENV_LOCAL_ROOT = "BAHNAR_LOCAL_ROOT"
ENV_DURABLE_ROOT = "BAHNAR_DURABLE_ROOT"
DEFAULT_LOCAL_ROOT = Path("/tmp/bahnar-runtime")
DEFAULT_DURABLE_ROOT = Path("/workspace/bahnar-s2tt-thesis")

PILOT_MARKER = "pilot"
RESUME_TEST_MARKER = "resume_test"
FULL_TRAIN_MARKER = "full_train"


@dataclass(frozen=True)
class DirectRuntimePaths:
    local_root: Path
    durable_root: Path
    project_root: Path

    @property
    def audio_cache_dir(self) -> Path:
        # Intentionally share the exact NB03 PCM cache. The Direct model uses the
        # same matched speech examples and must not duplicate ~tens of GB of WAVs.
        return self.local_root / "bahnar_full_audio_cache"

    @property
    def hf_parquet_cache_dir(self) -> Path:
        return self.local_root / "hf_parquet_cache"

    @property
    def local_ckpt_root(self) -> Path:
        return self.local_root / "bahnar_direct_checkpoints"

    @property
    def durable_state_root(self) -> Path:
        return self.durable_root / "bahnar_s2tt" / "direct_full_state"

    @property
    def export_root(self) -> Path:
        return self.durable_root / "exports" / "notebook05_runs"

    def state_dir(self, contract_hash: str) -> Path:
        h = str(contract_hash or "").strip()
        if len(h) < 12:
            raise ValueError(f"contract_hash too short: {contract_hash!r}")
        return self.durable_state_root / f"contract_{h[:16]}"

    def as_dict(self) -> Dict[str, str]:
        return {
            "local_root": str(self.local_root),
            "durable_root": str(self.durable_root),
            "project_root": str(self.project_root),
            "audio_cache_dir": str(self.audio_cache_dir),
            "hf_parquet_cache_dir": str(self.hf_parquet_cache_dir),
            "local_ckpt_root": str(self.local_ckpt_root),
            "durable_state_root": str(self.durable_state_root),
            "export_root": str(self.export_root),
        }


def resolve_direct_runtime_paths(
    *,
    project_root: Union[str, Path],
    env: Optional[Mapping[str, str]] = None,
    local_root: Optional[Union[str, Path]] = None,
    durable_root: Optional[Union[str, Path]] = None,
) -> DirectRuntimePaths:
    env = env if env is not None else os.environ
    return DirectRuntimePaths(
        local_root=Path(local_root or env.get(ENV_LOCAL_ROOT) or DEFAULT_LOCAL_ROOT),
        durable_root=Path(durable_root or env.get(ENV_DURABLE_ROOT) or DEFAULT_DURABLE_ROOT),
        project_root=Path(project_root).resolve(),
    )
