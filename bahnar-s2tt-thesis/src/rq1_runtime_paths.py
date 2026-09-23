"""Runtime paths for Notebook 06 — final frozen RQ1 comparison."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

ENV_LOCAL_ROOT = "BAHNAR_LOCAL_ROOT"
ENV_DURABLE_ROOT = "BAHNAR_DURABLE_ROOT"
DEFAULT_LOCAL_ROOT = Path("/tmp/bahnar-runtime")
DEFAULT_DURABLE_ROOT = Path("/workspace/bahnar-s2tt-thesis")


def find_bahnar_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve repository root from cwd, notebooks/, or any descendant.

    Requires ``requirements.txt``, ``src/``, and ``data/manifests/``.
    """
    node = Path(start or Path.cwd()).resolve()
    candidates = [node, *node.parents]
    for cand in candidates:
        if (
            (cand / "requirements.txt").is_file()
            and (cand / "src").is_dir()
            and (cand / "data" / "manifests").is_dir()
        ):
            return cand
    raise RuntimeError(
        f"Cannot locate bahnar-s2tt-thesis root from {node} "
        "(need requirements.txt + src/ + data/manifests/)"
    )


@dataclass(frozen=True)
class Rq1RuntimePaths:
    local_root: Path
    durable_root: Path
    project_root: Path

    @property
    def local_audio_dir(self) -> Path:
        return self.local_root / "rq1_frozen_audio"

    @property
    def hf_parquet_cache_dir(self) -> Path:
        return self.local_root / "hf_parquet_cache_rq1"

    @property
    def local_asr_ckpt_root(self) -> Path:
        return self.local_root / "rq1_restore" / "asr"

    @property
    def local_mt_ckpt_root(self) -> Path:
        return self.local_root / "rq1_restore" / "mt"

    @property
    def local_direct_ckpt_root(self) -> Path:
        return self.local_root / "rq1_restore" / "direct"

    @property
    def durable_state_root(self) -> Path:
        return self.durable_root / "bahnar_s2tt" / "rq1_final_state"

    @property
    def export_root(self) -> Path:
        return self.durable_root / "exports" / "notebook06_runs"

    def state_dir(self, contract_hash: str) -> Path:
        h = str(contract_hash or "").strip()
        if len(h) < 12:
            raise ValueError(f"rq1 contract hash too short: {contract_hash!r}")
        return self.durable_state_root / f"contract_{h[:16]}"


def resolve_rq1_runtime_paths(
    *,
    project_root: Union[str, Path],
    env: Optional[Mapping[str, str]] = None,
    local_root: Optional[Union[str, Path]] = None,
    durable_root: Optional[Union[str, Path]] = None,
) -> Rq1RuntimePaths:
    env = env if env is not None else os.environ
    return Rq1RuntimePaths(
        local_root=Path(local_root or env.get(ENV_LOCAL_ROOT) or DEFAULT_LOCAL_ROOT),
        durable_root=Path(durable_root or env.get(ENV_DURABLE_ROOT) or DEFAULT_DURABLE_ROOT),
        project_root=Path(project_root).resolve(),
    )
