"""Shared data helpers for Bahnar S2TT thesis notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"


def load_manifest(split: str) -> pd.DataFrame:
    """Load rq1_{split}.csv or .parquet from data/manifests."""
    csv_path = MANIFEST_DIR / f"rq1_{split}.csv"
    parquet_path = MANIFEST_DIR / f"rq1_{split}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Missing manifest for split={split!r}. Expected {parquet_path} or {csv_path}."
    )


def project_paths() -> dict[str, Path]:
    return {
        "root": PROJECT_ROOT,
        "manifests": MANIFEST_DIR,
        "audit": AUDIT_DIR,
        "checkpoints": PROJECT_ROOT / "checkpoints",
        "predictions": PROJECT_ROOT / "predictions",
        "metrics": PROJECT_ROOT / "metrics",
        "results": PROJECT_ROOT / "results",
        "configs": PROJECT_ROOT / "configs",
    }
