"""Small durable export helper for Notebook 05."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Union


def export_notebook05_run(
    *, export_root: Union[str, Path], run_id: str, artifacts_dir: Union[str, Path],
    results_dir: Union[str, Path], state_dir: Union[str, Path], pointer: Mapping[str, Any],
) -> Path:
    dst = Path(export_root) / str(run_id)
    dst.mkdir(parents=True, exist_ok=True)
    for name, src in (("artifacts", Path(artifacts_dir)), ("results", Path(results_dir))):
        if src.is_dir():
            out = dst / name
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(src, out)
    state_out = dst / "state"
    state_out.mkdir(exist_ok=True)
    state = Path(state_dir)
    for filename in (
        "direct_prepare_summary.json", "direct_target_tokenizer_audit.json",
        "direct_pilot_summary.json", "direct_resume_phase_a.json", "direct_resume_test_summary.json",
        "direct_training_contract.json", "direct_train_summary.json", "direct_evaluate_summary.json",
    ):
        p = state / filename
        if p.is_file():
            shutil.copy2(p, state_out / filename)
    (dst / "full_export_pointer.json").write_text(
        json.dumps(dict(pointer), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return dst
