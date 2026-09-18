"""
The export cell, executed for real against a stubbed namespace.

Unit tests on `src/` helpers cannot catch a stage the export cell simply does not
handle, which is how a successful `full_prepare` still ended the notebook with
"No full-train checkpoints found". These tests run the actual cell body for each
FULL_STAGE, so the wiring is checked rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "03_asr_baseline_training.ipynb"
EXPORT_CELL_MARKER = "Export run artifacts to DURABLE_ROOT"


def export_cell_source() -> str:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
        and EXPORT_CELL_MARKER in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1, f"expected exactly one export cell, found {len(matches)}"
    return matches[0]


def _namespace(tmp_path: Path, *, full_stage: str, contract=None, best=None):
    """A minimal stand-in for the notebook globals the export cell reads."""
    durable_root = tmp_path / "durable"
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    state_dir = durable_root / "bahnar_s2tt" / "full_state" / "contract_test"
    for d in (durable_root, artifacts, results, state_dir):
        d.mkdir(parents=True, exist_ok=True)
    (artifacts / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    (results / "run_summary.json").write_text("{}", encoding="utf-8")

    called = {"resolve_best": 0, "copied": []}

    def _resolve_best(state, *, experiment_id, train_summary, expected_contract, local_experiment_dir=None):
        called["resolve_best"] += 1
        called["local_experiment_dir"] = local_experiment_dir
        if best is None:
            raise RuntimeError("no valid best checkpoint")
        return best

    class _RuntimePaths:
        export_root = durable_root / "exports" / "notebook03_runs"

    return called, {
        "RUN_ID": "run-test",
        "RUN_MODE": "full",
        "FULL_STAGE": full_stage,
        "FULL_EXPERIMENT_ID": "full_xlsr300m_v1",
        "FULL_TRAIN_MARKER": "full_train",
        "DURABLE_ROOT": durable_root,
        "LOCAL_ROOT": tmp_path / "local",
        "RUNTIME_PATHS": _RuntimePaths(),
        "FULL_STATE_DIR": state_dir,
        "LOCAL_CKPT_ROOT": tmp_path / "ckpt",
        "PATHS": {
            "artifacts": artifacts,
            "results": results,
            "results_root": tmp_path / "results_root",
            "checkpoints": tmp_path / "pilot_ckpt",
        },
        "full_stage_contract": contract,
        "json": json,
        "Path": Path,
        "resolve_best_checkpoint_from_durable": _resolve_best,
        "experiment_checkpoint_dir": lambda root, exp, *, kind: Path(root) / kind / exp,
        "durable_experiment_dir": lambda root, exp, *, kind: Path(root) / kind / exp,
    }


def _run(ns: dict) -> dict:
    exec(compile(export_cell_source(), "<export_cell>", "exec"), ns)  # noqa: S102
    return ns


def _destination(ns: dict) -> Path:
    return Path(ns["RUNTIME_PATHS"].export_root) / ns["RUN_ID"]


class TestPrepareStage:
    def test_prepare_export_succeeds_without_any_checkpoint(self, tmp_path: Path):
        called, ns = _namespace(tmp_path, full_stage="prepare")
        (ns["FULL_STATE_DIR"] / "full_data_summary.json").write_text(
            json.dumps({"status": "SUCCESS_FULL_PREPARE"}), encoding="utf-8"
        )
        _run(ns)  # used to raise "No full-train checkpoints found"

        dest = _destination(ns)
        assert (dest / "artifacts" / "artifact_manifest.json").is_file()
        assert (dest / "results" / "run_summary.json").is_file()
        assert not (dest / "checkpoints").exists()
        assert called["resolve_best"] == 0  # no checkpoint gate at this stage

    def test_prepare_export_records_a_pointer_not_bytes(self, tmp_path: Path):
        called, ns = _namespace(tmp_path, full_stage="prepare")
        (ns["FULL_STATE_DIR"] / "full_data_summary.json").write_text(
            json.dumps({"status": "SUCCESS_FULL_PREPARE"}), encoding="utf-8"
        )
        _run(ns)

        pointer = json.loads(
            (_destination(ns) / "full_export_pointer.json").read_text(encoding="utf-8")
        )
        assert pointer["full_stage"] == "prepare"
        assert pointer["stage_status"] == "SUCCESS_FULL_PREPARE"
        assert pointer["full_state_dir"] == str(ns["FULL_STATE_DIR"])
        assert "best_checkpoint" not in pointer

    def test_prepare_export_still_requires_the_prepare_summary(self, tmp_path: Path):
        _called, ns = _namespace(tmp_path, full_stage="prepare")
        with pytest.raises(RuntimeError, match="full_data_summary.json"):
            _run(ns)


class TestResumeTestStages:
    @pytest.mark.parametrize("stage", ["resume_test_a", "resume_test_b"])
    def test_resume_test_export_does_not_need_full_train_artifacts(
        self, tmp_path: Path, stage: str
    ):
        called, ns = _namespace(tmp_path, full_stage=stage)
        _run(ns)

        pointer = json.loads(
            (_destination(ns) / "full_export_pointer.json").read_text(encoding="utf-8")
        )
        assert pointer["full_stage"] == stage
        assert called["resolve_best"] == 0
        assert not (_destination(ns) / "checkpoints").exists()


class TestTrainAndEvaluateStages:
    def _with_train_summary(self, tmp_path: Path, **kwargs):
        called, ns = _namespace(tmp_path, **kwargs)
        (ns["FULL_STATE_DIR"] / "full_train_summary.json").write_text(
            json.dumps({"status": "SUCCESS_FULL_TRAINING"}), encoding="utf-8"
        )
        return called, ns

    def test_train_export_gates_on_the_best_checkpoint(self, tmp_path: Path):
        best = tmp_path / "ckpt" / "full_train" / "full_xlsr300m_v1" / "checkpoint-500"
        best.mkdir(parents=True)
        called, ns = self._with_train_summary(
            tmp_path, full_stage="train", contract={"hparams": {"lr": 1e-4}}, best=best
        )
        _run(ns)

        assert called["resolve_best"] == 1
        assert called["local_experiment_dir"] is not None
        assert Path(called["local_experiment_dir"]).name == "full_xlsr300m_v1"
        pointer = json.loads(
            (_destination(ns) / "full_export_pointer.json").read_text(encoding="utf-8")
        )
        assert pointer["best_checkpoint"] == str(best)
        # The bytes stay in the durable versioned store; export must not copy GBs.
        assert not (_destination(ns) / "checkpoints").exists()

    def test_train_export_fails_when_best_checkpoint_is_invalid(self, tmp_path: Path):
        _called, ns = self._with_train_summary(
            tmp_path, full_stage="train", contract={"hparams": {"lr": 1e-4}}, best=None
        )
        with pytest.raises(RuntimeError, match="no valid best checkpoint"):
            _run(ns)

    def test_train_export_fails_without_a_contract(self, tmp_path: Path):
        _called, ns = self._with_train_summary(tmp_path, full_stage="train", contract=None)
        with pytest.raises(RuntimeError, match="without the canonical"):
            _run(ns)

    def test_evaluate_export_requires_its_own_summary(self, tmp_path: Path):
        _called, ns = self._with_train_summary(tmp_path, full_stage="evaluate", contract={"a": 1})
        with pytest.raises(RuntimeError, match="full_evaluate_summary.json"):
            _run(ns)


class TestPilotModeUnchanged:
    def test_pilot_still_exports_its_own_checkpoints_directory(self, tmp_path: Path):
        _called, ns = _namespace(tmp_path, full_stage="prepare")
        ns["RUN_MODE"] = "pilot"
        pilot_ckpt = ns["PATHS"]["checkpoints"]
        (pilot_ckpt / "checkpoint-10").mkdir(parents=True)
        _run(ns)

        dest = _destination(ns)
        assert (dest / "checkpoints" / "checkpoint-10").is_dir()
        assert not (dest / "full_export_pointer.json").exists()
