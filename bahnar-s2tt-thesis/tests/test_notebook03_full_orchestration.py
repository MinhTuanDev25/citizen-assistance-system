"""Orchestration / notebook-wiring integration tests for Notebook 03 full stages."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.asr_full_train import (
    FULL_TRAIN_MARKER,
    assert_checkpoint_allowed_for_full_train,
    assert_no_frozen_test_access,
    assert_ready_for_full_evaluate,
    build_training_contract,
    experiment_checkpoint_dir,
    make_drive_checkpoint_sync_callback,
    resolve_full_stage_status,
    resolve_resume_checkpoint,
    restore_experiment_checkpoints_from_drive,
    sync_experiment_checkpoints_to_drive,
    write_checkpoint_fingerprint,
    write_full_train_summary,
)


NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "03_asr_baseline_training.ipynb"


def _nb_code_cells() -> dict[str, str]:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    out = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for key in ("F1 —", "F2 —", "F3 —", "F4 —", "Cell 28 —", "Cell 31 —"):
            if key in src.split("\n", 1)[0] or (src.startswith("# Cell") and key.split("—")[0].strip() in src[:40]):
                if "F1" in key and "FULL_STAGE=prepare" in src:
                    out["F1"] = src
                elif "F2" in key:
                    out["F2"] = src
                elif "F3" in key:
                    out["F3"] = src
                elif "F4" in key:
                    out["F4"] = src
                elif "Cell 28" in key:
                    out["C28"] = src
        if src.startswith("# Cell F1"):
            out["F1"] = src
        if src.startswith("# Cell F2"):
            out["F2"] = src
        if src.startswith("# Cell F3"):
            out["F3"] = src
        if src.startswith("# Cell F4"):
            out["F4"] = src
        if src.startswith("# Cell 28"):
            out["C28"] = src
    return out


def _complete_ckpt(path: Path, step: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": 0.1}), encoding="utf-8"
    )
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(b"x")
    return path


def _contract(**overrides):
    # Overrides go into the builder so contract_hash stays consistent with the
    # fields; patching the dict afterwards would leave a stale hash.
    kwargs = dict(
        experiment_id="expA",
        dataset_id="ds",
        dataset_revision="rev",
        parquet_revision="a" * 40,
        train_manifest_content_hash="thash",
        validation_manifest_content_hash="vhash",
        vocab_fp="vfp",
        processing_version="notebook02_audio_v3",
        min_duration=0.5,
        max_duration=30.0,
        target_sr=16000,
        pretrained_model_id="facebook/wav2vec2-xls-r-300m",
        pretrained_model_revision="mrev",
        hparams={"max_steps": 10},
    )
    kwargs.update(overrides)
    return build_training_contract(**kwargs)


class TestNotebookOrchestrationWiring:
    def test_f2_f3_f4_use_opened_paths_and_full_audio_args(self):
        cells = _nb_code_cells()
        for name in ("F2", "F3", "F4"):
            src = cells[name]
            assert "OPENED_PATHS" in src
            assert "LOADED_SPLITS" in src
            assert "opened_paths" not in src or "opened_paths=OPENED_PATHS" in src
            assert "assert_eligible_audio_available(" in src
            assert "dataset_revision=EXPECTED_DATASET_REVISION" in src
            assert "target_sr=TARGET_SAMPLING_RATE" in src
            assert "min_duration=MIN_AUDIO_DURATION" in src
            assert "max_duration=MAX_AUDIO_DURATION" in src
            assert "expected_processing_version=EXPECTED_AUDIO_PROCESSING_VERSION" in src
            assert "register_path(FULL_STATE_DIR" in src
            assert "assert_eligible_frames_no_frozen_splits" in src
            ast.parse(src)

    def test_helpers_integrated_in_notebook(self):
        cells = _nb_code_cells()
        f2, f3, f4, c28 = cells["F2"], cells["F3"], cells["F4"], cells["C28"]
        assert "restore_experiment_checkpoints_from_drive" in f2
        assert "restore_experiment_checkpoints_from_drive" in f3
        assert "make_drive_checkpoint_sync_callback" in f3
        assert "callbacks=[drive_sync_cb]" in f3
        assert "assert_ready_for_full_evaluate" in f4
        assert "resolve_full_stage_status" in c28
        # restore before resolve in F3
        assert f3.index("restore_experiment_checkpoints_from_drive") < f3.index("resolve_resume_checkpoint")
        # true restart in F2
        assert "model_b = Wav2Vec2ForCTC.from_pretrained" in f2
        assert "del trainer_a, model_a" in f2
        assert "true_restart" in f2
        # F3 always constructs model even on resume
        resume_idx = f3.index("if resume_ck:")
        model_idx = f3.index("model = Wav2Vec2ForCTC.from_pretrained")
        assert model_idx > resume_idx
        # no hardcoded frozen_test_accessed=False in derive call
        assert "frozen_test_accessed=False" not in f3
        assert "frozen_test_accessed=bool(frozen_hits_full)" in f3

    def test_no_undefined_opened_paths_aliases(self):
        cells = _nb_code_cells()
        for name in ("F2", "F3", "F4"):
            src = cells[name]
            assert "list(opened_paths)" not in src
            assert "list(opened_splits)" not in src


class TestFingerprintAndDriveFailClosed:
    def test_fingerprint_preserves_contract_on_step_update(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        contract = _contract()
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER,
            global_step=0, extra=contract, overwrite=True,
        )
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER,
            global_step=500, overwrite=True, preserve_existing=True,
        )
        data = json.loads((root / "full_experiment_fingerprint.json").read_text(encoding="utf-8"))
        assert data["global_step"] == 500
        assert data["vocab_fp"] == "vfp"
        assert data["train_manifest_content_hash"] == "thash"

    def test_sync_fail_closed_missing_drive(self, tmp_path: Path):
        exp = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        exp.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="FULL_STATE_DIR missing|fail-closed"):
            sync_experiment_checkpoints_to_drive(
                exp, tmp_path / "missing_drive", experiment_id="expA", kind=FULL_TRAIN_MARKER,
                require_drive=True,
            )

    def test_on_save_callback_fail_closed(self, tmp_path: Path):
        exp = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(exp, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=0)
        _complete_ckpt(exp / "checkpoint-100", 100)
        cb = make_drive_checkpoint_sync_callback(
            local_experiment_dir=exp,
            full_state_dir=tmp_path / "no_drive",
            experiment_id="expA",
            kind=FULL_TRAIN_MARKER,
            training_contract=_contract(),
        )
        state = type("S", (), {"global_step": 100})()
        with pytest.raises(RuntimeError, match="Drive checkpoint sync failed"):
            cb.on_save(None, state, type("C", (), {})())

    def test_restore_before_resolve_order(self, tmp_path: Path):
        drive_state = tmp_path / "drive"
        drive_state.mkdir()
        local_root = tmp_path / "local"
        drive_exp = experiment_checkpoint_dir(drive_state / "checkpoints", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(drive_exp, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        _complete_ckpt(drive_exp / "checkpoint-100", 100)
        fresh = experiment_checkpoint_dir(local_root, "expA", kind=FULL_TRAIN_MARKER)
        restored = restore_experiment_checkpoints_from_drive(
            fresh, drive_state, experiment_id="expA", kind=FULL_TRAIN_MARKER,
        )
        assert restored is not None
        ck = resolve_resume_checkpoint(
            resume_policy="auto", experiment_dir=fresh, experiment_id="expA",
        )
        assert ck and ck.endswith("checkpoint-100")


class TestContractAndEvaluateGate:
    def test_checkpoint_contract_mismatch(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        ck = _complete_ckpt(root / "checkpoint-50", 50)
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER,
            global_step=50, extra=_contract(vocab_fp="old"),
        )
        with pytest.raises(RuntimeError, match="training_contract mismatch"):
            assert_checkpoint_allowed_for_full_train(
                ck, experiment_id="expA", expected_contract=_contract(vocab_fp="new"),
            )

    def test_stale_success_does_not_mask_pipeline_error(self, tmp_path: Path):
        write_full_train_summary(
            tmp_path,
            {"status": "SUCCESS_FULL_TRAINING", "experiment_id": "expA", **_contract()},
        )
        status = resolve_full_stage_status(
            pipeline_error={"message": "boom"},
            current_full_status=None,
            full_stage="train",
            state_dir=tmp_path,
            experiment_id="expA",
        )
        assert status == "FAILED"

    def test_evaluate_requires_matching_contract(self, tmp_path: Path):
        write_full_train_summary(
            tmp_path,
            {"status": "SUCCESS_FULL_TRAINING", **_contract()},
        )
        assert_ready_for_full_evaluate(
            tmp_path,
            expected_experiment_id="expA",
            expected_contract=_contract(),
        )
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_TRAINING"):
            assert_ready_for_full_evaluate(
                tmp_path,
                expected_experiment_id="expA",
                expected_contract=_contract(vocab_fp="other"),
            )


class TestFrozenCaseInsensitive:
    def test_assert_no_frozen_case_insensitive(self):
        with pytest.raises(RuntimeError, match="Frozen-test split"):
            assert_no_frozen_test_access([], ["RQ1_TEST"])
        with pytest.raises(RuntimeError, match="Frozen-test split"):
            assert_no_frozen_test_access([], ["Test"])
