"""
Regression coverage for the RunPod-only Notebook 03 review fixes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.asr_full_data import is_frozen_split_label, prefilter_clean_split
from src.asr_full_pcm import AUDIO_PCM_PIPELINE_VERSION
from src.asr_full_train import (
    AUDIO_PCM_PIPELINE_VERSION as TRAIN_PCM_ALIAS,
    DATA_CONTRACT_KEYS,
    FULL_TRAIN_MARKER,
    assert_checkpoint_allowed_for_full_train,
    assert_no_frozen_test_access,
    build_data_contract,
    experiment_checkpoint_dir,
    plan_local_checkpoint_disk_peak,
    write_checkpoint_fingerprint,
)
from src.asr_runtime_paths import OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP, resolve_runtime_paths

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "03_asr_baseline_training.ipynb"


def _nb_cells() -> dict[str, str]:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    out = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.startswith("# Cell F3"):
            out["F3"] = src
        elif src.startswith("# Cell F4"):
            out["F4"] = src
        elif src.startswith("# Cell F1"):
            out["F1"] = src
        elif src.startswith("# Cell F2"):
            out["F2"] = src
    return out


def _contract(**kwargs):
    base = dict(
        dataset_id="d",
        dataset_revision="r",
        parquet_revision="p" * 40,
        train_manifest_content_hash="t" * 64,
        validation_manifest_content_hash="v" * 64,
        vocab_fp="v" * 64,
        processing_version="notebook02_audio_v3",
        min_duration=0.5,
        max_duration=40.0,
        target_sr=16000,
        pretrained_model_id="m",
        pretrained_model_revision="mr",
        overlap_policy=OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
    )
    base.update(kwargs)
    return build_data_contract(**base)


class TestSessionFindsContractScopedState:
    def test_f3_and_f4_scope_state_before_readiness_or_csv(self):
        cells = _nb_cells()
        for name in ("F3", "F4"):
            src = cells[name]
            scoped = src.find('FULL_STATE_DIR = RUNTIME_PATHS.prepare_state_dir')
            assert scoped > 0
            gate = src.find("assert_ready_for_full_train" if name == "F3" else "full_train_summary.json")
            csv = src.find("full_train_eligible.csv" if name == "F3" else "full_validation_eligible.csv")
            assert scoped < gate < csv, f"{name} must scope FULL_STATE_DIR before gates/CSV"

    def test_new_session_resolves_same_prepare_state_dir(self, tmp_path: Path):
        paths = resolve_runtime_paths(
            project_root=tmp_path,
            local_root=tmp_path / "L",
            durable_root=tmp_path / "D",
        )
        c = _contract()
        a = paths.prepare_state_dir(c["contract_hash"])
        b = paths.prepare_state_dir(_contract()["contract_hash"])
        assert a == b
        assert a.parent == paths.durable_state_root


class TestPcmVersionInContract:
    def test_pcm_version_is_in_hash_keys(self):
        assert "audio_pcm_pipeline_version" in DATA_CONTRACT_KEYS
        assert TRAIN_PCM_ALIAS == AUDIO_PCM_PIPELINE_VERSION

    def test_pcm_mismatch_changes_contract_hash(self):
        a = _contract(audio_pcm_pipeline_version=AUDIO_PCM_PIPELINE_VERSION)
        b = _contract(audio_pcm_pipeline_version="other_pcm_v0")
        assert a["contract_hash"] != b["contract_hash"]

    def test_notebook_binds_pcm_version_in_all_full_stages(self):
        cells = _nb_cells()
        for name in ("F1", "F2", "F3", "F4"):
            assert "EXPECTED_AUDIO_PCM_PIPELINE_VERSION" in cells[name], name


class TestForeignCheckpointRoot:
    def _mk(self, root: Path, exp: str, step: int = 1) -> Path:
        exp_dir = experiment_checkpoint_dir(root, exp, kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(
            exp_dir, experiment_id=exp, kind=FULL_TRAIN_MARKER, global_step=step, overwrite=True,
        )
        cp = exp_dir / f"checkpoint-{step}"
        cp.mkdir(parents=True)
        (cp / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
            (cp / name).write_bytes(b"x")
        # fingerprint also under step dir via parent lookup
        return cp

    def test_foreign_root_rejected_even_with_matching_segment_names(self, tmp_path: Path):
        configured = experiment_checkpoint_dir(tmp_path / "local_a", "expA", kind=FULL_TRAIN_MARKER)
        foreign = self._mk(tmp_path / "other_host", "expA")
        write_checkpoint_fingerprint(
            configured, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=0, overwrite=True,
        )
        with pytest.raises(RuntimeError, match="exactly under configured"):
            assert_checkpoint_allowed_for_full_train(
                foreign, experiment_id="expA", experiment_root=configured, require_complete=False,
            )

    def test_configured_root_accepted(self, tmp_path: Path):
        cp = self._mk(tmp_path / "local", "expA")
        root = cp.parent
        assert assert_checkpoint_allowed_for_full_train(
            cp, experiment_id="expA", experiment_root=root, require_complete=False,
        )


class TestFrozenCaseInsensitive:
    def test_mixed_case_and_hyphen_labels(self):
        assert is_frozen_split_label("Test")
        assert is_frozen_split_label("RQ1-Test")
        assert is_frozen_split_label(" frozen_test ")
        assert not is_frozen_split_label("train")

    def test_assert_no_frozen_test_access_rejects_mixed_case(self):
        with pytest.raises(RuntimeError, match="Frozen-test split"):
            assert_no_frozen_test_access([], ["Train", "TEST"])

    def test_prefilter_rejects_mixed_case_split_column(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "record_uid": ["a"],
                "text_bahnar": ["x"],
                "duration_seconds": [1.0],
                "split": ["Test"],
                "parquet_file": ["hf://x/train-000.parquet"],
            }
        )
        with pytest.raises(RuntimeError, match="Forbidden split"):
            prefilter_clean_split(df, split="train")


class TestDiskBudgetPeak:
    def test_peak_does_not_collapse_to_max(self):
        peak = plan_local_checkpoint_disk_peak(
            existing_checkpoint_bytes=8_000_000_000,
            new_checkpoint_bytes=4_000_000_000,
            hydrated_wav_bytes=1_000_000_000,
        )
        assert peak["checkpoint_peak_bytes"] == 12_000_000_000
        assert peak["total_peak_bytes"] == 13_000_000_000

    def test_f3_preflight_before_trainer_train(self):
        f3 = _nb_cells()["F3"]
        assert "full_train before trainer.train" in f3
        assert f3.find("full_train before trainer.train") < f3.find("trainer.train(")
        assert "wav_hydrate" not in f3 or "hydrated_wav" in f3
        # Must not double-count both hydrate keys at once.
        assert ' "wav_hydrate":' not in f3
