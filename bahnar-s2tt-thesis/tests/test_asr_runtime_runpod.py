"""
RunPod-only runtime roots, pair_key train exclusion, MAX=40 contract isolation.

Also guards that Notebook 03 / related src no longer contain Colab or Google Drive
symbols, paths, or legacy ``*_to_drive`` APIs.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from src.asr_full_data import (
    PAIR_KEY_TRAIN_EXCLUSION_REASON,
    drop_train_pair_key_overlaps,
    estimate_hydrate_wav_bytes,
    write_effective_manifests,
)
from src.asr_full_pcm import WAV_HEADER_BYTES, wav_bytes_for_samples
from src.asr_full_train import (
    DATA_CONTRACT_KEYS,
    OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
    build_data_contract,
    plan_expected_resume_position,
)
from src.asr_runtime_paths import (
    DEFAULT_DURABLE_ROOT,
    DEFAULT_LOCAL_ROOT,
    ENV_DURABLE_ROOT,
    ENV_LOCAL_ROOT,
    assert_no_colab_drive_markers,
    find_colab_drive_markers,
    resolve_runtime_paths,
)

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "03_asr_baseline_training.ipynb"
SRC_ROOT = REPO / "src"

# Modules that Notebook 03 imports for the full pipeline.
NB03_SRC_MODULES = (
    "asr_runtime_paths.py",
    "asr_full_data.py",
    "asr_full_pcm.py",
    "asr_full_shards.py",
    "asr_full_train.py",
    "asr_utils.py",
    "seed.py",
)


def _nb_code() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )


class TestRunPodRoots:
    def test_defaults_are_runpod_paths(self):
        assert DEFAULT_LOCAL_ROOT == Path("/tmp/bahnar-runtime")
        assert DEFAULT_DURABLE_ROOT == Path("/workspace/bahnar-s2tt-thesis")

    def test_resolve_uses_defaults_without_env(self, tmp_path: Path):
        paths = resolve_runtime_paths(project_root=tmp_path, env={})
        assert paths.local_root == DEFAULT_LOCAL_ROOT
        assert paths.durable_root == DEFAULT_DURABLE_ROOT
        assert paths.profile == "runpod"

    def test_env_overrides_defaults(self, tmp_path: Path):
        local = tmp_path / "loc"
        durable = tmp_path / "dur"
        paths = resolve_runtime_paths(
            project_root=tmp_path,
            env={ENV_LOCAL_ROOT: str(local), ENV_DURABLE_ROOT: str(durable)},
        )
        assert paths.local_root == local
        assert paths.durable_root == durable
        assert paths.hf_parquet_cache_dir == local / "hf_parquet_cache"
        assert paths.export_root == durable / "exports" / "notebook03_runs"

    def test_contract_scoped_state_dir_isolates_hashes(self, tmp_path: Path):
        paths = resolve_runtime_paths(
            project_root=tmp_path,
            local_root=tmp_path / "L",
            durable_root=tmp_path / "D",
        )
        a = paths.prepare_state_dir("a" * 64)
        b = paths.prepare_state_dir("b" * 64)
        assert a != b
        assert a.parent == paths.durable_state_root

    def test_no_colab_profile_api(self):
        import src.asr_runtime_paths as m

        assert not hasattr(m, "detect_runtime_profile")
        assert not hasattr(m, "default_roots_for_profile")
        assert "colab" not in m.resolve_runtime_paths.__doc__.lower()


class TestNoColabDriveResidue:
    def test_notebook03_code_cells_have_no_colab_drive_markers(self):
        code = _nb_code()
        assert_no_colab_drive_markers(code, label="notebook03")
        assert "resolve_runtime_paths" in code
        assert "MAX_AUDIO_DURATION = 40.0" in code

    def test_nb03_src_modules_have_no_colab_drive_markers(self):
        # Precise allowlisting of the marker-definition node lives in
        # test_notebook03_anti_colab_drive.py; here we still scan every module
        # except allowing the definition assignment range in asr_runtime_paths.
        from tests.test_notebook03_anti_colab_drive import _production_hits

        for name in NB03_SRC_MODULES:
            hits = _production_hits(SRC_ROOT / name)
            assert not hits, f"{name} still has {hits}"

    def test_notebook_does_not_import_deleted_drive_symbols(self):
        code = _nb_code()
        forbidden = {
            "sync_experiment_checkpoints_to_drive",
            "restore_experiment_checkpoints_from_drive",
            "resolve_best_checkpoint_from_drive",
            "make_drive_checkpoint_sync_callback",
            "drive_experiment_dir",
            "DRIVE_ROOT",
            "DRIVE_MOUNT_POINT",
        }
        for name in forbidden:
            assert name not in code, f"notebook still references deleted symbol {name}"
        # Durable APIs must be present instead.
        assert "sync_experiment_checkpoints_to_durable" in code
        assert "restore_experiment_checkpoints_from_durable" in code


class TestPairKeyTrainExclusion:
    def _frames(self):
        train = pd.DataFrame(
            {
                "record_uid": ["t1", "t2", "t3"],
                "record_id": ["a", "b", "c"],
                "group_id": ["g1", "g2", "g3"],
                "recording_group_id": ["r1", "r2", "r3"],
                "source_split": ["train"] * 3,
                "split": ["train"] * 3,
                "parquet_file": ["p"] * 3,
                "shard_row_index": [0, 1, 2],
                "pair_key": ["same", "unique_train", "same"],
                "text_bahnar": ["x", "y", "x"],
                "duration_seconds": [1.0, 2.0, 1.0],
            }
        )
        val = pd.DataFrame(
            {
                "record_uid": ["v1", "v2"],
                "record_id": ["d", "e"],
                "group_id": ["g4", "g5"],
                "recording_group_id": ["r4", "r5"],
                "source_split": ["train"] * 2,
                "split": ["validation"] * 2,
                "parquet_file": ["p"] * 2,
                "shard_row_index": [3, 4],
                "pair_key": ["same", "unique_val"],
                "text_bahnar": ["x", "z"],
                "duration_seconds": [1.0, 2.0],
            }
        )
        return train, val

    def test_drops_only_train_rows_with_validation_pair_key(self):
        train, val = self._frames()
        kept, excl, report = drop_train_pair_key_overlaps(train, val)
        assert list(kept["record_uid"]) == ["t2"]
        assert set(excl["record_uid"]) == {"t1", "t3"}
        assert set(excl["reason"]) == {PAIR_KEY_TRAIN_EXCLUSION_REASON}
        assert report["train_rows_dropped"] == 2
        assert report["overlapping_pair_keys"] == 1

    def test_validation_frame_untouched_by_writer(self, tmp_path: Path):
        train, val = self._frames()
        kept, excl, _ = drop_train_pair_key_overlaps(train, val)
        summary = write_effective_manifests(
            state_dir=tmp_path,
            train_df=kept,
            val_df=val,
            train_exclusions=excl,
        )
        written_val = pd.read_csv(tmp_path / "effective_rq1_validation.csv")
        assert list(written_val["record_uid"]) == list(val["record_uid"])
        assert summary["validation_count"] == 2
        assert summary["exclusion_count"] == 2

    def test_count_is_data_driven_not_hardcoded(self):
        train, val = self._frames()
        _, excl, report = drop_train_pair_key_overlaps(train, val)
        assert report["train_rows_dropped"] == len(excl)


class TestMax40ContractIsolation:
    def _contract(self, **kwargs):
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

    def test_overlap_policy_and_duration_in_contract_keys(self):
        assert "overlap_policy" in DATA_CONTRACT_KEYS
        assert "max_duration" in DATA_CONTRACT_KEYS
        assert "audio_pcm_pipeline_version" in DATA_CONTRACT_KEYS

    def test_max30_and_max40_have_different_hashes(self):
        c30 = self._contract(max_duration=30.0)
        c40 = self._contract(max_duration=40.0)
        assert c30["contract_hash"] != c40["contract_hash"]

    def test_notebook_max_duration_is_40(self):
        assert "MAX_AUDIO_DURATION = 40.0" in _nb_code()
        assert "MAX_AUDIO_DURATION = 30.0" not in _nb_code()

    def test_state_dirs_for_max30_and_max40_differ(self, tmp_path: Path):
        paths = resolve_runtime_paths(
            project_root=tmp_path,
            local_root=tmp_path / "L",
            durable_root=tmp_path / "D",
        )
        d30 = paths.prepare_state_dir(self._contract(max_duration=30.0)["contract_hash"])
        d40 = paths.prepare_state_dir(self._contract(max_duration=40.0)["contract_hash"])
        assert d30 != d40


class TestHydrateDiskEstimate:
    def test_sums_n_samples_exactly(self):
        df = pd.DataFrame({"n_samples": [100, 250]})
        assert estimate_hydrate_wav_bytes(df) == (
            wav_bytes_for_samples(100) + wav_bytes_for_samples(250)
        )
        assert estimate_hydrate_wav_bytes(df) == (
            100 * 2 + WAV_HEADER_BYTES + 250 * 2 + WAV_HEADER_BYTES
        )

    def test_missing_n_samples_fails_closed(self):
        with pytest.raises(RuntimeError, match="n_samples"):
            estimate_hydrate_wav_bytes(pd.DataFrame({"record_uid": ["a"]}))


class TestResumePositionWrap:
    def test_never_returns_none_uid_on_long_phase_a(self):
        uids = [f"u{i}" for i in range(64)]
        plan = plan_expected_resume_position(
            uids,
            resume_step=100,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )
        assert plan["expected_first_uid"] == "u32"
