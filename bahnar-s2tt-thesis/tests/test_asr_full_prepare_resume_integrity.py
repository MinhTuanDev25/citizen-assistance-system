"""
Resume integrity of the metadata-only prepare pass.

Prepare for the full run spans several Colab sessions, so everything it commits
has to be provable on the next session rather than merely present. These tests
cover the two ways that used to fail silently or crash: a split that legitimately
has zero eligible rows, and durable artefacts whose bytes no longer match what
the state committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.asr_full_shards import (
    SHARD_STATE_JSON,
    load_shard_state,
    partial_csv_digest,
    shard_index_paths,
    verify_shard_sidecar_index,
)
from tests.test_asr_full_durability_behavior import (
    PQ_REV,
    REPO,
    SHARD_A,
    SHARD_B,
    FakeShardSource,
    _audio_bytes,
    _prepare,
    _row,
    _two_shard_source,
)


def _state(tmp_path: Path):
    return load_shard_state(tmp_path / "state" / SHARD_STATE_JSON)


def _partial(tmp_path: Path, split: str, which: str = "eligible") -> Path:
    return tmp_path / "state" / f"prepare_{split}" / f"{which}_partial.csv"


class TestZeroEligibleSplit:
    """A split can legitimately end a shard with no eligible rows at all."""

    @staticmethod
    def _splits_with_all_validation_excluded():
        # The validation row's parquet payload carries a different record_id, so
        # it is excluded rather than made eligible.
        train = pd.DataFrame([_row("u0", "id0", SHARD_A, 0)])
        val = pd.DataFrame([_row("u1", "MISMATCH", SHARD_A, 1, split="validation")])
        return {"train": train, "validation": val}

    def test_zero_row_partial_keeps_its_schema(self, tmp_path: Path):
        res = _prepare(
            tmp_path, _two_shard_source(),
            splits=self._splits_with_all_validation_excluded(),
        )
        assert len(res["validation"]["eligible_df"]) == 0

        path = _partial(tmp_path, "validation")
        # The old code wrote a column-less frame here, leaving b"\n" on disk.
        assert path.read_bytes().strip(), "zero-row partial must still carry a header"
        reloaded = pd.read_csv(path)  # would raise EmptyDataError before the fix
        assert len(reloaded) == 0
        assert "record_uid" in reloaded.columns

    def test_resume_after_reset_does_not_crash_and_skips_shards(self, tmp_path: Path):
        splits = self._splits_with_all_validation_excluded()
        source = _two_shard_source()
        _prepare(tmp_path, source, splits=splits)

        source.opens.clear()
        res = _prepare(tmp_path, source, splits=splits)

        assert source.opens == []  # completed shards verified, nothing re-read
        assert len(res["validation"]["eligible_df"]) == 0
        assert res["validation"]["accounting_ok"] is True
        assert res["train"]["accounting_ok"] is True


class TestPartialCsvIntegrity:
    def test_digest_is_committed_with_the_shard(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        state = _state(tmp_path)

        recorded = state.splits["train"]["partials"]
        assert set(recorded) == {"eligible", "exclusions", "commit_seq"}
        assert recorded["commit_seq"] == state.commit_seq
        for which in ("eligible", "exclusions"):
            on_disk = partial_csv_digest(_partial(tmp_path, "train", which))
            assert recorded[which]["sha256"] == on_disk["sha256"]
            assert recorded[which]["n_rows"] == on_disk["n_rows"]
            assert recorded[which]["uid_hash"] == on_disk["uid_hash"]

    def test_edited_transcript_with_same_uids_is_not_trusted(self, tmp_path: Path):
        """UID accounting alone cannot see this; the byte digest must."""
        source = _two_shard_source()
        _prepare(tmp_path, source)

        path = _partial(tmp_path, "train")
        frame = pd.read_csv(path)
        frame.loc[0, "text_bahnar_norm"] = "tampered text"
        frame.to_csv(path, index=False)  # same UIDs, different content

        source.opens.clear()
        res = _prepare(tmp_path, source)

        # Every shard is re-read: no row of a tampered partial may survive.
        assert sorted(source.opens) == [SHARD_A, SHARD_B]
        rebuilt = res["train"]["eligible_df"]
        assert "tampered text" not in set(rebuilt["text_bahnar_norm"])
        assert res["train"]["accounting_ok"] is True

    def test_truncated_partial_forces_restart(self, tmp_path: Path):
        source = _two_shard_source()
        _prepare(tmp_path, source)
        _partial(tmp_path, "train").write_text("record_uid\n", encoding="utf-8")

        source.opens.clear()
        res = _prepare(tmp_path, source)

        assert sorted(source.opens) == [SHARD_A, SHARD_B]
        assert len(res["train"]["eligible_df"]) == 2

    def test_partials_are_committed_atomically(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        leftovers = list((tmp_path / "state").rglob("*.tmp"))
        assert leftovers == []


class TestCompletedShardVerification:
    def test_corrupt_sidecar_reprocesses_only_that_shard(self, tmp_path: Path):
        source = _two_shard_source()
        _prepare(tmp_path, source)

        index_path, _ = shard_index_paths(tmp_path / "state", SHARD_B)
        blob = index_path.read_bytes()
        index_path.write_bytes(blob[: len(blob) // 2])  # interrupted Drive upload

        source.opens.clear()
        res = _prepare(tmp_path, source)

        assert source.opens == [SHARD_B]
        assert res["train"]["accounting_ok"] is True
        assert res["validation"]["accounting_ok"] is True
        assert len(res["train"]["eligible_df"]) == 2

    def test_sidecar_meta_hash_mismatch_is_detected(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        _, meta_path = shard_index_paths(tmp_path / "state", SHARD_A)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["sha256_index"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(RuntimeError, match="hash mismatch"):
            verify_shard_sidecar_index(tmp_path / "state", SHARD_A)

    def test_sidecar_wrong_dataset_revision_is_rejected(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        with pytest.raises(RuntimeError, match="dataset_revision"):
            verify_shard_sidecar_index(
                tmp_path / "state", SHARD_A,
                expected_parquet_revision=PQ_REV,
                expected_dataset_revision="a-different-revision",
            )

    def test_sidecar_wrong_target_sr_is_rejected(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        with pytest.raises(RuntimeError, match="sample_rate"):
            verify_shard_sidecar_index(
                tmp_path / "state", SHARD_A, expected_target_sr=8000,
            )

    def test_stale_sidecar_from_another_parquet_revision_is_rejected(self, tmp_path: Path):
        source = _two_shard_source()
        _prepare(tmp_path, source)

        _, meta_path = shard_index_paths(tmp_path / "state", SHARD_B)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["parquet_revision"] = "b" * 40
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        source.opens.clear()
        res = _prepare(tmp_path, source)

        assert source.opens == [SHARD_B]
        assert res["train"]["accounting_ok"] is True


class TestMeasuredShardBudget:
    """Prepare budgets disk from measured shard sizes, not from a hint."""

    def test_measured_sizes_are_accepted(self, tmp_path: Path):
        source = _two_shard_source()
        res = _prepare(
            tmp_path, source,
            shard_bytes={SHARD_A: 4096, SHARD_B: 8192},
        )
        assert res["train"]["all_shards_done"] is True
        assert sorted(source.opens) == [SHARD_A, SHARD_B]

    def test_a_shard_without_a_measured_size_is_refused(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="No measured parquet size"):
            _prepare(tmp_path, _two_shard_source(), shard_bytes={SHARD_A: 4096})


class TestSingleShardPass:
    def test_each_physical_shard_is_read_once_across_both_splits(self, tmp_path: Path):
        source = FakeShardSource({
            SHARD_A: {
                0: {"id": "id0", "audio": _audio_bytes(0)},
                1: {"id": "id1", "audio": _audio_bytes(1)},
            },
        })
        splits = {
            "train": pd.DataFrame([_row("u0", "id0", SHARD_A, 0)]),
            "validation": pd.DataFrame([
                _row("u1", "id1", SHARD_A, 1, split="validation")
            ]),
        }
        _prepare(tmp_path, source, splits=splits)
        assert source.opens == [SHARD_A]
        assert source.opens.count(SHARD_A) == 1
