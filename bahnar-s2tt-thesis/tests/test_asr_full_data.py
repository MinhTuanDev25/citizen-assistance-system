"""Tests for full-prepare data pipeline (Notebook 03 FULL_STAGE=prepare)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.asr_full_data import (
    ELIGIBLE_TRAIN_CSV,
    EXCLUSIONS_CSV,
    STATUS_FULL_PREPARE,
    assert_eligible_frames_no_frozen_splits,
    check_split_overlaps,
    filter_by_duration,
    filter_nonempty_normalized_text,
    finalize_full_prepare,
    load_prepare_success,
    prefilter_clean_split,
)
from src.asr_full_train import build_data_contract
from src.data_utils import normalize_bahnar_ctc_v1, safe_cache_filename


def _contract(**overrides):
    base = dict(
        dataset_id="ds", dataset_revision="rev", parquet_revision="a" * 40,
        train_manifest_content_hash="th", validation_manifest_content_hash="vh",
        vocab_fp="vfp", processing_version="notebook02_audio_v3",
        min_duration=0.5, max_duration=30.0, target_sr=16000,
        pretrained_model_id="facebook/wav2vec2-xls-r-300m",
        pretrained_model_revision="mrev",
    )
    base.update(overrides)
    return build_data_contract(**base)


def _row(uid, rid, pq, srow, dur, text, gid="g1", split="train"):
    return {
        "record_uid": uid,
        "record_id": rid,
        "group_id": gid,
        "recording_group_id": f"rg-{gid}",
        "pair_key": f"pk-{uid}",
        "source_split": "train",
        "split": split,
        "parquet_file": pq,
        "shard_row_index": srow,
        "text_bahnar": text,
        "duration_seconds": dur,
    }


def _decode_ok(audio):
    return {
        "ok": True,
        "reason": "ok",
        "duration_sec": 1.0,
        "sampling_rate": 16000,
        "waveform": np.zeros(16000, dtype=np.float32),
    }


def _vocab():
    return {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4, "t": 5, "e": 6, "x": 7}


class TestDurationAndTextFilters:
    def test_duration_filter_inclusive_bounds(self):
        df = pd.DataFrame([
            _row("a", "1", "s0.parquet", 0, 0.49, "ok"),
            _row("b", "2", "s0.parquet", 1, 0.5, "ok"),
            _row("c", "3", "s0.parquet", 2, 30.0, "ok"),
            _row("d", "4", "s0.parquet", 3, 30.01, "ok"),
        ])
        kept, excl = filter_by_duration(df, min_duration=0.5, max_duration=30.0)
        assert set(kept["record_uid"]) == {"b", "c"}
        assert set(excl["record_uid"]) == {"a", "d"}
        # Default gate is MAX=40: 30.01s is kept, 0.49s still excluded.
        kept40, excl40 = filter_by_duration(df)
        assert set(kept40["record_uid"]) == {"b", "c", "d"}
        assert set(excl40["record_uid"]) == {"a"}

    def test_empty_normalized_excluded(self):
        df = pd.DataFrame([
            _row("a", "1", "s0.parquet", 0, 1.0, "Xin chào"),
            _row("b", "2", "s0.parquet", 1, 1.0, "   "),
            _row("c", "3", "s0.parquet", 2, 1.0, "!!!"),
        ])
        kept, excl = filter_nonempty_normalized_text(df, normalize_fn=normalize_bahnar_ctc_v1)
        assert "a" in set(kept["record_uid"])
        assert "b" in set(excl["record_uid"])
        assert "c" in set(excl["record_uid"])

    def test_prefilter_order_deterministic(self):
        rows = [
            _row(f"u{i}", str(i), "a.parquet", i, 1.0 + i, f"text {i}", gid=f"g{i%2}")
            for i in range(5)
        ]
        rows[0]["duration_seconds"] = 100.0
        rows[1]["text_bahnar"] = "   "
        df = pd.DataFrame(rows)
        r1 = prefilter_clean_split(df, split="train")
        r2 = prefilter_clean_split(df, split="train")
        assert list(r1["candidates"]["record_uid"]) == list(r2["candidates"]["record_uid"])
        assert len(r1["exclusions"]) == len(r2["exclusions"]) == 2


class TestRowAccountingAndFinalize:
    def test_assert_row_accounting(self):
        from src.asr_full_data import assert_row_accounting
        assert_row_accounting(clean_count=10, eligible_count=7, exclusion_count=3, split="train")
        with pytest.raises(RuntimeError, match="Row accounting failed"):
            assert_row_accounting(clean_count=10, eligible_count=7, exclusion_count=2, split="train")

    def test_overlap_detected(self):
        train = pd.DataFrame([_row("u1", "1", "a.parquet", 0, 1.0, "ab", gid="gX")])
        val = pd.DataFrame([_row("u2", "2", "b.parquet", 0, 1.0, "ab", gid="gX")])
        ov = check_split_overlaps(train, val)
        assert ov["group_id"] == 1
        assert ov["record_uid"] == 0

    def test_missing_overlap_column_fails_no_pad(self):
        train = pd.DataFrame([{"record_uid": "u1", "group_id": "g1"}])
        val = pd.DataFrame([{"record_uid": "u2", "group_id": "g2"}])
        with pytest.raises(RuntimeError, match="missing required columns"):
            check_split_overlaps(train, val)

    def _result(self, df, split, shards, accounting_ok=True, clean_count=None):
        from src.asr_full_data import FullPrepareState
        empty_excl = pd.DataFrame(columns=[
            "split", "record_uid", "record_id", "group_id", "parquet_file", "shard_row_index", "reason", "detail"
        ])
        n = len(df)
        return {
            "eligible_df": df,
            "exclusions_df": empty_excl,
            "state": FullPrepareState(
                "ds", "rev", split,
                completed_shards=shards, pending_shards=[], finished=True,
                n_eligible=n, n_excluded=0,
            ),
            "all_shards_done": True,
            "accounting_ok": accounting_ok,
            "clean_count": n if clean_count is None else clean_count,
            "manifest_uid_hash": "hash",
            "vocab_fp": "vfp",
        }

    def test_finalize_success(self, tmp_path: Path):
        train_df = pd.DataFrame([_row("u1", "1", "a.parquet", 0, 1.0, "ab", gid="g1")])
        val_df = pd.DataFrame([_row("u2", "2", "b.parquet", 0, 1.0, "ab", gid="g2")])
        for df in (train_df, val_df):
            df["text_bahnar_norm"] = "ab"
            df["processed_duration_seconds"] = 1.0
            df["local_cache_relpath"] = "x.wav"
            df["audio_source"] = "prepared_local"
        summary = finalize_full_prepare(
            state_dir=tmp_path / "full_state",
            train_result=self._result(train_df, "train", ["a.parquet"]),
            val_result=self._result(val_df, "validation", ["b.parquet"]),
            dataset_id="ds", dataset_revision="rev",
            data_contract=_contract(),
        )
        assert summary["status"] == STATUS_FULL_PREPARE
        assert load_prepare_success(tmp_path / "full_state", expected_contract=_contract())
        # A summary from another dataset must not satisfy this run.
        assert not load_prepare_success(
            tmp_path / "full_state", expected_contract=_contract(dataset_id="other")
        )

    def test_finalize_rejects_frozen_eligible(self, tmp_path: Path):
        train_df = pd.DataFrame([_row("u1", "1", "a.parquet", 0, 1.0, "ab", gid="g1")])
        val_df = pd.DataFrame([_row("u2", "2", "b.parquet", 0, 1.0, "ab", gid="g2")])
        train_df["source_split"] = "rq1_test"
        for df in (train_df, val_df):
            df["text_bahnar_norm"] = "ab"
            df["processed_duration_seconds"] = 1.0
            df["local_cache_relpath"] = "x.wav"
            df["audio_source"] = "prepared_local"
        summary = finalize_full_prepare(
            state_dir=tmp_path / "st",
            train_result=self._result(train_df, "train", ["a.parquet"]),
            val_result=self._result(val_df, "validation", ["b.parquet"]),
            data_contract=_contract(),
        )
        assert summary["status"] != STATUS_FULL_PREPARE
        assert summary["frozen_eligible_ok"] is False

    def test_assert_eligible_frozen_helper(self):
        df = pd.DataFrame([_row("u1", "1", "a.parquet", 0, 1.0, "ab")])
        df["split"] = "frozen_test"
        with pytest.raises(RuntimeError, match="Forbidden split"):
            assert_eligible_frames_no_frozen_splits(df)

    def test_frozen_split_rejected_prefilter(self):
        df = pd.DataFrame([_row("u1", "1", "a.parquet", 0, 1.0, "ab")])
        df["source_split"] = "test"
        with pytest.raises(RuntimeError, match="Forbidden split"):
            prefilter_clean_split(df, split="train")

    def test_forbidden_parquet_path_checked(self):
        df = pd.DataFrame([_row("u1", "1", "data/rq1_test/shard.parquet", 0, 1.0, "ab")])
        with pytest.raises(RuntimeError, match="Frozen-test path"):
            prefilter_clean_split(df, split="train")
