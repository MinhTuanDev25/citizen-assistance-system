"""
Behavioural durability tests for full training.

Covers the scenarios that only show up after a session reset: a session reset wiping local
audio, a shard interrupted mid-write, a manifest edited in place, an interrupted
Drive sync, orphan/mixed checkpoints and a two-process resume.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.asr_full_data import (
    AUDIO_PCM_PIPELINE_VERSION,
    assert_sidecar_required_fields,
    assert_uid_set_accounting,
    compute_manifest_content_hash,
    finalize_full_prepare,
    hydrate_rows_audio,
    load_prepare_success,
    local_audio_ok,
    make_hf_parquet_stream_reader,
    hydrate_union_audio,
    plan_shard_hydrate,
    run_full_prepare_streaming,
    shard_index_paths,
    verify_eligible_audio_with_index,
    verify_shard_sidecar_index,
)
from src.asr_full_pcm import canonical_pcm16_from_bytes, write_wav_from_pcm16
from src.asr_full_shards import build_shard_plans
from src.asr_full_train import (
    FULL_TRAIN_MARKER,
    assert_checkpoint_allowed_for_full_train,
    assert_cross_session_resume,
    assert_ready_for_full_evaluate,
    build_data_contract,
    build_evaluate_contract,
    build_train_contract,
    derive_full_evaluate_status,
    derive_resume_test_status_from_proof,
    durable_experiment_dir,
    ensure_experiment_fingerprint,
    experiment_checkpoint_dir,
    metrics_are_finite,
    new_session_token,
    resolve_best_checkpoint_from_durable,
    restore_experiment_checkpoints_from_durable,
    summarize_resume_proof,
    sync_experiment_checkpoints_to_durable,
    write_checkpoint_fingerprint,
    write_full_train_summary,
)
from src.data_utils import safe_cache_filename

REPO = "cuong06/Bahnar_Vietnamese"
PQ_REV = "a" * 40
SHARD_A = "default/train/0000.parquet"
SHARD_B = "default/train/0001.parquet"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _vocab():
    return {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4}


def _audio_bytes(seed: int, seconds: float = 1.0, sr: int = 16000, subtype="PCM_16"):
    rng = np.random.default_rng(seed)
    wav = 0.2 * rng.standard_normal(int(seconds * sr))
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="FLAC", subtype=subtype)
    return buf.getvalue()


def _row(uid, rid, shard, srow, split="train", text="ab ba", dur=1.0, gid=None):
    gid = gid or f"g-{uid}"
    return {
        "record_uid": uid, "record_id": rid, "group_id": gid,
        "recording_group_id": f"rg-{gid}", "pair_key": f"pk-{uid}",
        "source_split": split, "split": split,
        "parquet_file": f"https://huggingface.co/datasets/{REPO}/resolve/"
                        f"refs%2Fconvert%2Fparquet/{shard}",
        "shard_row_index": srow, "text_bahnar": text, "duration_seconds": dur,
    }


class FakeShardSource:
    """In-memory stand-in for the parquet shards, counting physical opens."""

    def __init__(self, shards):
        self.shards = shards  # {shard_filename: {row_idx: payload}}
        self.opens = []

    def reader(self):
        def _reader(ref, needed):
            self.opens.append(ref.filename)
            rows = self.shards[ref.filename]
            for idx in sorted(int(i) for i in needed):
                if idx in rows:
                    yield idx, rows[idx]
        return _reader


def _two_shard_source():
    return FakeShardSource({
        SHARD_A: {
            0: {"id": "id0", "audio": _audio_bytes(0)},
            1: {"id": "id1", "audio": _audio_bytes(1)},
        },
        SHARD_B: {
            0: {"id": "id2", "audio": _audio_bytes(2)},
            1: {"id": "id3", "audio": _audio_bytes(3, subtype="PCM_24")},
        },
    })


def _splits():
    train = pd.DataFrame([
        _row("u0", "id0", SHARD_A, 0),
        _row("u2", "id2", SHARD_B, 0),
    ])
    val = pd.DataFrame([
        _row("u1", "id1", SHARD_A, 1, split="validation"),
        _row("u3", "id3", SHARD_B, 1, split="validation"),
    ])
    return {"train": train, "validation": val}


def _prepare(tmp_path, source, *, resume=True, splits=None, state_dir=None, audio_dir=None,
             write_audio=False, **kwargs):
    return run_full_prepare_streaming(
        **kwargs,
        splits=splits or _splits(),
        state_dir=state_dir or (tmp_path / "state"),
        vocab=_vocab(),
        dataset_id=REPO,
        dataset_revision="dsrev",
        parquet_revision=PQ_REV,
        shard_reader=source.reader(),
        local_audio_cache_dir=audio_dir or (tmp_path / "audio"),
        resume=resume,
        shard_bytes_hint=1024,
        write_audio=write_audio,
    )


def _prepare_contract(**overrides):
    from src.asr_full_train import build_data_contract

    base = dict(
        dataset_id=REPO, dataset_revision="dsrev", parquet_revision=PQ_REV,
        train_manifest_content_hash="th", validation_manifest_content_hash="vh",
        vocab_fp="vfp", processing_version="notebook02_audio_v3",
        min_duration=0.5, max_duration=30.0, target_sr=16000,
        pretrained_model_id="facebook/wav2vec2-xls-r-300m",
        pretrained_model_revision="mrev",
    )
    base.update(overrides)
    return build_data_contract(**base)


def _hydrate(tmp_path, source, frames, *, audio_dir=None):
    """Materialise local WAVs the way train/evaluate do, from the sidecar index."""
    return hydrate_rows_audio(
        pd.concat([f for f in frames if len(f)], ignore_index=True),
        state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
        shard_reader=source.reader(),
        local_audio_dir=audio_dir or (tmp_path / "audio"),
    )


def _complete_ckpt(path: Path, step: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(b"x")
    return path


def _data_contract(**kw):
    base = dict(
        dataset_id="ds", dataset_revision="rev", parquet_revision=PQ_REV,
        train_manifest_content_hash="th", validation_manifest_content_hash="vh",
        vocab_fp="vfp", processing_version="notebook02_audio_v3",
        min_duration=0.5, max_duration=30.0, target_sr=16000,
        pretrained_model_id="facebook/wav2vec2-xls-r-300m",
        pretrained_model_revision="mrev",
    )
    base.update(kw)
    return build_data_contract(**base)


# ---------------------------------------------------------------------------
# Streaming prepare
# ---------------------------------------------------------------------------

class TestSinglePassShardReading:
    def test_each_physical_shard_is_opened_once_for_both_splits(self, tmp_path: Path):
        """Train and validation share the same shards; a per-split loop reads twice."""
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        assert sorted(source.opens) == [SHARD_A, SHARD_B]
        assert len(res["train"]["eligible_df"]) == 2
        assert len(res["validation"]["eligible_df"]) == 2
        assert res["train"]["all_shards_done"] is True

    def test_prepare_is_metadata_only_and_writes_no_wav(self, tmp_path: Path):
        """Local WAVs die with the session, so prepare must not spend disk on them."""
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        frame = res["train"]["eligible_df"]
        assert {"sha256_pcm", "n_samples", "audio_pcm_pipeline_version"} <= set(frame.columns)
        assert set(frame["audio_pcm_pipeline_version"]) == {AUDIO_PCM_PIPELINE_VERSION}
        assert res["_shards"]["audio_written"] is False
        assert list((tmp_path / "audio").glob("*.wav")) == []
        # Metadata alone is not audio-ready; the stage that needs audio hydrates.
        with pytest.raises(RuntimeError, match="no local audio"):
            verify_eligible_audio_with_index(frame, [tmp_path / "audio"])
        _hydrate(tmp_path, source, [frame])
        verify_eligible_audio_with_index(frame, [tmp_path / "audio"])

    def test_non_finite_audio_row_is_excluded_not_written(self, tmp_path: Path):
        """A NaN row must land in exclusions, not quantise into fake silence."""
        buf = io.BytesIO()
        sf.write(buf, np.full(16000, np.nan), 16000, format="WAV", subtype="FLOAT")
        source = _two_shard_source()
        source.shards[SHARD_A][0] = {"id": "id0", "audio": buf.getvalue()}
        res = _prepare(tmp_path, source)
        assert list(res["train"]["eligible_df"]["record_uid"]) == ["u2"]
        excluded = res["train"]["exclusions_df"]
        reason = excluded.loc[excluded["record_uid"] == "u0", "reason"].iloc[0]
        assert reason == "audio_qa_non_finite_values"
        assert res["train"]["accounting_ok"] is True

    def test_resume_skips_completed_shards_without_reopening(self, tmp_path: Path):
        source = _two_shard_source()
        _prepare(tmp_path, source)
        source.opens.clear()
        res = _prepare(tmp_path, source)
        assert source.opens == []  # audio intact → no download, no re-QA
        assert len(res["train"]["eligible_df"]) == 2

    def test_missing_partial_csv_forces_reprocessing(self, tmp_path: Path):
        """State claiming completed shards without its partials must not lose rows."""
        source = _two_shard_source()
        _prepare(tmp_path, source)
        (tmp_path / "state" / "prepare_train" / "eligible_partial.csv").unlink()
        source.opens.clear()
        res = _prepare(tmp_path, source)
        assert sorted(source.opens) == [SHARD_A, SHARD_B]
        assert res["train"]["accounting_ok"] is True
        assert len(res["train"]["eligible_df"]) == 2

    def test_missing_shard_index_reprocesses_only_that_shard(self, tmp_path: Path):
        """A bad sidecar invalidates its own shard, not the whole prepare run."""
        source = _two_shard_source()
        _prepare(tmp_path, source)
        for path in shard_index_paths(tmp_path / "state", SHARD_B):
            path.unlink()
        source.opens.clear()
        res = _prepare(tmp_path, source)
        assert source.opens == [SHARD_B]
        assert res["train"]["all_shards_done"] is True
        assert res["train"]["accounting_ok"] is True
        assert len(res["train"]["eligible_df"]) == 2

    def test_record_id_mismatch_becomes_exclusion_not_a_crash(self, tmp_path: Path):
        source = _two_shard_source()
        source.shards[SHARD_A][0] = {"id": "WRONG", "audio": _audio_bytes(9)}
        res = _prepare(tmp_path, source)
        reasons = list(res["train"]["exclusions_df"]["reason"])
        assert "record_id_mismatch" in reasons
        assert len(res["train"]["eligible_df"]) == 1

    def test_missing_audio_row_is_accounted_as_exclusion(self, tmp_path: Path):
        source = _two_shard_source()
        del source.shards[SHARD_B][0]
        res = _prepare(tmp_path, source)
        assert "audio_row_missing_in_shard" in list(res["train"]["exclusions_df"]["reason"])
        # UID accounting still closes: nothing is silently dropped.
        assert res["train"]["accounting_ok"] is True

    def test_null_audio_payload_is_excluded(self, tmp_path: Path):
        source = _two_shard_source()
        source.shards[SHARD_A][0] = {"id": "id0", "audio": None}
        res = _prepare(tmp_path, source)
        assert any("null_audio" in r for r in res["train"]["exclusions_df"]["reason"])

    def test_shard_index_is_written_and_verified(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        index = verify_shard_sidecar_index(
            tmp_path / "state", SHARD_A, expected_parquet_revision=PQ_REV
        )
        assert set(index) == {"u0", "u1"}
        assert index["u0"]["sha256_source"] and index["u0"]["parquet_revision"] == PQ_REV

    def test_corrupt_shard_index_is_detected(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        index_path, _meta = shard_index_paths(tmp_path / "state", SHARD_A)
        index_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="corrupt"):
            verify_shard_sidecar_index(tmp_path / "state", SHARD_A)

    def test_shard_index_bound_to_parquet_revision(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        with pytest.raises(RuntimeError, match="parquet_revision"):
            verify_shard_sidecar_index(
                tmp_path / "state", SHARD_A, expected_parquet_revision="f" * 40
            )

    def test_metadata_only_on_drive_no_bulk_audio(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        state_dir = tmp_path / "state"
        assert not list(state_dir.rglob("*.wav"))
        # One sidecar index per shard, not one JSON per record.
        assert len(list((state_dir / "audio_index").glob("*.jsonl"))) == 2

    def test_disk_headroom_blocks_per_shard_even_metadata_only(self, tmp_path: Path):
        """Metadata-only still downloads parquet, so headroom is re-checked per shard."""
        with pytest.raises(RuntimeError, match="Insufficient disk headroom"):
            run_full_prepare_streaming(
                splits=_splits(), state_dir=tmp_path / "state", vocab=_vocab(),
                dataset_id=REPO, dataset_revision="dsrev", parquet_revision=PQ_REV,
                shard_reader=_two_shard_source().reader(),
                local_audio_cache_dir=tmp_path / "audio",
                shard_bytes_hint=10 ** 18,
            )

    def test_cleanup_callback_runs_per_shard(self, tmp_path: Path):
        cleaned = []
        run_full_prepare_streaming(
            splits=_splits(), state_dir=tmp_path / "state", vocab=_vocab(),
            dataset_id=REPO, dataset_revision="dsrev", parquet_revision=PQ_REV,
            shard_reader=_two_shard_source().reader(),
            local_audio_cache_dir=tmp_path / "audio",
            shard_bytes_hint=1024,
            cleanup_shard=lambda ref: cleaned.append(ref.filename),
        )
        assert sorted(cleaned) == [SHARD_A, SHARD_B]


class TestManifestContentBinding:
    def test_same_uids_different_content_changes_hash(self):
        base = pd.DataFrame([_row("u0", "id0", SHARD_A, 0)])
        edited = base.copy()
        edited.loc[0, "text_bahnar"] = "different transcript"
        assert compute_manifest_content_hash(base) != compute_manifest_content_hash(edited)

    def test_row_order_does_not_change_hash(self):
        rows = [_row("u0", "id0", SHARD_A, 0), _row("u1", "id1", SHARD_A, 1)]
        assert compute_manifest_content_hash(pd.DataFrame(rows)) == compute_manifest_content_hash(
            pd.DataFrame(list(reversed(rows)))
        )

    def test_duration_edit_changes_hash(self):
        base = pd.DataFrame([_row("u0", "id0", SHARD_A, 0, dur=1.0)])
        edited = pd.DataFrame([_row("u0", "id0", SHARD_A, 0, dur=1.5)])
        assert compute_manifest_content_hash(base) != compute_manifest_content_hash(edited)

    def test_missing_required_column_fails_closed(self):
        frame = pd.DataFrame([{"record_uid": "u0"}])
        with pytest.raises(RuntimeError, match="required columns"):
            compute_manifest_content_hash(frame)

    def test_edited_manifest_invalidates_resume_state(self, tmp_path: Path):
        """Same UIDs with new transcripts must not reuse the previous prepare state."""
        source = _two_shard_source()
        splits = _splits()
        _prepare(tmp_path, source, splits=splits)
        source.opens.clear()
        edited = {k: v.copy() for k, v in splits.items()}
        edited["train"].loc[0, "text_bahnar"] = "ba ab"
        _prepare(tmp_path, source, splits=edited)
        assert sorted(source.opens) == [SHARD_A, SHARD_B]

    def test_prepare_success_is_bound_to_content_hash(self, tmp_path: Path):
        source = _two_shard_source()
        splits = _splits()
        res = _prepare(tmp_path, source, splits=splits)
        state_dir = tmp_path / "state"
        train_hash = compute_manifest_content_hash(splits["train"])
        val_hash = compute_manifest_content_hash(splits["validation"])
        contract = _prepare_contract(
            train_manifest_content_hash=train_hash,
            validation_manifest_content_hash=val_hash,
        )
        summary = finalize_full_prepare(
            state_dir=state_dir, train_result=res["train"], val_result=res["validation"],
            data_contract=contract,
            dataset_id=REPO, dataset_revision="dsrev", parquet_revision=PQ_REV,
        )
        assert summary["status"] == "SUCCESS_FULL_PREPARE"
        assert load_prepare_success(state_dir, expected_contract=contract)
        # Same UIDs, edited transcripts: the content hash moves, so prepare is stale.
        assert not load_prepare_success(
            state_dir,
            expected_contract=_prepare_contract(
                train_manifest_content_hash="stale",
                validation_manifest_content_hash=val_hash,
            ),
        )
        assert not load_prepare_success(
            state_dir,
            expected_contract=_prepare_contract(
                train_manifest_content_hash=train_hash,
                validation_manifest_content_hash=val_hash,
                parquet_revision="f" * 40,
            ),
        )
        # Every other contract field is binding too, not just the hashes.
        for field, value in (
            ("dataset_revision", "other-rev"),
            ("vocab_fp", "other-vocab"),
            ("target_sr", 8000),
            ("max_duration", 20.0),
            ("pretrained_model_revision", "other-model-rev"),
        ):
            assert not load_prepare_success(
                state_dir,
                expected_contract=_prepare_contract(
                    train_manifest_content_hash=train_hash,
                    validation_manifest_content_hash=val_hash,
                    **{field: value},
                ),
            ), field


class TestSessionResetWithExclusions:
    """A wiped session must rehydrate eligible rows only, without re-running QA."""

    def _mixed_source(self):
        source = _two_shard_source()
        # One row fails audio QA, so its UID belongs in exclusions, never hydrate.
        source.shards[SHARD_A][0] = {"id": "id0", "audio": b""}
        return source

    def test_hydrate_after_reset_skips_excluded_uids(self, tmp_path: Path):
        source = self._mixed_source()
        res = _prepare(tmp_path, source)
        eligible = res["train"]["eligible_df"]
        excluded = set(res["train"]["exclusions_df"]["record_uid"].astype(str))
        assert excluded, "fixture must produce at least one exclusion"
        assert not (set(eligible["record_uid"].astype(str)) & excluded)

        # Simulate a new session: nothing local survives.
        for wav in (tmp_path / "audio").glob("*.wav"):
            wav.unlink()
        source.opens.clear()

        report = hydrate_rows_audio(
            eligible,
            state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
            shard_reader=source.reader(), local_audio_dir=tmp_path / "audio",
        )
        assert report["hydrated"] == len(eligible)
        verify_eligible_audio_with_index(eligible, [tmp_path / "audio"])
        # No WAV exists for an excluded record.
        for uid in excluded:
            assert not (tmp_path / "audio" / safe_cache_filename(uid)).exists()

    def test_hydrating_an_excluded_uid_is_refused(self, tmp_path: Path):
        source = self._mixed_source()
        res = _prepare(tmp_path, source)
        excluded_uid = str(res["train"]["exclusions_df"]["record_uid"].iloc[0])
        with pytest.raises(RuntimeError):
            plan_shard_hydrate(
                tmp_path / "state", SHARD_A,
                local_audio_dir=tmp_path / "audio", uids=[excluded_uid],
            )

    def test_union_hydrate_reads_each_physical_shard_once(self, tmp_path: Path):
        """Train and validation share shards; per-split hydrate would double the download."""
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        train_df, val_df = res["train"]["eligible_df"], res["validation"]["eligible_df"]
        source.opens.clear()
        report = hydrate_union_audio(
            [train_df, val_df],
            state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
            shard_reader=source.reader(), local_audio_dir=tmp_path / "audio",
        )
        assert sorted(source.opens) == sorted(set(source.opens))
        unique = len(set(train_df["record_uid"]) | set(val_df["record_uid"]))
        assert report["unique_records"] == unique
        verify_eligible_audio_with_index(train_df, [tmp_path / "audio"])
        verify_eligible_audio_with_index(val_df, [tmp_path / "audio"])


class TestUidSetAccounting:
    def test_partition_is_exact(self):
        report = assert_uid_set_accounting(
            clean_uids=["a", "b", "c"], eligible_uids=["a", "b"],
            exclusion_uids=["c"], split="train",
        )
        assert report == {"split": "train", "clean": 3, "eligible": 2, "exclusions": 1}

    def test_overlap_rejected(self):
        with pytest.raises(RuntimeError, match="both eligible and exclusions"):
            assert_uid_set_accounting(
                clean_uids=["a", "b"], eligible_uids=["a", "b"],
                exclusion_uids=["b"], split="train",
            )

    def test_missing_uid_rejected(self):
        with pytest.raises(RuntimeError, match="unaccounted"):
            assert_uid_set_accounting(
                clean_uids=["a", "b"], eligible_uids=["a"], exclusion_uids=[], split="train",
            )

    def test_duplicate_cannot_cancel_a_missing_row(self):
        """Count-only accounting would pass here; set accounting must not."""
        with pytest.raises(RuntimeError, match="duplicate|unaccounted"):
            assert_uid_set_accounting(
                clean_uids=["a", "b"], eligible_uids=["a", "a"],
                exclusion_uids=[], split="train",
            )

    def test_extra_uid_rejected(self):
        with pytest.raises(RuntimeError, match="not in clean manifest"):
            assert_uid_set_accounting(
                clean_uids=["a"], eligible_uids=["a"], exclusion_uids=["ghost"], split="train",
            )


# ---------------------------------------------------------------------------
# Session reset / hydrate
# ---------------------------------------------------------------------------

class TestSessionResetHydrate:
    def test_wiped_local_audio_is_rehydrated_without_requalifying(self, tmp_path: Path):
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        audio_dir = tmp_path / "audio"
        for wav in audio_dir.glob("*.wav"):
            wav.unlink()  # simulate a new session
        source.opens.clear()

        report = hydrate_rows_audio(
            res["train"]["eligible_df"],
            state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
            shard_reader=source.reader(), local_audio_dir=audio_dir,
        )
        assert report["hydrated"] == 2
        assert sorted(source.opens) == [SHARD_A, SHARD_B]
        verify_eligible_audio_with_index(res["train"]["eligible_df"], [audio_dir])

    def test_hydrate_is_a_noop_when_audio_is_intact(self, tmp_path: Path):
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        _hydrate(tmp_path, source, [res["train"]["eligible_df"]])
        source.opens.clear()
        report = hydrate_rows_audio(
            res["train"]["eligible_df"],
            state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
            shard_reader=source.reader(), local_audio_dir=tmp_path / "audio",
        )
        assert report["hydrated"] == 0 and source.opens == []

    def test_corrupt_local_audio_is_replaced(self, tmp_path: Path):
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        _hydrate(tmp_path, source, [res["train"]["eligible_df"]])
        target = tmp_path / "audio" / safe_cache_filename("u0")
        target.write_bytes(b"not a wav")
        plan = plan_shard_hydrate(
            tmp_path / "state", SHARD_A,
            local_audio_dir=tmp_path / "audio", uids=["u0"],
        )
        assert plan["missing"] == ["u0"]
        hydrate_rows_audio(
            res["train"]["eligible_df"],
            state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
            shard_reader=source.reader(), local_audio_dir=tmp_path / "audio",
        )
        ok, reason = local_audio_ok(
            target,
            expected_sha256_pcm=str(
                res["train"]["eligible_df"].set_index("record_uid").loc["u0", "sha256_pcm"]
            ),
        )
        assert ok, reason

    def test_hydrate_fails_closed_when_upstream_bytes_changed(self, tmp_path: Path):
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        for wav in (tmp_path / "audio").glob("*.wav"):
            wav.unlink()
        source.shards[SHARD_A][0] = {"id": "id0", "audio": _audio_bytes(999)}
        with pytest.raises(RuntimeError, match="upstream source bytes changed"):
            hydrate_rows_audio(
                res["train"]["eligible_df"],
                state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
                shard_reader=source.reader(), local_audio_dir=tmp_path / "audio",
            )

    def test_hydrate_without_index_is_refused(self, tmp_path: Path):
        source = _two_shard_source()
        res = _prepare(tmp_path, source)
        for path in shard_index_paths(tmp_path / "state", SHARD_A):
            path.unlink()
        with pytest.raises(RuntimeError, match="index missing"):
            hydrate_rows_audio(
                res["train"]["eligible_df"],
                state_dir=tmp_path / "state", dataset_id=REPO, parquet_revision=PQ_REV,
                shard_reader=source.reader(), local_audio_dir=tmp_path / "audio",
            )

    def test_uid_absent_from_index_is_refused(self, tmp_path: Path):
        _prepare(tmp_path, _two_shard_source())
        with pytest.raises(RuntimeError, match="absent from shard index"):
            plan_shard_hydrate(
                tmp_path / "state", SHARD_A,
                local_audio_dir=tmp_path / "audio", uids=["ghost"],
            )

    def test_missing_audio_for_eligible_row_is_reported(self, tmp_path: Path):
        res = _prepare(tmp_path, _two_shard_source())
        for wav in (tmp_path / "audio").glob("*.wav"):
            wav.unlink()
        with pytest.raises(RuntimeError, match="no local audio"):
            verify_eligible_audio_with_index(
                res["train"]["eligible_df"], [tmp_path / "audio"]
            )

    def test_swapped_audio_file_is_detected(self, tmp_path: Path):
        res = _prepare(tmp_path, _two_shard_source())
        frame = res["train"]["eligible_df"].set_index("record_uid")
        audio_dir = tmp_path / "audio"
        other = canonical_pcm16_from_bytes(_audio_bytes(1234), target_sr=16000)
        write_wav_from_pcm16(audio_dir / str(frame.loc["u0", "local_cache_relpath"]),
                             other["pcm"], 16000)
        with pytest.raises(RuntimeError, match="PCM hash verification"):
            verify_eligible_audio_with_index(
                res["train"]["eligible_df"].head(1), [audio_dir]
            )


class TestSidecarRequirements:
    def test_missing_sha_rejected(self):
        with pytest.raises(RuntimeError, match="cache_sha256"):
            assert_sidecar_required_fields(
                {
                    "record_uid": "u", "dataset_revision": "r",
                    "audio_processing_version": "notebook02_audio_v3",
                    "target_sampling_rate": 16000, "cache_sha256": "deadbeef",
                },
                record_uid="u", dataset_revision="r",
                expected_processing_version="notebook02_audio_v3", target_sr=16000,
            )

    def test_missing_fields_rejected(self):
        with pytest.raises(RuntimeError, match="missing fields"):
            assert_sidecar_required_fields(
                {"record_uid": "u"}, record_uid="u", dataset_revision="r",
                expected_processing_version="v", target_sr=16000,
            )

    def test_wrong_dataset_revision_rejected(self):
        with pytest.raises(RuntimeError, match="dataset_revision"):
            assert_sidecar_required_fields(
                {
                    "record_uid": "u", "dataset_revision": "OTHER",
                    "audio_processing_version": "v",
                    "target_sampling_rate": 16000, "cache_sha256": "a" * 64,
                },
                record_uid="u", dataset_revision="r",
                expected_processing_version="v", target_sr=16000,
            )


class TestParquetStreamReader:
    def test_reader_streams_only_requested_rows(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        payloads = [_audio_bytes(i, seconds=0.2) for i in range(6)]
        table = pa.table({
            "id": [f"id{i}" for i in range(6)],
            "audio": payloads,
            "text_bahnar": ["ab"] * 6,
        })
        shard = tmp_path / "shard.parquet"
        pq.write_table(table, shard)

        calls = []

        def download_fn(ref, cache_dir):
            calls.append((ref.filename, str(cache_dir)))
            return shard

        reader = make_hf_parquet_stream_reader(
            cache_dir=tmp_path / "hf", batch_size=2, download_fn=download_fn,
        )
        plans = build_shard_plans(
            {"train": pd.DataFrame([_row("u0", "id0", SHARD_A, 0), _row("u4", "id4", SHARD_A, 4)])},
            dataset_id=REPO, parquet_revision=PQ_REV,
        )
        got = list(reader(plans[0].ref, plans[0].needed_indices))
        assert [idx for idx, _ in got] == [0, 4]
        assert got[0][1]["id"] == "id0"
        assert bytes(got[1][1]["audio"]) == payloads[4]
        assert len(calls) == 1 and calls[0][0] == SHARD_A

    def test_reader_requires_audio_column(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard = tmp_path / "noaudio.parquet"
        pq.write_table(pa.table({"id": ["id0"]}), shard)
        reader = make_hf_parquet_stream_reader(
            cache_dir=tmp_path / "hf", download_fn=lambda ref, cd: shard,
        )
        plans = build_shard_plans(
            {"train": pd.DataFrame([_row("u0", "id0", SHARD_A, 0)])},
            dataset_id=REPO, parquet_revision=PQ_REV,
        )
        with pytest.raises(RuntimeError, match="no 'audio' column"):
            list(reader(plans[0].ref, [0]))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpointSnapshotLKG:
    def test_interrupted_sync_keeps_last_known_good(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        _complete_ckpt(local / "checkpoint-100", 100)
        assert sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        ) is not None
        dest = durable_experiment_dir(drive, "expA", kind=FULL_TRAIN_MARKER)
        assert (dest / "LATEST").read_text(encoding="utf-8").strip() == "v1"

        # An interrupted next sync leaves debris but must not move LATEST or
        # delete the checkpoint the current LATEST references.
        (dest / "snapshots" / "v2.tmp_copy").mkdir(parents=True)
        (dest / "ckpts" / "checkpoint-200.tmp_copy").mkdir(parents=True)
        assert (dest / "LATEST").read_text(encoding="utf-8").strip() == "v1"
        assert (dest / "ckpts" / "checkpoint-100" / "optimizer.pt").is_file()

        restored = restore_experiment_checkpoints_from_durable(
            experiment_checkpoint_dir(tmp_path / "fresh", "expA", kind=FULL_TRAIN_MARKER),
            drive, experiment_id="expA", kind=FULL_TRAIN_MARKER,
        )
        assert (restored / "checkpoint-100" / "optimizer.pt").is_file()

    def test_partial_upload_is_replaced_not_trusted(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        _complete_ckpt(local / "checkpoint-100", 100)
        sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        # Incomplete orphan (not snapshot-referenced) must be replaced atomically.
        dest = durable_experiment_dir(drive, "expA", kind=FULL_TRAIN_MARKER)
        truncated = dest / "ckpts" / "checkpoint-200"
        truncated.mkdir(parents=True)
        (truncated / "trainer_state.json").write_text("{}", encoding="utf-8")
        _complete_ckpt(local / "checkpoint-200", 200)
        write_checkpoint_fingerprint(local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=200)
        sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        assert (truncated / "optimizer.pt").is_file()

    def test_sync_without_fingerprint_is_refused(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100)
        with pytest.raises(RuntimeError, match="without local fingerprint"):
            sync_experiment_checkpoints_to_durable(
                local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
            )

    def test_missing_drive_is_not_silently_skipped(self, tmp_path: Path):
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=1)
        _complete_ckpt(local / "checkpoint-1", 1)
        with pytest.raises(RuntimeError, match="Durable FULL_STATE_DIR missing"):
            sync_experiment_checkpoints_to_durable(
                local, tmp_path / "no_drive", experiment_id="expA", kind=FULL_TRAIN_MARKER
            )

    def test_restore_prefers_newer_drive_over_stale_local(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=200)
        _complete_ckpt(local / "checkpoint-100", 100)
        _complete_ckpt(local / "checkpoint-200", 200)
        sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        # A resumed session where only the older checkpoint survived locally.
        stale = experiment_checkpoint_dir(tmp_path / "stale", "expA", kind=FULL_TRAIN_MARKER)
        stale.mkdir(parents=True)
        _complete_ckpt(stale / "checkpoint-100", 100)
        write_checkpoint_fingerprint(stale, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        restored = restore_experiment_checkpoints_from_durable(
            stale, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        assert (restored / "checkpoint-200" / "optimizer.pt").is_file()

    def test_drive_fingerprint_mismatch_blocks_restore(self, tmp_path: Path):
        """A snapshot whose fingerprint drifted must not be restored silently."""
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        _complete_ckpt(local / "checkpoint-100", 100)
        snap = sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        fp_path = snap / "full_experiment_fingerprint.json"
        payload = json.loads(fp_path.read_text(encoding="utf-8"))
        payload["kind"] = "resume_test"
        fp_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            restore_experiment_checkpoints_from_durable(
                experiment_checkpoint_dir(tmp_path / "fresh", "expA", kind=FULL_TRAIN_MARKER),
                drive, experiment_id="expA", kind=FULL_TRAIN_MARKER,
            )

    def test_orphan_checkpoint_never_auto_fingerprinted(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        _complete_ckpt(root / "checkpoint-1", 1)
        with pytest.raises(RuntimeError, match="missing fingerprint"):
            ensure_experiment_fingerprint(
                root, experiment_id="expA", kind=FULL_TRAIN_MARKER, allow_create_if_empty=True
            )

    def test_wrong_experiment_root_rejected(self, tmp_path: Path):
        wrong = tmp_path / "other" / "expA" / "checkpoint-1"
        _complete_ckpt(wrong, 1)
        write_checkpoint_fingerprint(
            wrong.parent, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=1
        )
        with pytest.raises(RuntimeError, match="full_train"):
            assert_checkpoint_allowed_for_full_train(wrong, experiment_id="expA")

    def test_pilot_and_resume_test_checkpoints_rejected(self, tmp_path: Path):
        for kind in ("pilot", "resume_test"):
            root = experiment_checkpoint_dir(tmp_path / kind, "expA", kind=kind)
            ck = _complete_ckpt(root / "checkpoint-100", 100)
            write_checkpoint_fingerprint(
                root, experiment_id="expA", kind=kind, global_step=100
            )
            with pytest.raises(RuntimeError):
                assert_checkpoint_allowed_for_full_train(ck, experiment_id="expA")


class TestContractsAndEvaluate:
    def test_hparams_mismatch_rejected(self, tmp_path: Path):
        data = _data_contract()
        good = build_train_contract(
            experiment_id="expA", data_contract=data, hparams={"max_steps": 10, "seed": 42}
        )
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        ck = _complete_ckpt(root / "checkpoint-10", 10)
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=10, extra=good
        )
        bad = build_train_contract(
            experiment_id="expA", data_contract=data, hparams={"max_steps": 99, "seed": 42}
        )
        with pytest.raises(RuntimeError, match="contract mismatch"):
            assert_checkpoint_allowed_for_full_train(ck, experiment_id="expA", expected_contract=bad)

    def test_parquet_revision_drift_rejected(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        ck = _complete_ckpt(root / "checkpoint-10", 10)
        old = build_train_contract(
            experiment_id="expA", data_contract=_data_contract(), hparams={"max_steps": 10}
        )
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=10, extra=old
        )
        reconverted = build_train_contract(
            experiment_id="expA",
            data_contract=_data_contract(parquet_revision="e" * 40),
            hparams={"max_steps": 10},
        )
        with pytest.raises(RuntimeError, match="contract mismatch"):
            assert_checkpoint_allowed_for_full_train(
                ck, experiment_id="expA", expected_contract=reconverted
            )

    def test_stale_contract_hash_cannot_fake_a_match(self, tmp_path: Path):
        """Field comparison wins: a payload edited after hashing must not pass."""
        data = _data_contract()
        contract = build_train_contract(
            experiment_id="expA", data_contract=data, hparams={"max_steps": 10}
        )
        tampered = dict(contract)
        tampered["vocab_fp"] = "someone_elses_vocab"  # contract_hash left untouched
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        ck = _complete_ckpt(root / "checkpoint-10", 10)
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=10, extra=tampered
        )
        with pytest.raises(RuntimeError, match="contract mismatch"):
            assert_checkpoint_allowed_for_full_train(
                ck, experiment_id="expA", expected_contract=contract
            )

    def test_stale_validation_contract_blocks_evaluate(self, tmp_path: Path):
        data = _data_contract()
        train_c = build_train_contract(
            experiment_id="expA", data_contract=data, hparams={"max_steps": 10}
        )
        write_full_train_summary(
            tmp_path,
            {"status": "SUCCESS_FULL_TRAINING", **train_c, "train_contract": train_c},
        )
        assert_ready_for_full_evaluate(
            tmp_path, expected_experiment_id="expA",
            expected_contract=build_evaluate_contract(train_contract=train_c),
        )
        stale = build_evaluate_contract(
            train_contract=build_train_contract(
                experiment_id="expA",
                data_contract=_data_contract(validation_manifest_content_hash="STALE"),
                hparams={"max_steps": 10},
            )
        )
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_TRAINING"):
            assert_ready_for_full_evaluate(
                tmp_path, expected_experiment_id="expA", expected_contract=stale
            )

    def test_best_checkpoint_restored_not_latest(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        train_c = build_train_contract(
            experiment_id="expA", data_contract=_data_contract(), hparams={"max_steps": 500}
        )
        write_checkpoint_fingerprint(
            local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=500, extra=train_c
        )
        _complete_ckpt(local / "checkpoint-100", 100)
        best = _complete_ckpt(local / "checkpoint-300", 300)
        _complete_ckpt(local / "checkpoint-500", 500)
        sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        fresh = experiment_checkpoint_dir(tmp_path / "fresh", "expA", kind=FULL_TRAIN_MARKER)
        resolved = resolve_best_checkpoint_from_durable(
            drive, experiment_id="expA",
            train_summary={"best_checkpoint": str(best), "status": "SUCCESS_FULL_TRAINING", **train_c},
            expected_contract=train_c, local_experiment_dir=fresh,
        )
        assert Path(resolved).name == "checkpoint-300"

    def test_missing_best_checkpoint_refuses_fallback(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", "expA", kind=FULL_TRAIN_MARKER)
        train_c = build_train_contract(
            experiment_id="expA", data_contract=_data_contract(), hparams={"max_steps": 500}
        )
        write_checkpoint_fingerprint(
            local, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=500, extra=train_c
        )
        _complete_ckpt(local / "checkpoint-500", 500)
        sync_experiment_checkpoints_to_durable(
            local, drive, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        fresh = experiment_checkpoint_dir(tmp_path / "fresh", "expA", kind=FULL_TRAIN_MARKER)
        with pytest.raises(RuntimeError, match="not found under restored tree"):
            resolve_best_checkpoint_from_durable(
                drive, experiment_id="expA",
                train_summary={"best_checkpoint": "checkpoint-300", **train_c},
                expected_contract=train_c, local_experiment_dir=fresh,
            )

    def test_summary_without_best_checkpoint_is_refused(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        with pytest.raises(RuntimeError, match="missing best_checkpoint"):
            resolve_best_checkpoint_from_durable(
                drive,
                experiment_id="expA",
                train_summary={"status": "SUCCESS_FULL_TRAINING"},
                local_experiment_dir=tmp_path / "local",
            )

    def test_evaluate_gate_is_fail_closed(self):
        ok = derive_full_evaluate_status(
            full_train_success=True, best_checkpoint_from_durable_valid=True,
            metrics_finite=True, frozen_test_accessed=False, contract_matches=True,
        )
        assert ok["status"] == "SUCCESS_FULL_EVALUATE" and ok["failed_checks"] == []
        for kwargs in (
            {"full_train_success": False},
            {"best_checkpoint_from_durable_valid": False},
            {"metrics_finite": False},
            {"frozen_test_accessed": True},
            {"contract_matches": False},
        ):
            base = dict(
                full_train_success=True, best_checkpoint_from_durable_valid=True,
                metrics_finite=True, frozen_test_accessed=False, contract_matches=True,
            )
            base.update(kwargs)
            verdict = derive_full_evaluate_status(**base)
            assert verdict["status"] == "FAILED" and verdict["failed_checks"]

    def test_metrics_finite_rejects_nan_and_missing(self):
        assert metrics_are_finite({"eval_wer": 0.5, "eval_cer": 0.2, "eval_loss": 1.0})
        assert not metrics_are_finite({"eval_wer": float("nan"), "eval_cer": 0.2, "eval_loss": 1.0})
        assert not metrics_are_finite({"eval_wer": 0.5, "eval_cer": 0.2})
        assert not metrics_are_finite({"eval_wer": "x", "eval_cer": 0.2, "eval_loss": 1.0})


# ---------------------------------------------------------------------------
# Two-process resume proof
# ---------------------------------------------------------------------------

class TestTwoSessionResumeProof:
    def test_same_process_resume_is_rejected(self):
        token = new_session_token()
        with pytest.raises(RuntimeError, match="reused the Phase A session token"):
            assert_cross_session_resume({"session": token}, current_session=token)

    def test_same_pid_and_start_time_is_rejected(self):
        a = {"pid": 1234, "nonce": "aaa", "process_started": 111.0}
        b = {"pid": 1234, "nonce": "bbb", "process_started": 111.0}
        with pytest.raises(RuntimeError, match="same process"):
            assert_cross_session_resume({"session": a}, current_session=b)

    def test_phase_a_without_session_token_is_rejected(self):
        with pytest.raises(RuntimeError, match="no session token"):
            assert_cross_session_resume({"global_step": 100}, current_session=new_session_token())

    def test_distinct_sessions_accepted(self):
        a = {"pid": 10, "nonce": "aaa", "process_started": 100.0}
        b = {"pid": 20, "nonce": "bbb", "process_started": 200.0}
        proof = assert_cross_session_resume({"session": a}, current_session=b)
        assert proof["two_sessions"] is True

    def test_status_requires_every_captured_flag(self):
        proof = {
            "model_restored": True, "optimizer_restored": True, "scheduler_restored": True,
            "rng_restored": True, "data_position_ok": True, "step_matches_checkpoint": True,
            "final_global_step": 200,
        }
        cross = {"two_sessions": True}
        good = derive_resume_test_status_from_proof(
            proof, phase_a_reached_target=True, cross_session=cross,
            used_separate_experiment_dir=True,
        )
        assert good["status"] == "SUCCESS_FULL_RESUME_TEST"

        for key in ("model_restored", "optimizer_restored", "scheduler_restored",
                    "rng_restored", "data_position_ok", "step_matches_checkpoint"):
            bad = dict(proof)
            bad[key] = False
            verdict = derive_resume_test_status_from_proof(
                bad, phase_a_reached_target=True, cross_session=cross,
                used_separate_experiment_dir=True,
            )
            assert verdict["status"] == "FAILED" and key in verdict["failed_checks"]

    def test_single_session_cannot_be_declared_a_true_restart(self):
        proof = {
            "model_restored": True, "optimizer_restored": True, "scheduler_restored": True,
            "rng_restored": True, "data_position_ok": True, "step_matches_checkpoint": True,
            "final_global_step": 200,
        }
        verdict = derive_resume_test_status_from_proof(
            proof, phase_a_reached_target=True, cross_session={},
            used_separate_experiment_dir=True,
        )
        assert verdict["status"] == "FAILED"
        assert "true_restart" in verdict["failed_checks"]

    def test_wrong_final_step_fails(self):
        proof = {
            "model_restored": True, "optimizer_restored": True, "scheduler_restored": True,
            "rng_restored": True, "data_position_ok": True, "step_matches_checkpoint": True,
            "final_global_step": 137,
        }
        verdict = derive_resume_test_status_from_proof(
            proof, phase_a_reached_target=True, cross_session={"two_sessions": True},
            used_separate_experiment_dir=True,
        )
        assert "reached_phase_b_steps" in verdict["failed_checks"]

    def test_summarize_refuses_missing_capture(self):
        with pytest.raises(RuntimeError, match="on_train_begin never fired"):
            summarize_resume_proof({})
        with pytest.raises(RuntimeError, match="on_step_begin never fired"):
            summarize_resume_proof({"restore": {"global_step": 100}})
