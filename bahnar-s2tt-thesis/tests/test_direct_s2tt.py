import ast
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.asr_full_train import ENV_DURABLE_CHECKPOINT_BUDGET_BYTES
from src.direct_contract import (
    DIRECT_CONFIG_RELPATH,
    LOCKED_ASR_PREPARE_CONTRACT_HASH,
    LOCKED_DECODER_ID,
    LOCKED_DECODER_REVISION,
    LOCKED_ENCODER_ID,
    LOCKED_ENCODER_REVISION,
    LOCKED_EXPERIMENT_ID,
    LOCKED_TARGET_LANG,
    SOURCE_FINGERPRINT_NOTEBOOK,
    SOURCE_FINGERPRINT_RELPATHS,
    STATUS_DIRECT_TRAINING,
    STATUS_FAILED,
    assert_direct_data_contract_self_consistent,
    assert_direct_training_contract_self_consistent,
    assert_locked_direct_baseline,
    assert_locked_direct_runtime,
    assert_source_fingerprint_matches,
    build_direct_data_contract,
    build_direct_training_contract,
    canonicalize_notebook_code,
    compute_source_fingerprint,
    load_direct_yaml_config,
)
from src.direct_data import (
    DIRECT_PREPARE_FILES,
    DirectAccessTracker,
    assert_direct_frames_match_contract,
    compute_ordered_row_hash,
    compute_pair_hash,
    huggingface_snapshot_dir,
    hydrate_direct_audio,
    load_asr_matched_direct_frames,
    load_direct_prepare_success,
    persist_direct_prepare,
    preflight_direct_local_disk,
    preflight_hydrate_disk,
    DIRECT_MONITOR_CSV,
    DIRECT_TRAINING_CONTRACT,
)
from src.direct_full_train import (
    assert_direct_checkpoint_budget,
    assert_direct_checkpoint_fingerprint,
    compute_direct_seq2seq_metrics,
    derive_direct_handoff,
    derive_direct_training_status,
    fixed_subset,
    frozen_flag_from_tracker,
    load_stage_success,
    plan_direct_resume_position,
    random_sampler_uid_order,
    restore_direct_best_checkpoint,
    validate_direct_best_checkpoint,
    assert_monitor_file_sha256,
    assert_monitor_matches_validation,
    monitor_manifest,
    _reference_from_frame,
)
from src.direct_model import (
    audit_direct_targets_by_split,
    audit_target_tokenizer,
    copy_mbart_seq2seq_embeddings,
    load_direct_model,
)
from src.direct_runtime_paths import resolve_direct_runtime_paths


def _contract_kw(**over):
    kw = dict(
        dataset_id="d", dataset_revision="a" * 40, parquet_revision="b" * 40,
        asr_prepare_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
        train_uid_set_hash="x", validation_uid_set_hash="y",
        train_ordered_uid_hash="ox", validation_ordered_uid_hash="oy",
        train_pair_hash="px", validation_pair_hash="py",
        train_ordered_row_hash="rx", validation_ordered_row_hash="ry",
        train_file_sha256="f" * 64, validation_file_sha256="g" * 64,
        locked_train_manifest_sha256="h" * 64, locked_validation_manifest_sha256="i" * 64,
        asr_train_eligible_uid_set_hash="j" * 64, asr_validation_eligible_uid_set_hash="k" * 64,
        asr_train_eligible_file_sha256="l" * 64, asr_validation_eligible_file_sha256="m" * 64,
        train_count=10, validation_count=2,
        encoder_id=LOCKED_ENCODER_ID, encoder_revision=LOCKED_ENCODER_REVISION,
        decoder_id=LOCKED_DECODER_ID, decoder_revision=LOCKED_DECODER_REVISION,
        target_lang=LOCKED_TARGET_LANG, tokenizer_fingerprint="tok", sample_rate=16000,
        max_audio_duration=40, max_target_length=256,
    )
    kw.update(over)
    return kw


def test_locked_direct_baseline_accepts_pins():
    assert_locked_direct_baseline(
        LOCKED_ENCODER_ID, LOCKED_ENCODER_REVISION,
        LOCKED_DECODER_ID, LOCKED_DECODER_REVISION, LOCKED_TARGET_LANG,
    )


def test_locked_direct_baseline_rejects_unpinned():
    with pytest.raises(RuntimeError):
        assert_locked_direct_baseline(LOCKED_ENCODER_ID, "main", LOCKED_DECODER_ID, LOCKED_DECODER_REVISION, LOCKED_TARGET_LANG)


def test_data_contract_is_deterministic():
    kw = _contract_kw()
    assert build_direct_data_contract(**kw)["contract_hash"] == build_direct_data_contract(**kw)["contract_hash"]


def test_data_contract_changes_when_target_lock_changes():
    a = build_direct_data_contract(**_contract_kw(train_pair_hash="aa"))
    b = build_direct_data_contract(**_contract_kw(train_pair_hash="bb"))
    assert a["contract_hash"] != b["contract_hash"]


def test_yaml_is_single_operational_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_direct_yaml_config(root / DIRECT_CONFIG_RELPATH)
    assert cfg["experiment_id"] == LOCKED_EXPERIMENT_ID
    assert cfg["encoder_id"] == LOCKED_ENCODER_ID


def test_data_contract_self_consistent_rejects_child_tamper():
    payload = build_direct_data_contract(**_contract_kw())
    assert_direct_data_contract_self_consistent(payload)
    payload["train_pair_hash"] = "tampered"
    with pytest.raises(RuntimeError, match="data contract hash mismatch"):
        assert_direct_data_contract_self_consistent(payload)


def test_load_prepare_rejects_inconsistent_embedded_contract(tmp_path):
    state = tmp_path / "st"
    state.mkdir()
    contract = build_direct_data_contract(**_contract_kw())
    embedded = dict(contract)
    embedded["train_pair_hash"] = "tampered"
    (state / "direct_prepare_summary.json").write_text(json.dumps({
        "status": "SUCCESS_DIRECT_PREPARE",
        "contract_hash": contract["contract_hash"],
        "data_contract": embedded,
    }))
    with pytest.raises(RuntimeError, match="data contract hash mismatch"):
        load_direct_prepare_success(state, expected_contract_hash=contract["contract_hash"])


def _train_kw(**over):
    kw = dict(
        direct_data_contract_hash="d", experiment_id="e", encoder_id="a", encoder_revision="b",
        decoder_id="c", decoder_revision="d", target_lang="vi_VN", tokenizer_fingerprint="t",
        train_uid_set_hash="x", validation_uid_set_hash="y", monitor_uid_set_hash="m",
        monitor_pair_hash="mp", monitor_ordered_row_hash="mo", monitor_file_sha256="n" * 64,
        monitor_size=2, source_fingerprint_sha256="s" * 64,
        torch_version="2.8.0", transformers_version="4.57.6", accelerate_version="1.10.1",
        seed=42, learning_rate=2e-5, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, num_train_epochs=3, warmup_ratio=.05, weight_decay=.01,
        fp16=True, bf16=False, gradient_checkpointing=True, save_steps=1000, eval_steps=1000,
        save_total_limit=2, max_target_length=256, generation_max_length=256, num_beams=4,
        metric_for_best_model="eval_sacrebleu", greater_is_better=True, freeze_feature_encoder=True,
    )
    kw.update(over)
    return kw


def test_training_contract_self_consistency():
    payload = build_direct_training_contract(**_train_kw())
    assert_direct_training_contract_self_consistent(payload)
    payload["num_beams"] = 5
    with pytest.raises(RuntimeError):
        assert_direct_training_contract_self_consistent(payload)


def _write_asr_state(tmp_path: Path, train_uids=("u1", "u2"), val_uids=("v1",)) -> Path:
    state = tmp_path / "asr"
    state.mkdir()
    (state / "full_data_summary.json").write_text(json.dumps({
        "status": "SUCCESS_FULL_PREPARE", "contract_hash": LOCKED_ASR_PREPARE_CONTRACT_HASH,
    }))
    base_cols = {
        "record_id": "r", "group_id": "g", "recording_group_id": "rg", "pair_key": "p", "source_split": "train",
        "parquet_file": "https://example.invalid/a.parquet", "shard_row_index": 0, "text_bahnar": "b",
        "text_bahnar_norm": "b", "duration_seconds": 1.0, "processed_duration_seconds": 1.0,
        "local_cache_relpath": "x.wav", "audio_source": "prepared_local", "sha256_pcm": "abc",
        "n_samples": 16000, "audio_pcm_pipeline_version": "v",
    }
    train = pd.DataFrame([{**base_cols, "record_uid": u, "split": "train", "group_id": f"tg{i}", "recording_group_id": f"tr{i}"} for i, u in enumerate(train_uids)])
    val = pd.DataFrame([{**base_cols, "record_uid": u, "split": "validation", "group_id": f"vg{i}", "recording_group_id": f"vr{i}"} for i, u in enumerate(val_uids)])
    train.to_csv(state / "full_train_eligible.csv", index=False)
    val.to_csv(state / "full_validation_eligible.csv", index=False)
    return state


def test_matched_direct_join_keeps_exact_asr_uids(tmp_path):
    state = _write_asr_state(tmp_path)
    train_manifest = tmp_path / "rq1_train.csv"
    val_manifest = tmp_path / "rq1_validation.csv"
    pd.DataFrame({"record_uid": ["u1", "u2", "extra"], "text_vi": ["xin chao", "tam biet", "x"]}).to_csv(train_manifest, index=False)
    pd.DataFrame({"record_uid": ["v1", "extra2"], "text_vi": ["cam on", "x"]}).to_csv(val_manifest, index=False)
    out = load_asr_matched_direct_frames(
        asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
        expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
        expected_train_count=2, expected_validation_count=1,
    )
    assert out["train"]["record_uid"].tolist() == ["u1", "u2"]
    assert out["validation"]["record_uid"].tolist() == ["v1"]
    assert out["train"]["text_vi_norm"].tolist() == ["xin chao", "tam biet"]


def test_matched_direct_join_fails_on_missing_target(tmp_path):
    state = _write_asr_state(tmp_path)
    train_manifest = tmp_path / "rq1_train.csv"
    val_manifest = tmp_path / "rq1_validation.csv"
    pd.DataFrame({"record_uid": ["u1", "u2"], "text_vi": ["ok", ""]}).to_csv(train_manifest, index=False)
    pd.DataFrame({"record_uid": ["v1"], "text_vi": ["ok"]}).to_csv(val_manifest, index=False)
    with pytest.raises(RuntimeError, match="empty Vietnamese targets"):
        load_asr_matched_direct_frames(
            asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
            expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
        )


def _frames():
    train = pd.DataFrame({
        "record_uid": ["u1", "u2"], "text_vi_norm": ["xin chao", "tam biet"],
        "split": ["train", "train"], "source_split": ["train", "train"],
    })
    val = pd.DataFrame({
        "record_uid": ["v1"], "text_vi_norm": ["cam on"],
        "split": ["validation"], "source_split": ["validation"],
    })
    return train, val


def _persist(state, train, val, **kwargs):
    kw = dict(
        train=train,
        validation=val,
        summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
        target_audit={"passed": True},
        training_contract={"direct_training_contract_hash": "t", "experiment_id": "e"},
        monitor=val,
    )
    kw.update(kwargs)
    return persist_direct_prepare(state, **kw)


def _locks(train, val, tmp_path):
    tpath, vpath = tmp_path / "t.csv", tmp_path / "v.csv"
    train.to_csv(tpath, index=False)
    val.to_csv(vpath, index=False)
    from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash, sha256_file
    return {
        "train_uid_set_hash": compute_uid_set_hash(train),
        "validation_uid_set_hash": compute_uid_set_hash(val),
        "train_ordered_uid_hash": compute_ordered_uid_hash(train),
        "validation_ordered_uid_hash": compute_ordered_uid_hash(val),
        "train_pair_hash": compute_pair_hash(train),
        "validation_pair_hash": compute_pair_hash(val),
        "train_ordered_row_hash": compute_ordered_row_hash(train),
        "validation_ordered_row_hash": compute_ordered_row_hash(val),
        "train_file_sha256": sha256_file(tpath),
        "validation_file_sha256": sha256_file(vpath),
        "train_count": len(train),
        "validation_count": len(val),
        "tpath": tpath,
        "vpath": vpath,
    }


def test_reload_detects_same_uid_target_change(tmp_path):
    train, val = _frames()
    locks = _locks(train, val, tmp_path)
    contract = {**locks}
    changed = train.copy()
    changed.loc[0, "text_vi_norm"] = "changed"
    with pytest.raises(RuntimeError, match="drifted"):
        assert_direct_frames_match_contract(changed, val, contract, train_path=locks["tpath"], validation_path=locks["vpath"])


def test_reload_detects_reorder(tmp_path):
    train, val = _frames()
    locks = _locks(train, val, tmp_path)
    reordered = train.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="drifted"):
        assert_direct_frames_match_contract(reordered, val, locks, train_path=locks["tpath"], validation_path=locks["vpath"])


def test_reload_detects_train_validation_swap(tmp_path):
    train, val = _frames()
    locks = _locks(train, val, tmp_path)
    with pytest.raises(RuntimeError):
        assert_direct_frames_match_contract(val, train, locks, train_path=locks["tpath"], validation_path=locks["vpath"])


class FakeTokenizer:
    pad_token_id = 1
    unk_token_id = 99

    def batch_decode(self, arr, skip_special_tokens=True):
        out = []
        for row in np.asarray(arr):
            vals = [str(int(x)) for x in row if int(x) not in (1, 2)]
            out.append(" ".join(vals))
        return out

    def __call__(self, text_target=None, add_special_tokens=True, truncation=False, return_attention_mask=False, **kwargs):
        ids = []
        for t in text_target:
            n = 10 if "short" in t else 400
            ids.append([3] * n)
        return {"input_ids": ids}


def test_validation_truncation_must_be_zero():
    tok = FakeTokenizer()
    train_ok = audit_target_tokenizer(tok, ["short a", "short b"], max_target_length=256, split="train")
    assert train_ok["passed"]
    val = audit_target_tokenizer(
        tok, ["this is a long target without the s-h-o-r-t marker"], max_target_length=256,
        split="validation", require_zero_truncation=True,
    )
    assert val["passed"] is False
    assert "validation_truncation_must_be_zero" in val["failed_checks"]
    both = audit_direct_targets_by_split(
        tok, train_targets=["short a"], validation_targets=["long target without marker"],
        max_target_length=256,
    )
    assert both["passed"] is False


def test_eval_reference_uses_full_dataframe_text():
    df = pd.DataFrame({"record_uid": ["u1"], "text_vi_norm": ["full " * 80]})
    assert _reference_from_frame(df, "u1") == "full " * 80
    with pytest.raises(RuntimeError, match="missing"):
        _reference_from_frame(df, "nope")


def _fake_metric(preds, refs):
    return {"sacrebleu": 12.5, "chrfpp": 30.0, "n": len(preds)}


def test_metrics_support_evalprediction_and_sanitize_minus100(monkeypatch):
    import src.direct_full_train as d
    monkeypatch.setattr(d, "mt_corpus_metrics", _fake_metric)
    class EP:
        predictions = np.array([[3, -100, 2], [4, 5, 2]])
        label_ids = np.array([[3, 1, 2], [4, -100, 2]])
    m = compute_direct_seq2seq_metrics(EP(), tokenizer=FakeTokenizer())
    assert m == {"sacrebleu": 12.5, "chrfpp": 30.0, "monitor_n": 2}


def test_metrics_support_tuple_and_nested_predictions(monkeypatch):
    import src.direct_full_train as d
    monkeypatch.setattr(d, "mt_corpus_metrics", _fake_metric)
    preds = (np.array([[3, -100, 2]]), np.array([0]))
    labels = np.array([[3, 1, 2]])
    m = compute_direct_seq2seq_metrics((preds, labels), tokenizer=FakeTokenizer())
    assert m["monitor_n"] == 1


def test_metrics_fail_closed_unknown_type():
    with pytest.raises(RuntimeError, match="Unexpected eval_preds"):
        compute_direct_seq2seq_metrics(object(), tokenizer=FakeTokenizer())


def test_fixed_subset_deterministic():
    df = pd.DataFrame({"record_uid": [f"u{i}" for i in range(100)]})
    a = fixed_subset(df, 10, seed=42)
    b = fixed_subset(df, 10, seed=42)
    assert a["record_uid"].tolist() == b["record_uid"].tolist()


def test_handoff_only_after_evaluate():
    assert derive_direct_handoff("SUCCESS_DIRECT_TRAINING", frozen_test_accessed=False)["ready_for_rq1_evaluation"] is False
    assert derive_direct_handoff("SUCCESS_DIRECT_EVALUATE", frozen_test_accessed=False)["ready_for_rq1_evaluation"] is True
    assert derive_direct_handoff("SUCCESS_DIRECT_EVALUATE", frozen_test_accessed=True)["ready_for_rq1_evaluation"] is False
    assert derive_direct_handoff("SUCCESS_DIRECT_EVALUATE", frozen_test_accessed=False)["ready_for_rq1_final"] is False


def test_frozen_flag_requires_tracker():
    with pytest.raises(RuntimeError, match="access tracker"):
        frozen_flag_from_tracker(None)
    t = DirectAccessTracker()
    assert frozen_flag_from_tracker(t) is False


def test_frozen_test_file_leakage(tmp_path):
    p = tmp_path / "rq1_test.csv"
    p.write_text("record_uid\n1\n")
    tr = DirectAccessTracker()
    with pytest.raises(RuntimeError, match="Frozen-test"):
        tr.record_file(p)


def test_frozen_test_parquet_leakage():
    tr = DirectAccessTracker()
    with pytest.raises(RuntimeError, match="Frozen-test"):
        tr.record_parquet("cuong06/Bahnar_Vietnamese/data/rq1_test.parquet@abc")


def test_runtime_paths_share_nb03_audio_cache(tmp_path):
    rt = resolve_direct_runtime_paths(project_root=tmp_path, local_root=tmp_path / "local", durable_root=tmp_path / "dur")
    assert rt.audio_cache_dir.name == "bahnar_full_audio_cache"
    assert "direct_full_state" in str(rt.durable_state_root)


def test_stale_stage_summary_rejected(tmp_path):
    p = tmp_path / "direct_pilot_summary.json"
    p.write_text(json.dumps({
        "status": "SUCCESS_DIRECT_PILOT", "contract_hash": "aaa",
        "experiment_id": LOCKED_EXPERIMENT_ID, "direct_training_contract_hash": "old",
    }))
    with pytest.raises(RuntimeError, match="direct_training_contract_hash"):
        load_stage_success(
            p, status="SUCCESS_DIRECT_PILOT", expected_contract_hash="aaa",
            expected_experiment_id=LOCKED_EXPERIMENT_ID, expected_training_contract_hash="new",
        )


def test_insufficient_checkpoint_budget():
    with pytest.raises(RuntimeError, match="Insufficient durable checkpoint budget"):
        assert_direct_checkpoint_budget(n_parameters=10 ** 9, env={ENV_DURABLE_CHECKPOINT_BUDGET_BYTES: "1000"})


def test_preflight_fails_when_wav_budget_impossible(monkeypatch):
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16000]})
    monkeypatch.setattr(
        "src.direct_data.missing_or_corrupt_wav_bytes",
        lambda frames, **k: {"wav_bytes": 10 ** 18, "unique_records": 1, "n_samples": 1},
    )
    with pytest.raises(RuntimeError, match="Insufficient"):
        preflight_hydrate_disk([df], audio_cache_dir="/tmp", parquet_cache_dir="/tmp")


def test_preflight_combines_wav_and_parquet_on_same_filesystem(monkeypatch):
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16000]})
    monkeypatch.setattr(
        "src.direct_data.missing_or_corrupt_wav_bytes",
        lambda frames, **k: {"wav_bytes": 40 * 10 ** 9},
    )
    seen = []

    def fake_budget(path, *, needs, reserve_bytes, label):
        seen.append(dict(needs))
        total = sum(int(v) for v in needs.values()) + int(reserve_bytes)
        if total > 71 * 10 ** 9:
            raise RuntimeError("Insufficient local disk for combined peak")
        return {"ok": True, "needs": needs, "label": label}

    monkeypatch.setattr("src.direct_data.assert_local_disk_budget", fake_budget)
    with pytest.raises(RuntimeError, match="combined peak"):
        preflight_hydrate_disk(
            [df], audio_cache_dir="/tmp/a", parquet_cache_dir="/tmp/b",
            parquet_headroom_bytes=40 * 10 ** 9, reserve_bytes=2 * 10 ** 9,
        )
    assert len(seen) == 1
    assert seen[0]["wav"] == 40 * 10 ** 9
    assert seen[0]["parquet_shard_headroom"] == 40 * 10 ** 9


def test_preflight_train_sums_parquet_and_local_checkpoint_copies(monkeypatch):
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16000]})
    monkeypatch.setattr(
        "src.direct_data.missing_or_corrupt_wav_bytes",
        lambda frames, **k: {"wav_bytes": 10 ** 9, "missing": 1, "corrupt": 0, "ok": 0},
    )
    monkeypatch.setattr(
        "src.direct_data.estimate_checkpoint_bytes",
        lambda **k: {"checkpoint_bytes": 20 * 10 ** 9, "n_parameters": k.get("n_parameters")},
    )
    monkeypatch.setattr("src.direct_data.huggingface_snapshot_is_present", lambda *a, **k: True)
    seen = []
    monkeypatch.setattr(
        "src.direct_data.assert_local_disk_budget",
        lambda path, *, needs, reserve_bytes, label: seen.append(dict(needs)) or {"ok": True, "needs": needs},
    )
    preflight_direct_local_disk(
        [df], audio_cache_dir="/tmp/a", parquet_cache_dir="/tmp/a",
        local_ckpt_dir="/tmp/a", n_parameters=100, parquet_headroom_bytes=4 * 10 ** 9,
        reserve_bytes=2 * 10 ** 9,
    )
    assert seen[0]["wav"] == 10 ** 9
    assert seen[0]["parquet_shard_headroom"] == 4 * 10 ** 9
    assert seen[0]["checkpoint_peak"] == 60 * 10 ** 9
    assert "peak_parquet_or_checkpoint" not in seen[0]
    assert "hf_model_cache" not in seen[0]


def test_hydrate_deletes_parquet_blob(tmp_path, monkeypatch):
    blob = tmp_path / "blob.parquet"
    blob.write_bytes(b"x" * 10)
    deleted = []

    def fake_reader(**kwargs):
        kwargs["downloaded"]["shard.parquet"] = str(blob)
        return object()

    def fake_hydrate(*args, **kwargs):
        class Ref:
            shard_key = "shard.parquet"
        kwargs["cleanup_shard"](Ref())
        return {"shards": 1, "hydrated": 1, "unique_records": 1}

    monkeypatch.setattr("src.direct_data.preflight_hydrate_disk", lambda *a, **k: {"ok": True})
    monkeypatch.setattr("src.direct_data.make_hf_parquet_stream_reader", fake_reader)
    monkeypatch.setattr("src.direct_data.hydrate_union_audio", fake_hydrate)
    monkeypatch.setattr("src.direct_data.cleanup_shard_download", lambda p: deleted.append(str(p)) or 10)
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16]})
    hydrate_direct_audio(
        [df], asr_state_dir=tmp_path, dataset_id="d", parquet_revision="r",
        audio_cache_dir=tmp_path / "wav", parquet_cache_dir=tmp_path / "pq", sample_rate=16000,
    )
    assert deleted == [str(blob)]


def test_corrupt_best_checkpoint_fingerprint(tmp_path):
    root = tmp_path / "full_train" / "exp"
    ckpt = root / "checkpoint-1"
    ckpt.mkdir(parents=True)
    (root / "full_experiment_fingerprint.json").write_text(json.dumps({
        "experiment_id": "exp", "direct_training_contract_hash": "aaa",
    }))
    with pytest.raises(RuntimeError, match="training-contract"):
        assert_direct_checkpoint_fingerprint(
            ckpt, experiment_id="exp", expected_training_contract_hash="bbb", experiment_root=root,
        )


def test_mixed_checkpoint_outside_experiment_root(tmp_path):
    root = tmp_path / "full_train" / "exp"
    other = tmp_path / "other" / "checkpoint-1"
    other.mkdir(parents=True)
    (other / "full_experiment_fingerprint.json").write_text(json.dumps({
        "experiment_id": "exp", "direct_training_contract_hash": "aaa",
    }))
    with pytest.raises(RuntimeError, match="not under experiment root"):
        assert_direct_checkpoint_fingerprint(
            other, experiment_id="exp", expected_training_contract_hash="aaa", experiment_root=root,
        )


def test_atomic_prepare_pointer_must_be_under_durable_root(tmp_path):
    root = tmp_path / "direct_full_state"
    state = root / "contract_abcd"
    train, val = _frames()
    _persist(state, train, val, durable_state_root=root)
    assert (root / "LATEST_PREPARE.json").is_file()
    other = tmp_path / "elsewhere"
    with pytest.raises(RuntimeError, match="not under durable state root"):
        _persist(state, train, val, durable_state_root=other)


def _sampler_worker(uids, seed, q):
    q.put(plan_direct_resume_position(
        uids, seed=seed, resume_step=20, per_device_train_batch_size=1, gradient_accumulation_steps=8,
    )["expected_first_uid"])


def test_resume_uid_uses_epoch1_seedable_sampler_not_epoch0():
    uids = [f"u{i}" for i in range(128)]
    plan = plan_direct_resume_position(
        uids, seed=42, resume_step=20, per_device_train_batch_size=1, gradient_accumulation_steps=8,
    )
    assert plan["epoch"] == 1
    assert plan["num_update_steps_per_epoch"] == 16
    assert plan["expected_sample_index"] == 32
    epoch0 = random_sampler_uid_order(uids, seed=42, epoch=0)
    epoch1 = random_sampler_uid_order(uids, seed=42, epoch=1)
    assert epoch0 != epoch1
    assert plan["expected_first_uid"] == epoch1[32]
    assert plan["expected_first_uid"] != epoch0[32]
    assert plan["epoch0_first_uid"] == epoch0[32]
    from accelerate.data_loader import SeedableRandomSampler

    class _DS:
        def __len__(self):
            return 128

    sampler = SeedableRandomSampler(_DS(), replacement=False, data_seed=42)
    sampler.set_epoch(1)
    actual = [uids[i] for i in list(sampler)]
    assert actual[32] == plan["expected_first_uid"]


def test_two_process_seedable_resume_position():
    uids = [f"u{i}" for i in range(128)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_sampler_worker, args=(uids, 42, q))
    p.start()
    p.join(timeout=30)
    assert p.exitcode == 0
    parent = plan_direct_resume_position(
        uids, seed=42, resume_step=20, per_device_train_batch_size=1, gradient_accumulation_steps=8,
    )
    assert q.get() == parent["expected_first_uid"]
    assert parent["epoch"] == 1


def test_source_fingerprint_stable():
    root = Path(__file__).resolve().parents[1]
    a = compute_source_fingerprint(root)
    b = compute_source_fingerprint(root)
    assert a["aggregate_sha256"] == b["aggregate_sha256"]
    paths = {f["path"] for f in a["files"]}
    assert "src/asr_full_train.py" in paths
    assert "src/asr_full_shards.py" in paths
    assert "src/asr_utils.py" in paths
    assert "src/asr_runtime_paths.py" in paths
    assert "src/audio_utils.py" in paths
    assert "src/mt_normalize.py" in paths
    assert "src/mt_tokenize.py" in paths
    assert "src/mt_runtime_paths.py" in paths
    assert "src/mt_full_train.py" not in paths
    assert "src/mt_contract.py" not in paths
    assert SOURCE_FINGERPRINT_NOTEBOOK in paths
    nb = next(f for f in a["files"] if f["path"] == SOURCE_FINGERPRINT_NOTEBOOK)
    assert nb["canonical"] is True
    for rel in SOURCE_FINGERPRINT_RELPATHS:
        assert rel in paths


def test_source_fingerprint_mismatch_fail_closed():
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        assert_source_fingerprint_matches({"aggregate_sha256": "aa"}, {"aggregate_sha256": "bb"})


def _notebook_code() -> str:
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "05_train_direct_s2tt.ipynb").read_text())
    parts = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            parts.append("".join(c.get("source") or []))
    return "\n".join(parts)


def test_notebook_ast_calls_only_existing_src_apis():
    src = _notebook_code()
    ast.parse(src)
    assert "make_sequential_sampler_trainer_cls" not in src
    assert "42949672960" not in src
    assert "315_000_000" not in src
    assert "plan_direct_resume_position" in src
    assert "derive_direct_training_status" in src
    assert "preflight_direct_local_disk" in src
    assert "validate_direct_best_checkpoint" in src
    assert "assert_monitor_matches_validation" in src
    assert "assert_monitor_file_sha256" in src
    assert "source_fingerprint_sha256" in src
    assert "best_checkpoint_name=validated_best" in src
    assert "frozen_test_accessed=False" not in src
    assert "os.environ[\"BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES\"] =" not in src
    tree = ast.parse(src)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
            imported.setdefault(node.module, []).extend(a.name for a in node.names)
    import importlib
    missing = []
    for mod, names in imported.items():
        m = importlib.import_module(mod)
        for n in names:
            if not hasattr(m, n):
                missing.append(f"{mod}.{n}")
    assert missing == []


def test_notebook_cells_compile():
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "05_train_direct_s2tt.ipynb").read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        ast.parse("".join(c.get("source") or []), filename=f"nb05_cell_{i}")


def _fake_complete_checkpoint(path: Path, step: int = 100) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"x")
    (path / "optimizer.pt").write_bytes(b"x")
    (path / "scheduler.pt").write_bytes(b"x")
    (path / "rng_state.pth").write_bytes(b"x")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return path


def test_train_success_gate_rejects_early_stop(tmp_path):
    ckpt = _fake_complete_checkpoint(tmp_path / "checkpoint-10", step=10)
    out = derive_direct_training_status(
        global_step=10, max_steps=100,
        metrics={"eval_sacrebleu": 1.0, "eval_chrfpp": 2.0},
        latest_checkpoint=ckpt, best_checkpoint=ckpt,
        durable_best_resolved=True, frozen_test_accessed=False,
    )
    assert out["status"] == STATUS_FAILED
    assert "reached_max_steps" in out["failed_checks"]


def test_train_success_gate_rejects_missing_durable(tmp_path):
    ckpt = _fake_complete_checkpoint(tmp_path / "checkpoint-100", step=100)
    out = derive_direct_training_status(
        global_step=100, max_steps=100,
        metrics={"eval_sacrebleu": 12.0, "eval_chrfpp": 30.0},
        latest_checkpoint=ckpt, best_checkpoint=ckpt,
        durable_best_resolved=False, frozen_test_accessed=False,
    )
    assert out["status"] == STATUS_FAILED
    assert "durable_best_resolved" in out["failed_checks"]


def test_train_success_gate_accepts_complete(tmp_path):
    ckpt = _fake_complete_checkpoint(tmp_path / "checkpoint-100", step=100)
    out = derive_direct_training_status(
        global_step=100, max_steps=100,
        metrics={"eval_sacrebleu": 12.0, "eval_chrfpp": 30.0},
        latest_checkpoint=ckpt, best_checkpoint=ckpt,
        durable_best_resolved=True, frozen_test_accessed=False,
    )
    assert out["status"] == STATUS_DIRECT_TRAINING
    assert out["failed_checks"] == []


def test_persist_restores_lkg_on_partial_commit(tmp_path, monkeypatch):
    import os
    from src.direct_data import DIRECT_TRAIN_CSV

    root = tmp_path / "state"
    train, val = _frames()
    persist_direct_prepare(
        root, train=train, validation=val,
        summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
        target_audit={"passed": True},
        training_contract={"direct_training_contract_hash": "t", "experiment_id": "e"},
        monitor=val,
    )
    old = (root / DIRECT_TRAIN_CSV).read_text()
    new_train = train.copy()
    new_train.loc[0, "text_vi_norm"] = "changed"
    real_replace = os.replace
    n = {"c": 0}

    def boom(src, dst):
        if ".staging_prepare" in str(src):
            n["c"] += 1
            if n["c"] == 2:
                raise OSError("simulated interrupt")
        return real_replace(src, dst)

    monkeypatch.setattr("src.direct_data.os.replace", boom)
    with pytest.raises(OSError, match="simulated interrupt"):
        persist_direct_prepare(
            root, train=new_train, validation=val,
            summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
            target_audit={"passed": True},
            training_contract={"direct_training_contract_hash": "t", "experiment_id": "e"},
            monitor=val,
        )
    assert (root / DIRECT_TRAIN_CSV).read_text() == old


def test_tracker_records_parquet_when_read_fails(tmp_path, monkeypatch):
    tracker = DirectAccessTracker()

    def fake_reader(**kwargs):
        def _r(ref, indices):
            raise RuntimeError("download failed")
        return _r

    def fake_hydrate(*args, **kwargs):
        class Ref:
            filename = "data/train/0.parquet"
            repo_id = "cuong06/Bahnar_Vietnamese"
            revision = "abc"
            shard_key = "data/train/0.parquet"
        kwargs["shard_reader"](Ref(), [0])
        return {}

    monkeypatch.setattr("src.direct_data.preflight_hydrate_disk", lambda *a, **k: {"ok": True})
    monkeypatch.setattr("src.direct_data.make_hf_parquet_stream_reader", fake_reader)
    monkeypatch.setattr("src.direct_data.hydrate_union_audio", fake_hydrate)
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16]})
    with pytest.raises(RuntimeError, match="download failed"):
        hydrate_direct_audio(
            [df], asr_state_dir=tmp_path, dataset_id="d", parquet_revision="r",
            audio_cache_dir=tmp_path / "wav", parquet_cache_dir=tmp_path / "pq",
            sample_rate=16000, tracker=tracker,
        )
    assert tracker.parquet_refs


def _copy_fingerprint_tree(dst: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in SOURCE_FINGERPRINT_RELPATHS:
        dest = dst / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest)


def test_canonical_notebook_masks_runtime_controls_and_outputs():
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 3,
                "outputs": [{"output_type": "stream", "text": ["hello\n"]}],
                "source": [
                    "FULL_STAGE = \"prepare\"\n",
                    "ALLOW_FULL_TRAINING = False\n",
                    "RUN_ID = str(uuid.uuid4())\n",
                    "x = 1\n",
                ],
            }
        ]
    }
    a = canonicalize_notebook_code(nb)
    nb["cells"][0]["source"] = [
        "FULL_STAGE = \"train\"\n",
        "ALLOW_FULL_TRAINING = True\n",
        "RUN_ID = \"other\"\n",
        "x = 1\n",
    ]
    nb["cells"][0]["execution_count"] = 99
    b = canonicalize_notebook_code(nb)
    assert a == b
    assert "<RUNTIME_CONTROL>" in a
    assert "prepare" not in a
    assert "train" not in a.split("x = 1")[0]


def test_source_fingerprint_stable_across_stage_and_autosave(tmp_path):
    _copy_fingerprint_tree(tmp_path)
    before = compute_source_fingerprint(tmp_path)
    nb_path = tmp_path / SOURCE_FINGERPRINT_NOTEBOOK
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["autosave\n"]}]
        cell["execution_count"] = 99
        src = cell.get("source") or []
        text = "".join(src) if isinstance(src, list) else str(src)
        text = text.replace('FULL_STAGE = "prepare"', 'FULL_STAGE = "train"')
        text = text.replace("ALLOW_FULL_TRAINING = False", "ALLOW_FULL_TRAINING = True")
        text = text.replace("RUN_ID = str(uuid.uuid4())", 'RUN_ID = "stage-switch"')
        cell["source"] = [text]
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    after = compute_source_fingerprint(tmp_path)
    assert before["aggregate_sha256"] == after["aggregate_sha256"]


def test_source_fingerprint_changes_when_production_code_changes(tmp_path):
    _copy_fingerprint_tree(tmp_path)
    before = compute_source_fingerprint(tmp_path)
    target = tmp_path / "src/direct_data.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# fingerprint-prod-change\n", encoding="utf-8")
    after = compute_source_fingerprint(tmp_path)
    assert before["aggregate_sha256"] != after["aggregate_sha256"]


def test_source_fingerprint_missing_dependency_fails(tmp_path):
    _copy_fingerprint_tree(tmp_path)
    (tmp_path / "src/asr_utils.py").unlink()
    with pytest.raises(RuntimeError, match="missing file"):
        compute_source_fingerprint(tmp_path)


def test_preflight_counts_only_missing_or_corrupt_wav(tmp_path, monkeypatch):
    from src.asr_full_pcm import wav_bytes_for_samples
    from src.data_utils import safe_cache_filename

    cache = tmp_path / "wav"
    cache.mkdir()
    (cache / safe_cache_filename("u1")).write_bytes(b"present")
    df = pd.DataFrame({"record_uid": ["u1", "u2"], "n_samples": [16000, 32000]})
    monkeypatch.setattr(
        "src.direct_data.assert_local_disk_budget",
        lambda path, *, needs, reserve_bytes, label: {"ok": True, "needs": needs},
    )
    report = preflight_direct_local_disk(
        [df], audio_cache_dir=cache, parquet_cache_dir=tmp_path / "pq",
        reserve_bytes=0, parquet_headroom_bytes=1,
    )
    assert report["wav"]["missing"] == 1
    assert report["wav"]["ok"] == 1
    assert report["wav"]["wav_bytes"] == wav_bytes_for_samples(32000)


def test_preflight_skips_existing_hf_cache_and_uses_two_local_copies(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    for repo, rev in (
        (LOCKED_ENCODER_ID, LOCKED_ENCODER_REVISION),
        (LOCKED_DECODER_ID, LOCKED_DECODER_REVISION),
    ):
        snap = huggingface_snapshot_dir(hub, repo, rev)
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "src.direct_data.missing_or_corrupt_wav_bytes",
        lambda frames, **k: {"wav_bytes": 0, "missing": 0, "corrupt": 0, "ok": 2},
    )
    monkeypatch.setattr(
        "src.direct_data.estimate_checkpoint_bytes",
        lambda **k: {"checkpoint_bytes": 10 * 10 ** 9, "n_parameters": k.get("n_parameters")},
    )
    seen = []
    monkeypatch.setattr(
        "src.direct_data.assert_local_disk_budget",
        lambda path, *, needs, reserve_bytes, label: seen.append((str(path), dict(needs), reserve_bytes))
        or {"ok": True, "needs": needs},
    )
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16000]})
    local = tmp_path / "local_ckpt"
    parquet = tmp_path / "pq"
    report = preflight_direct_local_disk(
        [df], audio_cache_dir=tmp_path / "wav", parquet_cache_dir=parquet,
        local_ckpt_dir=local, n_parameters=100, parquet_headroom_bytes=4 * 10 ** 9,
        reserve_bytes=2 * 10 ** 9, hf_cache_dir=hub,
    )
    assert report["hf_snapshots_missing"] == []
    assert report["peak_copies"] == 3
    needs = {k: v for _p, v, _r in seen for k, v in v.items()}
    assert "hf_model_cache" not in needs
    assert any("checkpoint_peak" in item[1] and item[1]["checkpoint_peak"] == 30 * 10 ** 9 for item in seen)
    assert any("parquet_shard_headroom" in item[1] for item in seen)


def test_preflight_local_and_durable_are_separate_filesystems(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.direct_data.missing_or_corrupt_wav_bytes",
        lambda frames, **k: {"wav_bytes": 10 ** 9},
    )
    monkeypatch.setattr(
        "src.direct_data.estimate_checkpoint_bytes",
        lambda **k: {"checkpoint_bytes": 5 * 10 ** 9, "n_parameters": 1},
    )
    monkeypatch.setattr("src.direct_data.huggingface_snapshot_is_present", lambda *a, **k: True)

    def fake_dev(path):
        return 2 if "durable" in str(path) else 1

    monkeypatch.setattr("src.direct_data.filesystem_device", fake_dev)
    seen = []
    monkeypatch.setattr(
        "src.direct_data.assert_local_disk_budget",
        lambda path, *, needs, reserve_bytes, label: seen.append((str(path), dict(needs)))
        or {"ok": True, "needs": needs},
    )
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16000]})
    preflight_direct_local_disk(
        [df], audio_cache_dir=tmp_path / "local" / "wav",
        parquet_cache_dir=tmp_path / "local" / "pq",
        local_ckpt_dir=tmp_path / "local" / "ckpt",
        n_parameters=1, parquet_headroom_bytes=4 * 10 ** 9, reserve_bytes=0,
        hf_cache_dir=tmp_path / "durable" / "hf",
    )
    local_needs = [needs for path, needs in seen if "durable" not in path]
    durable_needs = [needs for path, needs in seen if "durable" in path]
    assert local_needs
    combined = {}
    for needs in local_needs:
        combined.update(needs)
    assert combined.get("checkpoint_peak") == 15 * 10 ** 9
    assert durable_needs == [] or all("checkpoint_peak" not in n for n in durable_needs)


def test_preflight_does_not_false_reject_hydrated_pod(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    for repo, rev in (
        (LOCKED_ENCODER_ID, LOCKED_ENCODER_REVISION),
        (LOCKED_DECODER_ID, LOCKED_DECODER_REVISION),
    ):
        snap = huggingface_snapshot_dir(hub, repo, rev)
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "src.direct_data.missing_or_corrupt_wav_bytes",
        lambda frames, **k: {"wav_bytes": 0, "missing": 0, "corrupt": 0, "ok": 102486},
    )
    monkeypatch.setattr(
        "src.direct_data.estimate_checkpoint_bytes",
        lambda **k: {"checkpoint_bytes": 18 * 10 ** 9, "n_parameters": 1},
    )
    monkeypatch.setattr("src.asr_full_pcm.disk_free_bytes", lambda path: 71 * 10 ** 9)
    df = pd.DataFrame({"record_uid": ["u1"], "n_samples": [16000]})
    report = preflight_direct_local_disk(
        [df], audio_cache_dir=tmp_path / "wav", parquet_cache_dir=tmp_path / "pq",
        local_ckpt_dir=tmp_path / "ckpt", n_parameters=1,
        parquet_headroom_bytes=4 * 10 ** 9, reserve_bytes=2 * 10 ** 9, hf_cache_dir=hub,
    )
    required = max(r["required_bytes"] for r in report["reports"])
    assert required < 71 * 10 ** 9
    assert report["wav"]["wav_bytes"] == 0
    assert report["peak_copies"] == 3


def test_persist_requires_all_six_files_before_pointer(tmp_path):
    root = tmp_path / "direct_full_state"
    state = root / "contract_abcd"
    train, val = _frames()
    with pytest.raises(RuntimeError, match="training_contract"):
        persist_direct_prepare(
            state, train=train, validation=val,
            summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
            target_audit={"passed": True}, durable_state_root=root,
        )
    assert not (root / "LATEST_PREPARE.json").is_file()
    _persist(state, train, val, durable_state_root=root)
    for name in DIRECT_PREPARE_FILES:
        assert (state / name).is_file()
    assert (root / "LATEST_PREPARE.json").is_file()


def test_persist_interrupted_keeps_lkg_and_does_not_publish_new_pointer(tmp_path, monkeypatch):
    import os
    from src.direct_data import DIRECT_TRAIN_CSV

    root = tmp_path / "direct_full_state"
    state = root / "contract_abcd"
    train, val = _frames()
    _persist(state, train, val, durable_state_root=root)
    old_ptr = (root / "LATEST_PREPARE.json").read_text(encoding="utf-8")
    old_train = (state / DIRECT_TRAIN_CSV).read_text()
    new_train = train.copy()
    new_train.loc[0, "text_vi_norm"] = "changed"
    real_replace = os.replace
    n = {"c": 0}

    def boom(src, dst):
        if ".staging_prepare" in str(src):
            n["c"] += 1
            if n["c"] == 2:
                raise OSError("simulated interrupt")
        return real_replace(src, dst)

    monkeypatch.setattr("src.direct_data.os.replace", boom)
    with pytest.raises(OSError, match="simulated interrupt"):
        _persist(state, new_train, val, durable_state_root=root)
    assert (state / DIRECT_TRAIN_CSV).read_text() == old_train
    assert (root / "LATEST_PREPARE.json").read_text(encoding="utf-8") == old_ptr


def test_final_sync_keeps_best_when_not_latest_and_evaluate_restores_it(tmp_path):
    from src.asr_full_train import (
        FULL_TRAIN_MARKER,
        experiment_checkpoint_dir,
        resolve_snapshot_checkpoints,
        sync_experiment_checkpoints_to_durable,
        write_checkpoint_fingerprint,
    )

    exp_id = "direct_exp"
    local = experiment_checkpoint_dir(tmp_path / "local", exp_id, kind=FULL_TRAIN_MARKER)
    durable = tmp_path / "durable"
    durable.mkdir()
    contract = {
        "direct_training_contract_hash": "abc",
        "experiment_id": exp_id,
        "hparams": {"x": 1},
    }
    write_checkpoint_fingerprint(
        local, experiment_id=exp_id, kind=FULL_TRAIN_MARKER, global_step=200,
        extra=contract, overwrite=True,
    )
    for step in (50, 100, 200):
        _fake_complete_checkpoint(local / f"checkpoint-{step}", step)
    validated = validate_direct_best_checkpoint(
        experiment_dir=local, experiment_id=exp_id,
        best_checkpoint="checkpoint-50", expected_contract=contract,
    )
    assert validated["best_checkpoint_name"] == "checkpoint-50"
    snap = sync_experiment_checkpoints_to_durable(
        local, durable, experiment_id=exp_id, kind=FULL_TRAIN_MARKER,
        best_checkpoint_name=validated["best_checkpoint_name"],
        save_total_limit=2, budget_bytes=10 ** 12, require_durable=True,
    )
    names = {p.name for p in resolve_snapshot_checkpoints(snap)}
    assert "checkpoint-50" in names
    assert "checkpoint-200" in names
    restored = restore_direct_best_checkpoint(
        state_dir=durable, experiment_id=exp_id,
        train_summary={"best_checkpoint": "checkpoint-50", "best_checkpoint_name": "checkpoint-50"},
        local_experiment_dir=local, expected_contract=contract,
    )
    assert Path(restored).name == "checkpoint-50"


def test_monitor_target_change_is_detected():
    val = pd.DataFrame({
        "record_uid": ["u1", "u2", "u3"],
        "text_vi_norm": ["xin chao", "cam on", "tam biet"],
        "split": ["validation"] * 3,
        "source_split": ["validation"] * 3,
    })
    monitor = val.iloc[:2].copy()
    same = monitor_manifest(monitor)
    monitor_changed = monitor.copy()
    monitor_changed.loc[0, "text_vi_norm"] = "NOI DUNG DA BI THAY DOI"
    changed = monitor_manifest(monitor_changed)
    assert same["uid_set_hash"] == changed["uid_set_hash"]
    assert same["n"] == changed["n"]
    assert same["pair_hash"] != changed["pair_hash"]
    a = build_direct_training_contract(**_train_kw(monitor_pair_hash=same["pair_hash"], monitor_uid_set_hash=same["uid_set_hash"]))
    b = build_direct_training_contract(**_train_kw(monitor_pair_hash=changed["pair_hash"], monitor_uid_set_hash=changed["uid_set_hash"]))
    assert a["direct_training_contract_hash"] != b["direct_training_contract_hash"]


def test_monitor_must_match_locked_validation_subset():
    val = pd.DataFrame({
        "record_uid": [f"u{i}" for i in range(8)],
        "text_vi_norm": [f"t{i}" for i in range(8)],
        "split": ["validation"] * 8,
        "source_split": ["validation"] * 8,
    })
    monitor = fixed_subset(val, 3, seed=42)
    assert_monitor_matches_validation(monitor, val, size=3, seed=42)
    poisoned = monitor.copy()
    poisoned.loc[poisoned.index[0], "text_vi_norm"] = "TAMPERED"
    with pytest.raises(RuntimeError, match="monitor drifted"):
        assert_monitor_matches_validation(poisoned, val, size=3, seed=42)


def test_monitor_float_nan_duration_roundtrip_passes(tmp_path):
    from src.data_utils import sha256_file
    from src.direct_data import DIRECT_VAL_CSV

    val = pd.DataFrame({
        "record_uid": [f"u{i}" for i in range(8)],
        "text_vi_norm": [f"t{i}" for i in range(8)],
        "split": ["validation"] * 8,
        "source_split": ["validation"] * 8,
        "duration_seconds": [1.0, float("nan"), 1.23456789012345, 2.5, 3.0, 4.0, 5.0, 6.0],
        "n_samples": [16000.0, 16000, 16001.5, 16000, 16000, 16000, 16000, 16000],
    })
    train = pd.DataFrame({
        "record_uid": ["t1"], "text_vi_norm": ["x"],
        "split": ["train"], "source_split": ["train"],
        "duration_seconds": [1.0], "n_samples": [16000],
    })
    monitor = fixed_subset(val, 3, seed=42)
    state = tmp_path / "st"
    pointer = persist_direct_prepare(
        state, train=train, validation=val,
        summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
        target_audit={"passed": True},
        training_contract=build_direct_training_contract(**_train_kw()),
        monitor=monitor,
    )
    monitor_path = state / DIRECT_MONITOR_CSV
    loaded_mon = pd.read_csv(monitor_path)
    loaded_val = pd.read_csv(state / DIRECT_VAL_CSV)
    assert_monitor_matches_validation(loaded_mon, loaded_val, size=3, seed=42)
    locked = pointer["training_contract"]["monitor_file_sha256"]
    assert_monitor_file_sha256(monitor_path, locked)
    assert sha256_file(monitor_path) == locked


def test_monitor_target_change_fails():
    val = pd.DataFrame({
        "record_uid": [f"u{i}" for i in range(8)],
        "text_vi_norm": [f"t{i}" for i in range(8)],
        "split": ["validation"] * 8,
        "source_split": ["validation"] * 8,
    })
    monitor = fixed_subset(val, 3, seed=42)
    poisoned = monitor.copy()
    poisoned.loc[poisoned.index[0], "text_vi_norm"] = "TAMPERED"
    with pytest.raises(RuntimeError, match="monitor drifted"):
        assert_monitor_matches_validation(poisoned, val, size=3, seed=42)


def test_monitor_order_change_fails():
    val = pd.DataFrame({
        "record_uid": [f"u{i}" for i in range(8)],
        "text_vi_norm": [f"t{i}" for i in range(8)],
        "split": ["validation"] * 8,
        "source_split": ["validation"] * 8,
    })
    monitor = fixed_subset(val, 3, seed=42)
    reordered = monitor.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="monitor drifted"):
        assert_monitor_matches_validation(reordered, val, size=3, seed=42)


def test_monitor_byte_tamper_fails(tmp_path):
    val = pd.DataFrame({
        "record_uid": [f"u{i}" for i in range(8)],
        "text_vi_norm": [f"t{i}" for i in range(8)],
        "split": ["validation"] * 8,
        "source_split": ["validation"] * 8,
        "duration_seconds": [1.0] * 8,
    })
    train = pd.DataFrame({
        "record_uid": ["t1"], "text_vi_norm": ["x"],
        "split": ["train"], "source_split": ["train"],
    })
    monitor = fixed_subset(val, 3, seed=42)
    state = tmp_path / "st"
    pointer = persist_direct_prepare(
        state, train=train, validation=val,
        summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
        target_audit={"passed": True},
        training_contract=build_direct_training_contract(**_train_kw()),
        monitor=monitor,
    )
    monitor_path = state / DIRECT_MONITOR_CSV
    locked = pointer["training_contract"]["monitor_file_sha256"]
    lines = monitor_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0] + " "
    monitor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file SHA-256 mismatch"):
        assert_monitor_file_sha256(monitor_path, locked)


def test_persist_locks_staged_monitor_file_sha256(tmp_path):
    from src.data_utils import sha256_file

    train, val = _frames()
    state = tmp_path / "st"
    incoming = build_direct_training_contract(**_train_kw(monitor_file_sha256="0" * 64))
    pointer = persist_direct_prepare(
        state, train=train, validation=val,
        summary={"status": "SUCCESS_DIRECT_PREPARE", "contract_hash": "abcd"},
        target_audit={"passed": True},
        training_contract=incoming, monitor=val,
    )
    actual = sha256_file(state / DIRECT_MONITOR_CSV)
    locked = pointer["training_contract"]
    assert locked["monitor_file_sha256"] == actual
    assert locked["direct_training_contract_hash"] != incoming["direct_training_contract_hash"]
    on_disk = json.loads((state / DIRECT_TRAINING_CONTRACT).read_text(encoding="utf-8"))
    assert on_disk["monitor_file_sha256"] == actual
    assert pointer["summary"]["direct_training_contract_hash"] == locked["direct_training_contract_hash"]


def test_training_contract_binds_source_fingerprint():
    a = build_direct_training_contract(**_train_kw(source_fingerprint_sha256="a" * 64))
    b = build_direct_training_contract(**_train_kw(source_fingerprint_sha256="b" * 64))
    assert a["direct_training_contract_hash"] != b["direct_training_contract_hash"]


def test_local_checkpoint_peak_is_save_total_limit_plus_one():
    from src.direct_data import local_checkpoint_peak_copies
    assert local_checkpoint_peak_copies(2) == 3
    assert local_checkpoint_peak_copies(3) == 4


def _nb03_rq1_manifests(tmp_path: Path):
    train_manifest = tmp_path / "rq1_train.csv"
    val_manifest = tmp_path / "rq1_validation.csv"
    pd.DataFrame({"record_uid": ["u1", "u2"], "text_vi": ["xin chao", "tam biet"]}).to_csv(train_manifest, index=False)
    pd.DataFrame({"record_uid": ["v1"], "text_vi": ["cam on"]}).to_csv(val_manifest, index=False)
    return train_manifest, val_manifest


def _nb03_actual_pins(state: Path) -> dict:
    from src.asr_full_data import ELIGIBLE_TRAIN_CSV, ELIGIBLE_VAL_CSV
    from src.data_utils import compute_uid_set_hash, sha256_file

    train_p = state / ELIGIBLE_TRAIN_CSV
    val_p = state / ELIGIBLE_VAL_CSV
    return {
        "expected_asr_train_uid_set_hash": compute_uid_set_hash(pd.read_csv(train_p)),
        "expected_asr_validation_uid_set_hash": compute_uid_set_hash(pd.read_csv(val_p)),
        "expected_asr_train_file_sha256": sha256_file(train_p),
        "expected_asr_validation_file_sha256": sha256_file(val_p),
    }


def test_nb03_eligible_uid_hash_mismatch_fails(tmp_path):
    state = _write_asr_state(tmp_path)
    train_manifest, val_manifest = _nb03_rq1_manifests(tmp_path)
    with pytest.raises(RuntimeError, match="UID set hash mismatch"):
        load_asr_matched_direct_frames(
            asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
            expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
            expected_asr_train_uid_set_hash="a" * 64,
        )


def test_nb03_production_counts_0_of_4_pins_fail(tmp_path):
    from src.direct_contract import LOCKED_NB03_TRAIN_COUNT, LOCKED_NB03_VALIDATION_COUNT
    state = _write_asr_state(tmp_path)
    train_manifest, val_manifest = _nb03_rq1_manifests(tmp_path)
    with pytest.raises(RuntimeError, match="not fully pinned"):
        load_asr_matched_direct_frames(
            asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
            expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
            expected_train_count=LOCKED_NB03_TRAIN_COUNT,
            expected_validation_count=LOCKED_NB03_VALIDATION_COUNT,
        )


@pytest.mark.parametrize("keep", [1, 2, 3])
def test_nb03_production_counts_partial_pins_fail(tmp_path, keep):
    from src.direct_contract import LOCKED_NB03_TRAIN_COUNT, LOCKED_NB03_VALIDATION_COUNT
    state = _write_asr_state(tmp_path)
    train_manifest, val_manifest = _nb03_rq1_manifests(tmp_path)
    pins = _nb03_actual_pins(state)
    names = list(pins.keys())
    kwargs = {name: pins[name] for name in names[:keep]}
    with pytest.raises(RuntimeError, match="not fully pinned"):
        load_asr_matched_direct_frames(
            asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
            expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
            expected_train_count=LOCKED_NB03_TRAIN_COUNT,
            expected_validation_count=LOCKED_NB03_VALIDATION_COUNT,
            **kwargs,
        )


def test_nb03_four_pins_correct_pass(tmp_path):
    state = _write_asr_state(tmp_path)
    train_manifest, val_manifest = _nb03_rq1_manifests(tmp_path)
    out = load_asr_matched_direct_frames(
        asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
        expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
        expected_train_count=2, expected_validation_count=1,
        **_nb03_actual_pins(state),
    )
    assert out["train"]["record_uid"].tolist() == ["u1", "u2"]
    assert out["validation"]["record_uid"].tolist() == ["v1"]


def test_nb03_production_four_pins_correct_then_count_fails(tmp_path):
    from src.direct_contract import LOCKED_NB03_TRAIN_COUNT, LOCKED_NB03_VALIDATION_COUNT
    state = _write_asr_state(tmp_path)
    train_manifest, val_manifest = _nb03_rq1_manifests(tmp_path)
    with pytest.raises(RuntimeError, match="Unexpected NB03 eligible train count"):
        load_asr_matched_direct_frames(
            asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
            expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
            expected_train_count=LOCKED_NB03_TRAIN_COUNT,
            expected_validation_count=LOCKED_NB03_VALIDATION_COUNT,
            **_nb03_actual_pins(state),
        )


def test_nb03_production_one_wrong_hash_fails(tmp_path):
    from src.direct_contract import LOCKED_NB03_TRAIN_COUNT, LOCKED_NB03_VALIDATION_COUNT
    state = _write_asr_state(tmp_path)
    train_manifest, val_manifest = _nb03_rq1_manifests(tmp_path)
    pins = _nb03_actual_pins(state)
    pins["expected_asr_train_file_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="file sha256 mismatch"):
        load_asr_matched_direct_frames(
            asr_state_dir=state, train_manifest=train_manifest, validation_manifest=val_manifest,
            expected_asr_contract_hash=LOCKED_ASR_PREPARE_CONTRACT_HASH,
            expected_train_count=LOCKED_NB03_TRAIN_COUNT,
            expected_validation_count=LOCKED_NB03_VALIDATION_COUNT,
            **pins,
        )


def test_locked_runtime_rejects_wrong_transformers():
    with pytest.raises(RuntimeError, match="runtime version pin"):
        assert_locked_direct_runtime({
            "torch_version": "2.8.0",
            "transformers_version": "4.49.0",
            "accelerate_version": "1.10.1",
        })
    assert_locked_direct_runtime({
        "torch_version": "2.8.0+cu128",
        "transformers_version": "4.57.6",
        "accelerate_version": "1.10.1",
    })


def test_copy_mbart_seq2seq_embeddings_copies_shared_and_lm_head(monkeypatch):
    import torch

    class Weight:
        def __init__(self, fill, shape=(2, 3)):
            self.data = torch.full(shape, float(fill))
            self.shape = self.data.shape

    class Linear:
        def __init__(self, fill, shape=(2, 3)):
            self.weight = Weight(fill, shape=shape)

    class Seq2Seq:
        def __init__(self):
            self.model = type("M", (), {})()
            self.model.shared = Linear(7.0)
            self.lm_head = Linear(9.0)

    class Decoder:
        def __init__(self, shape=(2, 3)):
            self._emb = Linear(0.0, shape=shape)
            self.lm_head = Linear(0.0, shape=shape)

        def get_input_embeddings(self):
            return self._emb

    monkeypatch.setattr(
        "transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration.from_pretrained",
        classmethod(lambda cls, *a, **k: Seq2Seq()),
    )
    dec = Decoder()
    copy_mbart_seq2seq_embeddings(dec, decoder_id=LOCKED_DECODER_ID, decoder_revision=LOCKED_DECODER_REVISION)
    assert torch.equal(dec.get_input_embeddings().weight.data, torch.full((2, 3), 7.0))
    assert torch.equal(dec.lm_head.weight.data, torch.full((2, 3), 9.0))


def test_copy_mbart_seq2seq_embeddings_shape_mismatch_fails(monkeypatch):
    import torch

    class Weight:
        def __init__(self, fill, shape):
            self.data = torch.full(shape, float(fill))
            self.shape = self.data.shape

    class Linear:
        def __init__(self, fill, shape):
            self.weight = Weight(fill, shape)

    class Seq2Seq:
        def __init__(self):
            self.model = type("M", (), {})()
            self.model.shared = Linear(7.0, (4, 3))
            self.lm_head = Linear(9.0, (4, 3))

    class Decoder:
        def __init__(self):
            self._emb = Linear(0.0, (2, 3))
            self.lm_head = Linear(0.0, (2, 3))

        def get_input_embeddings(self):
            return self._emb

    monkeypatch.setattr(
        "transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration.from_pretrained",
        classmethod(lambda cls, *a, **k: Seq2Seq()),
    )
    with pytest.raises(RuntimeError, match="embed_tokens shape"):
        copy_mbart_seq2seq_embeddings(
            Decoder(), decoder_id=LOCKED_DECODER_ID, decoder_revision=LOCKED_DECODER_REVISION
        )


def test_load_direct_model_sets_trainer_accepts_loss_kwargs_false(monkeypatch, tmp_path):
    """Trainer must read model.accepts_loss_kwargs=False — not a decoder.forward monkey-patch."""
    import torch
    import torch.nn as nn
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    class Tok:
        lang_code_to_id = {LOCKED_TARGET_LANG: 250007}
        eos_token_id = 2
        pad_token_id = 1

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.frozen = False

        def freeze_feature_encoder(self):
            self.frozen = True

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, **kwargs):
            if "num_items_in_batch" in kwargs:
                raise TypeError(
                    "MBartForCausalLM.forward() got an unexpected keyword argument 'num_items_in_batch'"
                )
            return kwargs

    class Cfg:
        def __init__(self):
            self.decoder = type("D", (), {"vocab_size": 10})()
            self.pad_token_id = 1
            self.eos_token_id = 2
            self.decoder_start_token_id = 2
            self.vocab_size = 10
            self.is_encoder_decoder = True
            self.use_cache = False

    class Gen:
        decoder_start_token_id = 2
        pad_token_id = 1
        eos_token_id = 2
        forced_bos_token_id = 0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.config = Cfg()
            self.generation_config = Gen()

        def forward(self, **kwargs):
            # VAR_KEYWORD present: without accepts_loss_kwargs Trainer would inject loss kwargs.
            return type("O", (), {"loss": torch.tensor(0.0, requires_grad=True)})()

    monkeypatch.setattr(
        "transformers.SpeechEncoderDecoderModel.from_encoder_decoder_pretrained",
        staticmethod(lambda *a, **k: Model()),
    )
    monkeypatch.setattr("src.direct_model.copy_mbart_seq2seq_embeddings", lambda *a, **k: None)
    model = load_direct_model(
        encoder_id=LOCKED_ENCODER_ID, encoder_revision=LOCKED_ENCODER_REVISION,
        decoder_id=LOCKED_DECODER_ID, decoder_revision=LOCKED_DECODER_REVISION,
        tokenizer=Tok(), target_lang=LOCKED_TARGET_LANG, freeze_feature_encoder=True,
    )
    assert model.accepts_loss_kwargs is False
    assert getattr(model.encoder, "frozen", False) is True

    class TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "input_values": torch.zeros(8, dtype=torch.float32),
                "labels": torch.tensor([1, 2], dtype=torch.long),
            }

    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=1,
        max_steps=1,
        report_to=[],
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=TinyDS(),
    )
    assert trainer.model_accepts_loss_kwargs is False


def test_implementation_report_fingerprint_matches_live():
    root = Path(__file__).resolve().parents[1]
    report = (root / "IMPLEMENTATION_REPORT.md").read_text(encoding="utf-8")
    live = compute_source_fingerprint(root)["aggregate_sha256"]
    assert live in report, "IMPLEMENTATION_REPORT.md must be regenerated after the last source change"
