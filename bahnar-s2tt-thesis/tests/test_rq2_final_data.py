"""NB14 data-identity and equal-budget tests."""
from __future__ import annotations

import pytest

from src.rq2_final_contract import ARM_D0, ARM_QUALITY, ARM_RANDOM, ArmIsolationError, EvaluationError, Rq2FinalError, UpstreamGateError
from src.rq2_final_data import (
    assert_no_test_contamination,
    compose_arm_training_rows,
    data_contract_payload,
    load_frozen_supervised_splits,
    load_nb13_arm_manifest,
    load_pinned_nb13_arm_manifest,
    pair_identity_hash,
    validation_identity,
    verify_equal_budget_from_nb13,
    write_arm_manifest_csv,
)
from src.rq2_selection import verify_published_selection
from src.rq2_selection_contract import SELECTION_RELATIVE_DIR
from tests.rq2_nb14_fixtures import supervised_rows, validation_frame, world


def _compose(tmp_path, arm):
    env = world(tmp_path)
    from src.rq2_final_contract import verify_upstream_rq2

    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    selection = verify_published_selection(
        tmp_path / SELECTION_RELATIVE_DIR,
        project_root=tmp_path,
        durable_root=tmp_path,
        generation_id=upstream["nb13_generation_id"],
    )
    budget = verify_equal_budget_from_nb13(selection)
    pseudo = [] if arm == ARM_D0 else load_pinned_nb13_arm_manifest(tmp_path, arm=arm, upstream=upstream)
    composition = compose_arm_training_rows(
        arm=arm,
        supervised_rows=supervised_rows(),
        pseudo_rows=pseudo,
        selection_contract_sha256=upstream["nb13_selection_contract_sha256"],
        nb12_contract_sha256=upstream["nb12_contract_sha256"],
    )
    sha = write_arm_manifest_csv(tmp_path / f"{arm}.csv", composition["rows"])
    data = data_contract_payload(
        composition,
        manifest_sha256=sha,
        validation=validation_identity(validation_frame()),
        upstream=upstream,
        budget=budget,
    )
    return composition, data, budget, upstream


def test_d0_is_supervised_only(tmp_path):
    composition, data, budget, _ = _compose(tmp_path, ARM_D0)
    assert composition["pseudo"]["n_rows"] == 0
    assert composition["supervised"]["n_rows"] == 2
    assert data["realized_pseudo_duration_seconds"] == 0.0
    assert budget["target_budget_seconds"] == budget["d_random_realized_seconds"] + budget["d_random_unused_seconds"]


def test_random_manifest_cannot_be_used_as_quality(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import verify_upstream_rq2

    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    rows = load_pinned_nb13_arm_manifest(tmp_path, arm=ARM_RANDOM, upstream=upstream)
    with pytest.raises(ArmIsolationError):
        compose_arm_training_rows(
            arm=ARM_QUALITY,
            supervised_rows=supervised_rows(),
            pseudo_rows=rows,
            selection_contract_sha256=upstream["nb13_selection_contract_sha256"],
            nb12_contract_sha256=upstream["nb12_contract_sha256"],
        )


def test_changed_uid_order_changes_hash(tmp_path):
    composition, data, _, _ = _compose(tmp_path, ARM_RANDOM)
    reversed_rows = list(reversed(composition["rows"]))
    from src.rq2_final_data import ordered_example_uid_hash

    assert ordered_example_uid_hash([r["example_uid"] for r in reversed_rows]) != data["ordered_training_uid_hash"]


def test_changed_target_text_with_same_uid_fails_identity(tmp_path):
    composition, data, _, _ = _compose(tmp_path, ARM_RANDOM)
    mutated = [dict(row) for row in composition["rows"]]
    mutated[0]["target_text_norm"] = mutated[0]["target_text_norm"] + " extra"
    assert pair_identity_hash(mutated) != data["training_pair_hash"]


def test_changed_file_hash_is_visible(tmp_path):
    composition, data, _, _ = _compose(tmp_path, ARM_QUALITY)
    path = tmp_path / "d_quality.csv"
    original = data["data_manifest_sha256"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    from src.rq1_contract import sha256_file

    assert sha256_file(path) != original


def test_equal_budget_obeys_nb13(tmp_path):
    _, data, budget, _ = _compose(tmp_path, ARM_RANDOM)
    assert data["selection_budget_seconds"] == budget["target_budget_seconds"]
    assert budget["d_random_realized_seconds"] <= budget["target_budget_seconds"]
    assert budget["d_quality_realized_seconds"] <= budget["target_budget_seconds"]


def test_validation_rejects_test_split():
    import pandas as pd

    frame = pd.DataFrame([
        {"record_uid": "x", "text_vi": "a", "split": "test", "group_id": "g"},
    ])
    with pytest.raises(EvaluationError):
        validation_identity(frame)


def test_train_test_contamination_fails():
    with pytest.raises(EvaluationError):
        assert_no_test_contamination(["a", "t-1"], ["t-1", "t-2"])


def test_rq1_split_loader_refuses_g_test(tmp_path):
    from src.rq2_final_data import load_rq1_split_csv

    with pytest.raises(EvaluationError):
        load_rq1_split_csv(tmp_path, split="g_test")


def test_changed_train_text_same_uid_fails_d0_identity(tmp_path):
    env = world(tmp_path)
    identity = env["d0_contract"]
    from src.rq2_final_contract import bind_frozen_d0_identity

    d0 = bind_frozen_d0_identity(env["d0_dir"], project_root=tmp_path)
    path = tmp_path / "data" / "manifests" / "rq1_train.csv"
    frame = __import__("pandas").read_csv(path)
    frame.loc[0, "text_vi"] = str(frame.loc[0, "text_vi"]) + " extra"
    if "text_vi_norm" in frame.columns:
        frame.loc[0, "text_vi_norm"] = str(frame.loc[0, "text_vi_norm"]) + " extra"
    frame.to_csv(path, index=False)
    with pytest.raises(UpstreamGateError, match="UID->text"):
        load_frozen_supervised_splits(tmp_path, d0_identity=d0)


def test_changed_train_order_fails_d0_identity(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import bind_frozen_d0_identity

    d0 = bind_frozen_d0_identity(env["d0_dir"], project_root=tmp_path)
    path = tmp_path / "data" / "manifests" / "rq1_train.csv"
    frame = __import__("pandas").read_csv(path)
    frame.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(UpstreamGateError, match="ordered UID"):
        load_frozen_supervised_splits(tmp_path, d0_identity=d0)


def test_changed_validation_reference_fails_d0_identity(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import bind_frozen_d0_identity

    d0 = bind_frozen_d0_identity(env["d0_dir"], project_root=tmp_path)
    path = tmp_path / "data" / "manifests" / "rq1_validation.csv"
    frame = __import__("pandas").read_csv(path)
    frame.loc[0, "text_vi"] = "tham chiếu đã đổi"
    if "text_vi_norm" in frame.columns:
        frame.loc[0, "text_vi_norm"] = "tham chiếu đã đổi"
    frame.to_csv(path, index=False)
    with pytest.raises(UpstreamGateError, match="UID->reference"):
        load_frozen_supervised_splits(tmp_path, d0_identity=d0)


def test_nb13_pin_ignores_current_and_rejects_sha_mismatch(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import verify_upstream_rq2

    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    gen_a = upstream["nb13_generation_id"]
    rows = load_nb13_arm_manifest(
        tmp_path,
        arm=ARM_RANDOM,
        generation_id=gen_a,
        expected_sha256=upstream["d_random_manifest_sha256"],
    )
    assert rows
    out = tmp_path / SELECTION_RELATIVE_DIR
    (out / "CURRENT").write_text("generation-B\n", encoding="utf-8")
    again = load_pinned_nb13_arm_manifest(tmp_path, arm=ARM_RANDOM, upstream=upstream)
    assert [r["segment_uid"] for r in again] == [r["segment_uid"] for r in rows]
    path = out / "generations" / gen_a / "d_random_manifest.csv"
    path.write_text(path.read_text(encoding="utf-8") + "x", encoding="utf-8")
    with pytest.raises(UpstreamGateError, match="SHA256"):
        load_nb13_arm_manifest(
            tmp_path,
            arm=ARM_RANDOM,
            generation_id=gen_a,
            expected_sha256=upstream["d_random_manifest_sha256"],
        )


def test_strong_identity_without_optional_sidecar(tmp_path):
    env = world(tmp_path)
    sidecar = env["d0_dir"] / "direct_supervised_identity.json"
    assert sidecar.is_file()
    sidecar.unlink()
    from src.rq2_final_contract import bind_frozen_d0_identity

    d0 = bind_frozen_d0_identity(env["d0_dir"], project_root=tmp_path)
    loaded = load_frozen_supervised_splits(tmp_path, d0_identity=d0)
    assert loaded["identity"]["train"]["pair_hash"] == d0["train_pair_hash"]
    assert d0["train_ordered_uid_hash"]
    assert d0["train_audio_pair_hash"]


def test_changed_train_audio_same_uid_fails_d0_identity(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import bind_frozen_d0_identity

    d0 = bind_frozen_d0_identity(env["d0_dir"], project_root=tmp_path)
    path = tmp_path / "data" / "manifests" / "rq1_train.csv"
    frame = __import__("pandas").read_csv(path)
    frame.loc[0, "pcm16_sha256"] = "ff" * 32
    frame.to_csv(path, index=False)
    with pytest.raises(UpstreamGateError, match="UID->audio"):
        load_frozen_supervised_splits(tmp_path, d0_identity=d0)


def test_changed_train_manifest_bytes_fail_d0_identity(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import bind_frozen_d0_identity

    d0 = bind_frozen_d0_identity(env["d0_dir"], project_root=tmp_path)
    path = tmp_path / "data" / "manifests" / "rq1_train.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(UpstreamGateError, match="manifest SHA256|UID->text|ordered UID"):
        load_frozen_supervised_splits(tmp_path, d0_identity=d0)


def test_nb13_current_change_does_not_affect_budget_or_manifest(tmp_path):
    env = world(tmp_path)
    from src.rq2_final_contract import verify_upstream_rq2
    from src.rq2_final_data import verify_pinned_nb13_selection

    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    gen_a = upstream["nb13_generation_id"]
    first = verify_pinned_nb13_selection(tmp_path, generation_id=gen_a)
    budget_a = verify_equal_budget_from_nb13(first)
    rows_a = load_pinned_nb13_arm_manifest(tmp_path, arm=ARM_RANDOM, upstream=upstream)
    (tmp_path / SELECTION_RELATIVE_DIR / "CURRENT").write_text("generation-B\n", encoding="utf-8")
    second = verify_pinned_nb13_selection(tmp_path, generation_id=gen_a)
    budget_b = verify_equal_budget_from_nb13(second)
    rows_b = load_pinned_nb13_arm_manifest(tmp_path, arm=ARM_RANDOM, upstream=upstream)
    assert budget_a == budget_b
    assert first["contract"]["selection_contract_sha256"] == second["contract"]["selection_contract_sha256"]
    assert [r["segment_uid"] for r in rows_a] == [r["segment_uid"] for r in rows_b]
    assert second["summary"]["generation_id"] == gen_a


def _production_rq1_row(uid, *, split="train", index=0):
    return {
        "record_uid": uid,
        "record_id": uid,
        "audio_path": f"audio/{uid}.flac",
        "parquet_file": "default/train/0000.parquet",
        "shard_row_index": index,
        "duration_seconds": 1.0,
        "group_id": "gA",
        "text_vi": "câu huấn luyện",
        "split": split,
        "source_split": "g_train" if split == "train" else "g_validation",
    }


def _audio_index_record(uid, *, split="train", index=0, relpath=""):
    from src.asr_full_pcm import AUDIO_PCM_PIPELINE_VERSION

    pcm = ("a1" if uid.endswith("1") else "b2") * 32
    return {
        "record_uid": uid,
        "sha256_pcm": pcm,
        "sha256_source": "cd" * 32,
        "n_samples": 1600,
        "sample_rate": 16000,
        "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
        "parquet_revision": "deadbeefcafebabe",
        "dataset_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "split": split,
        "shard_key": "default/train/0000.parquet",
        "shard_row_index": index,
        "local_cache_relpath": relpath or f"{uid}.wav",
    }


def test_rq1_audio_identity_fail_closed_and_bind(tmp_path):
    import json

    import pandas as pd

    from src.data_utils import safe_cache_filename
    from src.rq2_final_data import bind_supervised_audio_identity, load_rq1_uid_audio_identity

    index_dir = tmp_path / "audio_index"
    index_dir.mkdir()
    recs = [
        _audio_index_record("sup-1", index=0),
        _audio_index_record("sup-2", index=1),
    ]
    (index_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    audio_map = load_rq1_uid_audio_identity(tmp_path, audio_index_dir=index_dir)
    assert audio_map["sup-1"]["sha256_pcm"] == "a1" * 32
    frame = pd.DataFrame([_production_rq1_row("sup-1"), _production_rq1_row("sup-2", index=1)])
    bound = bind_supervised_audio_identity(frame, audio_map)
    assert "pcm16_sha256" not in frame.columns
    assert bound["pcm16_sha256"].tolist() == ["a1" * 32, "b2" * 32]
    missing = pd.DataFrame([_production_rq1_row("missing")])
    with pytest.raises(UpstreamGateError, match="missing from RQ1 audio identity"):
        bind_supervised_audio_identity(missing, audio_map)
    recs.append(_audio_index_record("sup-1", index=9))
    recs[-1]["sha256_pcm"] = "ff" * 32
    (index_dir / "dup.jsonl").write_text(json.dumps(recs[-1]) + "\n", encoding="utf-8")
    with pytest.raises(UpstreamGateError, match="duplicate"):
        load_rq1_uid_audio_identity(tmp_path, audio_index_dir=index_dir)


def test_production_schema_direct_dataset_resolves_audio(tmp_path):
    import json
    import wave

    import numpy as np
    import pandas as pd

    from src.data_utils import safe_cache_filename
    from src.direct_dataset import DirectSpeechTranslationDataset
    from src.rq2_final_data import bind_supervised_audio_identity, load_rq1_uid_audio_identity

    cache = tmp_path / "audio_cache"
    cache.mkdir()
    uid = "sup-1"
    rel = safe_cache_filename(uid)
    wav_path = cache / rel
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
    index_dir = tmp_path / "audio_index"
    index_dir.mkdir()
    rec = _audio_index_record(uid, relpath=rel)
    (index_dir / "train.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    audio_map = load_rq1_uid_audio_identity(tmp_path, audio_index_dir=index_dir)
    frame = bind_supervised_audio_identity(pd.DataFrame([_production_rq1_row(uid)]), audio_map)
    frame["text_vi_norm"] = frame["text_vi"].map(__import__("src.mt_normalize", fromlist=["normalize_mt_text_v1"]).normalize_mt_text_v1)
    assert "pcm16_sha256" not in pd.DataFrame([_production_rq1_row(uid)]).columns
    assert "audio_path" in frame.columns
    assert "parquet_file" in frame.columns
    assert "shard_row_index" in frame.columns

    class DummyFE:
        def __call__(self, wav, sampling_rate=16000, return_attention_mask=True):
            arr = np.asarray(wav, dtype=np.float32).reshape(1, -1)
            return {"input_values": arr, "attention_mask": np.ones_like(arr, dtype=np.int64)}

    class DummyTok:
        def __call__(self, text_target=None, max_length=256, truncation=True, add_special_tokens=True, **kwargs):
            return {"input_ids": [2, 3]}

    ds = DirectSpeechTranslationDataset(
        frame,
        feature_extractor=DummyFE(),
        tokenizer=DummyTok(),
        audio_cache_roots=[cache],
        sample_rate=16000,
        max_target_length=8,
    )
    item = ds[0]
    assert item["record_uid"] == uid
    assert len(item["input_values"]) > 0

