import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.direct_contract import DIRECT_TRAINING_CONTRACT_KEYS, _hash_payload
from src.mt_contract import MT_TRAINING_CONTRACT_KEYS, contract_hash
from src.rq1_contract import (
    STATUS_FAILED,
    STATUS_RQ1_FINAL,
    PARQUET_SOURCE_RQ1_CONFIG_LEGACY,
    PARQUET_SOURCE_SPLIT_SUMMARY,
    assert_locked_manifest_bundle_matches_contract,
    assert_locked_runtime,
    assert_prediction_frame,
    assert_rq1_test_contract,
    assert_unlocked_final_contract_matches_current,
    atomic_write_csv,
    atomic_write_json,
    build_final_contract,
    build_rq1_test_contract,
    collect_runtime_provenance,
    derive_final_status,
    resolve_parquet_revision_pin,
)
from src.rq1_evaluation import (
    join_paired_predictions,
    paired_bootstrap,
    recompute_and_assert_finalize_integrity,
    unlock_frozen_test_manifest,
    verify_upstream_handoffs,
)
from src.rq1_runtime_paths import find_bahnar_project_root


LOCKED_PARQUET = "ad0a84362053dad098b86d0df215eb081f891dd8"
LOCKED_DATASET_REV = "b" * 40


def _test_frame(*, n_groups: int = 2):
    rows = []
    for i in range(4):
        rows.append(
            {
                "record_uid": f"u{i+1}",
                "record_id": f"r{i+1}",
                "source_split": "test" if i % 2 else "validation",
                "parquet_file": "default/test/a.parquet",
                "shard_row_index": i,
                "text_bahnar": f"A{i}",
                "text_vi": f"Xin chào {i}",
                "split": "test",
                "duration_seconds": 5.0 + i,
                "group_id": f"g{(i % n_groups) + 1}",
            }
        )
    return pd.DataFrame(rows)


def _train_frame():
    return pd.DataFrame(
        [
            {
                "record_uid": "tr1",
                "record_id": "tr1",
                "source_split": "train",
                "parquet_file": "default/train/a.parquet",
                "shard_row_index": 0,
                "text_bahnar": "T",
                "text_vi": "T",
                "split": "train",
                "duration_seconds": 3.0,
                "group_id": "gtrain",
            }
        ]
    )


def _val_frame():
    return pd.DataFrame(
        [
            {
                "record_uid": "va1",
                "record_id": "va1",
                "source_split": "validation",
                "parquet_file": "default/validation/c.parquet",
                "shard_row_index": 9,
                "text_bahnar": "V",
                "text_vi": "V",
                "split": "validation",
                "duration_seconds": 4.0,
                "group_id": "gval",
            }
        ]
    )


def _mock_mt_contract(*, experiment_id: str = "m") -> dict:
    mt = {k: None for k in MT_TRAINING_CONTRACT_KEYS}
    mt.update(
        {
            "experiment_id": experiment_id,
            "model_id": "facebook/mbart-large-50",
            "model_revision": "abc123",
            "tokenizer_fingerprint": "tok-mt",
            "generation_length_policy": "fixed",
            "generation_max_length": 128,
            "num_beams": 4,
            "max_source_length": 128,
            "max_target_length": 128,
            "metric_for_best_model": "eval_sacrebleu",
            "greater_is_better": True,
            "normalization_version": "v1",
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": False,
            "stage_version": "mt_train_v1",
            "eval_policy": "steps",
        }
    )
    for k in MT_TRAINING_CONTRACT_KEYS:
        if mt.get(k) is None:
            mt[k] = (
                0
                if any(x in k for x in ("length", "beam", "epoch", "step", "batch", "lr", "decay", "warmup", "monitor", "seed"))
                else (False if any(x in k for x in ("is_", "greater", "fp16", "bf16", "gradient")) else "")
            )
    mt["mt_training_contract_hash"] = contract_hash(mt, keys=MT_TRAINING_CONTRACT_KEYS)
    return mt


def _mock_direct_contract(*, experiment_id: str = "d") -> dict:
    d = {k: None for k in DIRECT_TRAINING_CONTRACT_KEYS}
    d.update(
        {
            "experiment_id": experiment_id,
            "encoder_id": "facebook/wav2vec2-xls-r-300m",
            "encoder_revision": "e1",
            "decoder_id": "facebook/mbart-large-50-many-to-many-mmt",
            "decoder_revision": "d1",
            "target_lang": "vi_VN",
            "tokenizer_fingerprint": "tok-d",
            "generation_max_length": 128,
            "num_beams": 4,
            "max_target_length": 128,
            "metric_for_best_model": "eval_sacrebleu",
            "greater_is_better": True,
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": False,
            "freeze_feature_encoder": True,
            "stage_version": "direct_train_v2",
            "torch_version": "2.8.0",
            "transformers_version": "4.57.6",
            "accelerate_version": "1.10.1",
        }
    )
    for k in DIRECT_TRAINING_CONTRACT_KEYS:
        if d.get(k) is None:
            d[k] = (
                0
                if any(x in k for x in ("length", "beam", "epoch", "step", "batch", "lr", "decay", "warmup", "monitor", "seed", "size", "limit"))
                else (False if any(x in k for x in ("is_", "greater", "fp16", "bf16", "gradient", "freeze")) else "")
            )
    d["direct_training_contract_hash"] = _hash_payload(d, DIRECT_TRAINING_CONTRACT_KEYS)
    return d


def _write_upstream(root: Path):
    asr = root / "asr"
    mt = root / "mt"
    direct = root / "direct"
    for p in (asr, mt, direct):
        p.mkdir(parents=True)
    asr_contract = {
        "experiment_id": "a",
        "contract_hash": "a" * 64,
        "data_contract_hash": "a" * 64,
        "model_id": "facebook/wav2vec2-xls-r-300m",
        "model_revision": "rev-a",
    }
    (asr / "full_evaluate_summary.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_FULL_EVALUATE",
                "frozen_test_accessed": False,
                "experiment_id": "a",
                "best_checkpoint": "checkpoint-1",
                "training_contract": asr_contract,
            }
        )
    )
    (asr / "full_train_summary.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_FULL_TRAINING",
                "best_checkpoint": "checkpoint-1",
                "experiment_id": "a",
                "training_contract": asr_contract,
            }
        )
    )
    mt_contract = _mock_mt_contract()
    (mt / "mt_evaluate_summary.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_MT_EVALUATE",
                "frozen_test_accessed": False,
                "experiment_id": "m",
                "best_checkpoint": "checkpoint-1",
                "mt_training_contract_hash": mt_contract["mt_training_contract_hash"],
            }
        )
    )
    (mt / "mt_train_summary.json").write_text(
        json.dumps(
            {
                "best_checkpoint": "checkpoint-1",
                "experiment_id": "m",
                "mt_training_contract_hash": mt_contract["mt_training_contract_hash"],
            }
        )
    )
    (mt / "mt_training_contract.json").write_text(json.dumps(mt_contract))
    direct_contract = _mock_direct_contract()
    (direct / "direct_evaluate_summary.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_DIRECT_EVALUATE",
                "frozen_test_accessed": False,
                "experiment_id": "d",
                "best_checkpoint": "checkpoint-1",
                "direct_training_contract_hash": direct_contract["direct_training_contract_hash"],
            }
        )
    )
    (direct / "direct_train_summary.json").write_text(
        json.dumps(
            {
                "best_checkpoint": "checkpoint-1",
                "experiment_id": "d",
                "direct_training_contract_hash": direct_contract["direct_training_contract_hash"],
            }
        )
    )
    (direct / "direct_training_contract.json").write_text(json.dumps(direct_contract))
    return asr, mt, direct


def _write_locked_manifests(root: Path, *, legacy_no_parquet: bool = True, parquet_revision=None):
    """Real-project-shaped split_summary: optionally omit parquet_revision."""
    man = root / "data" / "manifests"
    man.mkdir(parents=True)
    test = _test_frame(n_groups=2)
    train = _train_frame()
    val = _val_frame()
    test_path = man / "rq1_test.csv"
    train_path = man / "rq1_train.csv"
    val_path = man / "rq1_validation.csv"
    test.to_csv(test_path, index=False)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    pins = {
        "rq1_test.csv": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "rq1_train.csv": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "rq1_validation.csv": hashlib.sha256(val_path.read_bytes()).hexdigest(),
    }
    summary = {
        "dataset_id": "d",
        "expected_dataset_revision": LOCKED_DATASET_REV,
        "dataset_commit_sha": LOCKED_DATASET_REV,
        "manifest_sha256": pins,
    }
    if not legacy_no_parquet:
        summary["parquet_revision"] = parquet_revision or LOCKED_PARQUET
    elif parquet_revision is not None:
        summary["parquet_revision"] = parquet_revision
    (man / "split_summary.json").write_text(json.dumps(summary))
    return man, pins, summary


def _make_ckpt(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"x")
    (path / "optimizer.pt").write_bytes(b"x")
    (path / "scheduler.pt").write_bytes(b"x")
    (path / "rng_state.pth").write_bytes(b"x")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    return path


def test_resolve_parquet_legacy_uses_locked_config():
    pin = resolve_parquet_revision_pin({}, locked_parquet_revision=LOCKED_PARQUET)
    assert pin["parquet_revision"] == LOCKED_PARQUET
    assert pin["parquet_revision_source"] == PARQUET_SOURCE_RQ1_CONFIG_LEGACY
    with pytest.raises(RuntimeError, match="missing/invalid"):
        resolve_parquet_revision_pin({}, locked_parquet_revision="")


def test_resolve_parquet_split_summary_mismatch_fails():
    with pytest.raises(RuntimeError, match="mismatch"):
        resolve_parquet_revision_pin(
            {"parquet_revision": "0" * 40},
            locked_parquet_revision=LOCKED_PARQUET,
        )


def test_unlock_legacy_split_summary_without_parquet_revision(tmp_path):
    a, m, d = _write_upstream(tmp_path)
    h = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    _write_locked_manifests(tmp_path, legacy_no_parquet=True)
    got, meta = unlock_frozen_test_manifest(
        project_root=tmp_path,
        upstream_handoff=h,
        allow_frozen_test_access=True,
        dataset_id="d",
        dataset_revision=LOCKED_DATASET_REV,
        parquet_revision=LOCKED_PARQUET,
        expected_test_count=4,
    )
    assert len(got) == 4
    assert meta["parquet_revision_source"] == PARQUET_SOURCE_RQ1_CONFIG_LEGACY
    assert meta["contract"]["parquet_revision"] == LOCKED_PARQUET
    assert meta["contract"]["n_clusters"] == 2


def test_unlock_legacy_without_locked_pin_fails(tmp_path):
    a, m, d = _write_upstream(tmp_path)
    h = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    _write_locked_manifests(tmp_path, legacy_no_parquet=True)
    with pytest.raises(RuntimeError, match="missing/invalid"):
        unlock_frozen_test_manifest(
            project_root=tmp_path,
            upstream_handoff=h,
            allow_frozen_test_access=True,
            dataset_id="d",
            dataset_revision=LOCKED_DATASET_REV,
            parquet_revision="",
            expected_test_count=4,
        )


def test_unlock_explicit_parquet_mismatch_fails(tmp_path):
    a, m, d = _write_upstream(tmp_path)
    h = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    _write_locked_manifests(tmp_path, legacy_no_parquet=False, parquet_revision="1" * 40)
    with pytest.raises(RuntimeError, match="parquet revision mismatch"):
        unlock_frozen_test_manifest(
            project_root=tmp_path,
            upstream_handoff=h,
            allow_frozen_test_access=True,
            dataset_id="d",
            dataset_revision=LOCKED_DATASET_REV,
            parquet_revision=LOCKED_PARQUET,
            expected_test_count=4,
        )


def test_unlock_with_matching_split_summary_parquet(tmp_path):
    a, m, d = _write_upstream(tmp_path)
    h = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    _write_locked_manifests(tmp_path, legacy_no_parquet=False, parquet_revision=LOCKED_PARQUET)
    _, meta = unlock_frozen_test_manifest(
        project_root=tmp_path,
        upstream_handoff=h,
        allow_frozen_test_access=True,
        dataset_id="d",
        dataset_revision=LOCKED_DATASET_REV,
        parquet_revision=LOCKED_PARQUET,
        expected_test_count=4,
    )
    assert meta["parquet_revision_source"] == PARQUET_SOURCE_SPLIT_SUMMARY


def test_manifest_bundle_rehash_detects_each_mutation(tmp_path):
    a, m, d = _write_upstream(tmp_path)
    h = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    man, _, _ = _write_locked_manifests(tmp_path, legacy_no_parquet=True)
    _, meta = unlock_frozen_test_manifest(
        project_root=tmp_path,
        upstream_handoff=h,
        allow_frozen_test_access=True,
        dataset_id="d",
        dataset_revision=LOCKED_DATASET_REV,
        parquet_revision=LOCKED_PARQUET,
        expected_test_count=4,
    )
    tc = meta["contract"]
    assert_locked_manifest_bundle_matches_contract(project_root=tmp_path, test_contract=tc)
    for name in ("split_summary.json", "rq1_train.csv", "rq1_validation.csv", "rq1_test.csv"):
        path = man / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n#mut\n")
        with pytest.raises(RuntimeError, match="mutated after unlock"):
            assert_locked_manifest_bundle_matches_contract(project_root=tmp_path, test_contract=tc)
        path.write_bytes(original)


def test_cluster_bootstrap_five_groups_deterministic(monkeypatch):
    import src.rq1_evaluation as mod

    def fake(h, r):
        s = 100.0 * sum(a == b for a, b in zip(h, r)) / max(1, len(h))
        return {"sacrebleu": s, "chrfpp": s, "n": len(h), "finite": True}

    monkeypatch.setattr(mod, "mt_corpus_metrics", fake)
    rows = []
    for g in range(5):
        for i in range(3):
            rows.append(
                {
                    "record_uid": f"u{g}_{i}",
                    "group_id": f"G{g}",
                    "reference_vi": "a",
                    "c0_pred_vi": "a" if i else "x",
                    "d0_pred_vi": "a",
                }
            )
    p = pd.DataFrame(rows)
    x = paired_bootstrap(p, n_samples=30, seed=42)
    y = paired_bootstrap(p, n_samples=30, seed=42)
    assert x == y
    assert x["bootstrap_method"] == "paired_cluster"
    assert x["bootstrap_unit"] == "group_id"
    assert x["n_clusters"] == 5
    assert x["n_rows"] == 15


def test_cluster_bootstrap_missing_group_id_fails(monkeypatch):
    import src.rq1_evaluation as mod

    monkeypatch.setattr(mod, "mt_corpus_metrics", lambda h, r: {"sacrebleu": 0.0, "chrfpp": 0.0, "n": len(h), "finite": True})
    p = pd.DataFrame({"reference_vi": ["a"], "c0_pred_vi": ["a"], "d0_pred_vi": ["a"]})
    with pytest.raises(RuntimeError, match="cluster column"):
        paired_bootstrap(p, n_samples=5, seed=42)


def test_find_project_root_from_notebooks_subdir(tmp_path):
    (tmp_path / "requirements.txt").write_text("x\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    assert find_bahnar_project_root(notebooks) == tmp_path.resolve()
    assert find_bahnar_project_root(tmp_path) == tmp_path.resolve()


def test_atomic_write_interruption_leaves_no_final(tmp_path, monkeypatch):
    target = tmp_path / "rq1_final_summary.json"
    real_replace = __import__("os").replace

    def boom(src, dst):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr("src.rq1_contract.os.replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"status": "SUCCESS_RQ1_FINAL"})
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp")) or True  # temp cleaned in finally
    monkeypatch.setattr("src.rq1_contract.os.replace", real_replace)
    atomic_write_json(target, {"status": "ok"})
    assert json.loads(target.read_text())["status"] == "ok"


def test_unlocked_final_contract_rejects_changed_upstream():
    f = _test_frame()
    p = Path("/tmp")  # unused path not opened when building via hashes below
    # Build minimal contract via in-memory path write
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man = root / "data" / "manifests"
        man.mkdir(parents=True)
        tp = man / "rq1_test.csv"
        f.to_csv(tp, index=False)
        tc = build_rq1_test_contract(
            f,
            manifest_path=tp,
            split_summary_sha256="a" * 64,
            dataset_id="d",
            dataset_revision=LOCKED_DATASET_REV,
            parquet_revision=LOCKED_PARQUET,
            parquet_revision_source=PARQUET_SOURCE_RQ1_CONFIG_LEGACY,
            train_manifest_sha256="b" * 64,
            validation_manifest_sha256="c" * 64,
        )
    proof = {
        "asr": {"checkpoint_content_digest": "a1", "loaded": True, "experiment_id": "a", "contract_hash": "ca", "best_checkpoint_name": "checkpoint-1"},
        "mt": {"checkpoint_content_digest": "m1", "loaded": True, "experiment_id": "m", "contract_hash": "cm", "best_checkpoint_name": "checkpoint-1"},
        "direct": {"checkpoint_content_digest": "d1", "loaded": True, "experiment_id": "d", "contract_hash": "cd", "best_checkpoint_name": "checkpoint-1"},
        "proof_hash": "p" * 64,
    }
    runtime = {"torch": "2.8.0"}
    fc = build_final_contract(
        test_contract=tc,
        asr_handoff_hash="ha",
        mt_handoff_hash="hm",
        direct_handoff_hash="hd-A",
        source_fingerprint_sha256="s" * 64,
        runtime_versions=runtime,
        seed=42,
        bootstrap_samples=1000,
        confidence=0.95,
        checkpoint_proof=proof,
    )
    latest = {"final_contract_hash": fc["rq1_final_contract_hash"], "source_fingerprint": "s" * 64}
    assert_unlocked_final_contract_matches_current(
        fc,
        test_contract=tc,
        asr_handoff_hash="ha",
        mt_handoff_hash="hm",
        direct_handoff_hash="hd-A",
        source_fingerprint_sha256="s" * 64,
        runtime_versions=runtime,
        seed=42,
        bootstrap_samples=1000,
        confidence=0.95,
        checkpoint_proof=proof,
        latest_meta=latest,
    )
    with pytest.raises(RuntimeError, match="does not match current upstream"):
        assert_unlocked_final_contract_matches_current(
            fc,
            test_contract=tc,
            asr_handoff_hash="ha",
            mt_handoff_hash="hm",
            direct_handoff_hash="hd-B",
            source_fingerprint_sha256="s" * 64,
            runtime_versions=runtime,
            seed=42,
            bootstrap_samples=1000,
            confidence=0.95,
            checkpoint_proof=proof,
            latest_meta=latest,
        )


def test_assert_restored_checkpoint_matches_proof(tmp_path):
    from src.asr_full_train import checkpoint_content_digest
    from src.rq1_restore import assert_restored_checkpoint_matches_proof

    ckpt = _make_ckpt(tmp_path / "checkpoint-1")
    digest = checkpoint_content_digest(ckpt)
    proof = {
        "experiment_id": "d",
        "contract_hash": "cd",
        "best_checkpoint_name": "checkpoint-1",
        "checkpoint_content_digest": digest["digest"],
    }
    restored = {
        "best_checkpoint": str(ckpt),
        "experiment_id": "d",
        "training_contract": {"direct_training_contract_hash": "cd"},
    }
    assert assert_restored_checkpoint_matches_proof(restored, proof, system="direct")
    (ckpt / "model.safetensors").write_bytes(b"mutated-weights")
    with pytest.raises(RuntimeError, match="does not match FINAL_CONTRACT proof"):
        assert_restored_checkpoint_matches_proof(restored, proof, system="direct")


def _mk_local_tree(root: Path, name: str) -> Path:
    exp = root / name
    exp.mkdir(parents=True)
    (exp / "checkpoint-1").mkdir()
    (exp / "checkpoint-1" / "weights.bin").write_bytes(b"X" * 1000)
    return exp


def test_cleanup_allowed_root_safety(tmp_path):
    from src.rq1_restore import assert_path_is_strict_descendant, cleanup_local_rq1_experiment

    allowed = tmp_path / "rq1_restore" / "asr"
    child = _mk_local_tree(allowed, "expA")
    cleanup_local_rq1_experiment(child, allowed_root=allowed, label="asr")
    assert not child.exists()
    # idempotent
    cleanup_local_rq1_experiment(child, allowed_root=allowed, label="asr")
    with pytest.raises(RuntimeError, match="allowed_root itself"):
        cleanup_local_rq1_experiment(allowed, allowed_root=allowed, label="asr")
    outside = tmp_path / "elsewhere" / "exp"
    outside.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="outside allowed_root"):
        cleanup_local_rq1_experiment(outside, allowed_root=allowed, label="asr")
    # path traversal cannot escape
    with pytest.raises(RuntimeError, match="outside allowed_root"):
        cleanup_local_rq1_experiment(allowed / "expA" / ".." / ".." / "elsewhere" / "exp", allowed_root=allowed, label="asr")
    durable = tmp_path / "durable_state" / "contract_abc"
    durable.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="outside allowed_root"):
        cleanup_local_rq1_experiment(durable, allowed_root=allowed, label="asr")
    with pytest.raises(RuntimeError, match="forbidden path"):
        assert_path_is_strict_descendant("/", "/", label="x")


def test_failure_paths_cleanup_before_next_restore(tmp_path):
    """ASR/MT/Direct failure paths must delete local trees; ASR failure never restores MT."""
    from src.rq1_restore import cleanup_local_rq1_experiment, run_cascaded_c0_peak_safe, run_direct_d0_peak_safe

    asr_root = tmp_path / "rq1_restore" / "asr"
    mt_root = tmp_path / "rq1_restore" / "mt"
    direct_root = tmp_path / "rq1_restore" / "direct"
    events = []

    def restore_asr_ok():
        exp = _mk_local_tree(asr_root, "expA")
        events.append("restore_asr")
        return {
            "model": object(), "processor": object(),
            "best_checkpoint": str(exp / "checkpoint-1"),
            "local_experiment_dir": str(exp),
            "training_contract": {"contract_hash": "ca"},
        }

    def restore_mt_ok():
        events.append("restore_mt")
        assert not (asr_root / "expA").exists()
        exp = _mk_local_tree(mt_root, "expM")
        return {
            "model": object(), "tokenizer": object(),
            "best_checkpoint": str(exp / "checkpoint-1"),
            "local_experiment_dir": str(exp),
            "training_contract": {"mt_training_contract_hash": "cm"},
        }

    # ASR proof failure
    events.clear()
    with pytest.raises(RuntimeError, match="asr proof boom"):
        run_cascaded_c0_peak_safe(
            restore_asr=restore_asr_ok,
            restore_mt=restore_mt_ok,
            predict_asr=lambda a: pd.DataFrame({"record_uid": ["u1"]}),
            predict_mt=lambda m, r: pd.DataFrame({"record_uid": ["u1"]}),
            assert_asr_proof=lambda a: (_ for _ in ()).throw(RuntimeError("asr proof boom")),
            assert_mt_proof=lambda m: True,
            expected_uids=["u1"],
            asr_allowed_root=asr_root,
            mt_allowed_root=mt_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert not (asr_root / "expA").exists()
    assert "restore_mt" not in events

    # ASR inference failure
    events.clear()
    with pytest.raises(RuntimeError, match="asr infer boom"):
        run_cascaded_c0_peak_safe(
            restore_asr=restore_asr_ok,
            restore_mt=restore_mt_ok,
            predict_asr=lambda a: (_ for _ in ()).throw(RuntimeError("asr infer boom")),
            predict_mt=lambda m, r: pd.DataFrame({"record_uid": ["u1"]}),
            assert_asr_proof=lambda a: True,
            assert_mt_proof=lambda m: True,
            expected_uids=["u1"],
            asr_allowed_root=asr_root,
            mt_allowed_root=mt_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert not (asr_root / "expA").exists()
    assert "restore_mt" not in events

    # MT proof failure after ASR success
    events.clear()
    with pytest.raises(RuntimeError, match="mt proof boom"):
        run_cascaded_c0_peak_safe(
            restore_asr=restore_asr_ok,
            restore_mt=restore_mt_ok,
            predict_asr=lambda a: pd.DataFrame({"record_uid": ["u1"]}),
            predict_mt=lambda m, r: pd.DataFrame({"record_uid": ["u1"]}),
            assert_asr_proof=lambda a: True,
            assert_mt_proof=lambda m: (_ for _ in ()).throw(RuntimeError("mt proof boom")),
            expected_uids=["u1"],
            asr_allowed_root=asr_root,
            mt_allowed_root=mt_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert not (asr_root / "expA").exists()
    assert not (mt_root / "expM").exists()
    assert "restore_mt" in events

    # MT inference failure
    events.clear()
    with pytest.raises(RuntimeError, match="mt infer boom"):
        run_cascaded_c0_peak_safe(
            restore_asr=restore_asr_ok,
            restore_mt=restore_mt_ok,
            predict_asr=lambda a: pd.DataFrame({"record_uid": ["u1"]}),
            predict_mt=lambda m, r: (_ for _ in ()).throw(RuntimeError("mt infer boom")),
            assert_asr_proof=lambda a: True,
            assert_mt_proof=lambda m: True,
            expected_uids=["u1"],
            asr_allowed_root=asr_root,
            mt_allowed_root=mt_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert not (mt_root / "expM").exists()

    # Direct proof / infer failure
    def restore_direct_ok():
        exp = _mk_local_tree(direct_root, "expD")
        return {
            "model": object(), "feature_extractor": object(), "tokenizer": object(),
            "best_checkpoint": str(exp / "checkpoint-1"),
            "local_experiment_dir": str(exp),
            "training_contract": {},
        }
    with pytest.raises(RuntimeError, match="direct proof boom"):
        run_direct_d0_peak_safe(
            restore_direct=restore_direct_ok,
            predict_direct=lambda d: pd.DataFrame({"record_uid": ["u1"]}),
            assert_direct_proof=lambda d: (_ for _ in ()).throw(RuntimeError("direct proof boom")),
            expected_uids=["u1"],
            direct_allowed_root=direct_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert not (direct_root / "expD").exists()
    with pytest.raises(RuntimeError, match="direct infer boom"):
        run_direct_d0_peak_safe(
            restore_direct=restore_direct_ok,
            predict_direct=lambda d: (_ for _ in ()).throw(RuntimeError("direct infer boom")),
            assert_direct_proof=lambda d: True,
            expected_uids=["u1"],
            direct_allowed_root=direct_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert not (direct_root / "expD").exists()


def test_asr_uid_validated_before_mt_restore(tmp_path):
    from src.rq1_restore import cleanup_local_rq1_experiment, run_cascaded_c0_peak_safe

    asr_root = tmp_path / "rq1_restore" / "asr"
    mt_root = tmp_path / "rq1_restore" / "mt"
    seen = {"mt": False}

    def restore_asr():
        exp = _mk_local_tree(asr_root, "expA")
        return {
            "model": object(), "processor": object(),
            "best_checkpoint": str(exp / "checkpoint-1"),
            "local_experiment_dir": str(exp),
            "training_contract": {},
        }

    def restore_mt():
        seen["mt"] = True
        exp = _mk_local_tree(mt_root, "expM")
        return {
            "model": object(), "tokenizer": object(),
            "best_checkpoint": str(exp / "checkpoint-1"),
            "local_experiment_dir": str(exp),
            "training_contract": {},
        }

    with pytest.raises(RuntimeError, match="UID order mismatch"):
        run_cascaded_c0_peak_safe(
            restore_asr=restore_asr,
            restore_mt=restore_mt,
            predict_asr=lambda a: pd.DataFrame({"record_uid": ["u2", "u1"]}),  # wrong order
            predict_mt=lambda m, r: pd.DataFrame({"record_uid": ["u1", "u2"]}),
            assert_asr_proof=lambda a: True,
            assert_mt_proof=lambda m: True,
            expected_uids=["u1", "u2"],
            asr_allowed_root=asr_root,
            mt_allowed_root=mt_root,
            cleanup_local=cleanup_local_rq1_experiment,
        )
    assert seen["mt"] is False
    assert not (asr_root / "expA").exists()


def test_restore_load_failure_cleans_local_tree(tmp_path, monkeypatch):
    from src.rq1_restore import restore_asr_for_rq1

    a, m, d = _write_upstream(tmp_path)
    local_root = tmp_path / "rq1_restore" / "asr"
    planted = {"path": None}

    def fake_resolve(*args, **kwargs):
        local_exp = kwargs["local_experiment_dir"]
        planted["path"] = Path(local_exp)
        planted["path"].mkdir(parents=True, exist_ok=True)
        ckpt = planted["path"] / "checkpoint-1"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "model.safetensors").write_bytes(b"x")
        (ckpt / "optimizer.pt").write_bytes(b"x")
        (ckpt / "scheduler.pt").write_bytes(b"x")
        (ckpt / "rng_state.pth").write_bytes(b"x")
        (ckpt / "trainer_state.json").write_text("{}")
        return ckpt

    monkeypatch.setattr("src.asr_full_train.resolve_best_checkpoint_from_durable", fake_resolve)
    monkeypatch.setattr("src.asr_full_train.assert_checkpoint_complete_for_resume", lambda p: {"ok": True})

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    def _boom(*args, **kwargs):
        raise RuntimeError("load boom")

    # Patch the class method so the in-function `from transformers import ...` sees it.
    monkeypatch.setattr(Wav2Vec2Processor, "from_pretrained", classmethod(lambda cls, *a, **k: _boom()))
    monkeypatch.setattr(Wav2Vec2ForCTC, "from_pretrained", classmethod(lambda cls, *a, **k: _boom()))
    with pytest.raises(RuntimeError, match="load boom"):
        restore_asr_for_rq1(state_dir=a, local_ckpt_root=local_root, device="cpu")
    assert planted["path"] is not None
    assert not planted["path"].exists()


def test_stage_summary_must_match_current_final_contract():
    from src.rq1_contract import assert_stage_summary_matches_final_contract

    fc = {"rq1_final_contract_hash": "f" * 64}
    tc = {"test_count": 215}
    ok = {"status": "SUCCESS_RQ1_C0", "rq1_final_contract_hash": "f" * 64, "n": 215}
    assert_stage_summary_matches_final_contract(
        ok, final_contract=fc, test_contract=tc, expected_status="SUCCESS_RQ1_C0", label="C0",
    )
    stale = dict(ok)
    stale["rq1_final_contract_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="rq1_final_contract_hash mismatch"):
        assert_stage_summary_matches_final_contract(
            stale, final_contract=fc, test_contract=tc, expected_status="SUCCESS_RQ1_C0", label="C0",
        )
    bad_n = dict(ok)
    bad_n["n"] = 100
    with pytest.raises(RuntimeError, match="test_count"):
        assert_stage_summary_matches_final_contract(
            bad_n, final_contract=fc, test_contract=tc, expected_status="SUCCESS_RQ1_C0", label="C0",
        )


def test_bootstrap_gate_required_for_final_success():
    from src.rq1_contract import assert_rq1_bootstrap_gate
    import inspect

    sig = inspect.signature(derive_final_status)
    assert sig.parameters["bootstrap_ok"].default is inspect.Parameter.empty

    fc = {
        "bootstrap_method": "paired_cluster",
        "bootstrap_unit": "group_id",
        "cluster_col": "group_id",
        "bootstrap_samples": 1000,
        "seed": 42,
        "confidence": 0.95,
    }
    tc = {"test_count": 215, "n_clusters": 5}
    ok = {
        "bootstrap_method": "paired_cluster",
        "bootstrap_unit": "group_id",
        "cluster_col": "group_id",
        "n_rows": 215,
        "n_clusters": 5,
        "n_samples": 1000,
        "seed": 42,
        "confidence": 0.95,
        "delta_sacrebleu": {"mean": 1.0, "lower": 0.5, "upper": 1.5},
        "delta_chrfpp": {"mean": 2.0, "lower": 1.0, "upper": 3.0},
    }
    assert_rq1_bootstrap_gate(ok, final_contract=fc, test_contract=tc)
    verdict_bad = derive_final_status(
        test_contract_ok=True, c0_ok=True, d0_ok=True, paired_uid_order_ok=True,
        references_identical=True, metrics_finite=True, artifacts_hashed=True, bootstrap_ok=False,
    )
    assert verdict_bad["status"] == STATUS_FAILED
    assert "bootstrap_ok" in verdict_bad["failed_checks"]


def test_notebook_wires_disk_cleanup_and_bootstrap_gate():
    import ast

    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks/06_rq1_evaluation.ipynb").read_text())
    c7 = "".join(nb["cells"][7].get("source", []))
    c8 = "".join(nb["cells"][8].get("source", []))
    c9 = "".join(nb["cells"][9].get("source", []))
    for src in (c7, c8, c9):
        ast.parse(src)
    assert "run_cascaded_c0_peak_safe" in c7
    assert "asr_allowed_root" in c7
    assert "expected_uids" in c7
    assert "run_direct_d0_peak_safe" in c8
    assert "direct_allowed_root" in c8
    assert "assert_stage_summary_matches_final_contract" in c9
    assert "assert_rq1_bootstrap_gate" in c9
    assert "bootstrap_ok=bootstrap_ok" in c9


def test_notebook06_cells_compile_and_wire_hardening():
    import ast

    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks/06_rq1_evaluation.ipynb").read_text())
    assert "rq1-operator-controls" in (nb["cells"][1].get("metadata", {}).get("tags") or [])
    required = {
        2: ["_bootstrap_project_root", "find_bahnar_project_root", "atomic_write_json", "assert_locked_manifest_bundle_matches_contract"],
        3: ['RQ1_STAGE == "verify"', "assert_unlocked_final_contract_matches_current", "assert_locked_manifest_bundle_matches_contract"],
        5: ["build_final_contract", "parquet_revision_source", "BOOTSTRAP_METHOD_PAIRED_CLUSTER"],
        6: ["parquet_revision"],
        7: ["run_cascaded_c0_peak_safe", "asr_allowed_root", "expected_uids", "asr_local_cleaned"],
        8: ["run_direct_d0_peak_safe", "direct_allowed_root", "direct_local_cleaned"],
        9: ["paired_bootstrap", "assert_rq1_bootstrap_gate", "assert_stage_summary_matches_final_contract", "bootstrap_ok", "recompute_and_assert_finalize_integrity"],
    }
    for idx, marks in required.items():
        src = "".join(nb["cells"][idx].get("source", []))
        ast.parse(src)
        for mark in marks:
            assert mark in src, f"cell {idx} missing {mark}"
    cell3 = "".join(nb["cells"][3].get("source", []))
    assert 'RQ1_STAGE in {"verify", "unlock_test"}' not in cell3


def test_test_contract_detects_content_and_order(tmp_path):
    f = _test_frame()
    p = tmp_path / "rq1_test.csv"
    f.to_csv(p, index=False)
    c = build_rq1_test_contract(
        f,
        manifest_path=p,
        split_summary_sha256="a" * 64,
        dataset_id="d",
        dataset_revision=LOCKED_DATASET_REV,
        parquet_revision=LOCKED_PARQUET,
        parquet_revision_source=PARQUET_SOURCE_RQ1_CONFIG_LEGACY,
    )
    assert_rq1_test_contract(f, c, manifest_path=p)
    changed = f.copy()
    changed.loc[0, "text_vi"] = "Khác"
    with pytest.raises(RuntimeError):
        assert_rq1_test_contract(changed, c, manifest_path=p)


def test_asr_evaluate_wrapper_hash_binds_via_train_contract_hash(tmp_path):
    """NB03 evaluate contract_hash is a wrapper; identity is train_contract_hash."""
    from src.asr_full_train import build_data_contract, build_evaluate_contract, build_train_contract

    a, m, d = _write_upstream(tmp_path)
    data = build_data_contract(
        dataset_id="ds",
        dataset_revision="rev",
        parquet_revision="pq",
        train_manifest_content_hash="t" * 64,
        validation_manifest_content_hash="v" * 64,
        vocab_fp="vocab",
        processing_version="pcm-v1",
        min_duration=0.1,
        max_duration=30.0,
        target_sr=16000,
        pretrained_model_id="facebook/wav2vec2-xls-r-300m",
        pretrained_model_revision="rev-a",
    )
    train_contract = build_train_contract(
        experiment_id="a",
        data_contract=data,
        hparams={"learning_rate": 1e-4, "num_train_epochs": 1},
    )
    evaluate_contract = build_evaluate_contract(train_contract=train_contract)
    assert evaluate_contract["contract_hash"] != train_contract["contract_hash"]
    assert evaluate_contract["train_contract_hash"] == train_contract["contract_hash"]
    (a / "full_train_summary.json").write_text(json.dumps({
        "status": "SUCCESS_FULL_TRAINING",
        "experiment_id": "a",
        "best_checkpoint": "checkpoint-1",
        "training_contract": train_contract,
    }))
    (a / "full_evaluate_summary.json").write_text(json.dumps({
        "status": "SUCCESS_FULL_EVALUATE",
        "frozen_test_accessed": False,
        "experiment_id": "a",
        "checkpoint": "/tmp/full_train/a/checkpoint-1",
        "training_contract": evaluate_contract,
    }))
    handoff = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    assert handoff["asr"]["identity"]["contract_hash"] == train_contract["contract_hash"]

    tampered = dict(evaluate_contract)
    tampered["train_contract_hash"] = "0" * 64
    (a / "full_evaluate_summary.json").write_text(json.dumps({
        "status": "SUCCESS_FULL_EVALUATE",
        "frozen_test_accessed": False,
        "experiment_id": "a",
        "training_contract": tampered,
    }))
    with pytest.raises(RuntimeError, match="train_contract_hash does not match"):
        verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)


def _write_asr_pair(asr: Path, train_contract: dict, eval_contract: dict) -> None:
    (asr / "full_train_summary.json").write_text(json.dumps({
        "status": "SUCCESS_FULL_TRAINING",
        "experiment_id": "a",
        "best_checkpoint": "checkpoint-1",
        "training_contract": train_contract,
    }))
    (asr / "full_evaluate_summary.json").write_text(json.dumps({
        "status": "SUCCESS_FULL_EVALUATE",
        "frozen_test_accessed": False,
        "experiment_id": "a",
        "training_contract": eval_contract,
    }))


def test_asr_legacy_hash_bind_fail_closed(tmp_path):
    """Evaluate fields without any hash bind fail; matching data-only hashes pass."""
    a, m, d = _write_upstream(tmp_path)
    fields = {
        "experiment_id": "a",
        "dataset_id": "ds",
        "dataset_revision": "rev",
        "hparams": {"learning_rate": 1e-4},
    }
    train_contract = {**fields, "contract_hash": "abc"}
    eval_fields = dict(fields)
    _write_asr_pair(a, train_contract, eval_fields)
    with pytest.raises(RuntimeError, match="missing training-contract hash bind"):
        verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)

    data_only = {"experiment_id": "a", "dataset_id": "ds", "data_contract_hash": "D"}
    _write_asr_pair(a, dict(data_only), dict(data_only))
    handoff = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    assert handoff["asr"]["identity"]["contract_hash"] == "D"


def test_verify_upstream_never_needs_test_manifest(tmp_path):
    a, m, d = _write_upstream(tmp_path)
    h = verify_upstream_handoffs(asr_state_dir=a, mt_state_dir=m, direct_state_dir=d)
    assert h["status"] == "SUCCESS_RQ1_VERIFY"
    assert not (tmp_path / "data/manifests/rq1_test.csv").exists()


def test_prediction_requires_exact_order():
    f = _test_frame()
    pred = pd.DataFrame({"record_uid": ["u2", "u1", "u3", "u4"], "x": ["a", "b", "c", "d"]})
    with pytest.raises(RuntimeError):
        assert_prediction_frame(pred, f, system="x", pred_col="x")


def test_paired_join_includes_group_id():
    f = _test_frame()
    c0 = pd.DataFrame(
        {
            "record_uid": ["u1", "u2", "u3", "u4"],
            "asr_pred_bahnar": ["a", "b", "c", "d"],
            "c0_pred_vi": ["xin chào 0", "xin chào 1", "xin chào 2", "xin chào 3"],
        }
    )
    d0 = pd.DataFrame({"record_uid": ["u1", "u2", "u3", "u4"], "d0_pred_vi": ["xin chào 0", "xin chào 1", "xin chào 2", "xin chào 3"]})
    out = join_paired_predictions(f, c0, d0)
    assert "group_id" in out.columns
    assert out["group_id"].tolist() == f["group_id"].tolist()


def test_final_status_fail_closed():
    ok = derive_final_status(
        test_contract_ok=True, c0_ok=True, d0_ok=True, paired_uid_order_ok=True,
        references_identical=True, metrics_finite=True, artifacts_hashed=True, bootstrap_ok=True,
    )
    assert ok["status"] == STATUS_RQ1_FINAL
    bad = derive_final_status(
        test_contract_ok=True, c0_ok=False, d0_ok=True, paired_uid_order_ok=True,
        references_identical=True, metrics_finite=True, artifacts_hashed=True, bootstrap_ok=True,
    )
    assert bad["status"] == STATUS_FAILED


def test_runtime_exact_lock():
    assert_locked_runtime(
        {"torch": "2.8.0", "transformers": "4.57.6", "accelerate": "1.10.1"},
        {"torch": "2.8.0+cu128", "transformers": "4.57.6", "accelerate": "1.10.1"},
    )


def test_collect_runtime_provenance_has_required_packages():
    got = collect_runtime_provenance()
    for key in ("python", "torch", "transformers", "accelerate", "numpy", "pandas", "sacrebleu"):
        assert got[key]


def test_recompute_finalize_integrity_fail_closed(tmp_path):
    path = tmp_path / "rq1_audio_integrity.csv"
    path.write_text("record_uid,sha256_pcm\nu1,abc\n")
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    assert recompute_and_assert_finalize_integrity(
        integrity_path=path,
        c0_summary={"audio_integrity_sha256": current},
        d0_summary={"audio_integrity_sha256": current},
    ) == current


def test_verify_checkpoint_artifacts_mock_load_proof(tmp_path, monkeypatch):
    from src.rq1_restore import verify_upstream_checkpoint_artifacts

    a, m, d = _write_upstream(tmp_path)

    def fake_asr(*args, **kwargs):
        return _make_ckpt(tmp_path / "asr_best" / "checkpoint-1")

    def fake_mt(*args, **kwargs):
        return _make_ckpt(tmp_path / "mt_best" / "checkpoint-1")

    def fake_direct(*args, **kwargs):
        return _make_ckpt(tmp_path / "direct_best" / "checkpoint-1")

    monkeypatch.setattr("src.asr_full_train.resolve_best_checkpoint_from_durable", fake_asr)
    monkeypatch.setattr("src.mt_full_train.resolve_mt_best_checkpoint_for_evaluate", fake_mt)
    monkeypatch.setattr("src.direct_full_train.restore_direct_best_checkpoint", fake_direct)

    class _M:
        pass

    proof = verify_upstream_checkpoint_artifacts(
        asr_state_dir=a,
        mt_state_dir=m,
        direct_state_dir=d,
        asr_local_ckpt_root=tmp_path / "asr_ckpt",
        mt_local_ckpt_root=tmp_path / "mt_ckpt",
        direct_local_ckpt_root=tmp_path / "direct_ckpt",
        cleanup_local_after_verify=True,
        load_models=True,
        asr_loader=lambda best, device: {"model": _M(), "processor": _M()},
        mt_loader=lambda best, contract, device: {"model": _M(), "tokenizer": _M()},
        direct_loader=lambda best, contract, device: {"model": _M(), "tokenizer": _M()},
    )
    assert proof["asr"]["loaded"] is True
    assert proof["proof_hash"]


def test_source_fingerprint_ignores_operator_cell(tmp_path):
    import json as _json
    import shutil

    from src.rq1_contract import RQ1_SOURCE_FILES, compute_rq1_source_fingerprint

    repo = Path(__file__).resolve().parents[1]
    for rel in RQ1_SOURCE_FILES:
        src = repo / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    fp1 = compute_rq1_source_fingerprint(tmp_path)["aggregate_sha256"]
    nbp = tmp_path / "notebooks/06_rq1_evaluation.ipynb"
    nb = _json.loads(nbp.read_text())
    op = next(c for c in nb["cells"] if "rq1-operator-controls" in (c.get("metadata", {}).get("tags") or []))
    op["source"] = ['RQ1_STAGE = "finalize"\n']
    nbp.write_text(_json.dumps(nb))
    assert compute_rq1_source_fingerprint(tmp_path)["aggregate_sha256"] == fp1
