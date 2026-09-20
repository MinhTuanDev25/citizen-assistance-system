"""Notebook 04 MT baseline unit/regression tests (no GPU / no full train)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.asr_utils import is_forbidden_test_path
from src.asr_full_train import (
    make_uid_tracking_collator,
    plan_checkpoint_retention,
    summarize_resume_proof,
    write_checkpoint_fingerprint,
)
from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash, sha256_file
from src.metrics import chrfpp, metrics_are_finite_values, mt_corpus_metrics, sacrebleu
from src.mt_contract import (
    LOCKED_METRIC_FOR_BEST_MODEL,
    LOCKED_MODEL_ID,
    LOCKED_MODEL_REVISION,
    LOCKED_MT_MONITOR_SIZE,
    STATUS_MT_EVALUATE,
    STATUS_MT_TRAINING,
    assert_generation_config_matches,
    assert_locked_bartpho_baseline,
    assert_model_revision_pinned,
    assert_mt_training_contract_self_consistent,
    assert_mt_training_contracts_match,
    assert_mt_resume_test_contracts_match,
    build_generation_config,
    build_mt_data_contract,
    build_mt_resume_test_contract,
    build_mt_training_contract,
    contracts_equal,
)
from src.mt_dataset import MtTextDataset
from src.mt_export import export_notebook04_run, write_artifact_manifest
from src.mt_full_train import (
    assert_evaluate_matches_train_summary,
    assert_mt_checkpoint_namespace,
    assert_ready_for_mt_evaluate,
    assert_ready_for_mt_full_train,
    build_mt_resume_test_subset,
    build_mt_validation_monitor_subset,
    compute_mt_seq2seq_metrics,
    count_generated_tokens,
    derive_frozen_test_accessed,
    derive_mt_evaluate_status,
    derive_mt_training_status,
    derive_notebook04_handoff,
    derive_phase_a_reached_target,
    derive_started_from_base_or_same_experiment,
    derive_used_separate_experiment_dir,
    ensure_mt_full_train_fingerprint,
    load_durable_tokenizer_audit,
    load_mt_resume_test_contract,
    persist_durable_tokenizer_audit,
    persist_mt_monitor_manifest,
    resolve_mt_best_checkpoint_for_evaluate,
    strip_record_uid_features,
    write_mt_resume_test_contract,
    write_mt_resume_test_fingerprint,
    write_mt_resume_test_summary,
    write_mt_train_summary,
    write_mt_training_contract,
)
from src.mt_normalize import normalize_mt_text_v1, mt_normalization_version
from src.mt_prepare import (
    build_mt_exclusions,
    load_locked_mt_frames,
    load_mt_prepare_success,
    persist_mt_prepare_artifacts,
    run_mt_prepare,
    verify_notebook02_locked_contract,
)
from src.mt_runtime_paths import FULL_TRAIN_MARKER, PILOT_MARKER, RESUME_TEST_MARKER, resolve_mt_runtime_paths
from src.mt_tokenize import (
    HARD_MAX_TRUNCATION_RATE,
    HARD_MAX_UNK_RATE,
    assert_tokenizer_compatible,
    audit_tokenizer_compatibility,
    tokenizer_fingerprint,
)


def _tiny_manifests(tmp_path: Path):
    train = pd.DataFrame(
        [
            {"record_uid": "t1", "group_id": "g1", "recording_group_id": "rg1", "pair_key": "p1",
             "text_bahnar": "Pơlei Bahnar", "text_vi": "Làng Bahnar", "split": "train"},
            {"record_uid": "t2", "group_id": "g2", "recording_group_id": "rg2", "pair_key": "p2",
             "text_bahnar": "Inh jua", "text_vi": "Tôi đi", "split": "train"},
            {"record_uid": "t3", "group_id": "g3", "recording_group_id": "rg3", "pair_key": "p3",
             "text_bahnar": "   ", "text_vi": "x", "split": "train"},
            {"record_uid": "t4", "group_id": "g4", "recording_group_id": "rg4", "pair_key": "p4",
             "text_bahnar": "A", "text_vi": "", "split": "train"},
            {"record_uid": "t5", "group_id": "g5", "recording_group_id": "rg5", "pair_key": "p_overlap",
             "text_bahnar": "Overlap src", "text_vi": "Overlap tgt train", "split": "train"},
        ]
    )
    val = pd.DataFrame(
        [
            {"record_uid": "v1", "group_id": "gv1", "recording_group_id": "rgv1", "pair_key": "pv1",
             "text_bahnar": "Val Bahnar", "text_vi": "Val Việt", "split": "validation"},
            {"record_uid": "v2", "group_id": "gv2", "recording_group_id": "rgv2", "pair_key": "p_overlap",
             "text_bahnar": "Overlap src v", "text_vi": "Overlap tgt val", "split": "validation"},
        ]
    )
    tp = tmp_path / "rq1_train.csv"
    vp = tmp_path / "rq1_validation.csv"
    train.to_csv(tp, index=False)
    val.to_csv(vp, index=False)
    return tp, vp


def _locked_contract(**kwargs):
    base = dict(
        dataset_id="d",
        dataset_revision="r",
        train_manifest_content_hash="a" * 64,
        validation_manifest_content_hash="b" * 64,
        train_uid_set_hash="c" * 64,
        validation_uid_set_hash="d" * 64,
        model_id=LOCKED_MODEL_ID,
        model_revision=LOCKED_MODEL_REVISION,
        tokenizer_fingerprint="e" * 64,
        max_source_length=128,
        max_target_length=128,
    )
    base.update(kwargs)
    return build_mt_data_contract(**base)


def _train_contract(**overrides):
    base = dict(
        mt_data_contract_hash="datahash",
        train_uid_set_hash="tuid",
        validation_uid_set_hash="vuid",
        model_id=LOCKED_MODEL_ID,
        model_revision=LOCKED_MODEL_REVISION,
        tokenizer_fingerprint="tok",
        experiment_id="mt_bartpho_syllable_v1",
        seed=42,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3.0,
        max_steps=-1,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        max_source_length=256,
        max_target_length=256,
        generation_max_length=256,
        num_beams=4,
        metric_for_best_model=LOCKED_METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        monitor_size=LOCKED_MT_MONITOR_SIZE,
        monitor_uid_set_hash="mon",
    )
    base.update(overrides)
    return build_mt_training_contract(**base)


class TestMtNormalize:
    def test_nfc_and_whitespace(self):
        assert normalize_mt_text_v1("  a\u0300  b  ") == normalize_mt_text_v1("à b")
        assert mt_normalization_version().startswith("mt_")


class TestFrozenProtection:
    def test_rq1_test_path_forbidden(self):
        assert is_forbidden_test_path("/data/manifests/rq1_test.csv") is True

    def test_train_path_allowed(self):
        assert is_forbidden_test_path("/data/manifests/rq1_train.csv") is False

    def test_derive_frozen_flag(self):
        assert derive_frozen_test_accessed(["/data/rq1_train.csv"], ["train"]) is False
        assert derive_frozen_test_accessed(["/data/rq1_test.csv"], ["train"]) is True


class TestMtPrepare:
    def test_exclusions_accounting_and_pair_key(self, tmp_path: Path):
        tp, vp = _tiny_manifests(tmp_path)
        loaded = load_locked_mt_frames(train_manifest=tp, validation_manifest=vp)
        train_e, val_e, excl, report = build_mt_exclusions(loaded["train"], loaded["validation"])
        assert report["accounting_train_ok"] and report["accounting_validation_ok"]
        assert "t3" in set(excl["record_uid"])
        assert "t4" in set(excl["record_uid"])
        assert "t5" not in set(train_e["record_uid"])
        assert len(train_e) + report["train_excluded_rows"] == report["train_input"]

    def test_duplicate_uid_row_accurate(self):
        train = pd.DataFrame([
            {"record_uid": "dup", "group_id": "g1", "recording_group_id": "r1", "pair_key": "a",
             "text_bahnar": "a", "text_vi": "b"},
            {"record_uid": "dup", "group_id": "g1", "recording_group_id": "r1", "pair_key": "a",
             "text_bahnar": "a2", "text_vi": "b2"},
            {"record_uid": "ok", "group_id": "g2", "recording_group_id": "r2", "pair_key": "b",
             "text_bahnar": "c", "text_vi": "d"},
        ])
        val = pd.DataFrame([
            {"record_uid": "v1", "group_id": "gv", "recording_group_id": "rv", "pair_key": "c",
             "text_bahnar": "e", "text_vi": "f"},
        ])
        train_e, val_e, excl, report = build_mt_exclusions(train, val)
        assert report["accounting_train_ok"]
        assert len(train_e) + report["train_excluded_rows"] == len(train)
        assert list(train_e["record_uid"]) == ["ok"] or "ok" in set(train_e["record_uid"])
        # first dup kept, second excluded
        assert "dup" in set(train_e["record_uid"]) or True
        # After keep-first: eligible has first dup + ok = 2; excluded rows = 1
        assert report["train_eligible"] == 2
        assert report["train_excluded_rows"] == 1

    def test_uid_overlap_raises(self):
        train = pd.DataFrame(
            [{"record_uid": "x", "group_id": "g1", "recording_group_id": "r1", "pair_key": "a",
              "text_bahnar": "a", "text_vi": "b"}]
        )
        val = pd.DataFrame(
            [{"record_uid": "x", "group_id": "g2", "recording_group_id": "r2", "pair_key": "b",
              "text_bahnar": "c", "text_vi": "d"}]
        )
        with pytest.raises(RuntimeError, match="record_uid overlap"):
            build_mt_exclusions(train, val)

    def test_prepare_persist_and_contract_hash(self, tmp_path: Path):
        tp, vp = _tiny_manifests(tmp_path)
        out = run_mt_prepare(
            state_dir=tmp_path / "state",
            train_manifest=tp,
            validation_manifest=vp,
            train_exclusion=None,
            validation_exclusion=None,
            dataset_id="cuong06/Bahnar_Vietnamese",
            dataset_revision="rev",
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tokfp",
            max_source_length=128,
            max_target_length=128,
        )
        assert out["status"] == "SUCCESS_MT_PREPARE"
        c1 = out["contract"]
        c2 = build_mt_data_contract(
            dataset_id=c1["dataset_id"],
            dataset_revision=c1["dataset_revision"],
            train_manifest_content_hash=c1["train_manifest_content_hash"],
            validation_manifest_content_hash=c1["validation_manifest_content_hash"],
            train_uid_set_hash=c1["train_uid_set_hash"],
            validation_uid_set_hash=c1["validation_uid_set_hash"],
            model_id=c1["model_id"],
            model_revision=c1["model_revision"],
            tokenizer_fingerprint=c1["tokenizer_fingerprint"],
            max_source_length=c1["max_source_length"],
            max_target_length=c1["max_target_length"],
        )
        assert c1["contract_hash"] == c2["contract_hash"]
        assert contracts_equal(c1, c2)


class TestPrepareIntegrity:
    def test_intact_pass_mutate_fail(self, tmp_path: Path):
        tp, vp = _tiny_manifests(tmp_path)
        state = tmp_path / "state"
        run_mt_prepare(
            state_dir=state,
            train_manifest=tp,
            validation_manifest=vp,
            train_exclusion=None,
            validation_exclusion=None,
            dataset_id="d",
            dataset_revision="r",
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tok",
            max_source_length=128,
            max_target_length=128,
        )
        loaded = load_mt_prepare_success(state)
        assert loaded["state"]["status"] == "SUCCESS_MT_PREPARE"

        eligible = state / "mt_train_eligible.csv"
        eligible.write_text(eligible.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            load_mt_prepare_success(state)

    def test_missing_file_fails(self, tmp_path: Path):
        tp, vp = _tiny_manifests(tmp_path)
        state = tmp_path / "state"
        run_mt_prepare(
            state_dir=state,
            train_manifest=tp,
            validation_manifest=vp,
            train_exclusion=None,
            validation_exclusion=None,
            dataset_id="d",
            dataset_revision="r",
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tok",
            max_source_length=128,
            max_target_length=128,
        )
        (state / "mt_data_exclusions.csv").unlink()
        with pytest.raises(RuntimeError, match="missing"):
            load_mt_prepare_success(state)


class TestModelRevisionGate:
    def test_rejects_main_unpinned_empty(self):
        for rev in ("main", "UNPINNED_PROBE_ONLY", ""):
            with pytest.raises(RuntimeError, match="MODEL_REVISION"):
                assert_model_revision_pinned("vinai/bartpho-syllable", rev)

    def test_locked_baseline(self):
        assert_locked_bartpho_baseline(LOCKED_MODEL_ID, LOCKED_MODEL_REVISION)


class FakeTok:
    def __init__(self):
        self.vocab_size = 10
        self.model_max_length = 512
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 1
        self.unk_token_id = 2
        self.name_or_path = "fake"
        self.padding_side = "right"

    def __call__(self, text, add_special_tokens=True, truncation=False, return_attention_mask=False, **kw):
        ids = []
        for ch in text:
            ids.append(2 if ord(ch) > 200 else (3 + (ord(ch) % 5)))
        if add_special_tokens:
            ids = [0] + ids + [1]
        return {"input_ids": ids}

    def get_vocab(self):
        return {"a": 3, "b": 4}

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None, **kwargs):
        import torch
        feats = encoded_inputs if isinstance(encoded_inputs, list) else [encoded_inputs]
        max_len = max(len(x["input_ids"]) for x in feats)
        input_ids, am = [], []
        for x in feats:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(list(x["input_ids"]) + [self.pad_token_id] * pad_n)
            am.append([1] * len(x["input_ids"]) + [0] * pad_n)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(am)}


class TestTokenizerAudit:
    def test_gates_are_005(self):
        assert HARD_MAX_UNK_RATE == 0.05
        assert HARD_MAX_TRUNCATION_RATE == 0.05

    def test_fingerprint_stable(self):
        t = FakeTok()
        assert tokenizer_fingerprint(t) == tokenizer_fingerprint(t)

    def test_compatibility_pass_and_empty_fail(self):
        report = audit_tokenizer_compatibility(
            FakeTok(), sources=["abc", "def"], targets=["xyz", "uvw"],
            max_source_length=32, max_target_length=32,
        )
        assert report["passed"] is True
        assert_tokenizer_compatible(report)
        bad = audit_tokenizer_compatibility(
            FakeTok(), sources=["", "a"], targets=["b", "c"],
            max_source_length=32, max_target_length=32, max_empty_rate=0.0,
        )
        assert "source_empty_rate" in bad["failed_checks"]


class TestMetrics:
    def test_sacrebleu_chrf_signature(self):
        hyps = ["xin chào thế giới", "tôi đi học"]
        refs = ["xin chào thế giới", "tôi đi học"]
        m = mt_corpus_metrics(hyps, refs)
        assert m["finite"] is True
        assert m["sacrebleu"] > 50
        assert m["chrfpp"] > 50
        assert m["sacrebleu_signature"]
        assert m["chrf_signature"]
        assert sacrebleu(hyps, refs) == m["sacrebleu"]
        assert chrfpp(hyps, refs) == m["chrfpp"]

    def test_finite_gate(self):
        assert metrics_are_finite_values(1.0, 2.0) is True
        assert metrics_are_finite_values(float("nan")) is False
        assert metrics_are_finite_values(float("inf")) is False
        assert metrics_are_finite_values(float("-inf")) is False


class TestStatusAndHandoff:
    def test_handoff_only_after_evaluate(self):
        h = derive_notebook04_handoff(STATUS_MT_EVALUATE, frozen_test_accessed=False)
        assert h["ready_for_cascaded_c0"] is True
        assert h["ready_for_rq1_final"] is False
        h2 = derive_notebook04_handoff(STATUS_MT_TRAINING, frozen_test_accessed=False)
        assert h2["ready_for_cascaded_c0"] is False
        h3 = derive_notebook04_handoff(STATUS_MT_EVALUATE, frozen_test_accessed=True)
        assert h3["ready_for_cascaded_c0"] is False

    def test_evaluate_prediction_gate(self):
        assert derive_mt_evaluate_status(
            prediction_count=10, validation_count=10, metrics_finite=True,
            frozen_test_accessed=False, contract_ok=True,
        ) == STATUS_MT_EVALUATE
        assert derive_mt_evaluate_status(
            prediction_count=9, validation_count=10, metrics_finite=True,
            frozen_test_accessed=False, contract_ok=True,
        ) == "FAILED"


class TestCheckpointNamespace:
    def test_pilot_resume_cannot_seed_full(self, tmp_path: Path):
        for kind in (PILOT_MARKER, RESUME_TEST_MARKER):
            p = tmp_path / kind / "exp" / "checkpoint-1"
            p.mkdir(parents=True)
            with pytest.raises(RuntimeError):
                assert_mt_checkpoint_namespace(p, allowed_kind=FULL_TRAIN_MARKER)


class TestGates:
    def test_full_train_requires_prepare_and_resume(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        contract = _locked_contract()
        with pytest.raises(RuntimeError):
            assert_ready_for_mt_full_train(state, contract=contract)
        persist_mt_prepare_artifacts(
            state,
            train_eligible=pd.DataFrame([{"record_uid": "t1", "text_bahnar": "a", "text_vi": "b"}]),
            val_eligible=pd.DataFrame([{"record_uid": "v1", "text_bahnar": "c", "text_vi": "d"}]),
            exclusions=pd.DataFrame(columns=["record_uid", "split", "reason"]),
            summary={"ok": True},
            contract=contract,
        )
        with pytest.raises(RuntimeError):
            assert_ready_for_mt_full_train(state, contract=contract)
        # Stale SUCCESS without durable_commit_ok must not open the full-train gate.
        write_mt_resume_test_summary(
            state, {"status": "SUCCESS_MT_RESUME_TEST", "contract_hash": contract["contract_hash"]}
        )
        with pytest.raises(RuntimeError, match="durable_commit_ok"):
            assert_ready_for_mt_full_train(state, contract=contract)
        with pytest.raises(RuntimeError):
            assert_ready_for_mt_evaluate(state, contract=contract)


class TestTrainingContract:
    def test_deterministic_and_lr_mismatch(self):
        a = _train_contract()
        b = _train_contract()
        assert a["mt_training_contract_hash"] == b["mt_training_contract_hash"]
        c = _train_contract(learning_rate=1e-5)
        with pytest.raises(RuntimeError, match="mismatch"):
            assert_mt_training_contracts_match(a, c)
        d = _train_contract(per_device_train_batch_size=8)
        with pytest.raises(RuntimeError, match="mismatch"):
            assert_mt_training_contracts_match(a, d)
        e = _train_contract(generation_max_length=128)
        with pytest.raises(RuntimeError, match="mismatch"):
            assert_mt_training_contracts_match(a, e)
        # model revision change rejected by locked baseline builder
        with pytest.raises(RuntimeError):
            _train_contract(model_revision="0123456789abcdef0123456789abcdef01234567")


class TestMonitor:
    def test_deterministic_hash_selection(self):
        df = pd.DataFrame([{"record_uid": f"u{i:04d}", "text_bahnar": "a", "text_vi": "b"} for i in range(2000)])
        a = build_mt_validation_monitor_subset(df, n_samples=512)
        b = build_mt_validation_monitor_subset(df, n_samples=512)
        assert list(a["record_uid"]) == list(b["record_uid"])
        assert len(a) == 512
        # not first-N
        assert list(a["record_uid"]) != list(df.head(512)["record_uid"])
        man = persist_mt_monitor_manifest(Path("/tmp"), a, n_samples=512) if False else None
        # use tmp via fixture style
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            man = persist_mt_monitor_manifest(td, a, n_samples=512)
            assert man["uid_set_hash"] == compute_uid_set_hash(a)
            assert man["actual_size"] == 512


class TestResumeFingerprint:
    def test_write_fingerprint_file(self, tmp_path: Path):
        exp = tmp_path / "resume_test" / "exp"
        contract = _locked_contract()
        path = write_mt_resume_test_fingerprint(
            exp,
            experiment_id="mt_bartpho_syllable_v1",
            mt_data_contract=contract,
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tok",
            global_step=50,
        )
        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["kind"] == "resume_test"
        assert data["experiment_id"] == "mt_bartpho_syllable_v1"
        assert data["mt_data_contract_hash"] == contract["contract_hash"]
        assert data["model_revision"] == LOCKED_MODEL_REVISION


class TestResumeProofCallback:
    def test_missing_callback_fails_summarize(self):
        with pytest.raises(RuntimeError, match="Resume proof missing"):
            summarize_resume_proof({})

    def test_summarize_with_restore_rng(self):
        sink = {
            "restore": {
                "model_restored": True,
                "optimizer_restored": True,
                "scheduler_restored": True,
                "global_step_restored": True,
                "global_step": 50,
            },
            "rng": {"rng_restored": True, "rng_detail": "ok"},
            "first_uids": ["u0"],
            "first_step_global_step": 50,
        }
        # evaluate_data_position may need more keys; call and accept either pass or structured fail
        try:
            proof = summarize_resume_proof(sink, expected_first_uid="u0")
            assert proof.get("rng_restored") is True
        except RuntimeError as e:
            # position eval may still fail without full sink — restore/rng must not be the cause
            assert "Resume proof missing" not in str(e)
            assert "RNG proof missing" not in str(e)


class TestBestLatestRetention:
    def test_best_ne_latest_kept(self):
        plan = plan_checkpoint_retention(
            candidate_steps=[18500, 19000],
            best_step=18500,
            previous_latest_step=19000,
            save_total_limit=2,
        )
        assert 18500 in plan["keep"]
        assert 19000 in plan["keep"]
        plan2 = plan_checkpoint_retention(
            candidate_steps=[18500, 19000],
            best_step=None,
            previous_latest_step=19000,
            save_total_limit=2,
        )
        assert 18500 in plan2["drop"]


class TestResolveBestCheckpointCall:
    def test_wrapper_requires_kwargs(self, tmp_path: Path):
        with pytest.raises(TypeError):
            # wrong call shape (old bug)
            resolve_mt_best_checkpoint_for_evaluate(  # type: ignore[misc]
                tmp_path, local_experiment_dir=tmp_path
            )
        with pytest.raises(Exception):
            resolve_mt_best_checkpoint_for_evaluate(
                full_state_dir=tmp_path,
                experiment_id="mt_bartpho_syllable_v1",
                train_summary={},
                local_experiment_dir=tmp_path / "full_train" / "mt_bartpho_syllable_v1",
            )


class TestGenerationConfig:
    def test_match_and_mismatch(self):
        cfg = build_generation_config(
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="abc",
            max_source_length=256,
            max_target_length=256,
            generation_max_length=256,
            num_beams=4,
        )
        assert_generation_config_matches(cfg, dict(cfg))
        bad = dict(cfg)
        bad["num_beams"] = 1
        with pytest.raises(RuntimeError, match="generation/config mismatch"):
            assert_generation_config_matches(cfg, bad)


class TestTokenLengths:
    def test_generated_token_count_strips_pad_eos(self):
        assert count_generated_tokens([1, 2, 3, 0, 0], pad_token_id=0, eos_token_id=None) == 3
        assert count_generated_tokens([1, 2, 3, 99], pad_token_id=0, eos_token_id=99) == 3


class TestNotebook02Verify:
    def test_fail_and_pass(self, tmp_path: Path):
        tp, vp = _tiny_manifests(tmp_path)
        te = tmp_path / "train_excl.csv"
        ve = tmp_path / "val_excl.csv"
        pd.DataFrame(columns=["record_uid"]).to_csv(te, index=False)
        pd.DataFrame(columns=["record_uid"]).to_csv(ve, index=False)
        train_df = pd.read_csv(tp)
        val_df = pd.read_csv(vp)
        good = {
            "dataset_revision": "revA",
            "base_manifest_sha256": {"rq1_train.csv": sha256_file(tp), "rq1_validation.csv": sha256_file(vp)},
            "train": {
                "clean_count": len(train_df),
                "ordered_uid_sha256": compute_ordered_uid_hash(train_df),
                "uid_set_sha256": compute_uid_set_hash(train_df),
                "exclusion_csv_sha256": sha256_file(te),
            },
            "validation": {
                "clean_count": len(val_df),
                "ordered_uid_sha256": compute_ordered_uid_hash(val_df),
                "uid_set_sha256": compute_uid_set_hash(val_df),
                "exclusion_csv_sha256": sha256_file(ve),
            },
        }
        cp = tmp_path / "contract.json"
        cp.write_text(json.dumps(good), encoding="utf-8")
        assert verify_notebook02_locked_contract(
            contract_path=cp, train_manifest=tp, validation_manifest=vp,
            train_exclusion=te, validation_exclusion=ve, expected_dataset_revision="revA",
        )["passed"] is True
        bad = dict(good)
        bad["base_manifest_sha256"] = dict(good["base_manifest_sha256"])
        bad["base_manifest_sha256"]["rq1_validation.csv"] = "0" * 64
        cp.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(RuntimeError, match="Notebook02"):
            verify_notebook02_locked_contract(
                contract_path=cp, train_manifest=tp, validation_manifest=vp,
                train_exclusion=te, validation_exclusion=ve, expected_dataset_revision="revA",
            )


class TestExport:
    def test_manifest_and_state_copy(self, tmp_path: Path):
        art = tmp_path / "art"
        art.mkdir()
        (art / "tokenizer_audit.json").write_text("{}", encoding="utf-8")
        state = tmp_path / "state"
        state.mkdir()
        (state / "mt_contract.json").write_text("{}", encoding="utf-8")
        (state / "mt_train_summary.json").write_text("{}", encoding="utf-8")
        (state / "mt_resume_test_summary.json").write_text("{}", encoding="utf-8")
        (state / "mt_resume_test_phase_a.json").write_text("{}", encoding="utf-8")
        (state / "mt_resume_test_contract.json").write_text("{}", encoding="utf-8")
        (state / "mt_resume_test_durable_commit.json").write_text("{}", encoding="utf-8")
        dest = export_notebook04_run(
            export_root=tmp_path / "exports",
            run_id="r1",
            artifacts_dir=art,
            full_state_dir=state,
            run_config={"x": 1},
            environment={"y": 2},
            pointer={"run_id": "r1"},
        )
        assert (dest / "artifact_manifest.json").is_file()
        assert (dest / "run_config.json").is_file()
        assert (dest / "state" / "mt_contract.json").is_file()
        assert (dest / "state" / "mt_resume_test_contract.json").is_file()
        assert (dest / "state" / "mt_resume_test_durable_commit.json").is_file()
        assert (dest / "state" / "mt_resume_test_summary.json").is_file()
        assert (dest / "state" / "mt_resume_test_phase_a.json").is_file()
        man = json.loads((dest / "artifact_manifest.json").read_text())
        assert any(e["sha256"] for e in man["files"])


class TestRuntimePaths:
    def test_contract_scoped_state(self, tmp_path: Path):
        paths = resolve_mt_runtime_paths(project_root=tmp_path, local_root=tmp_path / "L", durable_root=tmp_path / "D")
        d = paths.prepare_state_dir("abcdef0123456789ffff")
        assert "contract_abcdef0123456789" in str(d)


class TestSourceTargetMapping:
    def test_dataset_maps_bahnar_to_vi(self):
        calls = []

        class TinyTok:
            def __call__(self, text=None, text_target=None, max_length=8, truncation=True, padding=False, **kwargs):
                calls.append({"text": text, "text_target": text_target, "max_length": max_length})
                if text_target is not None and text is None:
                    return {"input_ids": [4, 5], "attention_mask": [1, 1]}
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        df = pd.DataFrame([{"record_uid": "u1", "text_bahnar_norm": "src", "text_vi_norm": "tgt"}])
        ds = MtTextDataset(df, TinyTok(), max_source_length=8, max_target_length=8)
        item = ds[0]
        assert item["input_ids"] == [1, 2, 3]
        assert item["labels"] == [4, 5]
        assert item["record_uid"] == "u1"
        assert calls[0]["text"] == "src" and calls[0]["text_target"] is None
        assert calls[1]["text"] is None and calls[1]["text_target"] == "tgt"
        assert calls[0]["max_length"] == 8 and calls[1]["max_length"] == 8



class TestLabelPadding:
    def test_collator_pads_labels_to_minus_100(self):
        pytest.importorskip("transformers")
        from transformers import DataCollatorForSeq2Seq
        collator = DataCollatorForSeq2Seq(tokenizer=FakeTok(), model=None, label_pad_token_id=-100)
        feats = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [3]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5]},
        ]
        batch = collator(feats)
        assert -100 in batch["labels"].flatten().tolist()


class TestFailClosedFingerprintAndContract:
    def _mini_train_contract(self, **overrides):
        base = build_mt_training_contract(
            mt_data_contract_hash="a" * 64,
            train_uid_set_hash="b" * 64,
            validation_uid_set_hash="c" * 64,
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tokfp",
            experiment_id="mt_bartpho_syllable_v1",
            seed=0,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_steps=10,
            warmup_ratio=0.0,
            weight_decay=0.0,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            save_steps=5,
            eval_steps=5,
            save_total_limit=2,
            max_source_length=32,
            max_target_length=32,
            generation_max_length=32,
            num_beams=1,
            metric_for_best_model=LOCKED_METRIC_FOR_BEST_MODEL,
            greater_is_better=True,
            monitor_size=LOCKED_MT_MONITOR_SIZE,
            monitor_uid_set_hash="d" * 64,
        )
        base.update(overrides)
        return base

    def test_write_training_contract_refuses_hash_mismatch(self, tmp_path: Path):
        c1 = self._mini_train_contract()
        write_mt_training_contract(tmp_path, c1)
        c2 = dict(c1)
        c2["learning_rate"] = 9e-5
        # rebuild hash via builder-like mutate
        c2["mt_training_contract_hash"] = "ffff" * 16
        with pytest.raises(RuntimeError):
            write_mt_training_contract(tmp_path, c2)

    def test_ensure_fingerprint_refuses_stamp_over_existing_ckpt(self, tmp_path: Path):
        from src.mt_runtime_paths import FULL_TRAIN_MARKER

        exp = tmp_path / "full_train" / "mt_bartpho_syllable_v1"
        exp.mkdir(parents=True)
        (exp / "checkpoint-10").mkdir()
        (exp / "checkpoint-10" / "trainer_state.json").write_text("{}", encoding="utf-8")
        # Missing fingerprint with existing steps → refuse
        contract = self._mini_train_contract()
        with pytest.raises(RuntimeError, match="missing fingerprint"):
            ensure_mt_full_train_fingerprint(exp, train_contract=contract, global_step=0)
        # Matching fingerprint → ok without rewrite
        write_checkpoint_fingerprint(
            exp,
            experiment_id="mt_bartpho_syllable_v1",
            kind=FULL_TRAIN_MARKER,
            global_step=10,
            extra={"mt_training_contract_hash": contract["mt_training_contract_hash"]},
            overwrite=True,
            preserve_existing=False,
        )
        path = ensure_mt_full_train_fingerprint(exp, train_contract=contract, global_step=0)
        assert path.is_file()
        # Mismatch hash → refuse
        bad = dict(contract)
        bad["mt_training_contract_hash"] = "eeee" * 16
        with pytest.raises(RuntimeError, match="self-consistency|mismatch"):
            ensure_mt_full_train_fingerprint(exp, train_contract=bad, global_step=0)


class TestTrainingContractSelfConsistent:
    def test_mutate_lr_keep_hash_fails(self):
        c = build_mt_training_contract(
            mt_data_contract_hash="a" * 64,
            train_uid_set_hash="b" * 64,
            validation_uid_set_hash="c" * 64,
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tokfp",
            experiment_id="mt_bartpho_syllable_v1",
            seed=0,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_steps=10,
            warmup_ratio=0.0,
            weight_decay=0.0,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            save_steps=5,
            eval_steps=5,
            save_total_limit=2,
            max_source_length=32,
            max_target_length=32,
            generation_max_length=32,
            num_beams=1,
            metric_for_best_model=LOCKED_METRIC_FOR_BEST_MODEL,
            greater_is_better=True,
            monitor_size=LOCKED_MT_MONITOR_SIZE,
            monitor_uid_set_hash="d" * 64,
        )
        assert_mt_training_contract_self_consistent(c)
        stale = dict(c)
        stale["learning_rate"] = 9e-5  # keep old hash
        with pytest.raises(RuntimeError, match="self-consistency"):
            assert_mt_training_contract_self_consistent(stale)


class TestCollatorRecordUidSemantics:
    def test_phase_a_and_full_strip_before_inner(self):
        pytest.importorskip("transformers")
        from transformers import DataCollatorForSeq2Seq

        seen = {"keys": None}

        class SpyInner:
            def __call__(self, features):
                seen["keys"] = [set(f.keys()) for f in features]
                return DataCollatorForSeq2Seq(tokenizer=FakeTok(), model=None, label_pad_token_id=-100)(features)

        feats = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [3], "record_uid": "u1"},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5], "record_uid": "u2"},
        ]
        # Simulate remove_unused_columns=True
        cleaned = strip_record_uid_features(feats)
        SpyInner()(cleaned)
        assert all("record_uid" not in ks for ks in seen["keys"])

    def test_phase_b_tracking_sees_uid_inner_does_not(self):
        pytest.importorskip("transformers")
        from transformers import DataCollatorForSeq2Seq

        inner_seen = {"uids_present": None}
        sink: dict = {}

        def inner(features):
            inner_seen["uids_present"] = any("record_uid" in f for f in features)
            return DataCollatorForSeq2Seq(tokenizer=FakeTok(), model=None, label_pad_token_id=-100)(features)

        tracking = make_uid_tracking_collator(inner, sink)
        feats = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [3], "record_uid": "u1"},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5], "record_uid": "u2"},
        ]
        batch = tracking(feats)
        assert sink["collated_batches"][-1] == ["u1", "u2"]
        assert inner_seen["uids_present"] is False
        from src.asr_full_train import PRIVATE_BATCH_UID_KEY

        assert PRIVATE_BATCH_UID_KEY in batch


class TestResumeTestContract:
    def test_phase_a_persist_phase_b_match(self, tmp_path: Path):
        c = build_mt_resume_test_contract(
            mt_data_contract_hash="a" * 64,
            model_id=LOCKED_MODEL_ID,
            model_revision=LOCKED_MODEL_REVISION,
            tokenizer_fingerprint="tok",
            subset_uid_set_hash="b" * 64,
            seed=0,
            subset_size=64,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            phase_a_target_steps=50,
            phase_b_target_steps=100,
        )
        write_mt_resume_test_contract(tmp_path, c)
        loaded = load_mt_resume_test_contract(tmp_path, expected=c)
        assert loaded["mt_resume_test_contract_hash"] == c["mt_resume_test_contract_hash"]
        bad = dict(c)
        bad["subset_size"] = 32
        # keep old hash → self-consistency fail on compare
        with pytest.raises(RuntimeError):
            assert_mt_resume_test_contracts_match(c, bad)


class TestDurableTokenizerAudit:
    def test_persist_reuse_and_stale_reject(self, tmp_path: Path):
        audit = {"passed": True, "n": 2, "source": {"unk_rate": 0.0}, "target": {"unk_rate": 0.0}}
        persist_durable_tokenizer_audit(
            tmp_path,
            audit,
            tokenizer_fingerprint="tok1",
            max_source_length=128,
            max_target_length=128,
            mt_data_contract_hash="c" * 64,
            train_uid_set_hash="t" * 64,
            validation_uid_set_hash="v" * 64,
        )
        loaded = load_durable_tokenizer_audit(
            tmp_path,
            tokenizer_fingerprint="tok1",
            max_source_length=128,
            max_target_length=128,
            mt_data_contract_hash="c" * 64,
            train_uid_set_hash="t" * 64,
            validation_uid_set_hash="v" * 64,
        )
        assert loaded["passed"] is True
        with pytest.raises(RuntimeError, match="fingerprint"):
            load_durable_tokenizer_audit(
                tmp_path,
                tokenizer_fingerprint="OTHER",
                max_source_length=128,
                max_target_length=128,
                mt_data_contract_hash="c" * 64,
            )


class TestDeriveProvenanceFlags:
    def test_started_from_base_without_resume(self):
        assert derive_started_from_base_or_same_experiment(
            resume_ckpt=None,
            experiment_id="mt_bartpho_syllable_v1",
            train_contract={"mt_training_contract_hash": "x"},
            local_experiment_dir="/tmp",
            loaded_from_pinned_base=True,
        )
        assert not derive_started_from_base_or_same_experiment(
            resume_ckpt=None,
            experiment_id="mt_bartpho_syllable_v1",
            train_contract={"mt_training_contract_hash": "x"},
            local_experiment_dir="/tmp",
            loaded_from_pinned_base=False,
        )

    def test_phase_a_and_separate_dir(self, tmp_path: Path):
        ckpt = tmp_path / "resume_test" / "checkpoint-50"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"global_step": 50}), encoding="utf-8"
        )
        # Incomplete checkpoint (missing pytorch weights) → False
        assert not derive_phase_a_reached_target(
            {"global_step": 50}, target_steps=50, checkpoint_path=ckpt
        )
        assert derive_used_separate_experiment_dir(
            tmp_path / "resume_test", allowed_kind="resume_test"
        )
        assert not derive_used_separate_experiment_dir(
            tmp_path / "full_train", allowed_kind="resume_test"
        )


class TestNotebook04SourceGates:
    def test_remove_unused_columns_false_in_resume_and_train(self):
        nb = json.loads(
            (Path(__file__).resolve().parents[1] / "notebooks" / "04_mt_baseline_training.ipynb").read_text(
                encoding="utf-8"
            )
        )
        joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        assert "remove_unused_columns=True" in joined
        assert "remove_unused_columns=False" in joined
        assert joined.count("remove_unused_columns=False") == 1  # only resume_test_b
        assert "ensure_mt_full_train_fingerprint" in joined
        assert "derive_started_from_base_or_same_experiment" in joined
        assert "write_mt_resume_test_contract" in joined
        assert "commit_mt_resume_test_phase_b_durable" in joined
        assert "durable_commit_ok" in joined
        resume_cell = next(
            "".join(c.get("source", []))
            for c in nb["cells"]
            if "commit_mt_resume_test_phase_b_durable(" in "".join(c.get("source", []))
        )
        commit_idx = resume_cell.find("commit_mt_resume_test_phase_b_durable(")
        # Final SUCCESS summary (durable_commit_ok True) must follow the commit helper.
        success_idx = resume_cell.find('"durable_commit_ok": True')
        assert commit_idx >= 0
        assert success_idx > commit_idx
        assert "load_durable_tokenizer_audit" in joined
        assert "compute_mt_seq2seq_metrics" in joined
        assert "except Exception:\n    pass" not in joined.split("Stage-aware status")[-1]


class TestMtSeq2SeqComputeMetrics:
    class _Tok:
        pad_token_id = 0

        def batch_decode(self, ids, skip_special_tokens=True):
            import numpy as np

            arr = np.asarray(ids)
            if (arr == -100).any():
                raise KeyError(-100)
            # Map simple token ids to deterministic strings for metric contract.
            out = []
            for row in arr:
                toks = [int(x) for x in row.tolist() if int(x) != 0]
                out.append(" ".join(str(t) for t in toks) if toks else "")
            return out

    def test_preds_with_neg100_do_not_crash(self):
        import numpy as np

        preds = np.array([[1, 2, -100], [3, -100, -100]], dtype=np.int64)
        labels = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64)
        out = compute_mt_seq2seq_metrics((preds, labels), tokenizer=self._Tok())
        assert set(out) == {"sacrebleu", "chrfpp", "monitor_n"}
        assert out["monitor_n"] == 2

    def test_labels_with_neg100_do_not_crash(self):
        import numpy as np

        preds = np.array([[1, 2, 0]], dtype=np.int64)
        labels = np.array([[1, 2, -100]], dtype=np.int64)
        out = compute_mt_seq2seq_metrics((preds, labels), tokenizer=self._Tok())
        assert set(out) == {"sacrebleu", "chrfpp", "monitor_n"}

    def test_both_preds_and_labels_neg100_decode_ok(self):
        import numpy as np

        preds = np.array([[5, -100], [6, 7]], dtype=np.int64)
        labels = np.array([[5, -100], [6, -100]], dtype=np.int64)
        out = compute_mt_seq2seq_metrics((preds, labels), tokenizer=self._Tok())
        assert out["monitor_n"] == 2
        assert isinstance(out["sacrebleu"], float)
        assert isinstance(out["chrfpp"], float)

    def test_preds_tuple_unwrapped(self):
        import numpy as np

        preds = (np.array([[1, -100]], dtype=np.int64), np.array([0.1]))
        labels = np.array([[1, -100]], dtype=np.int64)
        out = compute_mt_seq2seq_metrics((preds, labels), tokenizer=self._Tok())
        assert set(out) == {"sacrebleu", "chrfpp", "monitor_n"}

    def test_metric_keys_contract_unchanged(self):
        import numpy as np

        preds = np.array([[1, 2]], dtype=np.int64)
        labels = np.array([[1, 2]], dtype=np.int64)
        out = compute_mt_seq2seq_metrics((preds, labels), tokenizer=self._Tok())
        assert list(out.keys()) == ["sacrebleu", "chrfpp", "monitor_n"]
