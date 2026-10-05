"""NB14 evaluation, paired predictions, bootstrap, and publication tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.metrics import mt_corpus_metrics
from src.mt_normalize import normalize_mt_text_v1
from src.rq2_final_contract import (
    ARM_D0,
    ARM_QUALITY,
    ARM_RANDOM,
    EvaluationError,
    FINAL_ARTIFACT_FILES,
    GTestFirewallError,
    Nb14Flags,
    STATUS_FAIL,
    STATUS_SUCCESS,
    default_nb14_flags,
    derive_final_status,
    evaluation_protocol,
    verify_upstream_rq2,
)
from src.rq2_final_evaluate import (
    arm_metrics,
    assert_paired_alignment,
    audit_target_truncation,
    canonical_references,
    derive_final_readiness,
    load_frozen_test_frame,
    paired_cluster_bootstrap_arms,
    persist_final_artifacts,
    prediction_row,
    publish_final_generation,
    require_g_test_unlocked,
    run_g_test_inference,
    write_predictions_parquet,
)
from tests.rq2_nb14_fixtures import (
    ARM_FINGERPRINTS,
    all_ready,
    g_test_proof_bundle,
    materialize_arm_proofs,
    predicting_frames,
    synthetic_best_checkpoints,
    synthetic_final_contract_fields,
    test_frame as g_test_rows,
    world,
)


def _frames():
    return predicting_frames()


def _ready_publish_inputs(tmp_path, frames=None):
    env = world(tmp_path, with_test=True)
    from src.rq2_final_contract import verify_upstream_rq2

    upstream = verify_upstream_rq2(tmp_path, flags=env["flags"])
    proofs = materialize_arm_proofs(env)
    best = proofs["best"]
    best[ARM_D0]["checkpoint_fingerprint"] = upstream["d0"]["checkpoint_fingerprint_sha256"]
    best[ARM_D0]["arm_training_contract_sha256"] = upstream["d0"]["direct_training_contract_hash"]
    best[ARM_D0]["direct_training_contract_hash"] = upstream["d0"]["direct_training_contract_hash"]
    fields = synthetic_final_contract_fields(tmp_path, upstream, proofs=proofs)
    return env, upstream, fields, best


def test_untruncated_references_are_canonical():
    frame = g_test_rows()
    refs = canonical_references(frame)
    assert refs[0] == normalize_mt_text_v1("đây là câu dài để kiểm tra tham chiếu không bị cắt")
    audit = audit_target_truncation([10, 400, 20], max_target_length=256)
    assert audit["n_target_truncated"] == 1
    assert audit["truncation_rate"] == pytest.approx(1 / 3)
    assert refs[0] not in {"", "truncated"}


def test_metrics_never_decode_labels_as_references():
    refs = canonical_references(g_test_rows())
    hyps = ["a", "b", "c"]
    payload = mt_corpus_metrics(hyps, refs)
    assert payload["n"] == 3
    assert "tokenizer.decode" not in json.dumps(payload)
    assert payload["finite"] is True


def test_paired_uid_and_reference_failures():
    frames = _frames()
    assert_paired_alignment(frames)
    bad_order = dict(frames)
    bad_order[ARM_RANDOM] = frames[ARM_RANDOM].iloc[::-1].reset_index(drop=True)
    with pytest.raises(EvaluationError, match="UID order"):
        assert_paired_alignment(bad_order)
    bad_ref = dict(frames)
    mutated = frames[ARM_QUALITY].copy()
    mutated.loc[0, "reference_vi"] = "khác"
    bad_ref[ARM_QUALITY] = mutated
    with pytest.raises(EvaluationError, match="references"):
        assert_paired_alignment(bad_ref)
    missing = dict(frames)
    dropped = frames[ARM_D0].copy()
    dropped.loc[1, "hypothesis_vi"] = ""
    missing[ARM_D0] = dropped
    with pytest.raises(EvaluationError, match="UID order|missing"):
        assert_paired_alignment(missing)


def test_paired_group_id_arm_and_fingerprint_failures():
    frames = _frames()
    bad_group = dict(frames)
    mutated = frames[ARM_QUALITY].copy()
    mutated.loc[0, "group_id"] = "other"
    bad_group[ARM_QUALITY] = mutated
    with pytest.raises(EvaluationError, match="group_id"):
        assert_paired_alignment(bad_group)
    bad_arm = dict(frames)
    mutated = frames[ARM_RANDOM].copy()
    mutated["arm"] = ARM_QUALITY
    bad_arm[ARM_RANDOM] = mutated
    with pytest.raises(EvaluationError, match="arm column"):
        assert_paired_alignment(bad_arm)
    with pytest.raises(EvaluationError, match="fingerprint"):
        prediction_row(
            record_uid="t-1",
            group_id="g1",
            hypothesis_vi="a",
            reference_vi="b",
            arm=ARM_D0,
            checkpoint_fingerprint="fp",
        )


def test_bootstrap_same_seed_and_cluster_identity():
    frames = _frames()
    left = paired_cluster_bootstrap_arms(frames, n_samples=20, seed=42)
    right = paired_cluster_bootstrap_arms(frames, n_samples=20, seed=42)
    assert left == right
    assert left["bootstrap_method"] == "paired_cluster"
    assert left["n_clusters"] == 2
    assert "d_quality-d_random" in left["comparisons"]
    broken = dict(frames)
    broken[ARM_D0] = frames[ARM_D0].drop(columns=["group_id"])
    with pytest.raises((RuntimeError, EvaluationError), match="cluster|group_id"):
        paired_cluster_bootstrap_arms(broken, n_samples=5, seed=1)


def test_g_test_load_blocked_until_unlock(tmp_path):
    world(tmp_path, with_test=True)
    flags = Nb14Flags(direct_state_dir=str(tmp_path / "direct_state"))
    with pytest.raises(GTestFirewallError):
        require_g_test_unlocked(flags, all_ready())
    with pytest.raises(GTestFirewallError):
        load_frozen_test_frame(project_root=tmp_path, flags=flags, readiness=all_ready())


def test_g_test_identity_rejects_order_reference_and_sha(tmp_path):
    bundle = g_test_proof_bundle(tmp_path)
    flags = bundle["flags"]
    path = tmp_path / "data" / "manifests" / "rq1_test.csv"
    original = path.read_text(encoding="utf-8")
    frame = pd.read_csv(path)
    frame.iloc[::-1].to_csv(path, index=False)
    kwargs = dict(
        project_root=tmp_path,
        flags=flags,
        readiness=all_ready(),
        layouts=bundle["layouts"],
        contracts=bundle["contracts"],
        best=bundle["best"],
        d0_identity=bundle["d0_identity"],
        upstream=bundle["upstream"],
    )
    with pytest.raises(EvaluationError, match="G_test identity"):
        load_frozen_test_frame(**kwargs)
    path.write_text(original, encoding="utf-8")
    frame = pd.read_csv(path)
    frame.loc[0, "text_vi"] = "tham chiếu đã đổi"
    frame.to_csv(path, index=False)
    with pytest.raises(EvaluationError, match="G_test identity"):
        load_frozen_test_frame(**kwargs)
    path.write_text(original, encoding="utf-8")
    contract_path = bundle["env"]["durable_rq1"]["state_dir"] / "rq1_test_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["manifest_sha256"] = "00" * 32
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(EvaluationError, match="G_test identity|durable|test contract"):
        load_frozen_test_frame(**kwargs)


def test_publish_does_not_move_current_on_failure(tmp_path):
    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    (artifact_dir / "predictions").mkdir()
    dummy = {"ok": True}
    mapping = {}
    for rel in FINAL_ARTIFACT_FILES:
        path = artifact_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".parquet"):
            write_predictions_parquet(path, _frames()[ARM_D0].to_dict("records"))
        else:
            path.write_text(json.dumps(dummy), encoding="utf-8")
        mapping[rel] = path
    result = publish_final_generation(tmp_path, artifacts=mapping, checks={})
    assert result["status"] == STATUS_FAIL
    assert result["wrote_current"] is False
    assert not (tmp_path / "artifacts" / "rq2" / "final" / "CURRENT").exists()


def test_fake_all_true_checks_cannot_publish(tmp_path):
    env = world(tmp_path, with_test=True)
    artifact_dir = tmp_path / "fake_arts"
    artifact_dir.mkdir()
    (artifact_dir / "predictions").mkdir()
    mapping = {}
    for rel in FINAL_ARTIFACT_FILES:
        path = artifact_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".parquet"):
            write_predictions_parquet(path, _frames()[rel.split("/")[1].replace(".parquet", "")].to_dict("records"))
        else:
            path.write_text(json.dumps({"ok": True, "rq2_final_contract_sha256": "11" * 32}), encoding="utf-8")
        mapping[rel] = path
    checks = {name: True for name in derive_final_status({}).get("checks")}
    result = publish_final_generation(tmp_path, artifacts=mapping, checks=checks)
    assert result["status"] == STATUS_FAIL
    assert result["wrote_current"] is False
    assert not (tmp_path / "artifacts" / "rq2" / "final" / "CURRENT").exists()
    assert env["flags"].run_real_training is False


def test_complete_synthetic_pipeline_reaches_success(tmp_path):
    env, upstream, fields, best = _ready_publish_inputs(tmp_path)
    bundle = g_test_proof_bundle(tmp_path)
    frames = _frames()
    protocol = evaluation_protocol(upstream["rq1_eval"])
    unlocked = load_frozen_test_frame(
        project_root=tmp_path,
        flags=bundle["flags"],
        readiness=all_ready(),
        layouts=bundle["layouts"],
        contracts=bundle["contracts"],
        best=bundle["best"],
        d0_identity=bundle["d0_identity"],
        upstream=bundle["upstream"],
    )
    assert unlocked["record_uid"].astype(str).tolist() == ["t-1", "t-2", "t-3"]

    def gen(arm):
        def _fn(frame):
            by = {str(u): h for u, h in zip(frames[arm]["record_uid"], frames[arm]["hypothesis_vi"])}
            return [{"record_uid": str(uid), "hypothesis_vi": by[str(uid)]} for uid in frame["record_uid"]]
        return _fn

    inferred = {
        arm: run_g_test_inference(
            arm=arm,
            test_frame=unlocked,
            checkpoint_fingerprint=ARM_FINGERPRINTS[arm],
            generate_fn=gen(arm),
        )
        for arm in (ARM_D0, ARM_RANDOM, ARM_QUALITY)
    }
    assert_paired_alignment(inferred)
    artifact_dir = tmp_path / "final_arts"
    mapping = persist_final_artifacts(
        artifact_dir,
        frames=inferred,
        contract_fields=fields,
        best=best,
        n_samples=10,
        seed=int(protocol["seed"]),
        confidence=float(protocol["confidence"]),
    )
    result = publish_final_generation(
        tmp_path,
        artifacts=mapping,
        checks={name: True for name in derive_final_status({}).get("checks")},
        durable_root=str(env["durable_root"]),
    )
    assert result["status"] == STATUS_SUCCESS
    assert result["wrote_current"] is True
    out = tmp_path / "artifacts" / "rq2" / "final"
    current = (out / "CURRENT").read_text(encoding="utf-8").strip()
    assert current
    from src.rq2_final_evaluate import _verify_staged_final_generation

    second = _verify_staged_final_generation(
        out / "generations" / current,
        project_root=tmp_path,
        durable_root=str(env["durable_root"]),
    )
    assert second["contract_sha256"]
    defaults = default_nb14_flags()
    assert defaults.run_real_training is False
    assert defaults.allow_g_test_evaluation is False
    readiness = derive_final_readiness(flags=defaults)
    assert readiness["all_ok"] is False


def test_notebook_default_run_all_source_is_wired():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks" / "14_RQ2_Final_Train_Evaluate.ipynb").read_text())
    source = "".join("".join(c.get("source") or []) for c in nb["cells"] if c["cell_type"] == "code")
    assert "RUN_REAL_TRAINING = False" in source
    assert "ALLOW_G_TEST_EVALUATION = False" in source
    assert "RQ2_FINAL_FROZEN = False" in source
    for name in (
        "verify_upstream_rq2",
        "load_frozen_supervised_splits",
        "load_pinned_nb13_arm_manifest",
        "resume_or_train_arm",
        "select_best_checkpoint",
        "derive_final_readiness",
        "run_g_test_inference",
        "paired_cluster_bootstrap_arms",
        "persist_final_artifacts",
        "publish_final_generation",
    ):
        assert name in source, name


def _unlocked_test_frame(tmp_path):
    bundle = g_test_proof_bundle(tmp_path)
    frame = load_frozen_test_frame(
        project_root=tmp_path,
        flags=bundle["flags"],
        readiness=all_ready(),
        layouts=bundle["layouts"],
        contracts=bundle["contracts"],
        best=bundle["best"],
        d0_identity=bundle["d0_identity"],
        upstream=bundle["upstream"],
    )
    return bundle["env"], frame


def test_inference_rejects_none_duplicate_extra_missing_and_wrong_group(tmp_path):
    env, frame = _unlocked_test_frame(tmp_path)
    fp = ARM_FINGERPRINTS[ARM_D0]

    def none_fn(_):
        return [{"record_uid": "t-1", "hypothesis_vi": None},
                {"record_uid": "t-2", "hypothesis_vi": "b"},
                {"record_uid": "t-3", "hypothesis_vi": "c"}]

    with pytest.raises(EvaluationError, match="missing hypothesis"):
        run_g_test_inference(arm=ARM_D0, test_frame=frame, checkpoint_fingerprint=fp, generate_fn=none_fn)

    def dup_fn(_):
        return [{"record_uid": "t-1", "hypothesis_vi": "a"},
                {"record_uid": "t-1", "hypothesis_vi": "aa"},
                {"record_uid": "t-2", "hypothesis_vi": "b"},
                {"record_uid": "t-3", "hypothesis_vi": "c"}]

    with pytest.raises(EvaluationError, match="duplicate"):
        run_g_test_inference(arm=ARM_D0, test_frame=frame, checkpoint_fingerprint=fp, generate_fn=dup_fn)

    def extra_fn(_):
        return [{"record_uid": "t-1", "hypothesis_vi": "a"},
                {"record_uid": "t-2", "hypothesis_vi": "b"},
                {"record_uid": "t-3", "hypothesis_vi": "c"},
                {"record_uid": "t-9", "hypothesis_vi": "x"}]

    with pytest.raises(EvaluationError, match="extra"):
        run_g_test_inference(arm=ARM_D0, test_frame=frame, checkpoint_fingerprint=fp, generate_fn=extra_fn)

    def missing_fn(_):
        return [{"record_uid": "t-1", "hypothesis_vi": "a"},
                {"record_uid": "t-2", "hypothesis_vi": "b"}]

    with pytest.raises(EvaluationError, match="missing"):
        run_g_test_inference(arm=ARM_D0, test_frame=frame, checkpoint_fingerprint=fp, generate_fn=missing_fn)

    def wrong_group_fn(_):
        return [{"record_uid": "t-1", "hypothesis_vi": "a", "group_id": "wrong"},
                {"record_uid": "t-2", "hypothesis_vi": "b"},
                {"record_uid": "t-3", "hypothesis_vi": "c"}]

    with pytest.raises(EvaluationError, match="group_id"):
        run_g_test_inference(arm=ARM_D0, test_frame=frame, checkpoint_fingerprint=fp, generate_fn=wrong_group_fn)

    def ok_fn(test):
        return [{"record_uid": str(uid), "hypothesis_vi": "ok", "group_id": "ignored-if-absent"} for uid in test["record_uid"]]

    # group_id omitted is fine; canonical frozen group_id is kept
    out = run_g_test_inference(arm=ARM_D0, test_frame=frame, checkpoint_fingerprint=fp, generate_fn=lambda t: [
        {"record_uid": str(uid), "hypothesis_vi": "ok"} for uid in t["record_uid"]
    ])
    assert out["group_id"].astype(str).tolist() == frame["group_id"].astype(str).tolist()


def test_durable_rq1_resolver_locates_final_and_test_contracts(tmp_path):
    env = world(tmp_path, with_test=True)
    from src.rq2_final_evaluate import resolve_frozen_rq1_final_state

    resolved = resolve_frozen_rq1_final_state(tmp_path, durable_root=env["durable_root"])
    assert (resolved["state_dir"] / "rq1_final_contract.json").is_file()
    assert (resolved["state_dir"] / "rq1_test_contract.json").is_file()
    assert "data/manifests/rq1_test_contract.json" not in str(resolved["state_dir"])
    assert resolved["rq1_final_contract_hash"] == resolved["final_contract"]["rq1_final_contract_hash"]
    assert resolved["rq1_test_contract_hash"] == resolved["test_contract"]["rq1_test_contract_hash"]


def _valid_mapping(tmp_path):
    env, upstream, fields, best = _ready_publish_inputs(tmp_path)
    artifact_dir = tmp_path / "final_arts"
    mapping = persist_final_artifacts(
        artifact_dir,
        frames=_frames(),
        contract_fields=fields,
        best=best,
        n_samples=10,
        seed=42,
        confidence=0.95,
    )
    return env, mapping, artifact_dir


def test_publisher_rejects_fake_upstream_sha_and_mutated_artifacts(tmp_path):
    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    contract_path = artifact_dir / "final_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["nb11_input_contract_sha256"] = "aa" * 32
    contract["rq2_final_contract_sha256"] = "bb" * 32
    from src.rq2_final_contract import build_final_contract

    rebuilt = build_final_contract({k: v for k, v in contract.items() if k != "rq2_final_contract_sha256"})
    contract_path.write_text(json.dumps(rebuilt), encoding="utf-8")
    from src.rq2_final_evaluate import hash_artifact_files
    from src.rq2_pseudo_contract import write_json

    write_json(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    mapping["final_contract.json"] = contract_path
    mapping["artifact_hashes.json"] = artifact_dir / "artifact_hashes.json"
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert result["wrote_current"] is False

    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    (artifact_dir / "arm_comparison.json").write_text((artifact_dir / "arm_comparison.json").read_text() + " ", encoding="utf-8")
    write_json = __import__("src.rq2_pseudo_contract", fromlist=["write_json"]).write_json
    from src.rq2_final_evaluate import hash_artifact_files
    write_json(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert "comparison" in str(result.get("error") or "").lower() or result["status"] == STATUS_FAIL

    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    (artifact_dir / "best_checkpoints.json").write_text((artifact_dir / "best_checkpoints.json").read_text() + " ", encoding="utf-8")
    from src.rq2_final_evaluate import hash_artifact_files
    from src.rq2_pseudo_contract import write_json as wj
    wj(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL

    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    summary["status"] = "NOT_SUCCESS"
    (artifact_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    from src.rq2_final_evaluate import hash_artifact_files
    from src.rq2_pseudo_contract import write_json as wj2
    wj2(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert "SUCCESS" in str(result.get("error") or "") or result["status"] == STATUS_FAIL

    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    contract = json.loads((artifact_dir / "final_contract.json").read_text(encoding="utf-8"))
    contract["source_fingerprint_sha256"] = "cc" * 32
    rebuilt = build_final_contract({k: v for k, v in contract.items() if k != "rq2_final_contract_sha256"})
    (artifact_dir / "final_contract.json").write_text(json.dumps(rebuilt), encoding="utf-8")
    from src.rq2_final_evaluate import hash_artifact_files
    from src.rq2_pseudo_contract import write_json as wj3
    wj3(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert "fingerprint" in str(result.get("error") or "").lower()


def test_finalized_generation_failure_does_not_update_current(tmp_path, monkeypatch):
    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    import src.rq2_final_evaluate as evaluate_mod

    real = evaluate_mod._verify_staged_final_generation
    calls = {"n": 0}

    def wrapped(gen_dir, *, project_root, durable_root=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise EvaluationError("forced finalize failure")
        return real(gen_dir, project_root=project_root, durable_root=durable_root)

    monkeypatch.setattr(evaluate_mod, "_verify_staged_final_generation", wrapped)
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert result["wrote_current"] is False
    assert not (tmp_path / "artifacts" / "rq2" / "final" / "CURRENT").exists()


def test_g_test_reader_never_called_when_arm_proof_invalid(tmp_path, monkeypatch):
    bundle = g_test_proof_bundle(tmp_path)
    original_random_fp = bundle["best"][ARM_RANDOM]["checkpoint_fingerprint"]
    called = {"reader": 0}
    import src.rq2_final_evaluate as evaluate_mod

    def boom(*args, **kwargs):
        called["reader"] += 1
        raise AssertionError("G_test reader invoked before proofs succeeded")

    monkeypatch.setattr(evaluate_mod, "resolve_frozen_rq1_final_state", boom)
    bad = dict(bundle["best"])
    bad[ARM_RANDOM] = dict(bundle["best"][ARM_RANDOM])
    bad[ARM_RANDOM]["checkpoint_fingerprint"] = "00" * 32
    with pytest.raises(GTestFirewallError):
        load_frozen_test_frame(
            project_root=tmp_path,
            flags=bundle["flags"],
            readiness=all_ready(),
            layouts=bundle["layouts"],
            contracts=bundle["contracts"],
            best=bad,
            d0_identity=bundle["d0_identity"],
            upstream=bundle["upstream"],
        )
    assert called["reader"] == 0
    empty_best = dict(bundle["best"][ARM_QUALITY])
    empty_best["checkpoint_fingerprint"] = ""
    best = {
        ARM_RANDOM: dict(bundle["best"][ARM_RANDOM], checkpoint_fingerprint=original_random_fp),
        ARM_QUALITY: empty_best,
        ARM_D0: bundle["best"].get(ARM_D0),
    }
    with pytest.raises(GTestFirewallError, match="empty"):
        load_frozen_test_frame(
            project_root=tmp_path,
            flags=bundle["flags"],
            readiness=all_ready(),
            layouts=bundle["layouts"],
            contracts=bundle["contracts"],
            best=best,
            d0_identity=bundle["d0_identity"],
            upstream=bundle["upstream"],
        )
    assert called["reader"] == 0


def test_publisher_rejects_bootstrap_mismatch(tmp_path):
    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    bootstrap = json.loads((artifact_dir / "bootstrap_results.json").read_text(encoding="utf-8"))
    first_key = next(iter(bootstrap["comparisons"]))
    bootstrap["comparisons"][first_key]["observed_sacrebleu_delta"] = 99.0
    (artifact_dir / "bootstrap_results.json").write_text(json.dumps(bootstrap), encoding="utf-8")
    from src.rq2_final_contract import build_final_contract
    from src.rq2_final_evaluate import hash_artifact_files
    from src.rq2_pseudo_contract import write_json
    from src.rq1_contract import sha256_file

    contract = json.loads((artifact_dir / "final_contract.json").read_text(encoding="utf-8"))
    contract["bootstrap_artifact_sha256"] = sha256_file(artifact_dir / "bootstrap_results.json")
    rebuilt = build_final_contract({k: v for k, v in contract.items() if k != "rq2_final_contract_sha256"})
    (artifact_dir / "final_contract.json").write_text(json.dumps(rebuilt), encoding="utf-8")
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    summary["rq2_final_contract_sha256"] = rebuilt["rq2_final_contract_sha256"]
    (artifact_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_json(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert "bootstrap" in str(result.get("error") or "").lower()
    assert "recompute" in str(result.get("error") or "").lower()


def test_publisher_rejects_missing_checkpoint_ownership(tmp_path):
    env, mapping, artifact_dir = _valid_mapping(tmp_path)
    best = json.loads((artifact_dir / "best_checkpoints.json").read_text(encoding="utf-8"))
    best[ARM_RANDOM].pop("arm_training_contract_sha256", None)
    best[ARM_RANDOM].pop("direct_training_contract_hash", None)
    (artifact_dir / "best_checkpoints.json").write_text(json.dumps(best), encoding="utf-8")
    from src.rq2_final_contract import build_final_contract
    from src.rq2_final_evaluate import hash_artifact_files
    from src.rq2_pseudo_contract import write_json
    from src.rq1_contract import sha256_file

    contract = json.loads((artifact_dir / "final_contract.json").read_text(encoding="utf-8"))
    contract["best_checkpoints_artifact_sha256"] = sha256_file(artifact_dir / "best_checkpoints.json")
    rebuilt = build_final_contract({k: v for k, v in contract.items() if k != "rq2_final_contract_sha256"})
    (artifact_dir / "final_contract.json").write_text(json.dumps(rebuilt), encoding="utf-8")
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    summary["rq2_final_contract_sha256"] = rebuilt["rq2_final_contract_sha256"]
    (artifact_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_json(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    result = publish_final_generation(tmp_path, artifacts=mapping, durable_root=str(env["durable_root"]))
    assert result["status"] == STATUS_FAIL
    assert "ownership" in str(result.get("error") or "").lower()


