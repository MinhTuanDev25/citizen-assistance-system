"""Notebook 14 single-seed and multi-seed orchestration without GPU or G_test."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.rq2_final_contract import (
    ARM_D0,
    ARM_QUALITY,
    ARM_RANDOM,
    GTestFirewallError,
    Nb14Flags,
    SEED_POLICY_MULTI,
    SEED_POLICY_SINGLE,
    STATUS_SUCCESS,
    STATUS_TRAINING_COMPLETE,
    TrainingContractError,
    evaluation_protocol,
    experiment_root_for_arm,
    iter_declared_seed_runs,
)
from src.rq2_final_contract import FINAL_ARTIFACT_FILES
from src.rq2_final_evaluate import (
    EvaluationError,
    STATUS_FAIL,
    derive_final_readiness,
    hash_artifact_files,
    persist_final_artifacts,
    prediction_row,
    publish_final_generation,
    require_g_test_unlocked,
)
from src.rq2_pseudo_contract import write_json
from src.rq2_final_train import (
    ArmIsolationError,
    bind_d0_arm,
    checkpoint_model_state_sha256,
    discover_latest_valid_checkpoint,
    ensure_arm_root,
    write_training_complete_proof,
)
from src.rq2_final_contract import assert_checkpoint_arm_isolation
from tests.rq2_nb14_fixtures import (
    synthetic_final_contract_fields,
    world,
    write_complete_checkpoint,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _frames(seed: int, *, d0_fp: str, random_fp: str, quality_fp: str) -> dict:
    uids = ["t-1", "t-2", "t-3"]
    refs = [
        "đây là câu dài để kiểm tra tham chiếu không bị cắt",
        "câu thứ hai",
        "câu thứ ba",
    ]
    groups = ["g1", "g1", "g2"]
    quality = {7: ["aa", "bb", "c"], 11: ["aaa", "b", "cc"], 13: ["aaaa", "bbb", "ccc"], 42: ["aa", "b", "c"]}
    hyps = {
        ARM_D0: ["a", "b", "c"],
        ARM_RANDOM: ["a", "b", "c"],
        ARM_QUALITY: quality[seed],
    }
    fps = {ARM_D0: d0_fp, ARM_RANDOM: random_fp, ARM_QUALITY: quality_fp}
    return {
        arm: pd.DataFrame([
            prediction_row(
                record_uid=uid,
                group_id=group,
                hypothesis_vi=hyp,
                reference_vi=ref,
                arm=arm,
                checkpoint_fingerprint=fps[arm],
            )
            for uid, hyp, ref, group in zip(uids, hyps[arm], refs, groups)
        ])
        for arm in (ARM_D0, ARM_RANDOM, ARM_QUALITY)
    }


def _orchestrate(tmp_path: Path, seeds: list):
    env = world(tmp_path, with_test=True)
    from src.rq2_final_contract import verify_upstream_rq2

    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    mode = "single" if len(seeds) == 1 else "multi"
    policy_name = SEED_POLICY_SINGLE if mode == "single" else SEED_POLICY_MULTI
    nb13 = str(upstream["nb13_generation_id"])
    d0_fp = str(upstream["d0"]["model_state_sha256"])
    d0_ckpt = str(upstream["d0"]["checkpoint_fingerprint_sha256"])
    manifests = {
        ARM_RANDOM: str(upstream["d_random_manifest_sha256"]),
        ARM_QUALITY: str(upstream["d_quality_manifest_sha256"]),
    }
    best = {}
    d0_layout = ensure_arm_root(tmp_path / "d0-production")
    bound = bind_d0_arm(
        layout=d0_layout,
        flags=env["flags"],
        training_contract={"arm": ARM_D0},
        fingerprint={"arm": ARM_D0},
    )
    best[ARM_D0] = bound["best"]
    layouts = {}
    contracts = {}
    completes = {}
    seed_runs = {}
    live = {}
    for arm in (ARM_RANDOM, ARM_QUALITY):
        best[arm] = {}
        layouts[arm] = {}
        contracts[arm] = {"contract": None, "by_seed": {}, "seed_policy": None}
        completes[arm] = []
        seed_runs[arm] = {}
        for seed in seeds:
            contract_hash = _hash(f"{arm}:{seed}")
            data_hash = manifests[arm]
            root = experiment_root_for_arm(
                durable_root=env["durable_root"], contract_hash=contract_hash, arm=arm,
            )
            layout = ensure_arm_root(root)
            ckpt = write_complete_checkpoint(
                layout["checkpoints"] / "checkpoint-100",
                arm=arm,
                contract_hash=contract_hash,
                data_hash=data_hash,
                step=100,
                weights=f"{arm}-{seed}-best".encode(),
                fingerprint_extra={"training_seed": seed, "nb13_generation_id": nb13},
            )
            terminal = write_complete_checkpoint(
                layout["checkpoints"] / "checkpoint-101",
                arm=arm,
                contract_hash=contract_hash,
                data_hash=data_hash,
                step=101,
                weights=f"{arm}-{seed}-terminal".encode(),
                fingerprint_extra={"training_seed": seed, "nb13_generation_id": nb13},
            )
            fingerprint = checkpoint_model_state_sha256(ckpt)
            live[(arm, seed)] = fingerprint
            record = {
                "arm": arm,
                "active_seed": seed,
                "checkpoint_name": "checkpoint-100",
                "checkpoint_fingerprint": fingerprint,
                "arm_training_contract_sha256": contract_hash,
                "data_manifest_sha256": data_hash,
                "data_contract_sha256": data_hash,
                "experiment_fingerprint_sha256": "99" * 32,
                "validation_metric": float(seed),
                "global_step": 100,
                "g_test_used": False,
                "nb13_generation_id": nb13,
                "d0_init_model_state_sha256": d0_fp,
                "selected_manifest_sha256": data_hash,
            }
            best[arm][seed] = record
            contract = {
                "arm": arm,
                "active_seed": seed,
                "arm_training_contract_sha256": contract_hash,
                "data_manifest_sha256": data_hash,
                "selected_manifest_sha256": data_hash,
                "nb13_generation_id": nb13,
                "nb13_selection_contract_sha256": upstream["nb13_selection_contract_sha256"],
                "d0_init_model_state_sha256": d0_fp,
                "d0_checkpoint_fingerprint_sha256": d0_ckpt,
                "seed_policy": policy_name,
                "seed_policy_seeds": list(seeds),
                "seed_policy_record": {"mode": mode, "seeds": list(seeds)},
            }
            run_fingerprint = {
                "nb13_generation_id": nb13,
                "selected_manifest_sha256": data_hash,
                "d0_init_model_state_sha256": d0_fp,
            }
            proof = write_training_complete_proof(
                layout,
                arm=arm,
                training_contract_sha256=contract_hash,
                final_global_step=101,
                checkpoint_path=terminal,
                best_checkpoint=record,
            )
            layouts[arm][seed] = layout
            contracts[arm]["by_seed"][seed] = {
                "contract": contract,
                "fingerprint": run_fingerprint,
                "layout": layout,
            }
            contracts[arm]["seed_policy"] = contract
            completes[arm].append(proof)
            seed_runs[arm][seed] = {"layout": layout, "contract": contract}
    if mode == "single":
        for arm in (ARM_RANDOM, ARM_QUALITY):
            proof = dict(completes[arm][0])
            proof["seed_runs"] = [dict(completes[arm][0])]
            completes[arm] = proof
    else:
        for arm in (ARM_RANDOM, ARM_QUALITY):
            completes[arm] = {
                "status": STATUS_TRAINING_COMPLETE,
                "arm": arm,
                "g_test_used": False,
                "seed_runs": completes[arm],
            }
    flags = Nb14Flags(
        run_real_training=False,
        allow_g_test_evaluation=True,
        rq2_final_frozen=True,
        direct_state_dir=str(env["d0_dir"]),
        durable_root=str(env["durable_root"]),
    )
    return {
        "env": env,
        "upstream": upstream,
        "seeds": list(seeds),
        "mode": mode,
        "policy_name": policy_name,
        "best": best,
        "layouts": layouts,
        "contracts": contracts,
        "completes": completes,
        "seed_runs": seed_runs,
        "live": live,
        "flags": flags,
        "d0_fp": d0_ckpt,
        "nb13": nb13,
    }


def _preflight(seed_runs):
    touched = []
    for arm, runs in seed_runs.items():
        for active_seed, run in runs.items():
            if int(run["contract"]["active_seed"]) != int(active_seed):
                raise AssertionError(active_seed)
            (run["layout"]["root"] / "target_truncation_audit.json").write_text("{}\n", encoding="utf-8")
            touched.append((arm, active_seed))
    assert len(iter_declared_seed_runs(seed_runs)) == len(touched)
    return touched


def _publish(tmp_path, built, *, mutate=None):
    seeds = built["seeds"]
    d0_fp = str(built["best"][ARM_D0]["checkpoint_fingerprint"])
    per_seed_frames = {
        seed: _frames(
            seed,
            d0_fp=d0_fp,
            random_fp=built["live"][(ARM_RANDOM, seed)],
            quality_fp=built["live"][(ARM_QUALITY, seed)],
        )
        for seed in seeds
    }
    fields = synthetic_final_contract_fields(tmp_path, built["upstream"])
    fields["d0_checkpoint_fingerprint"] = d0_fp
    fields["seed_policy_record"] = {"mode": built["mode"], "seeds": list(seeds)}
    fields["seed_policy"] = built["policy_name"]
    fields["seed_policy_seeds"] = list(seeds)
    fields["resolved_nb13_generation_id"] = built["nb13"]
    fields["d_random_training_complete"] = built["completes"][ARM_RANDOM]
    fields["d_quality_training_complete"] = built["completes"][ARM_QUALITY]
    if built["mode"] == "single":
        seed = seeds[0]
        fields["d_random_best_checkpoint_fingerprint"] = built["live"][(ARM_RANDOM, seed)]
        fields["d_quality_best_checkpoint_fingerprint"] = built["live"][(ARM_QUALITY, seed)]
        fields["d_random_training_contract_sha256"] = _hash(f"{ARM_RANDOM}:{seed}")
        fields["d_quality_training_contract_sha256"] = _hash(f"{ARM_QUALITY}:{seed}")
        fields["d_random_best_checkpoint"] = "checkpoint-100"
        fields["d_quality_best_checkpoint"] = "checkpoint-100"
    else:
        for key in (
            "d_random_best_checkpoint_fingerprint",
            "d_quality_best_checkpoint_fingerprint",
            "d_random_training_contract_sha256",
            "d_quality_training_contract_sha256",
            "d_random_best_checkpoint",
            "d_quality_best_checkpoint",
        ):
            fields.pop(key, None)
    protocol = evaluation_protocol(built["upstream"]["rq1_eval"])
    artifact_dir = tmp_path / "final_arts"
    mapping = persist_final_artifacts(
        artifact_dir,
        frames=per_seed_frames[seeds[0]],
        per_seed_frames=per_seed_frames,
        contract_fields=fields,
        best=built["best"],
        n_samples=int(protocol["bootstrap_samples"]),
        seed=int(protocol["seed"]),
        confidence=float(protocol["confidence"]),
    )
    if mutate is not None:
        mutate(artifact_dir, mapping)
    result = publish_final_generation(
        tmp_path,
        artifacts=mapping,
        durable_root=str(built["env"]["durable_root"]),
    )
    return result, mapping, artifact_dir


def _refresh_scientific_hashes(artifact_dir, mapping):
    """Keep semantic tamper tests on the verifier path after the hash manifest expanded."""
    write_json(artifact_dir / "artifact_hashes.json", {"files": hash_artifact_files(artifact_dir)})
    mapping["artifact_hashes.json"] = artifact_dir / "artifact_hashes.json"


def _dynamic_hash_paths(seeds):
    paths = ["aggregate/metrics.json", "evaluation/aggregate.json"]
    for seed in seeds:
        paths.extend([
            f"evaluation/seed-{seed}/d_random.parquet",
            f"evaluation/seed-{seed}/d_quality.parquet",
            f"evaluation/seed-{seed}/metrics.json",
            f"evaluation/seed-{seed}/bootstrap.json",
            f"training/{ARM_RANDOM}/seed-{seed}.json",
            f"training/{ARM_QUALITY}/seed-{seed}.json",
            f"evaluation/per_seed/seed-{seed}.json",
        ])
    return paths


def _assert_hash_coverage(generation: Path, seeds):
    payload = json.loads((generation / "artifact_hashes.json").read_text(encoding="utf-8"))
    files = payload["files"]
    assert "artifact_hashes.json" not in files
    assert "COMPLETE.json" not in files
    assert "artifact_manifest.json" not in files
    expected = [rel for rel in FINAL_ARTIFACT_FILES if rel != "artifact_hashes.json"]
    expected.extend(_dynamic_hash_paths(seeds))
    for rel in expected:
        assert rel in files, rel
        assert (generation / rel).is_file()


def _ready(built):
    protocol = evaluation_protocol(built["upstream"]["rq1_eval"])
    return derive_final_readiness(
        flags=built["flags"],
        upstream=built["upstream"],
        layouts=built["layouts"],
        contracts=built["contracts"],
        best=built["best"],
        protocol=protocol,
    )


def test_preflight_dict_iteration_single_and_multi(tmp_path):
    single = _orchestrate(tmp_path / "single", [42])
    multi = _orchestrate(tmp_path / "multi", [7, 11, 13])
    assert _preflight(single["seed_runs"]) == [(ARM_RANDOM, 42), (ARM_QUALITY, 42)]
    assert [seed for _arm, seed in _preflight(multi["seed_runs"])] == [7, 11, 13, 7, 11, 13]


def test_nb14_single_seed_orchestration_end_to_end(tmp_path):
    built = _orchestrate(tmp_path, [42])
    assert built["best"][ARM_D0]["validation_metric"] is None
    assert built["best"][ARM_D0]["global_step"] is None
    assert "checkpoint_fingerprint" not in built["best"][ARM_RANDOM]
    assert set(built["best"][ARM_RANDOM]) == {42}
    _preflight(built["seed_runs"])
    readiness = _ready(built)
    assert readiness["all_ok"] is True
    assert readiness["gates"]["d0_best_checkpoint_frozen"]["ok"] is True
    assert readiness["gates"]["best_checkpoints_frozen"]["ok"] is True
    assert readiness["gates"]["validation_selection_complete"]["ok"] is True
    invalid_d0 = copy.deepcopy(built["best"])
    invalid_d0[ARM_D0]["checkpoint_fingerprint"] = "zz"
    assert _ready({**built, "best": invalid_d0})["gates"]["d0_best_checkpoint_frozen"]["ok"] is False
    missing_metric = copy.deepcopy(built["best"])
    missing_metric[ARM_RANDOM][42]["validation_metric"] = None
    assert _ready({**built, "best": missing_metric})["all_ok"] is False
    missing_quality_fp = copy.deepcopy(built["best"])
    missing_quality_fp[ARM_QUALITY][42]["checkpoint_fingerprint"] = ""
    assert _ready({**built, "best": missing_quality_fp})["gates"]["d_quality_best_checkpoint_frozen"]["ok"] is False
    unlocked = require_g_test_unlocked(
        built["flags"],
        readiness,
        layouts=built["layouts"],
        contracts=built["contracts"],
        best=built["best"],
        d0_identity=built["upstream"]["d0"],
        upstream=built["upstream"],
    )
    assert unlocked["allow_g_test_evaluation"] is True
    result, _mapping, _artifact_dir = _publish(tmp_path, built)
    assert result["status"] == STATUS_SUCCESS, result.get("error")
    assert result["wrote_current"] is True
    current = (built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "CURRENT").read_text().strip()
    generation = built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "generations" / current
    assert (generation / "evaluation" / "seed-42" / "d_random.parquet").is_file()
    assert (generation / "evaluation" / "seed-42" / "d_quality.parquet").is_file()
    assert (generation / "training" / ARM_RANDOM / "seed-42.json").is_file()
    _assert_hash_coverage(generation, [42])
    aggregate = json.loads((generation / "aggregate" / "metrics.json").read_text())
    assert aggregate["n_valid_runs"] == 1
    assert aggregate["std_convention"] == "population_pstdev"
    assert aggregate[ARM_RANDOM]["sacrebleu"]["std"] is None
    from src.rq2_final_evaluate import _verify_staged_final_generation

    verified = _verify_staged_final_generation(
        generation, project_root=tmp_path, durable_root=str(built["env"]["durable_root"]),
    )
    assert verified["summary_status"] == STATUS_SUCCESS


def test_nb14_multi_seed_orchestration_end_to_end(tmp_path):
    built = _orchestrate(tmp_path, [7, 11, 13])
    assert set(built["best"][ARM_RANDOM]) == {7, 11, 13}
    assert set(built["best"][ARM_QUALITY]) == {7, 11, 13}
    assert len(_preflight(built["seed_runs"])) == 6
    partial = json.loads(json.dumps(built["best"]))
    del partial[ARM_QUALITY]["11"]
    with pytest.raises(GTestFirewallError, match="missing"):
        require_g_test_unlocked(
            built["flags"],
            _ready(built),
            layouts=built["layouts"],
            contracts=built["contracts"],
            best=partial,
            d0_identity=built["upstream"]["d0"],
            upstream=built["upstream"],
        )
    readiness = _ready(built)
    assert readiness["gates"]["d_random_complete"]["n_seed_runs"] == 3
    assert readiness["gates"]["d_quality_complete"]["n_seed_runs"] == 3
    assert readiness["gates"]["d0_best_checkpoint_frozen"]["ok"] is True
    assert built["best"][ARM_D0]["validation_metric"] is None
    assert readiness["all_ok"] is True
    invalid_d0 = copy.deepcopy(built["best"])
    invalid_d0[ARM_D0]["model_state_sha256"] = "not-a-sha"
    assert _ready({**built, "best": invalid_d0})["all_ok"] is False
    unlocked = require_g_test_unlocked(
        built["flags"],
        readiness,
        layouts=built["layouts"],
        contracts=built["contracts"],
        best=built["best"],
        d0_identity=built["upstream"]["d0"],
        upstream=built["upstream"],
    )
    assert set(unlocked["arm_proofs"][ARM_RANDOM]) == {7, 11, 13}
    assert set(unlocked["arm_proofs"][ARM_QUALITY]) == {7, 11, 13}
    result, _mapping, _artifact_dir = _publish(tmp_path, built)
    assert result["status"] == STATUS_SUCCESS, result.get("error")
    current = (built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "CURRENT").read_text().strip()
    generation = built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "generations" / current
    for seed in (7, 11, 13):
        assert (generation / "evaluation" / f"seed-{seed}" / "d_random.parquet").is_file()
        assert (generation / "evaluation" / f"seed-{seed}" / "metrics.json").is_file()
        assert (generation / "evaluation" / f"seed-{seed}" / "bootstrap.json").is_file()
        assert (generation / "training" / ARM_QUALITY / f"seed-{seed}.json").is_file()
    _assert_hash_coverage(generation, [7, 11, 13])
    aggregate = json.loads((generation / "aggregate" / "metrics.json").read_text())
    assert aggregate["n_valid_runs"] == 3
    assert aggregate[ARM_QUALITY]["sacrebleu"]["std"] is not None
    assert len(aggregate["treatment"]["per_seed_delta_sacrebleu"]) == 3
    from src.rq2_final_evaluate import _verify_staged_final_generation

    _verify_staged_final_generation(
        generation, project_root=tmp_path, durable_root=str(built["env"]["durable_root"]),
    )


def test_missing_fingerprint_and_wrong_seed_keep_g_test_locked(tmp_path):
    built = _orchestrate(tmp_path, [7, 11, 13])
    missing = json.loads(json.dumps(built["best"]))
    missing[ARM_QUALITY]["11"]["checkpoint_fingerprint"] = ""
    with pytest.raises(GTestFirewallError, match="fingerprint"):
        require_g_test_unlocked(
            built["flags"], _ready(built), layouts=built["layouts"], contracts=built["contracts"],
            best=missing, d0_identity=built["upstream"]["d0"], upstream=built["upstream"],
        )
    wrong = json.loads(json.dumps(built["best"]))
    wrong[ARM_RANDOM]["11"]["active_seed"] = 99
    with pytest.raises(GTestFirewallError, match="11"):
        require_g_test_unlocked(
            built["flags"], _ready(built), layouts=built["layouts"], contracts=built["contracts"],
            best=wrong, d0_identity=built["upstream"]["d0"], upstream=built["upstream"],
        )


def test_cross_seed_and_cross_arm_checkpoints_are_rejected(tmp_path):
    built = _orchestrate(tmp_path, [7, 11])
    with pytest.raises(TrainingContractError, match="active_seed"):
        discover_latest_valid_checkpoint(
            built["layouts"][ARM_RANDOM][7],
            arm=ARM_RANDOM,
            expected_contract_hash=_hash(f"{ARM_RANDOM}:7"),
            expected_data_hash=built["upstream"]["d_random_manifest_sha256"],
            expected_active_seed=11,
        )
    with pytest.raises(ArmIsolationError):
        assert_checkpoint_arm_isolation(
            built["layouts"][ARM_RANDOM][7]["checkpoints"] / "checkpoint-100",
            arm=ARM_QUALITY,
            expected_contract_hash=_hash(f"{ARM_RANDOM}:7"),
            expected_data_hash=built["upstream"]["d_random_manifest_sha256"],
        )


def test_seed_publication_rejects_alignment_ownership_aggregate_and_partial(tmp_path):

    def _failing(tmp, mutate):
        built = _orchestrate(tmp, [7, 11, 13])
        pointer = built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "CURRENT"
        assert not pointer.exists()
        result, _mapping, _arts = _publish(tmp, built, mutate=mutate)
        assert result["status"] == STATUS_FAIL
        assert result["wrote_current"] is False
        assert not pointer.exists()
        return result

    def uid_mismatch(artifact_dir, mapping):
        path = artifact_dir / "evaluation" / "seed-11" / "d_quality.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "record_uid"] = "other"
        frame.to_parquet(path, index=False)
        _refresh_scientific_hashes(artifact_dir, mapping)

    def reference_mismatch(artifact_dir, mapping):
        path = artifact_dir / "evaluation" / "seed-11" / "d_quality.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "reference_vi"] = "khác"
        frame.to_parquet(path, index=False)
        _refresh_scientific_hashes(artifact_dir, mapping)

    def fingerprint_mismatch(artifact_dir, mapping):
        path = artifact_dir / "evaluation" / "seed-11" / "d_random.parquet"
        frame = pd.read_parquet(path)
        frame["checkpoint_fingerprint"] = "ab" * 32
        frame.to_parquet(path, index=False)
        _refresh_scientific_hashes(artifact_dir, mapping)

    def aggregate_mismatch(artifact_dir, mapping):
        path = artifact_dir / "aggregate" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[ARM_RANDOM]["sacrebleu"]["mean"] = 12345.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        _refresh_scientific_hashes(artifact_dir, mapping)

    def partial(artifact_dir, mapping):
        path = artifact_dir / "evaluation" / "seed-11" / "d_quality.parquet"
        path.unlink()
        mapping.pop("evaluation/seed-11/d_quality.parquet", None)

    def chrf_delta_mismatch(artifact_dir, mapping):
        path = artifact_dir / "aggregate" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["treatment"]["per_seed_delta_chrfpp"][0] = float(payload["treatment"]["per_seed_delta_chrfpp"][0]) + 1.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        _refresh_scientific_hashes(artifact_dir, mapping)

    def bootstrap_ci_mismatch(artifact_dir, mapping):
        path = artifact_dir / "evaluation" / "seed-11" / "bootstrap.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["comparisons"]["d_quality-d_random"]["sacrebleu_ci_lower"] = -999.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        _refresh_scientific_hashes(artifact_dir, mapping)

    assert "UID order" in _failing(tmp_path / "uid", uid_mismatch)["error"]
    assert "reference" in _failing(tmp_path / "ref", reference_mismatch)["error"]
    assert "fingerprint" in _failing(tmp_path / "fp", fingerprint_mismatch)["error"]
    assert "mean" in _failing(tmp_path / "agg", aggregate_mismatch)["error"]
    assert "per_seed_delta_chrfpp" in _failing(tmp_path / "chrf", chrf_delta_mismatch)["error"]
    assert "sacrebleu_ci_lower" in _failing(tmp_path / "ci", bootstrap_ci_mismatch)["error"]
    partial_result = _failing(tmp_path / "partial", partial)
    assert "missing" in partial_result["error"].lower() or "No such file" in partial_result["error"]


def test_notebook_production_source_has_no_seed_skip_patterns():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks" / "14_RQ2_Final_Train_Evaluate.ipynb").read_text())
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"] if cell["cell_type"] == "code")
    assert "RUN_REAL_TRAINING = False" in source
    assert "ALLOW_G_TEST_EVALUATION = False" in source
    assert "RQ2_FINAL_FROZEN = False" in source
    assert 'GOLD_PSEUDO_MIX_POLICY = "configured_gold_pseudo_slot_ratio"' in source
    assert "GOLD_SLOTS = 5" in source
    assert "PSEUDO_SLOTS = 1" in source
    assert 'SEED_POLICY = "multi_seed"' in source
    assert "SEED_POLICY_SEEDS = [13, 17, 23]" in source
    assert "for active_seed, run in runs.items()" in source
    assert "for run in runs:" not in source
    assert "summarize_seed_runs([]" not in source
    assert "FRAMES = None" not in source
    assert "runs[0]" not in source
    assert "(not fp or is_sha256(fp))" not in source
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell.get("source") or []), "14_RQ2_Final_Train_Evaluate.ipynb", "exec")
