"""Synthetic fixtures for Notebook 14. No real training and no G_test access."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
from src.direct_contract import (
    LOCKED_ACCELERATE_VERSION,
    LOCKED_DECODER_ID,
    LOCKED_DECODER_REVISION,
    LOCKED_ENCODER_ID,
    LOCKED_ENCODER_REVISION,
    LOCKED_EXPERIMENT_ID,
    LOCKED_TARGET_LANG,
    LOCKED_TORCH_VERSION,
    LOCKED_TRANSFORMERS_VERSION,
    STATUS_DIRECT_EVALUATE,
    STATUS_DIRECT_TRAINING,
    build_direct_training_contract,
)
from src.direct_data import DIRECT_TRAIN_CSV, DIRECT_VAL_CSV, compute_pair_hash
from src.mt_normalize import normalize_mt_text_v1
from src.rq1_contract import (
    PARQUET_SOURCE_RQ1_CONFIG_LEGACY,
    build_rq1_test_contract,
    sha256_file,
    sha256_json,
)
from src.rq2_final_contract import ARM_D0, ARM_QUALITY, ARM_RANDOM, Nb14Flags
from src.rq2_selection import publish_same_budget_selection
from src.rq2_selection_contract import SELECTION_RELATIVE_DIR, resolve_frozen_u_prime
from tests.rq2_nb13_fixtures import seal_nb12_generation, u_prime_row


ARM_FINGERPRINTS = {
    ARM_D0: "a" * 64,
    ARM_RANDOM: "b" * 64,
    ARM_QUALITY: "c" * 64,
}


def pool_rows() -> List[Dict[str, Any]]:
    return [
        u_prime_row("a", 5, 0.1, text="xin chào, bạn"),
        u_prime_row("b", 4, 0.9, text="cảm ơn"),
        u_prime_row("c", 3, 0.9, text="tạm biệt"),
        u_prime_row("d", 1, 0.4, text="vâng"),
    ]


def supervised_rows() -> List[Dict[str, Any]]:
    return [
        {
            "record_uid": "sup-1",
            "text_vi": "đây là câu dài để kiểm tra tham chiếu không bị cắt",
            "text_vi_norm": normalize_mt_text_v1("đây là câu dài để kiểm tra tham chiếu không bị cắt"),
            "split": "train",
            "source_split": "g_train",
            "group_id": "gA",
            "duration_seconds": 2.0,
            "pcm16_sha256": "11" * 32,
        },
        {
            "record_uid": "sup-2",
            "text_vi": "câu huấn luyện thứ hai",
            "text_vi_norm": normalize_mt_text_v1("câu huấn luyện thứ hai"),
            "split": "train",
            "source_split": "g_train",
            "group_id": "gB",
            "duration_seconds": 1.5,
            "pcm16_sha256": "22" * 32,
        },
    ]


def validation_frame() -> pd.DataFrame:
    rows = [
        {
            "record_uid": "val-1",
            "text_vi": "tham chiếu validation",
            "text_vi_norm": normalize_mt_text_v1("tham chiếu validation"),
            "split": "validation",
            "group_id": "gV",
            "source_split": "g_validation",
            "pcm16_sha256": "c1" * 32,
        },
        {
            "record_uid": "val-2",
            "text_vi": "tham chiếu validation hai",
            "text_vi_norm": normalize_mt_text_v1("tham chiếu validation hai"),
            "split": "validation",
            "group_id": "gW",
            "source_split": "g_validation",
            "pcm16_sha256": "c2" * 32,
        },
    ]
    return pd.DataFrame(rows)


def test_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"record_uid": "t-1", "record_id": "r1", "source_split": "test", "parquet_file": "x.parquet", "shard_row_index": 0,
         "text_bahnar": "bah nar mot", "text_vi": "đây là câu dài để kiểm tra tham chiếu không bị cắt", "split": "test", "group_id": "g1"},
        {"record_uid": "t-2", "record_id": "r2", "source_split": "test", "parquet_file": "x.parquet", "shard_row_index": 1,
         "text_bahnar": "bah nar hai", "text_vi": "câu thứ hai", "split": "test", "group_id": "g1"},
        {"record_uid": "t-3", "record_id": "r3", "source_split": "test", "parquet_file": "x.parquet", "shard_row_index": 2,
         "text_bahnar": "bah nar ba", "text_vi": "câu thứ ba", "split": "test", "group_id": "g2"},
    ])


def _audio_pair_hash(frame: pd.DataFrame) -> str:
    pairs = [
        {"uid": str(uid), "audio": str(sha)}
        for uid, sha in zip(frame["record_uid"].astype(str), frame["pcm16_sha256"].astype(str))
    ]
    return sha256_json(pairs)


def write_locked_rq1_manifests(project_root: Path, *, with_test: bool = False) -> Dict[str, Any]:
    man = project_root / "data" / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(supervised_rows())
    val = validation_frame()
    train_path = man / "rq1_train.csv"
    val_path = man / "rq1_validation.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    summary_path = man / "split_summary.json"
    summary_path.write_text(json.dumps({"parquet_revision": "deadbeefcafebabe"}), encoding="utf-8")
    payload: Dict[str, Any] = {
        "train": train,
        "validation": val,
        "train_path": train_path,
        "validation_path": val_path,
        "summary_path": summary_path,
    }
    if with_test:
        frame = test_frame()
        test_path = man / "rq1_test.csv"
        frame.to_csv(test_path, index=False)
        contract = build_rq1_test_contract(
            frame,
            manifest_path=test_path,
            split_summary_sha256=sha256_file(summary_path),
            dataset_id="synthetic",
            dataset_revision="rev",
            parquet_revision="deadbeefcafebabe",
            parquet_revision_source=PARQUET_SOURCE_RQ1_CONFIG_LEGACY,
            train_manifest_sha256=sha256_file(train_path),
            validation_manifest_sha256=sha256_file(val_path),
        )
        payload["test"] = frame
        payload["test_path"] = test_path
        payload["test_contract"] = contract
    return payload


def write_frozen_d0_state(
    path: Path,
    *,
    manifests: Optional[Mapping[str, Any]] = None,
    write_sidecar: bool = True,
) -> Dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    train_src = (manifests or {}).get("train")
    val_src = (manifests or {}).get("validation")
    train = train_src.copy() if isinstance(train_src, pd.DataFrame) else pd.DataFrame(supervised_rows())
    val = val_src.copy() if isinstance(val_src, pd.DataFrame) else validation_frame()
    if "text_vi_norm" not in train.columns:
        train["text_vi_norm"] = train["text_vi"].map(normalize_mt_text_v1)
    if "text_vi_norm" not in val.columns:
        val["text_vi_norm"] = val["text_vi"].map(normalize_mt_text_v1)
    contract = build_direct_training_contract(
        direct_data_contract_hash="aa" * 32,
        experiment_id=LOCKED_EXPERIMENT_ID,
        encoder_id=LOCKED_ENCODER_ID,
        encoder_revision=LOCKED_ENCODER_REVISION,
        decoder_id=LOCKED_DECODER_ID,
        decoder_revision=LOCKED_DECODER_REVISION,
        target_lang=LOCKED_TARGET_LANG,
        tokenizer_fingerprint="bb" * 32,
        train_uid_set_hash=compute_uid_set_hash(train),
        validation_uid_set_hash=compute_uid_set_hash(val),
        monitor_uid_set_hash="ee" * 32,
        monitor_pair_hash="ff" * 32,
        monitor_ordered_row_hash="12" * 32,
        monitor_file_sha256="34" * 32,
        monitor_size=256,
        source_fingerprint_sha256="56" * 32,
        torch_version=LOCKED_TORCH_VERSION,
        transformers_version=LOCKED_TRANSFORMERS_VERSION,
        accelerate_version=LOCKED_ACCELERATE_VERSION,
        seed=42,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3.0,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        max_target_length=256,
        generation_max_length=256,
        num_beams=4,
        metric_for_best_model="eval_sacrebleu",
        greater_is_better=True,
        freeze_feature_encoder=True,
    )
    train_summary = {
        "status": STATUS_DIRECT_TRAINING,
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "direct_training_contract_hash": contract["direct_training_contract_hash"],
        "best_checkpoint_name": "checkpoint-100",
        "training_contract": contract,
    }
    evaluate = {
        "status": STATUS_DIRECT_EVALUATE,
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "direct_training_contract_hash": contract["direct_training_contract_hash"],
        "best_checkpoint_name": "checkpoint-100",
        "frozen_test_accessed": False,
        "training_contract": contract,
    }
    ckpt = path / "checkpoint-100"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(b"d0-weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")
    (ckpt / "scheduler.pt").write_bytes(b"sch")
    (ckpt / "rng_state.pth").write_bytes(b"rng")
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 100, "epoch": 1}), encoding="utf-8")
    fingerprint = {
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "kind": "direct",
        "direct_training_contract_hash": contract["direct_training_contract_hash"],
    }
    (ckpt / "full_experiment_fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    extra = {
        "train_ordered_uid_hash": compute_ordered_uid_hash(train),
        "validation_ordered_uid_hash": compute_ordered_uid_hash(val),
        "train_pair_hash": compute_pair_hash(train),
        "validation_pair_hash": compute_pair_hash(val),
        "train_audio_pair_hash": _audio_pair_hash(train) if "pcm16_sha256" in train.columns else "",
        "validation_audio_pair_hash": _audio_pair_hash(val) if "pcm16_sha256" in val.columns else "",
    }
    if manifests and manifests.get("train_path"):
        extra["train_file_sha256"] = sha256_file(manifests["train_path"])
    if manifests and manifests.get("validation_path"):
        extra["validation_file_sha256"] = sha256_file(manifests["validation_path"])
    train.to_csv(path / DIRECT_TRAIN_CSV, index=False)
    val.to_csv(path / DIRECT_VAL_CSV, index=False)
    (path / "direct_training_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (path / "direct_train_summary.json").write_text(json.dumps(train_summary), encoding="utf-8")
    (path / "direct_evaluate_summary.json").write_text(json.dumps(evaluate), encoding="utf-8")
    if write_sidecar:
        (path / "direct_supervised_identity.json").write_text(json.dumps(extra), encoding="utf-8")
    return contract


def write_complete_checkpoint(
    path: Path,
    *,
    arm: str,
    contract_hash: str,
    data_hash: str,
    step: int = 100,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "optimizer.pt").write_bytes(b"opt")
    (path / "scheduler.pt").write_bytes(b"sch")
    (path / "rng_state.pth").write_bytes(b"rng")
    (path / "trainer_state.json").write_text(json.dumps({
        "global_step": step,
        "epoch": 1,
        "log_history": [{"eval_sacrebleu": 10.0, "step": step}],
    }), encoding="utf-8")
    fingerprint = {
        "arm": arm,
        "arm_training_contract_sha256": contract_hash,
        "data_manifest_sha256": data_hash,
        "experiment_fingerprint_sha256": "99" * 32,
    }
    (path / "full_experiment_fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    return path


def seal_nb13(project_root: Path, hours: float = 0.2 / 3600.0) -> Dict[str, Any]:
    sealed = seal_nb12_generation(project_root, pool_rows())
    resolved = resolve_frozen_u_prime(project_root)
    out = project_root / SELECTION_RELATIVE_DIR
    out.mkdir(parents=True, exist_ok=True)
    published = publish_same_budget_selection(
        out,
        rows=resolved.rows,
        identity=resolved.identity,
        selection_budget_hours=hours,
        random_seed=42,
        policy_frozen=True,
        project_root=project_root,
    )
    return {"nb12": sealed, "nb13": published, "resolved": resolved}


def write_rq1_durable_final_state(project_root: Path, *, manifests: Mapping[str, Any], durable_root: Path) -> Dict[str, Any]:
    from src.rq1_contract import STATUS_RQ1_FINAL
    from src.rq1_contract import build_final_contract as build_rq1_final_contract
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths

    test_contract = dict(manifests["test_contract"])
    final = build_rq1_final_contract(
        test_contract=test_contract,
        asr_handoff_hash="aa" * 32,
        mt_handoff_hash="bb" * 32,
        direct_handoff_hash="cc" * 32,
        source_fingerprint_sha256="dd" * 32,
        runtime_versions={"torch": "2.8.0", "transformers": "4.57.6", "accelerate": "1.10.1"},
        seed=42,
        bootstrap_samples=10,
        confidence=0.95,
    )
    runtime = resolve_rq1_runtime_paths(project_root=project_root, durable_root=durable_root)
    state = runtime.state_dir(final["rq1_final_contract_hash"])
    state.mkdir(parents=True, exist_ok=True)
    (state / "rq1_final_contract.json").write_text(json.dumps(final), encoding="utf-8")
    (state / "rq1_test_contract.json").write_text(json.dumps(test_contract), encoding="utf-8")
    (state / "rq1_final_summary.json").write_text(
        json.dumps({
            "status": STATUS_RQ1_FINAL,
            "ready_rq1_final": True,
            "rq1_final_contract_hash": final["rq1_final_contract_hash"],
            "rq1_test_contract_hash": test_contract["rq1_test_contract_hash"],
        }),
        encoding="utf-8",
    )
    runtime.durable_state_root.mkdir(parents=True, exist_ok=True)
    (runtime.durable_state_root / "LATEST_UNLOCKED.json").write_text(
        json.dumps({"final_contract_hash": final["rq1_final_contract_hash"]}),
        encoding="utf-8",
    )
    return {"runtime": runtime, "final_contract": final, "test_contract": test_contract, "state_dir": state}


def world(tmp_path: Path, *, with_test: bool = False, hours: float = 0.2 / 3600.0) -> Dict[str, Any]:
    current = tmp_path / SELECTION_RELATIVE_DIR / "CURRENT"
    if current.is_file():
        sealed = {"nb13": {"reused": True}}
    else:
        sealed = seal_nb13(tmp_path, hours=hours)
    manifests = write_locked_rq1_manifests(tmp_path, with_test=with_test)
    durable_root = tmp_path / "durable"
    durable = None
    if with_test:
        durable = write_rq1_durable_final_state(tmp_path, manifests=manifests, durable_root=durable_root)
    d0_dir = tmp_path / "direct_state"
    if (d0_dir / "direct_evaluate_summary.json").is_file():
        d0_contract = json.loads((d0_dir / "direct_training_contract.json").read_text(encoding="utf-8"))
    else:
        d0_contract = write_frozen_d0_state(d0_dir, manifests=manifests)
    flags = Nb14Flags(direct_state_dir=str(d0_dir), durable_root=str(durable_root))
    return {
        "flags": flags,
        "d0_dir": d0_dir,
        "d0_contract": d0_contract,
        "manifests": manifests,
        "durable_root": durable_root,
        "durable_rq1": durable,
        **sealed,
    }


def all_ready() -> Dict[str, bool]:
    from src.rq2_final_contract import UNLOCK_REQUIRED

    return {name: True for name in UNLOCK_REQUIRED}


def predicting_frames():
    from src.rq2_final_evaluate import prediction_row

    uids = ["t-1", "t-2", "t-3"]
    refs = [
        "đây là câu dài để kiểm tra tham chiếu không bị cắt",
        "câu thứ hai",
        "câu thứ ba",
    ]
    groups = ["g1", "g1", "g2"]
    hyps = {
        ARM_D0: ["a", "b", "c"],
        ARM_RANDOM: ["a", "b", "c"],
        ARM_QUALITY: ["aa", "b", "c"],
    }
    out = {}
    for arm in (ARM_D0, ARM_RANDOM, ARM_QUALITY):
        out[arm] = pd.DataFrame([
            prediction_row(
                record_uid=uid,
                group_id=group,
                hypothesis_vi=hyp,
                reference_vi=ref,
                arm=arm,
                checkpoint_fingerprint=ARM_FINGERPRINTS[arm],
            )
            for uid, hyp, ref, group in zip(uids, hyps[arm], refs, groups)
        ])
    return out


def synthetic_best_checkpoints() -> Dict[str, Any]:
    return {
        arm: {
            "arm": arm,
            "checkpoint_name": "checkpoint-100",
            "checkpoint_fingerprint": ARM_FINGERPRINTS[arm],
            "arm_training_contract_sha256": ("11" * 32) if arm == ARM_RANDOM else (("22" * 32) if arm == ARM_QUALITY else "aa" * 32),
            "data_contract_sha256": "dd" * 32,
            "data_manifest_sha256": "dd" * 32,
            "experiment_fingerprint_sha256": "99" * 32,
            "validation_metric": 10.0,
            "g_test_used": False,
        }
        for arm in (ARM_D0, ARM_RANDOM, ARM_QUALITY)
    }


def synthetic_training_complete(arm: str, contract_sha256: str, *, fingerprint: Optional[str] = None) -> Dict[str, Any]:
    from src.rq2_final_contract import STATUS_TRAINING_COMPLETE

    fp = fingerprint or ARM_FINGERPRINTS[arm]
    return {
        "arm": arm,
        "training_contract_sha256": contract_sha256,
        "final_global_step": 100,
        "terminal_checkpoint": "checkpoint-101",
        "terminal_checkpoint_fingerprint": fp,
        "best_checkpoint": "checkpoint-100",
        "best_checkpoint_fingerprint": fp,
        "best_validation_metric": 10.0,
        "status": STATUS_TRAINING_COMPLETE,
    }


def explicit_test_mix_policy() -> Dict[str, Any]:
    from src.rq2_final_contract import GOLD_PSEUDO_MIX_POLICY_SLOTTED

    return {
        "gold_pseudo_mix_policy": GOLD_PSEUDO_MIX_POLICY_SLOTTED,
        "gold_slots": 1,
        "pseudo_slots": 1,
    }


def explicit_test_seed_policy() -> Dict[str, Any]:
    from src.rq2_final_contract import SEED_POLICY_SINGLE

    return {"seed_policy": SEED_POLICY_SINGLE, "seed_policy_seeds": [42]}


def materialize_arm_proofs(env: Mapping[str, Any], *, best: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write isolated experiment-root checkpoints so G_test/publisher proofs can re-open them."""
    from src.rq2_final_contract import experiment_root_for_arm
    from src.rq2_final_train import arm_state_layout, checkpoint_model_state_sha256, write_training_complete_proof
    from src.rq2_pseudo_contract import write_json

    durable = env["durable_root"]
    payloads = dict(best or synthetic_best_checkpoints())
    layouts = {}
    for arm, contract_hash in ((ARM_RANDOM, "11" * 32), (ARM_QUALITY, "22" * 32)):
        root = experiment_root_for_arm(durable_root=durable, contract_hash=contract_hash, arm=arm)
        layout = arm_state_layout(root)
        layout["root"].mkdir(parents=True, exist_ok=True)
        layout["checkpoints"].mkdir(parents=True, exist_ok=True)
        data_hash = str(payloads[arm].get("data_manifest_sha256") or "dd" * 32)
        ckpt = write_complete_checkpoint(
            layout["checkpoints"] / "checkpoint-100",
            arm=arm,
            contract_hash=contract_hash,
            data_hash=data_hash,
            step=100,
        )
        terminal = write_complete_checkpoint(
            layout["checkpoints"] / "checkpoint-101",
            arm=arm,
            contract_hash=contract_hash,
            data_hash=data_hash,
            step=101,
        )
        live_best = checkpoint_model_state_sha256(ckpt)
        live_terminal = checkpoint_model_state_sha256(terminal)
        payloads[arm]["checkpoint_fingerprint"] = live_best
        payloads[arm]["arm_training_contract_sha256"] = contract_hash
        payloads[arm]["data_manifest_sha256"] = data_hash
        payloads[arm]["data_contract_sha256"] = data_hash
        payloads[arm]["experiment_fingerprint_sha256"] = "99" * 32
        payloads[arm]["validation_metric"] = 10.0
        payloads[arm]["g_test_used"] = False
        write_json(layout["best_checkpoint"], payloads[arm])
        write_training_complete_proof(
            layout,
            arm=arm,
            training_contract_sha256=contract_hash,
            final_global_step=101,
            checkpoint_path=terminal,
            best_checkpoint=payloads[arm],
        )
        layouts[arm] = layout
        payloads[arm]["_terminal_fingerprint"] = live_terminal
    return {"best": payloads, "layouts": layouts}


def g_test_proof_bundle(tmp_path: Path) -> Dict[str, Any]:
    env = world(tmp_path, with_test=True)
    from src.rq2_final_contract import verify_upstream_rq2

    upstream = verify_upstream_rq2(tmp_path, flags=env["flags"])
    proofs = materialize_arm_proofs(env)
    contracts = {
        ARM_RANDOM: {
            "contract": {
                "arm": ARM_RANDOM,
                "arm_training_contract_sha256": "11" * 32,
                "data_manifest_sha256": "dd" * 32,
                "nb13_selection_contract_sha256": upstream["nb13_selection_contract_sha256"],
                "d0_init_model_state_sha256": upstream["d0"]["model_state_sha256"],
                "d0_checkpoint_fingerprint_sha256": upstream["d0"]["checkpoint_fingerprint_sha256"],
            }
        },
        ARM_QUALITY: {
            "contract": {
                "arm": ARM_QUALITY,
                "arm_training_contract_sha256": "22" * 32,
                "data_manifest_sha256": "dd" * 32,
                "nb13_selection_contract_sha256": upstream["nb13_selection_contract_sha256"],
                "d0_init_model_state_sha256": upstream["d0"]["model_state_sha256"],
                "d0_checkpoint_fingerprint_sha256": upstream["d0"]["checkpoint_fingerprint_sha256"],
            }
        },
    }
    flags = Nb14Flags(
        allow_g_test_evaluation=True,
        rq2_final_frozen=True,
        direct_state_dir=str(env["d0_dir"]),
        durable_root=str(env["durable_root"]),
    )
    return {
        "env": env,
        "upstream": upstream,
        "flags": flags,
        "layouts": proofs["layouts"],
        "contracts": contracts,
        "best": proofs["best"],
        "d0_identity": upstream["d0"],
    }


def synthetic_final_contract_fields(tmp_path: Path, upstream: Mapping[str, Any], proofs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    from src.rq1_contract import ordered_uid_hash, reference_content_hash
    from src.rq2_final_contract import compute_nb14_source_fingerprint, STATUS_TRAINING_COMPLETE

    test_path = tmp_path / "data" / "manifests" / "rq1_test.csv"
    test = pd.read_csv(test_path)
    fairness = {
        "supervised_ordered_uid_hash": "33" * 32,
        "supervised_uid_set_hash": "44" * 32,
        "supervised_pair_hash": "55" * 32,
        "supervised_audio_identity_hash": "66" * 32,
        "validation_ordered_uid_hash": "77" * 32,
        "validation_uid_set_hash": "88" * 32,
        "validation_pair_hash": "99" * 32,
        "validation_audio_identity_hash": "ab" * 32,
        "d_random_pseudo_ordered_uid_hash": "cd" * 32,
        "d_quality_pseudo_ordered_uid_hash": "ef" * 32,
        "d_random_pseudo_pair_hash": "a1" * 32,
        "d_quality_pseudo_pair_hash": "a2" * 32,
        "only_pseudo_labeled_component_differs": True,
    }
    fairness["fairness_proof_sha256"] = sha256_json({k: v for k, v in fairness.items() if k != "fairness_proof_sha256"})
    random_fp = ARM_FINGERPRINTS[ARM_RANDOM]
    quality_fp = ARM_FINGERPRINTS[ARM_QUALITY]
    if proofs:
        random_fp = str(proofs["best"][ARM_RANDOM]["checkpoint_fingerprint"])
        quality_fp = str(proofs["best"][ARM_QUALITY]["checkpoint_fingerprint"])
    random_complete = synthetic_training_complete(ARM_RANDOM, "11" * 32, fingerprint=random_fp)
    quality_complete = synthetic_training_complete(ARM_QUALITY, "22" * 32, fingerprint=quality_fp)
    if proofs:
        random_complete["terminal_checkpoint_fingerprint"] = proofs["best"][ARM_RANDOM].get("_terminal_fingerprint") or random_fp
        quality_complete["terminal_checkpoint_fingerprint"] = proofs["best"][ARM_QUALITY].get("_terminal_fingerprint") or quality_fp
    return {
        "nb11_generation_id": upstream["nb11_generation_id"],
        "nb11_input_contract_sha256": upstream["nb11_input_contract_sha256"],
        "nb12_generation_id": upstream["nb12_generation_id"],
        "nb12_contract_sha256": upstream["nb12_contract_sha256"],
        "nb13_selection_contract_sha256": upstream["nb13_selection_contract_sha256"],
        "nb13_generation_id": upstream["nb13_generation_id"],
        "d_random_manifest_sha256": upstream["d_random_manifest_sha256"],
        "d_quality_manifest_sha256": upstream["d_quality_manifest_sha256"],
        "d0_direct_training_contract_hash": upstream["d0"]["direct_training_contract_hash"],
        "d0_checkpoint_fingerprint": upstream["d0"]["checkpoint_fingerprint_sha256"],
        "d0_init_model_state_sha256": upstream["d0"]["model_state_sha256"],
        "d_random_training_contract_sha256": "11" * 32,
        "d_quality_training_contract_sha256": "22" * 32,
        "d_random_best_checkpoint": "checkpoint-100",
        "d_quality_best_checkpoint": "checkpoint-100",
        "d_random_best_checkpoint_fingerprint": random_fp,
        "d_quality_best_checkpoint_fingerprint": quality_fp,
        "g_test_manifest_sha256": sha256_file(test_path),
        "g_test_ordered_uid_hash": ordered_uid_hash(test),
        "g_test_reference_hash": reference_content_hash(test),
        "source_fingerprint_sha256": compute_nb14_source_fingerprint()["aggregate_sha256"],
        "bootstrap_config": upstream["rq1_eval"],
        "seed_policy": explicit_test_seed_policy()["seed_policy"],
        "seed_policy_seeds": list(explicit_test_seed_policy()["seed_policy_seeds"]),
        "gold_pseudo_mix_policy": explicit_test_mix_policy()["gold_pseudo_mix_policy"],
        "gold_slots": explicit_test_mix_policy()["gold_slots"],
        "pseudo_slots": explicit_test_mix_policy()["pseudo_slots"],
        "fairness_proof": fairness,
        "d_random_training_complete": random_complete,
        "d_quality_training_complete": quality_complete,
    }
