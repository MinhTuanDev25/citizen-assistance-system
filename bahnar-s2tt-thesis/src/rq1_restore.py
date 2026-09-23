"""Verified upstream checkpoint restore for Notebook 06."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from src.rq1_contract import sha256_file, sha256_json
from src.rq1_evaluation import verify_upstream_handoffs


def _json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Missing required JSON: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _assert_evaluate_clean(summary: Mapping[str, Any], *, expected_status: str, label: str) -> None:
    if summary.get("status") != expected_status:
        raise RuntimeError(f"{label} is not {expected_status}")
    if summary.get("frozen_test_accessed") is not False:
        raise RuntimeError(f"{label} evaluate summary is not frozen-test clean")


def _snapshot_manifest_digest(local_exp: Path) -> Optional[str]:
    from src.asr_full_train import read_snapshot_manifest

    # Prefer experiment-root fingerprint/manifest if present.
    for cand in (
        local_exp / "snapshot_manifest.json",
        local_exp.parent / "snapshot_manifest.json",
        local_exp / "full_experiment_fingerprint.json",
    ):
        if cand.is_file():
            return sha256_file(cand)
    # Durable snapshots may live beside the restored tree; optional.
    try:
        manifest = read_snapshot_manifest(local_exp)
    except Exception:
        manifest = None
    if manifest:
        return sha256_json(manifest)
    return None


def _proof_for_checkpoint(
    *,
    system: str,
    experiment_id: str,
    contract_hash: str,
    best_path: Union[str, Path],
    local_exp: Path,
) -> Dict[str, Any]:
    from src.asr_full_train import assert_checkpoint_complete_for_resume, checkpoint_content_digest

    best = Path(best_path)
    assert_checkpoint_complete_for_resume(best)
    digest = checkpoint_content_digest(best)
    return {
        "system": system,
        "verified": True,
        "loaded": False,
        "experiment_id": str(experiment_id),
        "contract_hash": str(contract_hash),
        "best_checkpoint_name": best.name,
        "best_checkpoint_path": str(best),
        "checkpoint_content_digest": digest["digest"],
        "checkpoint_n_files": digest["n_files"],
        "snapshot_manifest_digest": _snapshot_manifest_digest(local_exp),
    }


def _contract_hash_from_restored(system: str, restored: Mapping[str, Any]) -> str:
    contract = restored.get("training_contract") or {}
    if system == "asr":
        return str(
            contract.get("contract_hash")
            or contract.get("train_contract_hash")
            or contract.get("data_contract_hash")
            or ""
        )
    if system == "mt":
        return str(contract.get("mt_training_contract_hash") or "")
    if system == "direct":
        return str(contract.get("direct_training_contract_hash") or "")
    raise RuntimeError(f"Unknown RQ1 system for contract hash: {system}")


def assert_checkpoint_proof_matches_upstream(
    proof: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> None:
    """Unlock/post-verify: durable proof must still bind to current handoff identity."""
    if proof.get("status") != "SUCCESS_RQ1_CHECKPOINT_PROOF":
        raise RuntimeError("checkpoint_proof.json is not SUCCESS_RQ1_CHECKPOINT_PROOF")
    recomputed = sha256_json(
        {
            "asr": proof.get("asr") or {},
            "mt": proof.get("mt") or {},
            "direct": proof.get("direct") or {},
        }
    )
    if str(proof.get("proof_hash") or "") != recomputed:
        raise RuntimeError("checkpoint_proof.proof_hash self-consistency failed")
    hh = proof.get("handoff_hashes") or {}
    for key in ("asr_handoff_hash", "mt_handoff_hash", "direct_handoff_hash"):
        if key in hh and str(hh.get(key) or "") != str(upstream.get(key) or ""):
            raise RuntimeError(f"checkpoint_proof handoff_hashes.{key} mismatch vs current UPSTREAM")
    for system in ("asr", "mt", "direct"):
        entry = proof.get(system) or {}
        if not entry.get("loaded"):
            raise RuntimeError(f"Checkpoint proof incomplete for {system}; re-run verify")
        identity = (upstream.get(system) or {}).get("identity") or {}
        for field in ("experiment_id", "contract_hash", "best_checkpoint_name"):
            if str(entry.get(field) or "") != str(identity.get(field) or ""):
                raise RuntimeError(
                    f"checkpoint_proof.{system}.{field} mismatch vs current UPSTREAM identity"
                )


def assert_restored_checkpoint_matches_proof(
    restored: Mapping[str, Any],
    proof_entry: Mapping[str, Any],
    *,
    system: str,
) -> Dict[str, Any]:
    """
    Re-hash the checkpoint actually restored for inference and require equality
    with the digest bound into the unlocked final contract.
    """
    from src.asr_full_train import assert_checkpoint_complete_for_resume, checkpoint_content_digest

    best = Path(str(restored.get("best_checkpoint") or ""))
    if not best.is_dir():
        raise RuntimeError(f"{system} restored best_checkpoint missing: {best}")
    assert_checkpoint_complete_for_resume(best)
    digest = checkpoint_content_digest(best)
    got = {
        "experiment_id": str(restored.get("experiment_id") or ""),
        "contract_hash": _contract_hash_from_restored(system, restored),
        "best_checkpoint_name": best.name,
        "checkpoint_content_digest": digest["digest"],
    }
    want = {
        "experiment_id": str(proof_entry.get("experiment_id") or ""),
        "contract_hash": str(proof_entry.get("contract_hash") or ""),
        "best_checkpoint_name": str(proof_entry.get("best_checkpoint_name") or ""),
        "checkpoint_content_digest": str(proof_entry.get("checkpoint_content_digest") or ""),
    }
    bad = {k: {"got": got[k], "want": want[k]} for k in got if got[k] != want[k]}
    if bad:
        raise RuntimeError(
            f"{system} restored checkpoint does not match FINAL_CONTRACT proof: {bad}"
        )
    return got


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


_FORBIDDEN_DELETE_EXACT = {
    Path("/").resolve(),
    Path("/tmp").resolve(),
    Path("/var").resolve(),
    Path("/home").resolve(),
    Path("/Users").resolve(),
}


def assert_path_is_strict_descendant(
    path: Union[str, Path],
    allowed_root: Union[str, Path],
    *,
    label: str,
) -> Path:
    """Require ``path`` is a resolved strict descendant of ``allowed_root``."""
    if not str(path or "").strip():
        raise RuntimeError(f"{label}: path missing")
    if not str(allowed_root or "").strip():
        raise RuntimeError(f"{label}: allowed_root required")
    child = _resolve_path(path)
    root = _resolve_path(allowed_root)
    if child in _FORBIDDEN_DELETE_EXACT:
        raise RuntimeError(f"{label}: refuse to delete forbidden path {child}")
    if root in _FORBIDDEN_DELETE_EXACT and child == root:
        raise RuntimeError(f"{label}: refuse to delete forbidden root {root}")
    if child == root:
        raise RuntimeError(f"{label}: refuse to delete allowed_root itself: {root}")
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label}: refuse to delete path outside allowed_root: {child} not under {root}"
        ) from exc
    # Block path-traversal style equality tricks after resolve (already resolved).
    return child


def cleanup_local_rq1_experiment(
    local_experiment_dir: Union[str, Path],
    *,
    allowed_root: Union[str, Path],
    label: str = "rq1",
) -> None:
    """
    Delete a local RQ1 restore experiment tree under ``allowed_root`` only.

    Idempotent when the valid child path is already absent. Never ignores
    cleanup failures, never deletes the allowed root itself, and never escapes
    via ``..`` / symlink resolution tricks (paths are resolved first).
    """
    import shutil

    target = assert_path_is_strict_descendant(
        local_experiment_dir, allowed_root, label=label,
    )
    if not target.exists():
        return
    shutil.rmtree(target)
    if target.exists():
        raise RuntimeError(f"Failed to cleanup local {label} restore tree: {target}")


def _release_restored_handles(restored: Optional[Mapping[str, Any]]) -> None:
    if not restored:
        return
    # Best-effort drop of heavy objects; caller should also del the dict.
    for key in ("model", "processor", "tokenizer", "feature_extractor"):
        if hasattr(restored, "pop"):
            try:
                restored.pop(key, None)  # type: ignore[attr-defined]
            except Exception:
                pass


def run_cascaded_c0_peak_safe(
    *,
    restore_asr: Callable[[], Mapping[str, Any]],
    restore_mt: Callable[[], Mapping[str, Any]],
    predict_asr: Callable[[Mapping[str, Any]], Any],
    predict_mt: Callable[[Mapping[str, Any], Any], Any],
    assert_asr_proof: Callable[[Mapping[str, Any]], Any],
    assert_mt_proof: Callable[[Mapping[str, Any]], Any],
    expected_uids: Sequence[str],
    asr_allowed_root: Union[str, Path],
    mt_allowed_root: Union[str, Path],
    cleanup_local: Callable[..., None] = cleanup_local_rq1_experiment,
    release_cuda: Optional[Callable[[], None]] = None,
    validate_uids: Optional[Callable[[Any, Sequence[str], str], None]] = None,
) -> Dict[str, Any]:
    """
    Exception-safe C0 peak control:

    ASR restore → proof → infer → UID validate → finally CUDA+disk cleanup
    then (only if ASR succeeded and disk gone)
    MT restore → proof → infer → UID validate → finally CUDA+disk cleanup
    """
    from src.rq1_contract import assert_prediction_uid_order

    def _validate(frame: Any, label: str) -> None:
        if validate_uids is not None:
            validate_uids(frame, expected_uids, label)
        else:
            assert_prediction_uid_order(frame, expected_uids, label=label)

    order: List[str] = []
    asr_local = ""
    asr_best = ""
    asr_rows = None
    asr_obj: Optional[Dict[str, Any]] = None
    asr_ok = False
    try:
        asr_obj = dict(restore_asr())
        order.append("restore_asr")
        asr_local = str(asr_obj.get("local_experiment_dir") or "")
        asr_best = str(asr_obj.get("best_checkpoint") or "")
        assert_asr_proof(asr_obj)
        order.append("asr_proof")
        asr_rows = predict_asr(asr_obj)
        order.append("asr_infer")
        _validate(asr_rows, "ASR")
        order.append("asr_uid_ok")
        asr_ok = True
    finally:
        _release_restored_handles(asr_obj)
        asr_obj = None
        if asr_local:
            cleanup_local(asr_local, allowed_root=asr_allowed_root, label="asr")
            order.append("cleanup_asr_disk")
        if release_cuda is not None:
            release_cuda()
            order.append("release_asr_cuda")

    if not asr_ok or asr_rows is None:
        raise RuntimeError("ASR stage failed before producing validated predictions")
    if asr_local and Path(asr_local).exists():
        raise RuntimeError(f"ASR local restore still present before MT: {asr_local}")

    mt_local = ""
    mt_best = ""
    c0 = None
    mt_obj: Optional[Dict[str, Any]] = None
    mt_ok = False
    try:
        mt_obj = dict(restore_mt())
        order.append("restore_mt")
        if asr_local and Path(asr_local).exists():
            raise RuntimeError(f"ASR local restore reappeared during MT restore: {asr_local}")
        mt_local = str(mt_obj.get("local_experiment_dir") or "")
        mt_best = str(mt_obj.get("best_checkpoint") or "")
        assert_mt_proof(mt_obj)
        order.append("mt_proof")
        c0 = predict_mt(mt_obj, asr_rows)
        order.append("mt_infer")
        _validate(c0, "C0")
        order.append("c0_uid_ok")
        mt_ok = True
    finally:
        _release_restored_handles(mt_obj)
        mt_obj = None
        if mt_local:
            cleanup_local(mt_local, allowed_root=mt_allowed_root, label="mt")
            order.append("cleanup_mt_disk")
        if release_cuda is not None:
            release_cuda()
            order.append("release_mt_cuda")

    if not mt_ok or c0 is None:
        raise RuntimeError("MT stage failed before producing validated C0 predictions")
    if mt_local and Path(mt_local).exists():
        raise RuntimeError(f"MT local restore still present after cleanup: {mt_local}")

    return {
        "asr_rows": asr_rows,
        "c0": c0,
        "asr_best_checkpoint": asr_best,
        "mt_best_checkpoint": mt_best,
        "restore_order": order,
        "asr_local_experiment_dir": asr_local,
        "mt_local_experiment_dir": mt_local,
    }


def run_direct_d0_peak_safe(
    *,
    restore_direct: Callable[[], Mapping[str, Any]],
    predict_direct: Callable[[Mapping[str, Any]], Any],
    assert_direct_proof: Callable[[Mapping[str, Any]], Any],
    expected_uids: Sequence[str],
    direct_allowed_root: Union[str, Path],
    cleanup_local: Callable[..., None] = cleanup_local_rq1_experiment,
    release_cuda: Optional[Callable[[], None]] = None,
    validate_uids: Optional[Callable[[Any, Sequence[str], str], None]] = None,
) -> Dict[str, Any]:
    """Exception-safe Direct restore → proof → infer → UID validate → finally cleanup."""
    from src.rq1_contract import assert_prediction_uid_order

    def _validate(frame: Any, label: str) -> None:
        if validate_uids is not None:
            validate_uids(frame, expected_uids, label)
        else:
            assert_prediction_uid_order(frame, expected_uids, label=label)

    order: List[str] = []
    direct_local = ""
    direct_best = ""
    d0 = None
    direct_obj: Optional[Dict[str, Any]] = None
    ok = False
    try:
        direct_obj = dict(restore_direct())
        order.append("restore_direct")
        direct_local = str(direct_obj.get("local_experiment_dir") or "")
        direct_best = str(direct_obj.get("best_checkpoint") or "")
        assert_direct_proof(direct_obj)
        order.append("direct_proof")
        d0 = predict_direct(direct_obj)
        order.append("direct_infer")
        _validate(d0, "D0")
        order.append("d0_uid_ok")
        ok = True
    finally:
        _release_restored_handles(direct_obj)
        direct_obj = None
        if direct_local:
            cleanup_local(direct_local, allowed_root=direct_allowed_root, label="direct")
            order.append("cleanup_direct_disk")
        if release_cuda is not None:
            release_cuda()
            order.append("release_direct_cuda")

    if not ok or d0 is None:
        raise RuntimeError("Direct stage failed before producing validated D0 predictions")
    if direct_local and Path(direct_local).exists():
        raise RuntimeError(f"Direct local restore still present after cleanup: {direct_local}")
    return {
        "d0": d0,
        "direct_best_checkpoint": direct_best,
        "restore_order": order,
        "direct_local_experiment_dir": direct_local,
    }


def restore_asr_for_rq1(*, state_dir: Union[str, Path], local_ckpt_root: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
    """Restore verified NB03 best checkpoint and load Wav2Vec2 processor/model."""
    from src.asr_full_train import (
        FULL_TRAIN_MARKER,
        assert_checkpoint_complete_for_resume,
        experiment_checkpoint_dir,
        extract_training_contract,
        resolve_best_checkpoint_from_durable,
    )
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    state = Path(state_dir)
    ev = _json(state / "full_evaluate_summary.json")
    tr = _json(state / "full_train_summary.json")
    _assert_evaluate_clean(ev, expected_status="SUCCESS_FULL_EVALUATE", label="NB03")
    exp = str(tr.get("experiment_id") or ev.get("experiment_id") or "")
    if not exp:
        raise RuntimeError("NB03 summary missing experiment_id")
    contract = extract_training_contract(tr)
    if not contract:
        raise RuntimeError("NB03 train summary missing training contract")
    allowed_root = Path(local_ckpt_root)
    local_exp = experiment_checkpoint_dir(local_ckpt_root, exp, kind=FULL_TRAIN_MARKER)
    ok = False
    try:
        best = resolve_best_checkpoint_from_durable(
            state,
            experiment_id=exp,
            train_summary=tr,
            local_experiment_dir=local_exp,
            expected_contract=contract,
        )
        assert_checkpoint_complete_for_resume(best)
        processor = Wav2Vec2Processor.from_pretrained(str(best))
        model = Wav2Vec2ForCTC.from_pretrained(str(best)).to(device)
        ok = True
        return {
            "model": model,
            "processor": processor,
            "best_checkpoint": str(best),
            "experiment_id": exp,
            "training_contract": contract,
            "local_experiment_dir": str(local_exp),
        }
    finally:
        if not ok:
            cleanup_local_rq1_experiment(local_exp, allowed_root=allowed_root, label="asr")


def restore_mt_for_rq1(*, state_dir: Union[str, Path], local_ckpt_root: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
    from src.mt_full_train import resolve_mt_best_checkpoint_for_evaluate
    from src.mt_tokenize import load_mt_tokenizer, tokenizer_fingerprint
    from transformers import AutoModelForSeq2SeqLM

    state = Path(state_dir)
    ev = _json(state / "mt_evaluate_summary.json")
    tr = _json(state / "mt_train_summary.json")
    contract = _json(state / "mt_training_contract.json")
    _assert_evaluate_clean(ev, expected_status="SUCCESS_MT_EVALUATE", label="NB04")
    exp = str(contract.get("experiment_id") or tr.get("experiment_id") or "")
    if not exp:
        raise RuntimeError("NB04 summary missing experiment_id")
    allowed_root = Path(local_ckpt_root)
    local_exp = Path(local_ckpt_root) / "full_train" / exp
    ok = False
    try:
        best = resolve_mt_best_checkpoint_for_evaluate(
            full_state_dir=state,
            experiment_id=exp,
            train_summary=tr,
            local_experiment_dir=local_exp,
            expected_training_contract=contract,
        )
        tokenizer = load_mt_tokenizer(str(contract["model_id"]), str(contract["model_revision"]))
        got_fp = tokenizer_fingerprint(tokenizer)
        if got_fp != str(contract.get("tokenizer_fingerprint") or ""):
            raise RuntimeError("NB04 tokenizer fingerprint mismatch at RQ1 restore")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(best)).to(device)
        ok = True
        return {
            "model": model,
            "tokenizer": tokenizer,
            "best_checkpoint": str(best),
            "experiment_id": exp,
            "training_contract": contract,
            "local_experiment_dir": str(local_exp),
        }
    finally:
        if not ok:
            cleanup_local_rq1_experiment(local_exp, allowed_root=allowed_root, label="mt")


def restore_direct_for_rq1(*, state_dir: Union[str, Path], local_ckpt_root: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
    from src.direct_full_train import restore_direct_best_checkpoint
    from src.direct_model import load_direct_processors
    from src.mt_tokenize import tokenizer_fingerprint
    from transformers import SpeechEncoderDecoderModel

    state = Path(state_dir)
    ev = _json(state / "direct_evaluate_summary.json")
    tr = _json(state / "direct_train_summary.json")
    contract = _json(state / "direct_training_contract.json")
    _assert_evaluate_clean(ev, expected_status="SUCCESS_DIRECT_EVALUATE", label="NB05")
    exp = str(contract.get("experiment_id") or tr.get("experiment_id") or "")
    if not exp:
        raise RuntimeError("NB05 summary missing experiment_id")
    allowed_root = Path(local_ckpt_root)
    local_exp = Path(local_ckpt_root) / "full_train" / exp
    ok = False
    try:
        best = restore_direct_best_checkpoint(
            state_dir=state,
            experiment_id=exp,
            train_summary=tr,
            local_experiment_dir=local_exp,
            expected_contract=contract,
        )
        feat, tok = load_direct_processors(
            encoder_id=str(contract["encoder_id"]),
            encoder_revision=str(contract["encoder_revision"]),
            decoder_id=str(contract["decoder_id"]),
            decoder_revision=str(contract["decoder_revision"]),
            target_lang=str(contract["target_lang"]),
        )
        if tokenizer_fingerprint(tok) != str(contract.get("tokenizer_fingerprint") or ""):
            raise RuntimeError("NB05 tokenizer fingerprint mismatch at RQ1 restore")
        model = SpeechEncoderDecoderModel.from_pretrained(str(best)).to(device)
        forced_bos = int(tok.lang_code_to_id[str(contract["target_lang"])])
        model.generation_config.forced_bos_token_id = forced_bos
        ok = True
        return {
            "model": model,
            "feature_extractor": feat,
            "tokenizer": tok,
            "best_checkpoint": str(best),
            "experiment_id": exp,
            "training_contract": contract,
            "local_experiment_dir": str(local_exp),
        }
    finally:
        if not ok:
            cleanup_local_rq1_experiment(local_exp, allowed_root=allowed_root, label="direct")


def _default_load_asr(best: Path, device: str) -> Dict[str, Any]:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(str(best))
    model = Wav2Vec2ForCTC.from_pretrained(str(best)).to(device)
    return {"model": model, "processor": processor}


def _default_load_mt(best: Path, contract: Mapping[str, Any], device: str) -> Dict[str, Any]:
    from src.mt_tokenize import load_mt_tokenizer, tokenizer_fingerprint
    from transformers import AutoModelForSeq2SeqLM

    tokenizer = load_mt_tokenizer(str(contract["model_id"]), str(contract["model_revision"]))
    if tokenizer_fingerprint(tokenizer) != str(contract.get("tokenizer_fingerprint") or ""):
        raise RuntimeError("NB04 tokenizer fingerprint mismatch at RQ1 verify-load")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(best)).to(device)
    return {"model": model, "tokenizer": tokenizer}


def _default_load_direct(best: Path, contract: Mapping[str, Any], device: str) -> Dict[str, Any]:
    from src.direct_model import load_direct_processors
    from src.mt_tokenize import tokenizer_fingerprint
    from transformers import SpeechEncoderDecoderModel

    feat, tok = load_direct_processors(
        encoder_id=str(contract["encoder_id"]),
        encoder_revision=str(contract["encoder_revision"]),
        decoder_id=str(contract["decoder_id"]),
        decoder_revision=str(contract["decoder_revision"]),
        target_lang=str(contract["target_lang"]),
    )
    if tokenizer_fingerprint(tok) != str(contract.get("tokenizer_fingerprint") or ""):
        raise RuntimeError("NB05 tokenizer fingerprint mismatch at RQ1 verify-load")
    model = SpeechEncoderDecoderModel.from_pretrained(str(best)).to(device)
    return {"model": model, "feature_extractor": feat, "tokenizer": tok}


def verify_upstream_checkpoint_artifacts(
    *,
    asr_state_dir: Union[str, Path],
    mt_state_dir: Union[str, Path],
    direct_state_dir: Union[str, Path],
    asr_local_ckpt_root: Union[str, Path],
    mt_local_ckpt_root: Union[str, Path],
    direct_local_ckpt_root: Union[str, Path],
    cleanup_local_after_verify: bool = True,
    load_models: bool = True,
    device: str = "cpu",
    asr_loader: Optional[Callable[..., Dict[str, Any]]] = None,
    mt_loader: Optional[Callable[..., Dict[str, Any]]] = None,
    direct_loader: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Restore + optionally load each best model/processor on CPU one system at a time.

    Does not run inference. Unlock requires loaded=True for every system when
    ``load_models`` is enabled.
    """
    import shutil

    from src.asr_full_train import (
        FULL_TRAIN_MARKER,
        experiment_checkpoint_dir,
        extract_training_contract,
        resolve_best_checkpoint_from_durable,
    )
    from src.direct_full_train import restore_direct_best_checkpoint
    from src.mt_full_train import resolve_mt_best_checkpoint_for_evaluate

    # Fail closed on identity before touching durable blobs.
    handoff = verify_upstream_handoffs(
        asr_state_dir=asr_state_dir,
        mt_state_dir=mt_state_dir,
        direct_state_dir=direct_state_dir,
    )

    proof: Dict[str, Any] = {"status": "SUCCESS_RQ1_CHECKPOINT_PROOF", "systems": {}}
    astate = Path(asr_state_dir)
    mstate = Path(mt_state_dir)
    dstate = Path(direct_state_dir)

    atr = _json(astate / "full_train_summary.json")
    aev = _json(astate / "full_evaluate_summary.json")
    aexp = str(handoff["asr"]["identity"]["experiment_id"])
    acontract = extract_training_contract(atr)
    alocal = experiment_checkpoint_dir(asr_local_ckpt_root, aexp, kind=FULL_TRAIN_MARKER)
    try:
        abest = resolve_best_checkpoint_from_durable(
            astate,
            experiment_id=aexp,
            train_summary=atr,
            local_experiment_dir=alocal,
            expected_contract=acontract,
        )
        entry = _proof_for_checkpoint(
            system="asr",
            experiment_id=aexp,
            contract_hash=str(handoff["asr"]["identity"]["contract_hash"]),
            best_path=abest,
            local_exp=Path(alocal),
        )
        if load_models:
            loaded = (asr_loader or _default_load_asr)(Path(abest), device)
            if not loaded.get("model") or not loaded.get("processor"):
                raise RuntimeError("NB03 verify-load did not return model/processor")
            entry["loaded"] = True
            del loaded
            gc.collect()
        proof["systems"]["asr"] = entry
        proof["asr"] = entry
    finally:
        if cleanup_local_after_verify:
            cleanup_local_rq1_experiment(alocal, allowed_root=asr_local_ckpt_root, label="asr")

    mtr = _json(mstate / "mt_train_summary.json")
    mcontract = _json(mstate / "mt_training_contract.json")
    mexp = str(handoff["mt"]["identity"]["experiment_id"])
    mlocal = Path(mt_local_ckpt_root) / "full_train" / mexp
    try:
        mbest = resolve_mt_best_checkpoint_for_evaluate(
            full_state_dir=mstate,
            experiment_id=mexp,
            train_summary=mtr,
            local_experiment_dir=mlocal,
            expected_training_contract=mcontract,
        )
        entry = _proof_for_checkpoint(
            system="mt",
            experiment_id=mexp,
            contract_hash=str(handoff["mt"]["identity"]["contract_hash"]),
            best_path=mbest,
            local_exp=mlocal,
        )
        if load_models:
            loaded = (mt_loader or _default_load_mt)(Path(mbest), mcontract, device)
            if not loaded.get("model") or not loaded.get("tokenizer"):
                raise RuntimeError("NB04 verify-load did not return model/tokenizer")
            entry["loaded"] = True
            del loaded
            gc.collect()
        proof["systems"]["mt"] = entry
        proof["mt"] = entry
    finally:
        if cleanup_local_after_verify:
            cleanup_local_rq1_experiment(mlocal, allowed_root=mt_local_ckpt_root, label="mt")

    dtr = _json(dstate / "direct_train_summary.json")
    dcontract = _json(dstate / "direct_training_contract.json")
    dexp = str(handoff["direct"]["identity"]["experiment_id"])
    dlocal = Path(direct_local_ckpt_root) / "full_train" / dexp
    try:
        dbest = restore_direct_best_checkpoint(
            state_dir=dstate,
            experiment_id=dexp,
            train_summary=dtr,
            local_experiment_dir=dlocal,
            expected_contract=dcontract,
        )
        entry = _proof_for_checkpoint(
            system="direct",
            experiment_id=dexp,
            contract_hash=str(handoff["direct"]["identity"]["contract_hash"]),
            best_path=dbest,
            local_exp=dlocal,
        )
        if load_models:
            loaded = (direct_loader or _default_load_direct)(Path(dbest), dcontract, device)
            if not loaded.get("model") or not loaded.get("tokenizer"):
                raise RuntimeError("NB05 verify-load did not return model/tokenizer")
            entry["loaded"] = True
            del loaded
            gc.collect()
        proof["systems"]["direct"] = entry
        proof["direct"] = entry
    finally:
        if cleanup_local_after_verify:
            cleanup_local_rq1_experiment(dlocal, allowed_root=direct_local_ckpt_root, label="direct")

    if load_models:
        for name, entry in proof["systems"].items():
            if not entry.get("loaded"):
                raise RuntimeError(f"RQ1 checkpoint proof incomplete: {name} not loaded")

    proof["proof_hash"] = sha256_json(
        {
            "asr": proof["asr"],
            "mt": proof["mt"],
            "direct": proof["direct"],
        }
    )
    proof["handoff_hashes"] = {
        "asr_handoff_hash": handoff["asr_handoff_hash"],
        "mt_handoff_hash": handoff["mt_handoff_hash"],
        "direct_handoff_hash": handoff["direct_handoff_hash"],
    }
    return proof
