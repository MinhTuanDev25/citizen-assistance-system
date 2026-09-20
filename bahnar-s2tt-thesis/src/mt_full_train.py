"""
Notebook 04 MT full-stage orchestration: resume proof, train/eval gates, statuses.

Reuses durable checkpoint primitives from ``asr_full_train`` (namespace-isolated via
paths). Does not reuse CTC/audio/ASR contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.asr_full_train import (
    CHECKPOINT_STORE_DIRNAME,
    assert_checkpoint_allowed_for_full_train,
    assert_checkpoint_complete_for_resume,
    assert_cross_session_resume,
    assert_durable_checkpoint_budget,
    assert_no_frozen_test_access,
    checkpoint_content_digest,
    derive_resume_test_status_from_proof,
    durable_experiment_dir,
    ensure_experiment_fingerprint,
    experiment_checkpoint_dir,
    is_forbidden_init_checkpoint,
    list_snapshot_checkpoint_names,
    make_resume_proof_callback,
    make_sequential_sampler_trainer_cls,
    make_uid_tracking_collator,
    make_uid_tracking_trainer_cls,
    measure_dir_bytes,
    new_session_token,
    plan_expected_resume_position,
    resolve_durable_checkpoint_budget_bytes,
    resolve_durable_latest_snapshot,
    restore_experiment_checkpoints_from_durable,
    resolve_best_checkpoint_from_durable,
    summarize_resume_proof,
    sync_experiment_checkpoints_to_durable,
    write_checkpoint_fingerprint,
)
from src.metrics import mt_corpus_metrics
from src.mt_contract import (
    LOCKED_MT_MONITOR_SIZE,
    STATUS_FAILED,
    STATUS_MT_EVALUATE,
    STATUS_MT_PREPARE,
    STATUS_MT_RESUME_TEST,
    STATUS_MT_TRAINING,
    STAGE_VERSION_EVALUATE,
    STAGE_VERSION_RESUME_TEST,
    STAGE_VERSION_TRAIN,
    assert_generation_config_matches,
    assert_locked_bartpho_baseline,
    assert_model_revision_pinned,
    assert_mt_resume_test_contract_self_consistent,
    assert_mt_resume_test_contracts_match,
    assert_mt_training_contract_self_consistent,
    assert_mt_training_contracts_match,
    build_generation_config,
    build_mt_data_contract,
    build_mt_resume_test_contract,
    build_mt_training_contract,
    contracts_equal,
    fingerprint_extra_from_mt_training_contract,
)
from src.mt_dataset import MtTextDataset, make_seq2seq_collator
from src.mt_normalize import normalize_mt_text_v1
from src.mt_prepare import load_mt_prepare_success
from src.mt_runtime_paths import FULL_TRAIN_MARKER, PILOT_MARKER, RESUME_TEST_MARKER, SOURCE_FIELD, TARGET_FIELD
from src.mt_tokenize import load_mt_tokenizer, tokenizer_fingerprint
from src.data_utils import compute_uid_set_hash

RESUME_TEST_PHASE_A_STEPS = 50
RESUME_TEST_PHASE_B_STEPS = 100
RESUME_TEST_SUBSET_SIZE = 64
PHASE_B_ATTEMPTS_DIRNAME = "phase_b_attempts"


def _atomic_write_json(path: Union[str, Path], payload: Any) -> Path:
    """Write JSON via temp file + fsync + atomic replace (fail-closed on I/O errors)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return path


def mt_experiment_dir(
    root: Union[str, Path],
    *,
    kind: str,
    experiment_id: str,
) -> Path:
    return experiment_checkpoint_dir(root, experiment_id, kind=kind)


def assert_mt_checkpoint_namespace(path: Union[str, Path], *, allowed_kind: str) -> None:
    p = str(Path(path).resolve())
    if f"/{allowed_kind}/" not in p and not p.endswith(f"/{allowed_kind}"):
        raise RuntimeError(f"Checkpoint path not under namespace {allowed_kind!r}: {path}")
    if allowed_kind == FULL_TRAIN_MARKER and is_forbidden_init_checkpoint(path):
        raise RuntimeError(f"Forbidden init checkpoint for full MT train: {path}")


def build_mt_resume_test_subset(df: pd.DataFrame, *, n_samples: int, seed: int) -> pd.DataFrame:
    if len(df) < n_samples:
        raise RuntimeError(f"Need >= {n_samples} MT rows for resume_test, got {len(df)}")
    return df.sample(n=n_samples, random_state=seed).sort_index().reset_index(drop=True)


def write_mt_resume_test_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    return _atomic_write_json(state_dir / "mt_resume_test_summary.json", payload)


def _mt_resume_test_snap_ref_names(dest_root: Path) -> set:
    names: set = set()
    snap_root = dest_root / "snapshots"
    if not snap_root.is_dir():
        return names
    for p in snap_root.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
            names.update(list_snapshot_checkpoint_names(p))
    return names


def _mt_checkpoint_complete(path: Path) -> bool:
    try:
        from src.asr_full_train import _verify_checkpoint_dir_complete

        _verify_checkpoint_dir_complete(path)
        return True
    except Exception:
        return False


def _is_safe_phase_b_attempt_id(name: str) -> bool:
    """Attempt ids are digest prefixes (hex). Reject path-like / hidden names."""
    if not name or name.startswith("."):
        return False
    if "/" in name or "\\" in name or name in {".", ".."}:
        return False
    return all(c in "0123456789abcdef" for c in name.lower())


def measure_mt_resume_test_durable_bytes(dest_root: Union[str, Path]) -> Dict[str, Any]:
    """
    Bytes under the resume_test durable experiment that count toward budget:
    immutable ``ckpts/`` store + ``phase_b_attempts/`` full checkpoints.
    """
    root = Path(dest_root)
    store = root / CHECKPOINT_STORE_DIRNAME
    attempts = root / PHASE_B_ATTEMPTS_DIRNAME
    store_bytes = measure_dir_bytes(store) if store.is_dir() else 0
    attempt_bytes = measure_dir_bytes(attempts) if attempts.is_dir() else 0
    per_attempt: Dict[str, int] = {}
    if attempts.is_dir():
        for child in sorted(attempts.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                per_attempt[child.name] = measure_dir_bytes(child)
    return {
        "store_bytes": int(store_bytes),
        "attempt_bytes": int(attempt_bytes),
        "per_attempt_bytes": per_attempt,
        "total_bytes": int(store_bytes) + int(attempt_bytes),
    }


def _is_hex_digest(value: Any, *, min_len: int = 12) -> bool:
    s = str(value or "")
    if len(s) < int(min_len):
        return False
    return all(c in "0123456789abcdef" for c in s.lower())


def _load_existing_durable_commit(
    state_dir: Path,
    *,
    experiment_id: str,
    dest_root: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """
    Load + semantically validate prior durable commit metadata for retention.

    Missing file → None (first commit). Corrupt JSON or semantic inconsistency →
    raise (fail-closed: never prune while references are unknown/unproven).
    """
    path = Path(state_dir) / "mt_resume_test_durable_commit.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Corrupt/unreadable mt_resume_test_durable_commit.json at {path}: {exc}. "
            f"Refusing to prune phase_b_attempts fail-closed."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Invalid mt_resume_test_durable_commit.json at {path}: expected object, "
            f"got {type(data).__name__}. Refusing to prune phase_b_attempts fail-closed."
        )
    validate_mt_resume_test_durable_commit_references(
        data,
        experiment_id=str(experiment_id),
        dest_root=Path(dest_root),
        meta_path=path,
    )
    return data


def validate_mt_resume_test_durable_commit_references(
    commit: Mapping[str, Any],
    *,
    experiment_id: str,
    dest_root: Union[str, Path],
    meta_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Prove previous durable-commit metadata is semantically consistent with on-disk
    artifacts before any phase_b_attempts prune.

    Fail-closed on any missing/inconsistent field.
    """
    label = str(meta_path or "mt_resume_test_durable_commit.json")
    root = Path(dest_root)

    def _fail(msg: str) -> None:
        raise RuntimeError(
            f"Invalid durable commit metadata at {label}: {msg}. "
            f"Refusing to prune phase_b_attempts fail-closed."
        )

    if commit.get("durable_commit_ok") is not True:
        _fail("durable_commit_ok is not True")
    if str(commit.get("experiment_id") or "") != str(experiment_id):
        _fail(
            f"experiment_id mismatch: meta={commit.get('experiment_id')!r} "
            f"expected={experiment_id!r}"
        )
    if str(commit.get("kind") or "") != str(RESUME_TEST_MARKER):
        _fail(f"kind must be {RESUME_TEST_MARKER!r}, got {commit.get('kind')!r}")

    mode = str(commit.get("mode") or "")
    if mode not in {"attempt", "reuse", "sync"}:
        _fail(f"unsupported mode {mode!r}")

    phase_b_digest = str(commit.get("phase_b_digest") or "")
    if not _is_hex_digest(phase_b_digest):
        _fail(f"phase_b_digest missing/invalid: {phase_b_digest!r}")

    phase_b_checkpoint = str(commit.get("phase_b_checkpoint") or "")
    if not phase_b_checkpoint.startswith("checkpoint-"):
        _fail(f"phase_b_checkpoint missing/invalid: {phase_b_checkpoint!r}")

    rel = str(commit.get("phase_b_durable_relpath") or "")
    if not rel:
        _fail("phase_b_durable_relpath missing")

    if mode == "attempt":
        aid = str(commit.get("attempt_id") or "")
        if not _is_safe_phase_b_attempt_id(aid):
            _fail(f"attempt_id missing/unsafe: {aid!r}")
        if not phase_b_digest.lower().startswith(aid.lower()):
            _fail(
                f"attempt_id {aid!r} is not a prefix of phase_b_digest {phase_b_digest[:16]}…"
            )
        expected_rel = f"{PHASE_B_ATTEMPTS_DIRNAME}/{aid}"
        if rel != expected_rel:
            _fail(
                f"phase_b_durable_relpath {rel!r} != expected {expected_rel!r}"
            )
        attempt_path = root / expected_rel
        if not attempt_path.is_dir():
            _fail(f"referenced attempt missing: {attempt_path}")
        if not _mt_checkpoint_complete(attempt_path):
            _fail(f"referenced attempt incomplete: {attempt_path}")
        on_disk = str(checkpoint_content_digest(attempt_path)["digest"])
        if on_disk != phase_b_digest:
            _fail(
                f"referenced attempt digest mismatch: on_disk={on_disk} meta={phase_b_digest}"
            )
    else:
        # reuse / sync → canonical store checkpoint
        aid = commit.get("attempt_id")
        if aid is not None and str(aid) != "":
            _fail(f"mode={mode!r} must not set attempt_id, got {aid!r}")
        expected_rel = f"{CHECKPOINT_STORE_DIRNAME}/{phase_b_checkpoint}"
        if rel != expected_rel:
            _fail(
                f"phase_b_durable_relpath {rel!r} != expected {expected_rel!r}"
            )
        ckpt_path = root / expected_rel
        if not ckpt_path.is_dir():
            _fail(f"referenced durable checkpoint missing: {ckpt_path}")
        if not _mt_checkpoint_complete(ckpt_path):
            _fail(f"referenced durable checkpoint incomplete: {ckpt_path}")
        on_disk = str(checkpoint_content_digest(ckpt_path)["digest"])
        if on_disk != phase_b_digest:
            _fail(
                f"referenced durable checkpoint digest mismatch: "
                f"on_disk={on_disk} meta={phase_b_digest}"
            )

    return dict(commit)


def _attempt_ids_referenced_by_commit(commit: Optional[Mapping[str, Any]]) -> set:
    ids: set = set()
    if not commit:
        return ids
    aid = commit.get("attempt_id")
    if aid and _is_safe_phase_b_attempt_id(str(aid)):
        ids.add(str(aid))
    rel = str(commit.get("phase_b_durable_relpath") or "")
    prefix = f"{PHASE_B_ATTEMPTS_DIRNAME}/"
    if rel.startswith(prefix):
        cand = rel[len(prefix) :].split("/")[0]
        if _is_safe_phase_b_attempt_id(cand):
            ids.add(cand)
    return ids


def plan_mt_resume_test_phase_b_attempt_retention(
    *,
    current_attempt_id: Optional[str],
    previous_commit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Keep the attempt referenced by the commit being written, plus at most one
    previous rollback attempt from the prior durable commit metadata.
    """
    keep: set = set()
    if current_attempt_id and _is_safe_phase_b_attempt_id(str(current_attempt_id)):
        keep.add(str(current_attempt_id))
    prev_ids = _attempt_ids_referenced_by_commit(previous_commit)
    # At most one previous rollback (stable pick).
    for pid in sorted(prev_ids):
        if pid not in keep:
            keep.add(pid)
            break
    return {"keep_attempt_ids": sorted(keep), "previous_attempt_ids": sorted(prev_ids)}


def prune_mt_resume_test_phase_b_attempts(
    dest_root: Union[str, Path],
    *,
    keep_attempt_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Fail-closed prune of orphan ``phase_b_attempts/<id>/`` directories.

    Never touches ``ckpts/`` or ``snapshots/``. Only deletes attempt dirs whose
    names are safe digest-prefix ids and not in ``keep_attempt_ids``. Unknown /
    unsafe names are left in place and reported (caller must budget-check).
    """
    root = Path(dest_root)
    attempts = root / PHASE_B_ATTEMPTS_DIRNAME
    keep = {str(x) for x in keep_attempt_ids if _is_safe_phase_b_attempt_id(str(x))}
    deleted: List[str] = []
    retained: List[str] = []
    unsafe: List[str] = []
    if not attempts.is_dir():
        return {
            "deleted": deleted,
            "retained": retained,
            "unsafe_left": unsafe,
            "keep_attempt_ids": sorted(keep),
        }
    for child in sorted(attempts.iterdir()):
        if child.name.startswith("."):
            # Staging / parked temps — remove only clear tmp_copy leftovers.
            if child.name.endswith(".tmp_copy") and child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                deleted.append(child.name)
            continue
        if not child.is_dir():
            unsafe.append(child.name)
            continue
        if not _is_safe_phase_b_attempt_id(child.name):
            unsafe.append(child.name)
            continue
        if child.name in keep:
            retained.append(child.name)
            continue
        # Orphan: safe to delete only if it looks like a checkpoint tree or empty.
        shutil.rmtree(child)
        deleted.append(child.name)
    if unsafe:
        raise RuntimeError(
            f"Refusing to prune unsafe phase_b_attempts entries under {attempts}: {unsafe}. "
            f"Manual inspection required; durable commit aborted fail-closed."
        )
    return {
        "deleted": deleted,
        "retained": retained,
        "unsafe_left": unsafe,
        "keep_attempt_ids": sorted(keep),
    }


def enforce_mt_resume_test_durable_budget(
    dest_root: Union[str, Path],
    *,
    budget_bytes: int,
    extra_upload_bytes: int = 0,
    label: str = "mt_resume_test durable",
) -> Dict[str, Any]:
    """Budget gate including ``ckpts/`` + ``phase_b_attempts/`` (+ pending upload)."""
    measured = measure_mt_resume_test_durable_bytes(dest_root)
    report = assert_durable_checkpoint_budget(
        protected_bytes=int(measured["total_bytes"]),
        upload_bytes=int(extra_upload_bytes),
        budget_bytes=int(budget_bytes),
        label=label,
        detail={
            "store_bytes": measured["store_bytes"],
            "attempt_bytes": measured["attempt_bytes"],
            "per_attempt_bytes": measured["per_attempt_bytes"],
            "extra_upload_bytes": int(extra_upload_bytes),
        },
    )
    report["measured"] = measured
    return report


def commit_mt_resume_test_phase_b_durable(
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    phase_a_steps: int,
    phase_b_steps: int,
    budget_bytes: Optional[int] = None,
    save_total_limit: int = 2,
) -> Dict[str, Any]:
    """
    Retry-safe durable commit for MT resume_test Phase B.

    Strategies (never force-overwrite snapshot-referenced store entries):
      * reuse — durable checkpoint-{B} complete and digest matches local
      * sync — normal sync_experiment_checkpoints_to_durable (no collision)
      * attempt — snapshot-referenced name collides with different digest:
        publish Phase B under phase_b_attempts/<attempt_id>/, park local
        checkpoint-{B} during sync so the immutable store slot is untouched,
        then restore the local Phase B directory.

    Attempt bytes count toward ``BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES``. Orphan
    attempts are pruned after a successful commit (keep current + ≤1 previous).
    """
    local = Path(local_experiment_dir)
    state_dir = Path(full_state_dir)
    phase_a_name = f"checkpoint-{int(phase_a_steps)}"
    phase_b_name = f"checkpoint-{int(phase_b_steps)}"
    local_a = local / phase_a_name
    local_b = local / phase_b_name
    if not local_a.is_dir() or not _mt_checkpoint_complete(local_a):
        raise RuntimeError(f"Phase A checkpoint missing/incomplete for durable commit: {local_a}")
    if not local_b.is_dir() or not _mt_checkpoint_complete(local_b):
        raise RuntimeError(f"Phase B checkpoint missing/incomplete for durable commit: {local_b}")

    phase_a_digest = str(checkpoint_content_digest(local_a)["digest"])
    phase_b_digest = str(checkpoint_content_digest(local_b)["digest"])

    dest_root = durable_experiment_dir(state_dir, experiment_id, kind=RESUME_TEST_MARKER)
    store = dest_root / CHECKPOINT_STORE_DIRNAME
    store.mkdir(parents=True, exist_ok=True)
    snap_refs = _mt_resume_test_snap_ref_names(dest_root)
    store_b = store / phase_b_name
    previous_commit = _load_existing_durable_commit(
        state_dir,
        experiment_id=experiment_id,
        dest_root=dest_root,
    )

    mode = "sync"
    attempt_id: Optional[str] = None
    phase_b_durable_rel: str = f"{CHECKPOINT_STORE_DIRNAME}/{phase_b_name}"

    store_b_complete = store_b.is_dir() and _mt_checkpoint_complete(store_b)
    store_b_digest = (
        str(checkpoint_content_digest(store_b)["digest"]) if store_b_complete else None
    )
    if store_b_complete and store_b_digest == phase_b_digest:
        mode = "reuse"
    elif phase_b_name in snap_refs and (not store_b_complete or store_b_digest != phase_b_digest):
        mode = "attempt"
        attempt_id = phase_b_digest[:12]
        if not _is_safe_phase_b_attempt_id(attempt_id):
            raise RuntimeError(f"Refusing unsafe phase_b attempt_id derived from digest: {attempt_id!r}")
        phase_b_durable_rel = f"{PHASE_B_ATTEMPTS_DIRNAME}/{attempt_id}"

    budget = resolve_durable_checkpoint_budget_bytes(budget_bytes, required=True)
    assert budget is not None

    retention = plan_mt_resume_test_phase_b_attempt_retention(
        current_attempt_id=attempt_id,
        previous_commit=previous_commit,
    )
    # Prune orphans before budget math so stale attempts cannot inflate peak forever.
    # Only reached when previous_commit loaded cleanly (or was missing).
    prune_mt_resume_test_phase_b_attempts(
        dest_root, keep_attempt_ids=retention["keep_attempt_ids"]
    )

    extra_upload = 0
    attempt_already_present = False
    if mode == "attempt":
        assert attempt_id is not None
        attempt_root = dest_root / PHASE_B_ATTEMPTS_DIRNAME / attempt_id
        attempt_already_present = (
            attempt_root.is_dir()
            and _mt_checkpoint_complete(attempt_root)
            and str(checkpoint_content_digest(attempt_root)["digest"]) == phase_b_digest
        )
        if not attempt_already_present:
            extra_upload = int(measure_dir_bytes(local_b))
    enforce_mt_resume_test_durable_budget(
        dest_root,
        budget_bytes=int(budget),
        extra_upload_bytes=int(extra_upload),
        label=f"mt_resume_test/{experiment_id} before durable mutate",
    )

    parked: Optional[Path] = None
    try:
        if mode == "attempt":
            assert attempt_id is not None
            attempt_root = dest_root / PHASE_B_ATTEMPTS_DIRNAME / attempt_id
            attempt_root.parent.mkdir(parents=True, exist_ok=True)

            if attempt_already_present:
                # Reuse existing same-digest attempt: verify only — no tmp / no copytree.
                if not _mt_checkpoint_complete(attempt_root):
                    raise RuntimeError(f"Existing phase_b attempt incomplete: {attempt_root}")
                got = str(checkpoint_content_digest(attempt_root)["digest"])
                if got != phase_b_digest:
                    raise RuntimeError(
                        f"Existing phase_b attempt digest drift at {attempt_root}: "
                        f"{got} != {phase_b_digest}"
                    )
            else:
                tmp = dest_root / PHASE_B_ATTEMPTS_DIRNAME / f".{attempt_id}.tmp_copy"
                if tmp.exists():
                    shutil.rmtree(tmp)
                shutil.copytree(local_b, tmp)
                if not _mt_checkpoint_complete(tmp):
                    raise RuntimeError(f"Attempt Phase B copy incomplete: {tmp}")
                got = str(checkpoint_content_digest(tmp)["digest"])
                if got != phase_b_digest:
                    raise RuntimeError("Attempt Phase B digest mismatch after copy")
                if attempt_root.exists():
                    raise RuntimeError(
                        f"Refusing to mutate existing phase_b attempt dir {attempt_root}"
                    )
                os.replace(str(tmp), str(attempt_root))

            enforce_mt_resume_test_durable_budget(
                dest_root,
                budget_bytes=int(budget),
                extra_upload_bytes=0,
                label=f"mt_resume_test/{experiment_id} after attempt materialize",
            )

            parked = local / f".mt_phase_b_parked_{attempt_id}"
            if parked.exists():
                shutil.rmtree(parked, ignore_errors=True)
            os.replace(str(local_b), str(parked))

            sync_experiment_checkpoints_to_durable(
                local,
                state_dir,
                experiment_id=experiment_id,
                kind=RESUME_TEST_MARKER,
                require_durable=True,
                save_total_limit=int(save_total_limit),
                budget_bytes=budget,
            )
        elif mode == "sync":
            sync_experiment_checkpoints_to_durable(
                local,
                state_dir,
                experiment_id=experiment_id,
                kind=RESUME_TEST_MARKER,
                require_durable=True,
                save_total_limit=int(save_total_limit),
                budget_bytes=budget,
            )
        # mode == "reuse": durable Phase B already matches; verify only (no mutate).
    finally:
        if parked is not None and parked.exists() and not local_b.exists():
            os.replace(str(parked), str(local_b))
        elif parked is not None and parked.exists():
            shutil.rmtree(parked, ignore_errors=True)

    # Verify Phase A still resolvable / Phase B durable artifact matches.
    if not (store / phase_a_name).is_dir() or not _mt_checkpoint_complete(store / phase_a_name):
        latest = resolve_durable_latest_snapshot(state_dir, experiment_id, kind=RESUME_TEST_MARKER)
        if latest is None:
            raise RuntimeError("Durable LATEST missing after resume_test Phase B commit")
    if mode in {"reuse", "sync"}:
        if not store_b.is_dir() or not _mt_checkpoint_complete(store_b):
            raise RuntimeError(f"Durable Phase B checkpoint missing after sync: {store_b}")
        got_b = str(checkpoint_content_digest(store_b)["digest"])
        if got_b != phase_b_digest:
            raise RuntimeError(
                f"Durable Phase B digest mismatch after sync: {got_b} != {phase_b_digest}"
            )
        phase_b_durable_rel = f"{CHECKPOINT_STORE_DIRNAME}/{phase_b_name}"
    else:
        attempt_path = dest_root / PHASE_B_ATTEMPTS_DIRNAME / str(attempt_id)
        if not attempt_path.is_dir() or not _mt_checkpoint_complete(attempt_path):
            raise RuntimeError(f"Durable Phase B attempt missing/incomplete: {attempt_path}")
        got_b = str(checkpoint_content_digest(attempt_path)["digest"])
        if got_b != phase_b_digest:
            raise RuntimeError(
                f"Durable Phase B attempt digest mismatch: {got_b} != {phase_b_digest}"
            )
        if store_b_complete and store_b_digest is not None:
            still = str(checkpoint_content_digest(store_b)["digest"])
            if still != store_b_digest:
                raise RuntimeError(
                    f"Immutable store {phase_b_name} was mutated during attempt commit"
                )

    latest = resolve_durable_latest_snapshot(state_dir, experiment_id, kind=RESUME_TEST_MARKER)
    if latest is None:
        raise RuntimeError("Durable LATEST missing after resume_test Phase B commit")

    prune_report = prune_mt_resume_test_phase_b_attempts(
        dest_root, keep_attempt_ids=retention["keep_attempt_ids"]
    )
    budget_report = enforce_mt_resume_test_durable_budget(
        dest_root,
        budget_bytes=int(budget),
        extra_upload_bytes=0,
        label=f"mt_resume_test/{experiment_id} after prune",
    )

    commit = {
        "durable_commit_ok": True,
        "mode": mode,
        "attempt_id": attempt_id,
        "experiment_id": experiment_id,
        "kind": RESUME_TEST_MARKER,
        "phase_a_checkpoint": phase_a_name,
        "phase_a_digest": phase_a_digest,
        "phase_b_checkpoint": phase_b_name,
        "phase_b_digest": phase_b_digest,
        "phase_b_durable_relpath": phase_b_durable_rel,
        "snapshot": str(latest),
        "snapshot_name": latest.name,
        "attempt_retention": retention,
        "attempt_prune": prune_report,
        "budget": {
            "budget_bytes": int(budget),
            "total_bytes": budget_report["measured"]["total_bytes"],
            "store_bytes": budget_report["measured"]["store_bytes"],
            "attempt_bytes": budget_report["measured"]["attempt_bytes"],
        },
    }
    _atomic_write_json(state_dir / "mt_resume_test_durable_commit.json", commit)
    return commit


def verify_mt_resume_test_durable_commit(
    state_dir: Union[str, Path],
    *,
    experiment_id: str,
    summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fail-closed verification that a durable Phase B commit artifact is intact."""
    state_dir = Path(state_dir)
    commit_path = state_dir / "mt_resume_test_durable_commit.json"
    if not commit_path.is_file():
        raise RuntimeError(f"Missing mt_resume_test_durable_commit.json under {state_dir}")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("durable_commit_ok") is not True:
        raise RuntimeError("mt_resume_test_durable_commit.durable_commit_ok is not True")
    if str(commit.get("experiment_id")) != str(experiment_id):
        raise RuntimeError("durable commit experiment_id mismatch")
    if summary is not None:
        if summary.get("durable_commit_ok") is not True:
            raise RuntimeError("resume_test summary missing durable_commit_ok")
        if str(summary.get("phase_b_digest") or "") != str(commit.get("phase_b_digest") or ""):
            raise RuntimeError("resume_test summary phase_b_digest != durable commit")
        if str(summary.get("attempt_id") or "") != str(commit.get("attempt_id") or ""):
            raise RuntimeError("resume_test summary attempt_id != durable commit")

    dest_root = durable_experiment_dir(state_dir, experiment_id, kind=RESUME_TEST_MARKER)
    rel = str(commit.get("phase_b_durable_relpath") or "")
    phase_b_path = dest_root / rel
    if not phase_b_path.is_dir() or not _mt_checkpoint_complete(phase_b_path):
        raise RuntimeError(f"Durable Phase B artifact missing/incomplete: {phase_b_path}")
    got = str(checkpoint_content_digest(phase_b_path)["digest"])
    if got != str(commit.get("phase_b_digest")):
        raise RuntimeError(
            f"Durable Phase B digest drift: on_disk={got} commit={commit.get('phase_b_digest')}"
        )
    latest = resolve_durable_latest_snapshot(state_dir, experiment_id, kind=RESUME_TEST_MARKER)
    if latest is None:
        raise RuntimeError("Durable LATEST missing while verifying resume_test commit")
    return commit


def load_mt_resume_test_success(
    state_dir: Union[str, Path],
    *,
    expected_contract_hash: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_resume_test_summary.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT resume_test summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_MT_RESUME_TEST:
        raise RuntimeError(f"MT resume_test not successful: {data.get('status')}")
    if expected_contract_hash and data.get("contract_hash") != expected_contract_hash:
        raise RuntimeError("MT resume_test contract_hash mismatch")
    if data.get("durable_commit_ok") is not True:
        raise RuntimeError(
            "MT resume_test summary lacks durable_commit_ok "
            "(stale SUCCESS before durable commit is not accepted)"
        )
    if data.get("failed_checks"):
        raise RuntimeError(f"MT resume_test summary has failed_checks: {data.get('failed_checks')}")
    proof = data.get("proof") or {}
    for key in (
        "model_restored",
        "optimizer_restored",
        "scheduler_restored",
        "rng_restored",
        "data_position_ok",
        "true_restart",
    ):
        if proof.get(key) is not True and data.get(key) is not True:
            raise RuntimeError(f"MT resume_test proof missing/false: {key}")
    cross = data.get("cross_session") or {}
    if cross.get("two_sessions") is not True:
        raise RuntimeError("MT resume_test cross_session.two_sessions is not True")
    exp_id = experiment_id or data.get("experiment_id")
    if not exp_id:
        raise RuntimeError("MT resume_test summary missing experiment_id for durable verify")
    verify_mt_resume_test_durable_commit(state_dir, experiment_id=str(exp_id), summary=data)
    return data


def write_mt_train_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_train_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_mt_train_success(
    state_dir: Union[str, Path],
    *,
    expected_contract_hash: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_train_summary.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT train summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_MT_TRAINING:
        raise RuntimeError(f"MT training not successful: {data.get('status')}")
    if expected_contract_hash and data.get("contract_hash") != expected_contract_hash:
        raise RuntimeError("MT train contract_hash mismatch")
    return data


def write_mt_evaluate_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_evaluate_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def assert_ready_for_mt_full_train(state_dir: Union[str, Path], *, contract: Mapping[str, Any]) -> None:
    assert_locked_bartpho_baseline(str(contract.get("model_id")), str(contract.get("model_revision")))
    load_mt_prepare_success(state_dir, expected_contract=dict(contract))
    load_mt_resume_test_success(
        state_dir,
        expected_contract_hash=str(contract.get("contract_hash")),
        experiment_id=str(contract.get("experiment_id") or "") or None,
    )


def assert_ready_for_mt_evaluate(state_dir: Union[str, Path], *, contract: Mapping[str, Any]) -> None:
    assert_ready_for_mt_full_train(state_dir, contract=contract)
    load_mt_train_success(state_dir, expected_contract_hash=str(contract.get("contract_hash")))


def derive_mt_training_status(
    *,
    global_step: int,
    expected_max_steps: int,
    best_checkpoint_exists: bool,
    metrics_finite: bool,
    frozen_test_accessed: bool,
    started_from_base_or_same_experiment: bool,
) -> str:
    ok = (
        int(global_step) >= int(expected_max_steps)
        and best_checkpoint_exists
        and bool(metrics_finite)
        and frozen_test_accessed is False
        and started_from_base_or_same_experiment
    )
    return STATUS_MT_TRAINING if ok else STATUS_FAILED


def derive_mt_evaluate_status(
    *,
    prediction_count: int,
    validation_count: int,
    metrics_finite: bool,
    frozen_test_accessed: bool,
    contract_ok: bool,
) -> str:
    ok = (
        int(prediction_count) == int(validation_count)
        and metrics_finite
        and frozen_test_accessed is False
        and contract_ok
    )
    return STATUS_MT_EVALUATE if ok else STATUS_FAILED


def resolve_mt_stage_status(
    *,
    full_stage: str,
    pipeline_error: Optional[Any],
    full_status: Optional[str],
) -> str:
    if pipeline_error is not None:
        return STATUS_FAILED
    if full_status:
        return str(full_status)
    return STATUS_FAILED


def derive_notebook04_handoff(
    status: str,
    *,
    frozen_test_accessed: bool,
) -> Dict[str, bool]:
    """Cascaded C0 handoff only after independent SUCCESS_MT_EVALUATE."""
    return {
        "ready_for_cascaded_c0": status == STATUS_MT_EVALUATE and frozen_test_accessed is False,
        "ready_for_rq1_final": False,
        "frozen_test_accessed": bool(frozen_test_accessed),
    }


def build_mt_training_hparams(
    *,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
    max_steps: int,
    num_train_epochs: float,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
    max_source_length: int,
    max_target_length: int,
    generation_max_length: int,
    num_beams: int,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> Dict[str, Any]:
    return {
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "num_train_epochs": float(num_train_epochs),
        "save_steps": int(save_steps),
        "eval_steps": int(eval_steps),
        "save_total_limit": int(save_total_limit),
        "fp16": bool(fp16),
        "bf16": bool(bf16),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "generation_max_length": int(generation_max_length),
        "num_beams": int(num_beams),
        "metric_for_best_model": str(metric_for_best_model),
        "greater_is_better": bool(greater_is_better),
        "stage_version": STAGE_VERSION_TRAIN,
    }


def load_mt_seq2seq_model(model_id: str, model_revision: str) -> Any:
    from transformers import AutoModelForSeq2SeqLM

    assert_model_revision_pinned(model_id, model_revision)
    return AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=model_revision)


def count_tokens_no_special(tokenizer: Any, text: str) -> int:
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
    return int(len(list(enc["input_ids"])))


def count_generated_tokens(token_ids: Sequence[int], *, pad_token_id: Optional[int], eos_token_id: Optional[int]) -> int:
    ids = list(int(x) for x in token_ids)
    if pad_token_id is not None:
        while ids and ids[-1] == int(pad_token_id):
            ids.pop()
    if eos_token_id is not None and ids and ids[-1] == int(eos_token_id):
        ids = ids[:-1]
    return int(len(ids))


def build_mt_validation_monitor_subset(
    eligible_val_df: pd.DataFrame,
    *,
    n_samples: int = LOCKED_MT_MONITOR_SIZE,
    salt: str = "mt_monitor_v1",
) -> pd.DataFrame:
    """Deterministic hash-based UID selection (not first-N)."""
    if len(eligible_val_df) == 0:
        raise RuntimeError("Cannot build MT validation-monitor subset from empty validation")
    n = min(int(n_samples), int(len(eligible_val_df)))
    scored = []
    for uid in eligible_val_df["record_uid"].astype(str).tolist():
        h = hashlib.sha256(f"{salt}:{uid}".encode("utf-8")).hexdigest()
        scored.append((h, uid))
    scored.sort(key=lambda x: (x[0], x[1]))
    selected = {uid for _, uid in scored[:n]}
    out = eligible_val_df.loc[eligible_val_df["record_uid"].astype(str).isin(selected)].copy()
    out = out.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    if len(out) != n:
        raise RuntimeError(f"MT monitor subset size mismatch: got {len(out)} expected {n}")
    return out


def persist_mt_monitor_manifest(
    state_dir: Union[str, Path],
    monitor_df: pd.DataFrame,
    *,
    n_samples: int,
    salt: str = "mt_monitor_v1",
) -> Dict[str, Any]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    uids = monitor_df["record_uid"].astype(str).tolist()
    payload = {
        "n_samples": int(n_samples),
        "actual_size": int(len(uids)),
        "salt": salt,
        "selection_policy": "sha256_salt_uid_sorted",
        "ordered_uids": uids,
        "uid_set_hash": compute_uid_set_hash(monitor_df),
        "note": (
            "Training checkpoint selection uses monitor SacreBLEU only; "
            "final FULL_STAGE=evaluate reports full-validation SacreBLEU."
        ),
    }
    path = state_dir / "mt_validation_monitor_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_mt_training_contract(state_dir: Union[str, Path], contract: Mapping[str, Any]) -> Path:
    """Atomic write; refuses overwrite when an existing contract hash differs."""
    assert_mt_training_contract_self_consistent(contract)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_training_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert_mt_training_contracts_match(
            existing, dict(contract), label="persisted mt_training_contract.json"
        )
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_mt_training_contract(
    state_dir: Union[str, Path],
    *,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_training_contract.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT training contract: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_mt_training_contract_self_consistent(data)
    if expected_hash and str(data.get("mt_training_contract_hash")) != str(expected_hash):
        raise RuntimeError("MT training contract hash mismatch on load")
    return data


def write_mt_resume_test_contract(state_dir: Union[str, Path], contract: Mapping[str, Any]) -> Path:
    assert_mt_resume_test_contract_self_consistent(contract)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_resume_test_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert_mt_resume_test_contracts_match(
            existing, dict(contract), label="persisted mt_resume_test_contract.json"
        )
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_mt_resume_test_contract(
    state_dir: Union[str, Path],
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_resume_test_contract.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT resume-test contract: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_mt_resume_test_contract_self_consistent(data)
    if expected is not None:
        assert_mt_resume_test_contracts_match(data, dict(expected), label="resume-test contract")
    return data


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_canonical_json(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def persist_durable_tokenizer_audit(
    state_dir: Union[str, Path],
    audit: Mapping[str, Any],
    *,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    mt_data_contract_hash: str,
    train_uid_set_hash: Optional[str] = None,
    validation_uid_set_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist tokenizer_audit.json + sidecar checksum into durable contract state."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(audit)
    payload["tokenizer_fingerprint"] = str(tokenizer_fingerprint)
    payload["max_source_length"] = int(max_source_length)
    payload["max_target_length"] = int(max_target_length)
    payload["mt_data_contract_hash"] = str(mt_data_contract_hash)
    if train_uid_set_hash is not None:
        payload["train_uid_set_hash"] = str(train_uid_set_hash)
    if validation_uid_set_hash is not None:
        payload["validation_uid_set_hash"] = str(validation_uid_set_hash)
    digest = sha256_canonical_json(payload)
    payload["audit_sha256"] = digest
    path = state_dir / "tokenizer_audit.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Pretty file for humans; checksum is over canonical form of identity+audit body
    # Recompute digest excluding audit_sha256 itself for stable verify.
    body = {k: v for k, v in payload.items() if k != "audit_sha256"}
    digest = sha256_canonical_json(body)
    payload["audit_sha256"] = digest
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    meta = {
        "path": str(path),
        "audit_sha256": digest,
        "tokenizer_fingerprint": str(tokenizer_fingerprint),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "mt_data_contract_hash": str(mt_data_contract_hash),
    }
    (state_dir / "tokenizer_audit.sha256").write_text(digest + "\n", encoding="utf-8")
    return meta


def load_durable_tokenizer_audit(
    state_dir: Union[str, Path],
    *,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    mt_data_contract_hash: str,
    train_uid_set_hash: Optional[str] = None,
    validation_uid_set_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load durable tokenizer_audit.json if identity+checksum match.
    Raises RuntimeError on missing/mismatch (caller may fall back to fresh audit).
    """
    path = Path(state_dir) / "tokenizer_audit.json"
    if not path.is_file():
        raise RuntimeError(f"Missing durable tokenizer_audit.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in data.items() if k != "audit_sha256"}
    digest = sha256_canonical_json(body)
    stored = str(data.get("audit_sha256") or "")
    if stored != digest:
        raise RuntimeError(
            f"tokenizer_audit SHA256 mismatch: stored={stored!r} recomputed={digest!r}"
        )
    if str(data.get("tokenizer_fingerprint") or "") != str(tokenizer_fingerprint):
        raise RuntimeError("tokenizer_audit tokenizer_fingerprint mismatch")
    if int(data.get("max_source_length") or -1) != int(max_source_length):
        raise RuntimeError("tokenizer_audit max_source_length mismatch")
    if int(data.get("max_target_length") or -1) != int(max_target_length):
        raise RuntimeError("tokenizer_audit max_target_length mismatch")
    if str(data.get("mt_data_contract_hash") or "") != str(mt_data_contract_hash):
        raise RuntimeError("tokenizer_audit mt_data_contract_hash mismatch")
    if train_uid_set_hash is not None and data.get("train_uid_set_hash") is not None:
        if str(data.get("train_uid_set_hash")) != str(train_uid_set_hash):
            raise RuntimeError("tokenizer_audit train_uid_set_hash mismatch")
    if validation_uid_set_hash is not None and data.get("validation_uid_set_hash") is not None:
        if str(data.get("validation_uid_set_hash")) != str(validation_uid_set_hash):
            raise RuntimeError("tokenizer_audit validation_uid_set_hash mismatch")
    if data.get("passed") is not True:
        raise RuntimeError("durable tokenizer_audit did not pass")
    return data


def try_load_durable_tokenizer_audit(
    state_dir: Optional[Union[str, Path]],
    *,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    mt_data_contract_hash: str,
    train_uid_set_hash: Optional[str] = None,
    validation_uid_set_hash: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if state_dir is None:
        return None
    try:
        return load_durable_tokenizer_audit(
            state_dir,
            tokenizer_fingerprint=tokenizer_fingerprint,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            mt_data_contract_hash=mt_data_contract_hash,
            train_uid_set_hash=train_uid_set_hash,
            validation_uid_set_hash=validation_uid_set_hash,
        )
    except Exception:
        return None


def strip_record_uid_features(features: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate HF Trainer remove_unused_columns for record_uid before Seq2Seq collator."""
    return [{k: v for k, v in dict(f).items() if k != "record_uid"} for f in features]


def ensure_mt_full_train_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    train_contract: Mapping[str, Any],
    global_step: int = 0,
    update_global_step: bool = False,
) -> Path:
    """
    Fail-closed fingerprint policy for full_train:

    - If step checkpoints exist: existing fingerprint must match current training
      contract BEFORE any rewrite. Never stamp a new identity onto old weights.
    - If directory empty: create fingerprint for the current contract.
    - If update_global_step=True and identity already matches: merge global_step only.
    """
    from src.asr_full_train import list_step_checkpoints, read_checkpoint_fingerprint

    root = Path(experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    assert_mt_training_contract_self_consistent(train_contract)
    steps = list_step_checkpoints(root)
    fp = read_checkpoint_fingerprint(root)
    expected_hash = str(train_contract.get("mt_training_contract_hash") or "")
    if steps:
        if not fp:
            raise RuntimeError(
                f"Local full_train dir has checkpoints but missing fingerprint "
                f"(refuse to stamp new identity): {root}"
            )
        got = str(
            fp.get("mt_training_contract_hash")
            or (fp.get("mt_training_contract") or {}).get("mt_training_contract_hash")
            or ""
        )
        if got != expected_hash:
            raise RuntimeError(
                "Local full_train fingerprint mt_training_contract_hash mismatch "
                f"(existing={got!r} current={expected_hash!r}); refuse resume/stamp"
            )
        if str(fp.get("experiment_id")) != str(train_contract.get("experiment_id")):
            raise RuntimeError("Local full_train fingerprint experiment_id mismatch")
        if str(fp.get("kind")) != FULL_TRAIN_MARKER:
            raise RuntimeError(f"Local fingerprint kind mismatch: {fp.get('kind')}")
        if update_global_step:
            return write_checkpoint_fingerprint(
                root,
                experiment_id=str(train_contract["experiment_id"]),
                kind=FULL_TRAIN_MARKER,
                global_step=int(global_step),
                extra=fingerprint_extra_from_mt_training_contract(train_contract),
                overwrite=True,
                preserve_existing=True,
            )
        # Identity already valid — do not restamp before resume/train.
        return root / "full_experiment_fingerprint.json"
    # Empty dir: create fresh fingerprint for current contract
    return write_checkpoint_fingerprint(
        root,
        experiment_id=str(train_contract["experiment_id"]),
        kind=FULL_TRAIN_MARKER,
        global_step=int(global_step),
        extra=fingerprint_extra_from_mt_training_contract(train_contract),
        overwrite=True,
        preserve_existing=False,
    )


def write_mt_resume_test_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    experiment_id: str,
    mt_data_contract: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    global_step: int,
) -> Path:
    """Create a real full_experiment_fingerprint.json before durable sync (resume_test)."""
    extra = {
        "mt_data_contract_hash": mt_data_contract.get("contract_hash"),
        "mt_data_contract": dict(mt_data_contract),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "contract_hash": mt_data_contract.get("contract_hash"),
    }
    return write_checkpoint_fingerprint(
        experiment_dir,
        experiment_id=experiment_id,
        kind=RESUME_TEST_MARKER,
        global_step=int(global_step),
        extra=extra,
        overwrite=True,
        preserve_existing=False,
    )


def write_mt_full_train_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    train_contract: Mapping[str, Any],
    global_step: int = 0,
    update_global_step: bool = False,
) -> Path:
    """Backward-compatible alias — prefer ensure_mt_full_train_fingerprint."""
    return ensure_mt_full_train_fingerprint(
        experiment_dir,
        train_contract=train_contract,
        global_step=global_step,
        update_global_step=update_global_step,
    )


def derive_started_from_base_or_same_experiment(
    *,
    resume_ckpt: Optional[str],
    experiment_id: str,
    train_contract: Mapping[str, Any],
    local_experiment_dir: Union[str, Path],
    loaded_from_pinned_base: bool,
) -> bool:
    """
    Provenance gate:
    - no resume → True only if model was loaded from pinned base weights
    - resume → True only if checkpoint is allowed for this full_train experiment
      and matches the current MT training contract hash
    """
    if not resume_ckpt:
        return bool(loaded_from_pinned_base)
    expected = {
        "experiment_id": experiment_id,
        "hparams": fingerprint_extra_from_mt_training_contract(train_contract).get("hparams"),
        "mt_training_contract_hash": train_contract.get("mt_training_contract_hash"),
        "contract_hash": train_contract.get("mt_training_contract_hash"),
    }
    assert_checkpoint_allowed_for_full_train(
        resume_ckpt,
        experiment_id=experiment_id,
        expected_contract=expected,
        experiment_root=local_experiment_dir,
    )
    assert_checkpoint_complete_for_resume(resume_ckpt)
    return True


def derive_phase_a_reached_target(
    phase_a_payload: Mapping[str, Any],
    *,
    target_steps: int,
    checkpoint_path: Union[str, Path],
) -> bool:
    if int(phase_a_payload.get("global_step") or -1) != int(target_steps):
        return False
    try:
        assert_checkpoint_complete_for_resume(checkpoint_path)
    except Exception:
        return False
    return True


def derive_used_separate_experiment_dir(
    experiment_dir: Union[str, Path],
    *,
    allowed_kind: str = RESUME_TEST_MARKER,
) -> bool:
    p = str(Path(experiment_dir).resolve())
    if f"/{allowed_kind}/" not in p and not p.endswith(f"/{allowed_kind}"):
        return False
    if f"/{FULL_TRAIN_MARKER}/" in p or f"/{PILOT_MARKER}/" in p:
        return False
    return True


def resolve_latest_checkpoint_dir(experiment_dir: Union[str, Path]) -> Optional[Path]:
    from src.asr_full_train import list_step_checkpoints

    steps = list_step_checkpoints(experiment_dir)
    return steps[-1] if steps else None


def resolve_mt_best_checkpoint_for_evaluate(
    *,
    full_state_dir: Union[str, Path],
    experiment_id: str,
    train_summary: Mapping[str, Any],
    local_experiment_dir: Union[str, Path],
    expected_training_contract: Optional[Mapping[str, Any]] = None,
) -> str:
    """Correct wrapper around resolve_best_checkpoint_from_durable + MT contract checks."""
    expected = None
    if expected_training_contract is not None:
        expected = {
            "experiment_id": expected_training_contract.get("experiment_id"),
            "hparams": fingerprint_extra_from_mt_training_contract(expected_training_contract).get("hparams"),
            "mt_training_contract_hash": expected_training_contract.get("mt_training_contract_hash"),
            "contract_hash": expected_training_contract.get("mt_training_contract_hash"),
        }
    best = resolve_best_checkpoint_from_durable(
        full_state_dir,
        experiment_id=experiment_id,
        train_summary=dict(train_summary),
        local_experiment_dir=local_experiment_dir,
        expected_contract=expected,
    )
    best_path = Path(best)
    if not best_path.is_dir():
        raise RuntimeError(f"Resolved best checkpoint missing: {best_path}")
    assert_checkpoint_complete_for_resume(best_path)
    assert_checkpoint_allowed_for_full_train(
        best_path,
        experiment_id=experiment_id,
        expected_contract=expected,
        experiment_root=local_experiment_dir,
    )
    if expected_training_contract is not None:
        # Fingerprint may live at experiment root
        from src.asr_full_train import read_checkpoint_fingerprint

        fp = read_checkpoint_fingerprint(local_experiment_dir) or {}
        got_hash = fp.get("mt_training_contract_hash") or (fp.get("mt_training_contract") or {}).get(
            "mt_training_contract_hash"
        )
        if got_hash and str(got_hash) != str(expected_training_contract.get("mt_training_contract_hash")):
            raise RuntimeError(
                "Restored experiment fingerprint mt_training_contract_hash mismatch: "
                f"{got_hash} != {expected_training_contract.get('mt_training_contract_hash')}"
            )
    return str(best_path)


def assert_evaluate_config_consistency(
    *,
    train_summary: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    current_training_contract: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
) -> None:
    assert_mt_training_contracts_match(
        training_contract, current_training_contract, label="evaluate vs persisted training contract"
    )
    if str(train_summary.get("mt_training_contract_hash")) != str(
        training_contract.get("mt_training_contract_hash")
    ):
        raise RuntimeError("train_summary mt_training_contract_hash != training contract")
    data_h = train_summary.get("mt_data_contract_hash") or train_summary.get("contract_hash")
    if str(data_h) != str(training_contract.get("mt_data_contract_hash")):
        raise RuntimeError("evaluate data contract hash mismatch vs training contract")
    assert_evaluate_matches_train_summary(
        train_summary,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_fingerprint=tokenizer_fingerprint,
        generation_config=generation_config,
    )


def derive_frozen_test_accessed(opened_paths: Sequence[str], loaded_splits: Sequence[str]) -> bool:
    """Return True if any opened path/split looks like frozen test (should already have raised)."""
    from src.asr_utils import is_forbidden_test_path
    from src.asr_full_data import is_frozen_split_label

    for p in opened_paths:
        if is_forbidden_test_path(p):
            return True
    for s in loaded_splits:
        if is_frozen_split_label(str(s)):
            return True
    return False


def generate_mt_predictions(
    *,
    model: Any,
    tokenizer: Any,
    dataset: MtTextDataset,
    generation_max_length: int,
    num_beams: int,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Greedy/beam generate; uses generation_max_length (same semantics as Trainer)."""
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.eval()
    collator = make_seq2seq_collator(tokenizer, model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    def _collate(feats):
        uids = [f["record_uid"] for f in feats]
        batch = collator([{k: v for k, v in f.items() if k != "record_uid"} for f in feats])
        batch["_uids"] = uids
        return batch

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    src_col = f"{SOURCE_FIELD}_norm" if f"{SOURCE_FIELD}_norm" in dataset.df.columns else SOURCE_FIELD
    tgt_col = f"{TARGET_FIELD}_norm" if f"{TARGET_FIELD}_norm" in dataset.df.columns else TARGET_FIELD
    uid_to_idx = {
        str(uid): i for i, uid in enumerate(dataset.df["record_uid"].astype(str).tolist())
    }
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            uids = batch.pop("_uids")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=int(generation_max_length),
                num_beams=int(num_beams),
            )
            preds = tokenizer.batch_decode(outs, skip_special_tokens=True)
            for i, uid in enumerate(uids):
                idx = uid_to_idx.get(str(uid))
                if idx is None:
                    raise RuntimeError(f"UID not found in dataset index: {uid}")
                row = dataset.df.iloc[idx]
                src_text = normalize_mt_text_v1(row[src_col])
                tgt_text = normalize_mt_text_v1(row[tgt_col])
                pred_ids = outs[i].detach().cpu().tolist()
                rows.append(
                    {
                        "record_uid": str(uid),
                        "text_bahnar": src_text,
                        "text_vi_reference": tgt_text,
                        "text_vi_prediction": normalize_mt_text_v1(preds[i]),
                        "source_token_length": count_tokens_no_special(tokenizer, src_text),
                        "target_token_length": count_tokens_no_special(tokenizer, tgt_text),
                        "prediction_token_length": count_generated_tokens(
                            pred_ids, pad_token_id=pad_id, eos_token_id=eos_id
                        ),
                        "target_char_length": int(len(tgt_text)),
                        "source_char_length": int(len(src_text)),
                    }
                )
    return rows


def evaluate_mt_predictions(pred_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hyps = [str(r["text_vi_prediction"]) for r in pred_rows]
    refs = [str(r["text_vi_reference"]) for r in pred_rows]
    metrics = mt_corpus_metrics(hyps, refs)
    metrics["prediction_count"] = len(pred_rows)
    return metrics


def assert_evaluate_matches_train_summary(
    train_summary: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    generation_config: Mapping[str, Any],
) -> None:
    """Fail-closed: evaluate must reuse train generation/model settings."""
    from src.mt_contract import assert_generation_config_matches, build_generation_config

    expected = train_summary.get("generation_config") or {}
    if not expected:
        expected = build_generation_config(
            model_id=str(train_summary.get("model_id")),
            model_revision=str(train_summary.get("model_revision")),
            tokenizer_fingerprint=str(train_summary.get("tokenizer_fingerprint")),
            max_source_length=int((train_summary.get("hparams") or {}).get("max_source_length") or generation_config["max_source_length"]),
            max_target_length=int((train_summary.get("hparams") or {}).get("max_target_length") or generation_config["max_target_length"]),
            generation_max_length=int((train_summary.get("hparams") or {}).get("generation_max_length") or generation_config["generation_max_length"]),
            num_beams=int((train_summary.get("hparams") or {}).get("num_beams") or generation_config["num_beams"]),
            metric_for_best_model=str((train_summary.get("hparams") or {}).get("metric_for_best_model") or generation_config["metric_for_best_model"]),
            greater_is_better=bool((train_summary.get("hparams") or {}).get("greater_is_better", generation_config["greater_is_better"])),
        )
    if str(train_summary.get("model_id")) != str(model_id):
        raise RuntimeError("evaluate MODEL_ID != train summary")
    if str(train_summary.get("model_revision")) != str(model_revision):
        raise RuntimeError("evaluate MODEL_REVISION != train summary")
    if str(train_summary.get("tokenizer_fingerprint")) != str(tokenizer_fingerprint):
        raise RuntimeError("evaluate tokenizer_fingerprint != train summary")
    assert_generation_config_matches(expected, generation_config)


def select_best_checkpoint_by_metric(
    history: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    greater_is_better: bool,
) -> Optional[str]:
    best_name = None
    best_val = None
    for row in history:
        if metric_key not in row:
            continue
        val = row[metric_key]
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            continue
        if best_val is None:
            best_val, best_name = val, row.get("checkpoint")
            continue
        better = val > best_val if greater_is_better else val < best_val
        if better:
            best_val, best_name = val, row.get("checkpoint")
    return str(best_name) if best_name else None
