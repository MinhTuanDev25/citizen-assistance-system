"""RQ2 Notebook 13: same-budget D-Random and D-Quality selection.

Both arms walk one frozen U′. D-Random's order is SHA256(seed, segment_uid)
and never reads quality_score. D-Quality's order is quality_score descending,
then segment_uid ascending. A segment is taken only when adding its full
duration does not exceed the shared target. Audio is never cut.

The walk does not stop at the first segment that does not fit. Later segments
in the same order are still taken when they fit. Segments are not reordered
by duration to pack the budget more tightly.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.rq1_contract import sha256_file
from src.rq2_pseudo_contract import (
    assert_not_g_test_path,
    finalize_generation,
    stage_generation,
    verify_generation,
    write_json,
)
from src.rq2_selection_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    DURATION_COLUMN,
    MANIFEST_COLUMNS,
    PSEUDO_LABEL_COLUMN,
    QUALITY_COLUMN,
    SELECTION_SCHEMA_VERSION,
    STATUS_SELECTION_FROZEN,
    UID_COLUMN,
    FrozenUPrime,
    SelectionIntegrityError,
    SelectionPolicyError,
    U_PRIME_COLUMNS,
    _FLOAT_U_PRIME_COLUMNS,
    assert_nb13_output_dir,
    assert_nb13_sources_clean,
    build_selection_contract,
    ordered_uid_sha256,
    resolve_frozen_u_prime,
    selection_source_paths,
    validate_u_prime_rows,
    verify_selection_contract,
)

GATE_NAMES = (
    "nb12_frozen_input_verified",
    "d_random_subset_of_u_prime",
    "d_quality_subset_of_u_prime",
    "no_duplicate_uid_within_arm",
    "random_ordering_deterministic",
    "quality_ordering_deterministic",
    "same_target_budget",
    "target_budget_within_u_prime_capacity",
    "nonempty_selection_both_arms",
    "no_arm_exceeds_target",
    "underfill_within_segment_granularity",
    "same_realized_budget_within_tolerance",
    "pseudo_labels_unchanged",
    "durations_unchanged",
    "quality_scores_unchanged",
    "artifacts_hashed",
    "contract_hashed",
    "no_g_test_access",
    "no_training_or_teacher_inference",
)

SCIENTIFIC_GATE_NAMES = (
    "d_random_subset_of_u_prime",
    "d_quality_subset_of_u_prime",
    "no_duplicate_uid_within_arm",
    "random_ordering_deterministic",
    "quality_ordering_deterministic",
    "same_target_budget",
    "target_budget_within_u_prime_capacity",
    "nonempty_selection_both_arms",
    "no_arm_exceeds_target",
    "underfill_within_segment_granularity",
    "same_realized_budget_within_tolerance",
    "pseudo_labels_unchanged",
    "durations_unchanged",
    "quality_scores_unchanged",
)

SELECTION_ARTIFACT_FILES = (
    "d_random_manifest.csv",
    "d_quality_manifest.csv",
    "selection_audit.json",
    "contract.json",
    "summary.json",
)


def random_selection_key(seed: int, segment_uid: str) -> str:
    """Platform-stable key. The digest does not include quality_score or a path."""
    payload = f"{int(seed)}\n{segment_uid}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def order_d_random(rows: Sequence[Mapping[str, Any]], seed: int) -> List[Dict[str, Any]]:
    if type(seed) is not int:
        raise SelectionPolicyError("random_seed must be an int")
    keyed = []
    for row in rows:
        uid = str(row[UID_COLUMN])
        keyed.append((random_selection_key(seed, uid), uid, dict(row)))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed]


def order_d_quality(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = []
    for row in rows:
        keyed.append((-float(row[QUALITY_COLUMN]), str(row[UID_COLUMN]), dict(row)))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed]


def fill_duration_budget(
    ordered_rows: Sequence[Mapping[str, Any]],
    target_seconds: float,
    *,
    arm: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Greedy fill. ``used + duration <= target`` is required; audio is not cut."""
    if not math.isfinite(target_seconds) or float(target_seconds) <= 0.0:
        raise SelectionPolicyError("target budget must be finite and > 0 seconds")
    target = float(target_seconds)
    selected: List[Dict[str, Any]] = []
    used = 0.0
    for row in ordered_rows:
        duration = float(row[DURATION_COLUMN])
        if used + duration <= target:
            used += duration
            enriched = {column: row.get(column) for column in U_PRIME_COLUMNS}
            uid = str(row[UID_COLUMN])
            if arm == ARM_RANDOM:
                key = random_selection_key(int(seed), uid)
            else:
                key = f"{float(row[QUALITY_COLUMN])!r}|{uid}"
            enriched.update({
                "selection_arm": arm,
                "selection_rank": len(selected),
                "selection_key": key,
                "cumulative_duration_seconds": used,
            })
            selected.append(enriched)
    unused = target - used
    return {
        "arm": arm,
        "rows": selected,
        "considered_uids": [str(row[UID_COLUMN]) for row in ordered_rows],
        "selected_uids": [row[UID_COLUMN] for row in selected],
        "target_budget_seconds": target,
        "selected_duration_seconds": used,
        "unused_budget_seconds": unused,
        "n_selected": len(selected),
    }


def _pool_duration_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(sum(float(row[DURATION_COLUMN]) for row in rows))


def _underfill_within_granularity(
    ordered_rows: Sequence[Mapping[str, Any]],
    selected_uids: Sequence[str],
    used: float,
    target: float,
) -> bool:
    """Unused budget must be strictly shorter than every segment that was not selected."""
    if used > target:
        return False
    unused = target - used
    chosen = set(selected_uids)
    for row in ordered_rows:
        if str(row[UID_COLUMN]) in chosen:
            continue
        if not (unused < float(row[DURATION_COLUMN])):
            return False
    return True


def select_same_budget(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_seconds: float,
    random_seed: int,
) -> Dict[str, Any]:
    """Select both arms from copies of ``rows``. The input sequence is not modified."""
    pool = validate_u_prime_rows(rows)
    random_order = order_d_random(pool, random_seed)
    quality_order = order_d_quality(pool)
    d_random = fill_duration_budget(random_order, target_seconds, arm=ARM_RANDOM, seed=random_seed)
    d_quality = fill_duration_budget(quality_order, target_seconds, arm=ARM_QUALITY)
    by_uid = {row[UID_COLUMN]: row for row in pool}
    overlap_uids = sorted(set(d_random["selected_uids"]) & set(d_quality["selected_uids"]))
    intersection_seconds = sum(float(by_uid[uid][DURATION_COLUMN]) for uid in overlap_uids)
    union = len(set(d_random["selected_uids"]) | set(d_quality["selected_uids"]))
    jaccard = 1.0 if union == 0 else len(overlap_uids) / union
    return {
        "pool": pool,
        "by_uid": by_uid,
        ARM_RANDOM: d_random,
        ARM_QUALITY: d_quality,
        "overlap": {
            "n_intersection": len(overlap_uids),
            "intersection_uids": overlap_uids,
            "intersection_seconds": intersection_seconds,
            "intersection_hours": intersection_seconds / 3600.0,
            "jaccard_uid": jaccard,
        },
    }


def _field_equal(column: str, got: Any, expected: Any) -> bool:
    if column in _FLOAT_U_PRIME_COLUMNS or column == "cumulative_duration_seconds":
        if got is None or expected is None:
            return got is None and expected is None
        return float(got) == float(expected)
    if got is None or expected is None:
        return got is None and expected is None
    if column == "selection_rank":
        return int(got) == int(expected)
    return got == expected


def _field_unchanged(arm: Mapping[str, Any], by_uid: Mapping[str, Mapping[str, Any]], column: str) -> bool:
    for row in arm["rows"]:
        source = by_uid[row[UID_COLUMN]]
        if not _field_equal(column, row.get(column), source.get(column)):
            return False
    return True


def realized_duration_audit(result: Mapping[str, Any], tolerance_seconds: float) -> Dict[str, Any]:
    """Cross-arm duration audit. The tolerance is an input, never measured from the arms."""
    d_random = result[ARM_RANDOM]
    d_quality = result[ARM_QUALITY]
    target = float(d_random["target_budget_seconds"])
    if float(d_quality["target_budget_seconds"]) != target:
        raise SelectionIntegrityError("D-Random and D-Quality do not share one target budget")
    random_selected = float(d_random["selected_duration_seconds"])
    quality_selected = float(d_quality["selected_duration_seconds"])
    gap = abs(random_selected - quality_selected)
    tolerance = float(tolerance_seconds)
    return {
        "target_budget_seconds": target,
        "random_selected_duration_seconds": random_selected,
        "quality_selected_duration_seconds": quality_selected,
        "random_underfill_seconds": float(d_random["unused_budget_seconds"]),
        "quality_underfill_seconds": float(d_quality["unused_budget_seconds"]),
        "realized_duration_gap_seconds": gap,
        "realized_duration_tolerance_seconds": tolerance,
        "same_realized_budget_within_tolerance": gap <= tolerance,
    }


def bind_realized_duration_contract(contract: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach the realized-duration audit and rehash. This does not change either arm."""
    from src.rq1_contract import sha256_json

    body = {key: value for key, value in contract.items() if key != "selection_contract_sha256"}
    body.update(realized_duration_audit(result, float(contract["realized_duration_tolerance_seconds"])))
    body["selection_contract_sha256"] = sha256_json(body)
    return body


def selection_gates(
    result: Mapping[str, Any],
    *,
    random_seed: int,
    realized_duration_tolerance_seconds: float,
) -> Dict[str, bool]:
    """Scientific selection gates. Trust gates stay false until publication proves them."""
    pool = result["pool"]
    d_random = result[ARM_RANDOM]
    d_quality = result[ARM_QUALITY]
    pool_uids = {row[UID_COLUMN] for row in pool}
    random_expected = [row[UID_COLUMN] for row in order_d_random(pool, random_seed)]
    quality_expected = [row[UID_COLUMN] for row in order_d_quality(pool)]
    same_target = (
        float(d_random["target_budget_seconds"]) == float(d_quality["target_budget_seconds"])
    )
    target = float(d_random["target_budget_seconds"])
    total = _pool_duration_seconds(pool)
    realized = realized_duration_audit(result, realized_duration_tolerance_seconds)
    gates = {
        "nb12_frozen_input_verified": False,
        "d_random_subset_of_u_prime": set(d_random["selected_uids"]) <= pool_uids,
        "d_quality_subset_of_u_prime": set(d_quality["selected_uids"]) <= pool_uids,
        "no_duplicate_uid_within_arm": (
            len(set(d_random["selected_uids"])) == len(d_random["selected_uids"])
            and len(set(d_quality["selected_uids"])) == len(d_quality["selected_uids"])
        ),
        "random_ordering_deterministic": d_random["considered_uids"] == random_expected,
        "quality_ordering_deterministic": d_quality["considered_uids"] == quality_expected,
        "same_target_budget": same_target,
        "target_budget_within_u_prime_capacity": bool(same_target and 0.0 < target <= total),
        "nonempty_selection_both_arms": int(d_random["n_selected"]) > 0 and int(d_quality["n_selected"]) > 0,
        "no_arm_exceeds_target": (
            float(d_random["selected_duration_seconds"]) <= target
            and float(d_quality["selected_duration_seconds"]) <= target
        ),
        "underfill_within_segment_granularity": (
            _underfill_within_granularity(order_d_random(pool, random_seed), d_random["selected_uids"], d_random["selected_duration_seconds"], target)
            and _underfill_within_granularity(order_d_quality(pool), d_quality["selected_uids"], d_quality["selected_duration_seconds"], target)
        ),
        "same_realized_budget_within_tolerance": realized["same_realized_budget_within_tolerance"],
        "pseudo_labels_unchanged": _field_unchanged(d_random, result["by_uid"], PSEUDO_LABEL_COLUMN) and _field_unchanged(d_quality, result["by_uid"], PSEUDO_LABEL_COLUMN),
        "durations_unchanged": _field_unchanged(d_random, result["by_uid"], DURATION_COLUMN) and _field_unchanged(d_quality, result["by_uid"], DURATION_COLUMN),
        "quality_scores_unchanged": _field_unchanged(d_random, result["by_uid"], QUALITY_COLUMN) and _field_unchanged(d_quality, result["by_uid"], QUALITY_COLUMN),
        "artifacts_hashed": False,
        "contract_hashed": False,
        "no_g_test_access": False,
        "no_training_or_teacher_inference": False,
    }
    return {name: bool(gates[name]) for name in GATE_NAMES}


def _csv_cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column in _FLOAT_U_PRIME_COLUMNS or column == "cumulative_duration_seconds":
        return repr(float(value))
    if column == "selection_rank":
        return str(int(value))
    return str(value)


def _parse_csv_cell(column: str, text: str) -> Any:
    if text == "":
        return None
    if column in _FLOAT_U_PRIME_COLUMNS or column == "cumulative_duration_seconds":
        return float(text)
    if column == "selection_rank":
        return int(text)
    return text


def write_manifest_csv(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> None:
    from src.rq2_pseudo_contract import atomic_write_text

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _csv_cell(column, row.get(column)) for column in MANIFEST_COLUMNS})
    atomic_write_text(path, buffer.getvalue())


def read_manifest_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != MANIFEST_COLUMNS:
            raise SelectionIntegrityError(f"{Path(path).name} schema drift")
        return [
            {column: _parse_csv_cell(column, raw[column]) for column in MANIFEST_COLUMNS}
            for raw in reader
        ]


def assert_manifest_round_trip(loaded: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> None:
    """Every NB12 field and every NB13 audit field must survive the CSV write."""
    if [row[UID_COLUMN] for row in loaded] != list(arm["selected_uids"]):
        raise SelectionIntegrityError("manifest UID order does not match the selection")
    if len(loaded) != len(arm["rows"]):
        raise SelectionIntegrityError("manifest row count does not match the selection")
    checked = list(U_PRIME_COLUMNS) + ["selection_arm", "selection_rank", "selection_key", "cumulative_duration_seconds"]
    for got, expected in zip(loaded, arm["rows"]):
        uid = str(expected[UID_COLUMN])
        for column in checked:
            if not _field_equal(column, got.get(column), expected.get(column)):
                raise SelectionIntegrityError(f"manifest field {column} drifted", uid)


def _arm_report(arm: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "target_budget_seconds": arm["target_budget_seconds"],
        "selected_duration_seconds": arm["selected_duration_seconds"],
        "unused_budget_seconds": arm["unused_budget_seconds"],
        "n_selected": arm["n_selected"],
        "ordered_uid_sha256": ordered_uid_sha256(arm["selected_uids"]),
        "considered_uid_sha256": ordered_uid_sha256(arm["considered_uids"]),
    }


def _duplicate_pcm_groups(rows: Sequence[Mapping[str, Any]]) -> int:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row["segment_pcm16_sha256"])
        counts[key] = counts.get(key, 0) + 1
    return sum(1 for count in counts.values() if count > 1)


_IDENTITY_KEYS = (
    "nb11_generation_id",
    "nb11_input_contract_sha256",
    "nb12_generation_id",
    "nb12_contract_sha256",
    "u_prime_manifest_sha256",
    "u_prime_ordered_uid_sha256",
    "quality_score_contract_sha256",
)


def _assert_supplied_matches_frozen(
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    frozen: FrozenUPrime,
) -> None:
    supplied = validate_u_prime_rows(rows)
    resolved = list(frozen.rows)
    if len(supplied) != len(resolved):
        raise SelectionIntegrityError("supplied U′ rows do not match the resolved frozen pool")
    for got, expected in zip(supplied, resolved):
        for column in U_PRIME_COLUMNS:
            if not _field_equal(column, got.get(column), expected.get(column)):
                raise SelectionIntegrityError(f"supplied U′ field {column} does not match frozen U′", str(expected[UID_COLUMN]))
    for key in _IDENTITY_KEYS:
        if str(identity.get(key) or "") != str(frozen.identity.get(key) or ""):
            raise SelectionIntegrityError(f"supplied identity {key} does not match the resolved frozen U′")


_TRUST_GATES_AFTER_RESOLVE = (
    "nb12_frozen_input_verified",
    "no_g_test_access",
    "no_training_or_teacher_inference",
)
_TRUST_GATES_AFTER_HASH = ("artifacts_hashed", "contract_hashed")


def prepare_same_budget_selection(
    frozen: FrozenUPrime,
    *,
    selection_budget_hours: Any,
    random_seed: int,
) -> Dict[str, Any]:
    """Build the contract and both arms. This function writes nothing and does not move CURRENT."""
    if not isinstance(frozen, FrozenUPrime):
        raise SelectionIntegrityError("selection requires a resolved FrozenUPrime")
    policy = build_selection_contract(
        frozen.identity, selection_budget_hours=selection_budget_hours, random_seed=random_seed,
    )
    result = select_same_budget(
        frozen.rows, target_seconds=float(policy["selection_budget_seconds"]), random_seed=random_seed,
    )
    contract = bind_realized_duration_contract(policy, result)
    verify_selection_contract(contract)
    gates = selection_gates(
        result,
        random_seed=random_seed,
        realized_duration_tolerance_seconds=float(contract["realized_duration_tolerance_seconds"]),
    )
    if not gates["target_budget_within_u_prime_capacity"]:
        total = _pool_duration_seconds(result["pool"])
        raise SelectionPolicyError(
            f"target budget {policy['selection_budget_seconds']} seconds is outside (0, U′ duration {total}]"
        )
    if not gates["nonempty_selection_both_arms"]:
        raise SelectionIntegrityError("each real selection arm must select at least one segment")
    deferred = set(_TRUST_GATES_AFTER_RESOLVE) | set(_TRUST_GATES_AFTER_HASH)
    pre_publish = [name for name in GATE_NAMES if name not in deferred and not gates[name]]
    if pre_publish:
        raise SelectionIntegrityError("selection gates failed before publication: " + ", ".join(pre_publish))
    return {"result": result, "contract": contract, "gates": gates}


def publish_same_budget_selection(
    out_dir: Union[str, Path],
    *,
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    selection_budget_hours: Any,
    random_seed: int,
    policy_frozen: bool,
    project_root: Optional[Union[str, Path]] = None,
    durable_root: Optional[Union[str, Path]] = None,
    pseudo_dir: Optional[Union[str, Path]] = None,
    u_clean_dir: Optional[Union[str, Path]] = None,
    expected_generation_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve frozen U′ and publish. This is the only NB13 function that can move CURRENT."""
    if policy_frozen is not True:
        raise SelectionPolicyError("SELECTION_POLICY_FROZEN is false; refusing to publish")
    out = Path(out_dir)
    assert_not_g_test_path(out)
    if project_root is None:
        raise SelectionPolicyError("publication requires project_root so frozen NB12 U′ can be re-resolved")
    assert_nb13_output_dir(out, project_root, durable_root=durable_root, env=env)
    resolved = resolve_frozen_u_prime(
        project_root,
        generation_id=expected_generation_id,
        expected_generation_id=expected_generation_id,
        pseudo_dir=pseudo_dir,
        u_clean_dir=u_clean_dir,
        durable_root=durable_root,
        env=env,
    )
    _assert_supplied_matches_frozen(rows, identity, resolved)
    assert_nb13_sources_clean(selection_source_paths())
    prepared = prepare_same_budget_selection(
        resolved, selection_budget_hours=selection_budget_hours, random_seed=random_seed,
    )
    gates = prepared["gates"]
    for name in _TRUST_GATES_AFTER_RESOLVE:
        gates[name] = True
    result = prepared["result"]
    contract = prepared["contract"]
    identity = resolved.identity
    staged = stage_generation(out)
    gen_dir: Path = staged["staging_dir"]
    write_manifest_csv(gen_dir / "d_random_manifest.csv", result[ARM_RANDOM]["rows"])
    write_manifest_csv(gen_dir / "d_quality_manifest.csv", result[ARM_QUALITY]["rows"])
    assert_manifest_round_trip(read_manifest_csv(gen_dir / "d_random_manifest.csv"), result[ARM_RANDOM])
    assert_manifest_round_trip(read_manifest_csv(gen_dir / "d_quality_manifest.csv"), result[ARM_QUALITY])
    realized = {
        key: contract[key]
        for key in (
            "target_budget_seconds",
            "random_selected_duration_seconds",
            "quality_selected_duration_seconds",
            "random_underfill_seconds",
            "quality_underfill_seconds",
            "realized_duration_gap_seconds",
            "realized_duration_tolerance_seconds",
            "same_realized_budget_within_tolerance",
        )
    }
    audit = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "same_budget_definition": contract["same_budget_definition"],
        **realized,
        "arms": {ARM_RANDOM: _arm_report(result[ARM_RANDOM]), ARM_QUALITY: _arm_report(result[ARM_QUALITY])},
        "overlap": result["overlap"],
        "n_duplicate_pcm16_groups": _duplicate_pcm_groups(result["pool"]),
        "u_prime_ordered_uid_sha256": identity["u_prime_ordered_uid_sha256"],
        "n_u_prime": len(result["pool"]),
        "expected_nb12_generation_id": contract["expected_nb12_generation_id"],
        "resolved_nb12_generation_id": contract["resolved_nb12_generation_id"],
        "nb12_current_generation_id": contract["nb12_current_generation_id"],
    }
    write_json(gen_dir / "selection_audit.json", audit)
    write_json(gen_dir / "contract.json", contract)
    disk_contract = _read_back_json(gen_dir / "contract.json")
    verify_selection_contract(disk_contract)
    if disk_contract["selection_contract_sha256"] != contract["selection_contract_sha256"]:
        raise SelectionIntegrityError("staged selection contract hash drifted")
    manifest_hashes = {
        "d_random_manifest_sha256": sha256_file(gen_dir / "d_random_manifest.csv"),
        "d_quality_manifest_sha256": sha256_file(gen_dir / "d_quality_manifest.csv"),
        "selection_audit_sha256": sha256_file(gen_dir / "selection_audit.json"),
    }
    summary = {
        "status": STATUS_SELECTION_FROZEN,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "generation_id": str(staged["generation_id"]),
        "selection_contract_sha256": contract["selection_contract_sha256"],
        "nb11_input_contract_sha256": identity["nb11_input_contract_sha256"],
        "nb12_contract_sha256": identity["nb12_contract_sha256"],
        "u_prime_manifest_sha256": identity["u_prime_manifest_sha256"],
        "u_prime_ordered_uid_sha256": identity["u_prime_ordered_uid_sha256"],
        "selection_budget_hours": contract["selection_budget_hours"],
        "selection_budget_seconds": contract["selection_budget_seconds"],
        "random_seed": contract["random_seed"],
        "expected_nb12_generation_id": contract["expected_nb12_generation_id"],
        "resolved_nb12_generation_id": contract["resolved_nb12_generation_id"],
        **realized,
        **manifest_hashes,
        "d_random": audit["arms"][ARM_RANDOM],
        "d_quality": audit["arms"][ARM_QUALITY],
        "overlap": {
            "n_intersection": result["overlap"]["n_intersection"],
            "intersection_hours": result["overlap"]["intersection_hours"],
            "jaccard_uid": result["overlap"]["jaccard_uid"],
        },
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for name in _TRUST_GATES_AFTER_HASH:
        gates[name] = True
    failed = [name for name in GATE_NAMES if gates.get(name) is not True]
    if failed:
        raise SelectionIntegrityError("selection gates failed before CURRENT moved: " + ", ".join(failed))
    summary["gates"] = {name: True for name in GATE_NAMES}
    write_json(gen_dir / "summary.json", summary)
    verified = finalize_generation(out, staged, SELECTION_ARTIFACT_FILES)
    return {"summary": summary, "generation": verified, "contract": contract, "audit": audit}


def _read_back_json(path: Path) -> Dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelectionIntegrityError(f"JSON object required: {path.name}")
    return payload


def run_offline_synthetic_self_check() -> Dict[str, Any]:
    """In-memory check used by the notebook when real selection is off. Writes nothing."""
    rows = []
    for index, (duration, quality) in enumerate(((5.0, 0.2), (4.0, 0.9), (3.0, 0.4), (1.0, 0.8))):
        uid = f"synthetic-{index}"
        rows.append({
            "segment_uid": uid,
            "source_id": "synthetic",
            "source_group_id": "synthetic",
            "segment_local_path": f"artifacts/rq2/u_clean/segments/synthetic/{uid}.wav",
            "segment_pcm16_sha256": f"{index:064x}",
            "segment_wav_sha256": f"{index + 1:064x}",
            "duration_seconds": duration,
            "asr_text_norm": "bahnar",
            "pseudo_vi_norm": f"pseudo {uid}",
            "asr_mean_logprob": -0.2,
            "mt_mean_logprob": -0.3,
            "teacher_d0_agreement_chrf": None,
            "asr_confidence_norm": 0.5,
            "mt_confidence_norm": 0.5,
            "d0_agreement_norm": None,
            "quality_penalty": 0.0,
            "quality_score": quality,
            "nb11_generation_id": "synthetic",
            "nb11_input_contract_sha256": "ab" * 32,
            "teacher_contract_sha256": "cd" * 32,
            "d0_agreement_contract_sha256": "ef" * 32,
            "quality_score_contract_sha256": "01" * 32,
        })
    result = select_same_budget(rows, target_seconds=6.0, random_seed=42)
    again = select_same_budget(rows, target_seconds=6.0, random_seed=42)
    if result[ARM_RANDOM]["selected_uids"] != again[ARM_RANDOM]["selected_uids"]:
        raise SelectionIntegrityError("synthetic D-Random was not deterministic")
    if result[ARM_QUALITY]["selected_uids"] != again[ARM_QUALITY]["selected_uids"]:
        raise SelectionIntegrityError("synthetic D-Quality was not deterministic")
    changed = [dict(row) for row in rows]
    for row in changed:
        row["quality_score"] = 1.0 - float(row["quality_score"])
    rerandom = select_same_budget(changed, target_seconds=6.0, random_seed=42)
    if rerandom[ARM_RANDOM]["selected_uids"] != result[ARM_RANDOM]["selected_uids"]:
        raise SelectionIntegrityError("synthetic D-Random changed when quality_score changed")
    return {
        "target_budget_seconds": 6.0,
        "d_random_uids": result[ARM_RANDOM]["selected_uids"],
        "d_quality_uids": result[ARM_QUALITY]["selected_uids"],
        "d_random_seconds": result[ARM_RANDOM]["selected_duration_seconds"],
        "d_quality_seconds": result[ARM_QUALITY]["selected_duration_seconds"],
        "d_random_unused_seconds": result[ARM_RANDOM]["unused_budget_seconds"],
        "d_quality_unused_seconds": result[ARM_QUALITY]["unused_budget_seconds"],
        "n_intersection": result["overlap"]["n_intersection"],
        "jaccard_uid": result["overlap"]["jaccard_uid"],
        "wrote_current": False,
    }


_PINNED_IDENTITY_KEYS = (
    "nb11_generation_id",
    "nb11_input_contract_sha256",
    "nb12_generation_id",
    "nb12_contract_sha256",
    "u_prime_manifest_sha256",
    "u_prime_ordered_uid_sha256",
)


def _same_audit_value(got: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return got is expected
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return float(got) == float(expected)
    return got == expected


def verify_published_selection(
    out_dir: Union[str, Path],
    *,
    project_root: Union[str, Path],
    generation_id: Optional[str] = None,
    durable_root: Optional[Union[str, Path]] = None,
    pseudo_dir: Optional[Union[str, Path]] = None,
    u_clean_dir: Optional[Union[str, Path]] = None,
    expected_generation_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Rehash the generation, then recompute both arms from the pinned frozen U′."""
    verified = verify_generation(out_dir, generation_id)
    gen_dir = Path(out_dir) / "generations" / verified["generation_id"]
    contract = _read_back_json(gen_dir / "contract.json")
    verify_selection_contract(contract)
    summary = _read_back_json(gen_dir / "summary.json")
    if summary.get("status") != STATUS_SELECTION_FROZEN:
        raise SelectionIntegrityError("published selection status is not frozen")
    if str(summary.get("generation_id") or "") != str(verified["generation_id"]):
        raise SelectionIntegrityError("summary generation_id does not match the verified NB13 generation")
    for key in (
        "schema_version",
        "selection_contract_sha256",
        "selection_budget_hours",
        "selection_budget_seconds",
        "random_seed",
        "nb11_input_contract_sha256",
        "nb12_contract_sha256",
        "u_prime_manifest_sha256",
        "u_prime_ordered_uid_sha256",
    ):
        if not _same_audit_value(summary.get(key), contract.get(key)):
            raise SelectionIntegrityError(f"summary {key} does not match the selection contract")
    gates = summary.get("gates") or {}
    for name in GATE_NAMES:
        if gates.get(name) is not True:
            raise SelectionIntegrityError(f"summary gate {name} is not true")
    for name, key in (
        ("d_random_manifest.csv", "d_random_manifest_sha256"),
        ("d_quality_manifest.csv", "d_quality_manifest_sha256"),
        ("selection_audit.json", "selection_audit_sha256"),
    ):
        if sha256_file(gen_dir / name) != summary.get(key):
            raise SelectionIntegrityError(f"summary hash does not match {name}")
    for key in _PINNED_IDENTITY_KEYS:
        if key not in contract:
            raise SelectionIntegrityError(f"selection contract is missing {key}")
        if key in summary and str(summary.get(key) or "") != str(contract.get(key) or ""):
            raise SelectionIntegrityError(f"summary {key} does not match the selection contract")
    pin = str(expected_generation_id or contract["expected_nb12_generation_id"])
    if pin != str(contract["expected_nb12_generation_id"]) or pin != str(contract["resolved_nb12_generation_id"]):
        raise RuntimeError(f"pinned NB12 generation {pin} does not match the published selection contract")
    frozen = resolve_frozen_u_prime(
        project_root,
        generation_id=str(contract["nb12_generation_id"]),
        expected_generation_id=pin,
        pseudo_dir=pseudo_dir,
        u_clean_dir=u_clean_dir,
        durable_root=durable_root,
        env=env,
    )
    for key in _PINNED_IDENTITY_KEYS:
        if str(contract.get(key) or "") != str(frozen.identity.get(key) or ""):
            raise SelectionIntegrityError(f"selection contract {key} does not match the pinned frozen U′")
    result = select_same_budget(
        frozen.rows,
        target_seconds=float(contract["selection_budget_seconds"]),
        random_seed=int(contract["random_seed"]),
    )
    recomputed = realized_duration_audit(result, float(contract["realized_duration_tolerance_seconds"]))
    for key, expected_value in recomputed.items():
        if not _same_audit_value(contract.get(key), expected_value):
            raise SelectionIntegrityError(f"selection contract {key} does not match the recomputed selection")
        if key in summary and not _same_audit_value(summary.get(key), expected_value):
            raise SelectionIntegrityError(f"summary {key} does not match the recomputed selection")
    recomputed_gates = selection_gates(
        result,
        random_seed=int(contract["random_seed"]),
        realized_duration_tolerance_seconds=float(contract["realized_duration_tolerance_seconds"]),
    )
    failed_gates = [name for name in SCIENTIFIC_GATE_NAMES if recomputed_gates.get(name) is not True]
    if failed_gates:
        raise SelectionIntegrityError("recomputed selection gates failed: " + ", ".join(failed_gates))
    audit = _read_back_json(gen_dir / "selection_audit.json")
    for key, expected_value in recomputed.items():
        if not _same_audit_value(audit.get(key), expected_value):
            raise SelectionIntegrityError(f"audit {key} does not match the recomputed selection")
    if audit.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise SelectionIntegrityError("audit schema_version does not match the selection schema")
    if not _same_audit_value(audit.get("target_budget_seconds"), contract["selection_budget_seconds"]):
        raise SelectionIntegrityError("audit target_budget_seconds does not match the selection contract")
    if not _same_audit_value(audit.get("n_u_prime"), len(frozen.rows)):
        raise SelectionIntegrityError("audit n_u_prime does not match frozen U′")
    if not _same_audit_value(audit.get("n_duplicate_pcm16_groups"), _duplicate_pcm_groups(frozen.rows)):
        raise SelectionIntegrityError("audit n_duplicate_pcm16_groups does not match frozen U′")
    if str(audit.get("u_prime_ordered_uid_sha256") or "") != str(frozen.identity["u_prime_ordered_uid_sha256"]):
        raise SelectionIntegrityError("audit u_prime_ordered_uid_sha256 does not match frozen U′")
    by_uid = {row[UID_COLUMN]: row for row in frozen.rows}
    manifest_names = {ARM_RANDOM: "d_random_manifest.csv", ARM_QUALITY: "d_quality_manifest.csv"}
    for arm_name, filename in manifest_names.items():
        loaded = read_manifest_csv(gen_dir / filename)
        expected = result[arm_name]
        if [row[UID_COLUMN] for row in loaded] != list(expected["selected_uids"]):
            raise SelectionIntegrityError(f"{arm_name} manifest order does not match the recomputed selection")
        for got, selected in zip(loaded, expected["rows"]):
            uid = str(got[UID_COLUMN])
            source = by_uid[uid]
            for column in U_PRIME_COLUMNS:
                if not _field_equal(column, got.get(column), source.get(column)):
                    raise SelectionIntegrityError(f"{arm_name} field {column} differs from frozen U′", uid)
            for column in ("selection_arm", "selection_rank", "selection_key", "cumulative_duration_seconds"):
                if not _field_equal(column, got.get(column), selected.get(column)):
                    raise SelectionIntegrityError(f"{arm_name} audit field {column} drifted", uid)
        expected_report = _arm_report(expected)
        report = (audit.get("arms") or {}).get(arm_name) or {}
        summary_arm = summary.get(arm_name) or {}
        if set(report) != set(expected_report) or set(summary_arm) != set(expected_report):
            raise SelectionIntegrityError(f"{arm_name} audit report keys do not match the recomputed selection")
        for field, expected_value in expected_report.items():
            if not _same_audit_value(report.get(field), expected_value):
                raise SelectionIntegrityError(f"{arm_name} audit {field} does not match the recomputed selection")
            if not _same_audit_value(summary_arm.get(field), expected_value):
                raise SelectionIntegrityError(f"{arm_name} summary {field} does not match the recomputed selection")
    audit_overlap = audit.get("overlap") or {}
    expected_overlap = result["overlap"]
    if list(audit_overlap.get("intersection_uids") or []) != list(expected_overlap["intersection_uids"]):
        raise SelectionIntegrityError("audit overlap intersection_uids does not match the recomputed selection")
    for field in ("n_intersection", "intersection_seconds", "intersection_hours", "jaccard_uid"):
        if not _same_audit_value(audit_overlap.get(field), expected_overlap[field]):
            raise SelectionIntegrityError(f"audit overlap {field} does not match the recomputed selection")
    summary_overlap = summary.get("overlap") or {}
    for field in ("n_intersection", "intersection_hours", "jaccard_uid"):
        if not _same_audit_value(summary_overlap.get(field), expected_overlap[field]):
            raise SelectionIntegrityError(f"summary overlap {field} does not match the recomputed selection")
    return {
        "summary": summary,
        "contract": contract,
        "generation": verified,
        "frozen_generation_id": frozen.generation_id,
        "nb11_generation_id": frozen.identity["nb11_generation_id"],
    }
