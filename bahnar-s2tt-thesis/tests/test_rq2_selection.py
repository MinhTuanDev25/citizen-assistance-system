"""Notebook 13 same-budget selection tests. Synthetic pools only; no real U′ and no training."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import GTestAccessError, Nb11InputError
from src.rq2_quality import U_PRIME_COLUMNS
from src.rq2_selection import (
    GATE_NAMES,
    assert_manifest_round_trip,
    fill_duration_budget,
    order_d_quality,
    order_d_random,
    prepare_same_budget_selection,
    _arm_report,
    publish_same_budget_selection,
    random_selection_key,
    read_manifest_csv,
    run_offline_synthetic_self_check,
    select_same_budget,
    verify_published_selection,
    write_manifest_csv,
    realized_duration_audit,
)
from src.rq2_selection_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    MANIFEST_COLUMNS,
    STATUS_SELECTION_FROZEN,
    FrozenUPrime,
    SelectionIntegrityError,
    SelectionPolicyError,
    assert_nb13_output_dir,
    assert_nb13_sources_clean,
    build_selection_contract,
    budget_seconds_from_hours,
    forbidden_operations,
    load_u_prime_parquet,
    resolve_frozen_u_prime,
    selection_source_paths,
    validate_u_prime_rows,
)
from tests.rq2_nb12_fixtures import build_nb11_generation
from tests.rq2_nb13_fixtures import seal_nb12_generation, selection_identity, u_prime_row, write_u_prime_parquet

ROOT = Path(__file__).resolve().parents[1]
NB13_NOTEBOOK = ROOT / "notebooks" / "13_RQ2_SameBudget_Selection_Freeze.ipynb"
NB13_MODULES = ("src.rq2_selection_contract", "src.rq2_selection")


def _pool():
    return [
        u_prime_row("a", 5, 0.1, text="xin chào, bạn"),
        u_prime_row("b", 4, 0.9),
        u_prime_row("c", 3, 0.9),
        u_prime_row("d", 1, 0.4),
    ]


def _uids(rows):
    return [row["segment_uid"] for row in rows]


def _selection_dir(tmp_path):
    out = tmp_path / "artifacts" / "rq2" / "selection"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sealed(tmp_path):
    seal_nb12_generation(tmp_path, _pool())
    return resolve_frozen_u_prime(tmp_path, durable_root=tmp_path)


def _publish(tmp_path, resolved, hours, **overrides):
    return publish_same_budget_selection(
        overrides.pop("out", _selection_dir(tmp_path)),
        rows=overrides.pop("rows", resolved.rows),
        identity=overrides.pop("identity", resolved.identity),
        selection_budget_hours=hours,
        random_seed=overrides.pop("seed", 42),
        policy_frozen=overrides.pop("policy_frozen", True),
        project_root=overrides.pop("project_root", tmp_path),
        durable_root=overrides.pop("durable_root", tmp_path),
        pseudo_dir=overrides.pop("pseudo_dir", tmp_path / "artifacts" / "rq2" / "pseudo_labels"),
        u_clean_dir=overrides.pop("u_clean_dir", tmp_path / "artifacts" / "rq2" / "u_clean"),
    )


# --------------------------------------------------------------------------- #
# A. D-Random                                                                  #
# --------------------------------------------------------------------------- #
def test_random_selection_is_deterministic_and_ignores_quality():
    rows = _pool()
    first = order_d_random(rows, 42)
    second = order_d_random(list(reversed(rows)), 42)
    assert _uids(first) == _uids(second)
    changed = [dict(row, quality_score=1.0 - float(row["quality_score"])) for row in rows]
    assert _uids(order_d_random(changed, 42)) == _uids(first)
    selected = select_same_budget(rows, target_seconds=6, random_seed=42)
    again = select_same_budget(changed, target_seconds=6, random_seed=42)
    assert selected[ARM_RANDOM]["selected_uids"] == again[ARM_RANDOM]["selected_uids"]
    keys = [random_selection_key(42, uid) for uid in _uids(first)]
    assert keys == sorted(keys)
    assert "quality_score" not in random_selection_key(42, "a")


def test_random_seed_changes_order():
    rows = _pool()
    assert _uids(order_d_random(rows, 1)) != _uids(order_d_random(rows, 2))
    assert [random_selection_key(1, "a"), random_selection_key(1, "b")] != [
        random_selection_key(2, "a"), random_selection_key(2, "b"),
    ]


# --------------------------------------------------------------------------- #
# B. D-Quality                                                                 #
# --------------------------------------------------------------------------- #
def test_quality_order_is_score_desc_then_uid():
    ordered = order_d_quality(_pool())
    assert _uids(ordered) == ["b", "c", "d", "a"]
    assert [row["quality_score"] for row in ordered] == [0.9, 0.9, 0.4, 0.1]
    flipped = [dict(row) for row in _pool()]
    flipped[0]["quality_score"] = 5.0
    assert _uids(order_d_quality(flipped))[0] == "a"


# --------------------------------------------------------------------------- #
# C. Duration budget                                                           #
# --------------------------------------------------------------------------- #
def test_duration_budget_never_exceeds_target_and_keeps_later_fitting_item():
    rows = _pool()
    filled = fill_duration_budget(rows, 6, arm=ARM_QUALITY)
    assert filled["selected_uids"] == ["a", "d"]
    assert filled["selected_duration_seconds"] == 6
    assert filled["unused_budget_seconds"] == 0
    assert filled["n_selected"] == 2
    assert filled["selected_duration_seconds"] <= filled["target_budget_seconds"]


def test_duration_budget_edges():
    rows = _pool()
    larger_than_remainder = fill_duration_budget(
        [u_prime_row("big", 5, 0.1), u_prime_row("mid", 4, 0.2), u_prime_row("small", 1, 0.3)],
        6,
        arm=ARM_RANDOM,
        seed=1,
    )
    assert larger_than_remainder["selected_uids"] == ["big", "small"]
    assert larger_than_remainder["unused_budget_seconds"] == 0

    exact = fill_duration_budget([u_prime_row("x", 2, 0.1), u_prime_row("y", 2, 0.2)], 4, arm=ARM_QUALITY)
    assert exact["n_selected"] == 2 and exact["unused_budget_seconds"] == 0

    small = fill_duration_budget([u_prime_row("x", 1, 0.1), u_prime_row("y", 1, 0.2)], 10, arm=ARM_QUALITY)
    assert small["n_selected"] == 2 and small["unused_budget_seconds"] == 8

    empty = fill_duration_budget([], 10, arm=ARM_QUALITY)
    assert empty["n_selected"] == 0 and empty["unused_budget_seconds"] == 10 and empty["selected_duration_seconds"] == 0

    both = select_same_budget(rows, target_seconds=6, random_seed=7)
    assert both[ARM_RANDOM]["target_budget_seconds"] == both[ARM_QUALITY]["target_budget_seconds"] == 6
    assert both[ARM_RANDOM]["selected_duration_seconds"] <= 6
    assert both[ARM_QUALITY]["selected_duration_seconds"] <= 6


def test_unfixed_budget_is_refused():
    with pytest.raises(SelectionPolicyError):
        budget_seconds_from_hours(None)
    with pytest.raises(SelectionPolicyError):
        fill_duration_budget(_pool(), 0, arm=ARM_QUALITY)


# --------------------------------------------------------------------------- #
# D. Common-pool integrity                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate,message", [
    (lambda rows: rows.append(dict(rows[0])), "duplicate segment_uid"),
    (lambda rows: rows[0].update(duration_seconds=0), "duration_seconds"),
    (lambda rows: rows[0].update(duration_seconds=-1), "duration_seconds"),
    (lambda rows: rows[0].update(duration_seconds=float("nan")), "duration_seconds"),
    (lambda rows: rows[0].update(duration_seconds=float("inf")), "duration_seconds"),
    (lambda rows: rows[0].update(quality_score=float("nan")), "quality_score"),
    (lambda rows: rows[0].update(quality_score=float("inf")), "quality_score"),
    (lambda rows: rows[0].update(pseudo_vi_norm="  "), "pseudo_vi_norm"),
    (lambda rows: rows[0].update(pseudo_vi_norm=None), "pseudo_vi_norm"),
])
def test_malformed_u_prime_fails_closed(mutate, message):
    rows = _pool()
    mutate(rows)
    with pytest.raises(SelectionIntegrityError, match=message) as caught:
        validate_u_prime_rows(rows)
    assert caught.value.uid


def test_missing_column_fails_closed():
    rows = _pool()
    del rows[1]["pseudo_vi_norm"]
    with pytest.raises(SelectionIntegrityError, match="missing columns"):
        validate_u_prime_rows(rows)


def test_tampered_u_prime_hash_fails(tmp_path):
    sealed = seal_nb12_generation(tmp_path, _pool())
    resolved = resolve_frozen_u_prime(tmp_path, durable_root=tmp_path)
    assert resolved.identity["nb12_contract_sha256"] == sealed["contract"]["nb12_contract_sha256"]
    assert resolved.identity["u_prime_ordered_uid_sha256"]
    parquet = sealed["pseudo_dir"] / "generations" / "nb12-fixture-gen" / "u_prime_manifest.parquet"
    before = parquet.read_bytes()
    parquet.write_bytes(before[:-1] + bytes([before[-1] ^ 0x01]))
    with pytest.raises((SelectionIntegrityError, RuntimeError)):
        resolve_frozen_u_prime(tmp_path, durable_root=tmp_path)


def test_loaded_columns_match_nb12_schema(tmp_path):
    path = tmp_path / "u_prime_manifest.parquet"
    write_u_prime_parquet(path, _pool())
    loaded = load_u_prime_parquet(path)
    assert list(loaded[0]) == list(U_PRIME_COLUMNS)
    assert MANIFEST_COLUMNS[: len(U_PRIME_COLUMNS)] == list(U_PRIME_COLUMNS)


# --------------------------------------------------------------------------- #
# E. Contract sensitivity                                                      #
# --------------------------------------------------------------------------- #
def test_contract_hash_changes_with_every_scientific_parameter():
    rows = _pool()
    identity = selection_identity(rows)
    base = build_selection_contract(identity, selection_budget_hours=1, random_seed=42)
    variants = [
        build_selection_contract(
            {**identity, "nb12_contract_sha256": "ab" * 32}, selection_budget_hours=1, random_seed=42,
        ),
        build_selection_contract(
            {**identity, "u_prime_manifest_sha256": "cd" * 32}, selection_budget_hours=1, random_seed=42,
        ),
        build_selection_contract(identity, selection_budget_hours=2, random_seed=42),
        build_selection_contract(identity, selection_budget_hours=1, random_seed=7),
        build_selection_contract(identity, selection_budget_hours=1, random_seed=42, random_algorithm="other-random"),
        build_selection_contract(identity, selection_budget_hours=1, random_seed=42, quality_algorithm="other-quality"),
        build_selection_contract(identity, selection_budget_hours=1, random_seed=42, duration_budget_algorithm="other-budget"),
    ]
    hashes = {item["selection_contract_sha256"] for item in variants}
    assert base["selection_contract_sha256"] not in hashes
    assert len(hashes) == len(variants)
    blob = json.dumps(base)
    assert "/Users/" not in blob and "selection_budget_seconds" in base
    assert base["quality_score_used_by_random"] is False and base["audio_cut_allowed"] is False


# --------------------------------------------------------------------------- #
# F. Immutability                                                              #
# --------------------------------------------------------------------------- #
def test_selected_fields_match_source_and_source_file_is_not_rewritten(tmp_path):
    rows = _pool()
    parquet = tmp_path / "incoming" / "u_prime_manifest.parquet"
    write_u_prime_parquet(parquet, rows)
    before = parquet.read_bytes()
    loaded = load_u_prime_parquet(parquet)
    result = select_same_budget(loaded, target_seconds=6, random_seed=42)
    by_uid = {row["segment_uid"]: row for row in loaded}
    for arm in (ARM_RANDOM, ARM_QUALITY):
        for row in result[arm]["rows"]:
            source = by_uid[row["segment_uid"]]
            assert row["pseudo_vi_norm"] == source["pseudo_vi_norm"]
            assert row["duration_seconds"] == source["duration_seconds"]
            assert row["quality_score"] == source["quality_score"]
    assert parquet.read_bytes() == before


# --------------------------------------------------------------------------- #
# G. Publication                                                               #
# --------------------------------------------------------------------------- #
def _keep_current(out):
    (out / "CURRENT").write_text("keep-me\n", encoding="utf-8")


def _assert_current_untouched(out):
    assert (out / "CURRENT").read_text(encoding="utf-8") == "keep-me\n"
    assert not list(out.glob("generations/*/COMPLETE.json"))


def test_prepare_exact_integer_pool_has_zero_unused():
    seconds = budget_seconds_from_hours(2 / 3600)
    rows = validate_u_prime_rows([u_prime_row("a", seconds / 2, 0.2), u_prime_row("b", seconds / 2, 0.8)])
    frozen = FrozenUPrime("synthetic", tuple(rows), selection_identity(rows))
    prepared = prepare_same_budget_selection(frozen, selection_budget_hours=2 / 3600, random_seed=42)
    for arm in (ARM_RANDOM, ARM_QUALITY):
        report = prepared["result"][arm]
        assert report["n_selected"] == 2
        assert report["unused_budget_seconds"] == 0
        assert report["selected_duration_seconds"] == seconds


def test_target_above_pool_fails_and_leaves_current(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    _keep_current(out)
    with pytest.raises(SelectionPolicyError, match="outside"):
        _publish(tmp_path, resolved, 10 / 3600, out=out)
    _assert_current_untouched(out)


def test_target_equal_to_pool_publishes_the_full_pool(tmp_path):
    resolved = _sealed(tmp_path)
    total = sum(float(row["duration_seconds"]) for row in resolved.rows)
    published = _publish(tmp_path, resolved, total / 3600.0)
    verified = verify_published_selection(_selection_dir(tmp_path), project_root=tmp_path, durable_root=tmp_path)
    assert verified["frozen_generation_id"] == resolved.generation_id
    for arm in (ARM_RANDOM, ARM_QUALITY):
        report = published["audit"]["arms"][arm]
        assert report["n_selected"] == len(resolved.rows)
        assert report["selected_duration_seconds"] <= total
        assert report["unused_budget_seconds"] < min(float(row["duration_seconds"]) for row in resolved.rows)


def test_target_below_pool_publishes_and_self_verifies(tmp_path):
    resolved = _sealed(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005)
    verified = verify_published_selection(_selection_dir(tmp_path), project_root=tmp_path, durable_root=tmp_path)
    assert verified["summary"]["status"] == STATUS_SELECTION_FROZEN
    assert verified["contract"]["selection_contract_sha256"] == published["contract"]["selection_contract_sha256"]
    assert verified["frozen_generation_id"] == resolved.generation_id
    for arm in (ARM_RANDOM, ARM_QUALITY):
        report = published["audit"]["arms"][arm]
        assert report["n_selected"] >= 1
        assert report["selected_duration_seconds"] <= report["target_budget_seconds"]


def test_target_smaller_than_every_segment_fails_and_leaves_current(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    _keep_current(out)
    with pytest.raises(SelectionIntegrityError, match="at least one segment"):
        _publish(tmp_path, resolved, 0.05 / 3600, out=out)
    _assert_current_untouched(out)


def test_policy_failure_does_not_publish_success(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    _keep_current(out)
    with pytest.raises(SelectionPolicyError):
        _publish(tmp_path, resolved, 0.00005, out=out, policy_frozen=False)
    _assert_current_untouched(out)


def test_fake_rows_and_trust_boolean_cannot_publish(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    _keep_current(out)
    with pytest.raises(TypeError):
        publish_same_budget_selection(
            out,
            rows=resolved.rows,
            identity=resolved.identity,
            selection_budget_hours=0.00005,
            random_seed=42,
            policy_frozen=True,
            project_root=tmp_path,
            frozen_input_verified=True,
        )
    forged_rows = [dict(row) for row in resolved.rows]
    forged_rows[0] = dict(forged_rows[0], pseudo_vi_norm="not the frozen label")
    with pytest.raises(SelectionIntegrityError, match="pseudo_vi_norm"):
        _publish(tmp_path, resolved, 0.00005, out=out, rows=forged_rows)
    forged_identity = dict(resolved.identity)
    forged_identity["u_prime_manifest_sha256"] = "ab" * 32
    with pytest.raises(SelectionIntegrityError, match="u_prime_manifest_sha256"):
        _publish(tmp_path, resolved, 0.00005, out=out, identity=forged_identity)
    _assert_current_untouched(out)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    assert published["summary"]["status"] == STATUS_SELECTION_FROZEN
    assert published["summary"]["nb12_contract_sha256"] == resolved.identity["nb12_contract_sha256"]


def test_noncanonical_output_cannot_publish(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    _keep_current(out)
    with pytest.raises(SelectionPolicyError, match="output dir"):
        _publish(tmp_path, resolved, 0.00005, out=tmp_path / "elsewhere")
    _assert_current_untouched(out)
    assert not (tmp_path / "elsewhere" / "CURRENT").exists()


def test_manifest_round_trip_rejects_changed_audio_path_and_contract(tmp_path):
    resolved = _sealed(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005)
    seconds = float(published["contract"]["selection_budget_seconds"])
    result = select_same_budget(resolved.rows, target_seconds=seconds, random_seed=42)
    generation_id = published["generation"]["generation_id"]
    manifest = _selection_dir(tmp_path) / "generations" / generation_id / "d_random_manifest.csv"
    loaded = read_manifest_csv(manifest)
    assert_manifest_round_trip(loaded, result[ARM_RANDOM])
    source = result[ARM_RANDOM]["rows"][0]
    assert loaded[0]["segment_local_path"] == source["segment_local_path"]
    assert loaded[0]["segment_pcm16_sha256"] == source["segment_pcm16_sha256"]
    assert loaded[0]["segment_wav_sha256"] == source["segment_wav_sha256"]
    assert loaded[0]["teacher_d0_agreement_chrf"] is None
    assert loaded[0]["d0_agreement_norm"] is None
    assert loaded[0]["asr_text_norm"] == source["asr_text_norm"]
    assert loaded[0]["nb11_input_contract_sha256"] == source["nb11_input_contract_sha256"]
    assert float(loaded[0]["duration_seconds"]) == float(source["duration_seconds"])
    for column, value in (
        ("segment_pcm16_sha256", "ff" * 32),
        ("segment_local_path", "artifacts/rq2/u_clean/segments/src-a/other.wav"),
        ("nb11_input_contract_sha256", "ee" * 32),
    ):
        drifted = [dict(row) for row in loaded]
        drifted[0][column] = value
        with pytest.raises(SelectionIntegrityError, match=column):
            assert_manifest_round_trip(drifted, result[ARM_RANDOM])


def test_finalize_failure_leaves_current_untouched(tmp_path, monkeypatch):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    _keep_current(out)

    def boom(*_args, **_kwargs):
        raise RuntimeError("seal failed")

    monkeypatch.setattr("src.rq2_selection.finalize_generation", boom)
    with pytest.raises(RuntimeError, match="seal failed"):
        _publish(tmp_path, resolved, 0.00005, out=out)
    _assert_current_untouched(out)


def _reseal_generation(gen_dir):
    summary_path = gen_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["d_random_manifest_sha256"] = sha256_file(gen_dir / "d_random_manifest.csv")
    summary["d_quality_manifest_sha256"] = sha256_file(gen_dir / "d_quality_manifest.csv")
    summary["selection_audit_sha256"] = sha256_file(gen_dir / "selection_audit.json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = gen_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, meta in manifest["files"].items():
        path = gen_dir / rel
        meta["sha256"] = sha256_file(path)
        meta["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    complete_path = gen_dir / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["artifact_manifest_sha256"] = sha256_file(manifest_path)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")


def test_historical_verification_keeps_pinned_nb11_after_current_moves(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    pinned_nb11 = resolved.identity["nb11_generation_id"]
    pinned_nb12 = resolved.generation_id
    later = build_nb11_generation(
        tmp_path,
        gen_id="20261002T000000000000Z-dddd5678",
        n_segments=4,
        write_current=True,
    )
    current = (tmp_path / "artifacts" / "rq2" / "u_clean" / "CURRENT").read_text(encoding="utf-8").strip()
    assert current == later["gen_id"]
    assert current != pinned_nb11
    verified = verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)
    assert verified["frozen_generation_id"] == pinned_nb12
    assert verified["nb11_generation_id"] == pinned_nb11
    assert verified["generation"]["generation_id"] == published["generation"]["generation_id"]
    manifest = tmp_path / "artifacts" / "rq2" / "u_clean" / "generations" / pinned_nb11 / "u_clean_manifest.jsonl"
    manifest.unlink()
    assert (tmp_path / "artifacts" / "rq2" / "u_clean" / "CURRENT").read_text(encoding="utf-8").strip() == later["gen_id"]
    with pytest.raises(Nb11InputError, match="missing or empty"):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("mutate,message", [
    ("considered_uid_sha256", "considered_uid_sha256"),
    ("arm_target_budget_seconds", "target_budget_seconds"),
    ("n_u_prime", "n_u_prime"),
    ("u_prime_ordered_uid_sha256", "u_prime_ordered_uid_sha256"),
    ("summary_random_seed", "random_seed"),
    ("summary_budget_seconds", "selection_budget_seconds"),
    ("contract_random_algorithm", "random_algorithm"),
    ("contract_schema_version", "schema_version"),
    ("contract_code_version", "code_version"),
])
def test_resealed_scientific_field_fails_verification(tmp_path, mutate, message):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    gen_dir = out / "generations" / published["generation"]["generation_id"]
    audit_path = gen_dir / "selection_audit.json"
    summary_path = gen_dir / "summary.json"
    contract_path = gen_dir / "contract.json"
    if mutate == "considered_uid_sha256":
        audit = _load(audit_path)
        audit["arms"]["d_random"]["considered_uid_sha256"] = "ab" * 32
        _dump(audit_path, audit)
    elif mutate == "arm_target_budget_seconds":
        audit = _load(audit_path)
        audit["arms"]["d_random"]["target_budget_seconds"] = 123.0
        _dump(audit_path, audit)
    elif mutate == "n_u_prime":
        audit = _load(audit_path)
        audit["n_u_prime"] = 0
        _dump(audit_path, audit)
    elif mutate == "u_prime_ordered_uid_sha256":
        audit = _load(audit_path)
        audit["u_prime_ordered_uid_sha256"] = "cd" * 32
        _dump(audit_path, audit)
    elif mutate == "summary_random_seed":
        summary = _load(summary_path)
        summary["random_seed"] = 99
        _dump(summary_path, summary)
    elif mutate == "summary_budget_seconds":
        summary = _load(summary_path)
        summary["selection_budget_seconds"] = 1.0
        _dump(summary_path, summary)
    else:
        contract = _load(contract_path)
        contract.pop("selection_contract_sha256")
        if mutate == "contract_random_algorithm":
            contract["random_algorithm"] = "rq2_random_other"
        elif mutate == "contract_schema_version":
            contract["schema_version"] = "rq2-selection-other"
        else:
            contract["code_version"] = "rq2_same_budget_selection_other"
        contract["selection_contract_sha256"] = sha256_json(contract)
        _dump(contract_path, contract)
    _reseal_generation(gen_dir)
    with pytest.raises(SelectionIntegrityError, match=message):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


@pytest.mark.parametrize("mutate,message", [
    ("intersection_uids", "intersection_uids"),
    ("intersection_seconds", "intersection_seconds"),
    ("summary_generation_id", "generation_id"),
    ("nb12_schema_version", "nb12_schema_version"),
    ("u_prime_columns", "u_prime_columns"),
    ("pseudo_label_column", "pseudo_label_column"),
])
def test_resealed_audit_metadata_fails_verification(tmp_path, mutate, message):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    gen_dir = out / "generations" / published["generation"]["generation_id"]
    if mutate == "intersection_uids":
        audit = _load(gen_dir / "selection_audit.json")
        audit["overlap"]["intersection_uids"] = ["not-a-selected-uid"]
        _dump(gen_dir / "selection_audit.json", audit)
    elif mutate == "intersection_seconds":
        audit = _load(gen_dir / "selection_audit.json")
        audit["overlap"]["intersection_seconds"] = 123.0
        _dump(gen_dir / "selection_audit.json", audit)
    elif mutate == "summary_generation_id":
        summary = _load(gen_dir / "summary.json")
        summary["generation_id"] = "other-nb13-generation"
        _dump(gen_dir / "summary.json", summary)
    else:
        contract = _load(gen_dir / "contract.json")
        contract.pop("selection_contract_sha256")
        if mutate == "nb12_schema_version":
            contract["nb12_schema_version"] = "rq2-pseudo-other"
        elif mutate == "u_prime_columns":
            contract["u_prime_columns"] = ["segment_uid"]
        else:
            contract["pseudo_label_column"] = "other_text"
        contract["selection_contract_sha256"] = sha256_json(contract)
        _dump(gen_dir / "contract.json", contract)
    _reseal_generation(gen_dir)
    with pytest.raises(SelectionIntegrityError, match=message):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


def test_resealed_oversize_budget_fails_capacity_gate(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)
    gen_dir = out / "generations" / published["generation"]["generation_id"]
    hours = 10 / 3600
    seconds = budget_seconds_from_hours(hours)
    contract = _load(gen_dir / "contract.json")
    contract.pop("selection_contract_sha256")
    contract["selection_budget_hours"] = hours
    contract["selection_budget_seconds"] = seconds
    result = select_same_budget(resolved.rows, target_seconds=seconds, random_seed=42)
    contract.update(realized_duration_audit(result, float(contract["realized_duration_tolerance_seconds"])))
    contract["selection_contract_sha256"] = sha256_json(contract)
    _dump(gen_dir / "contract.json", contract)
    write_manifest_csv(gen_dir / "d_random_manifest.csv", result[ARM_RANDOM]["rows"])
    write_manifest_csv(gen_dir / "d_quality_manifest.csv", result[ARM_QUALITY]["rows"])
    audit = _load(gen_dir / "selection_audit.json")
    audit["target_budget_seconds"] = seconds
    audit["arms"] = {ARM_RANDOM: _arm_report(result[ARM_RANDOM]), ARM_QUALITY: _arm_report(result[ARM_QUALITY])}
    audit["overlap"] = result["overlap"]
    _dump(gen_dir / "selection_audit.json", audit)
    summary = _load(gen_dir / "summary.json")
    summary["selection_contract_sha256"] = contract["selection_contract_sha256"]
    summary["selection_budget_hours"] = hours
    summary["selection_budget_seconds"] = seconds
    for key in (
        "target_budget_seconds",
        "random_selected_duration_seconds",
        "quality_selected_duration_seconds",
        "random_underfill_seconds",
        "quality_underfill_seconds",
        "realized_duration_gap_seconds",
        "realized_duration_tolerance_seconds",
        "same_realized_budget_within_tolerance",
    ):
        summary[key] = contract[key]
        audit[key] = contract[key]
    _dump(gen_dir / "selection_audit.json", audit)
    summary["d_random"] = audit["arms"][ARM_RANDOM]
    summary["d_quality"] = audit["arms"][ARM_QUALITY]
    summary["overlap"] = {
        "n_intersection": result["overlap"]["n_intersection"],
        "intersection_hours": result["overlap"]["intersection_hours"],
        "jaccard_uid": result["overlap"]["jaccard_uid"],
    }
    summary["gates"]["target_budget_within_u_prime_capacity"] = True
    _dump(gen_dir / "summary.json", summary)
    _reseal_generation(gen_dir)
    with pytest.raises(SelectionIntegrityError, match="target_budget_within_u_prime_capacity"):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


def test_resealed_budget_unit_disagreement_fails(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    gen_dir = out / "generations" / published["generation"]["generation_id"]
    contract = _load(gen_dir / "contract.json")
    contract.pop("selection_contract_sha256")
    contract["selection_budget_seconds"] = float(contract["selection_budget_seconds"]) + 1.0
    contract["selection_contract_sha256"] = sha256_json(contract)
    _dump(gen_dir / "contract.json", contract)
    summary = _load(gen_dir / "summary.json")
    summary["selection_contract_sha256"] = contract["selection_contract_sha256"]
    summary["selection_budget_seconds"] = contract["selection_budget_seconds"]
    _dump(gen_dir / "summary.json", summary)
    _reseal_generation(gen_dir)
    with pytest.raises(SelectionIntegrityError, match="selection_budget_seconds does not match selection_budget_hours"):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


def test_verification_uses_pinned_nb12_and_rejects_resealed_drift(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    pseudo_current = tmp_path / "artifacts" / "rq2" / "pseudo_labels" / "CURRENT"
    pseudo_current.write_text("newer-nb12\n", encoding="utf-8")
    verified = verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)
    assert verified["frozen_generation_id"] == resolved.generation_id
    assert verified["frozen_generation_id"] != "newer-nb12"
    generation_id = published["generation"]["generation_id"]
    manifest = out / "generations" / generation_id / "d_random_manifest.csv"
    text = manifest.read_text(encoding="utf-8")
    needle = next(row["segment_pcm16_sha256"] for row in resolved.rows if row["segment_pcm16_sha256"] in text)
    manifest.write_text(text.replace(needle, "ff" * 32, 1), encoding="utf-8")
    _reseal_generation(manifest.parent)
    with pytest.raises(SelectionIntegrityError, match="segment_pcm16_sha256"):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


def test_publication_self_verifies_and_tamper_fails(tmp_path):
    resolved = _sealed(tmp_path)
    out = _selection_dir(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005, out=out)
    verified = verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)
    assert verified["summary"]["status"] == STATUS_SELECTION_FROZEN
    assert verified["contract"]["selection_contract_sha256"] == published["contract"]["selection_contract_sha256"]
    assert set(GATE_NAMES) <= set(verified["summary"]["gates"])
    assert all(verified["summary"]["gates"].values())
    manifest = out / "generations" / published["generation"]["generation_id"] / "d_random_manifest.csv"
    blob = bytearray(manifest.read_bytes())
    blob[-2] ^= 0x01
    manifest.write_bytes(blob)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_published_selection(out, project_root=tmp_path, durable_root=tmp_path)


# --------------------------------------------------------------------------- #
# H. Anti-leakage                                                              #
# --------------------------------------------------------------------------- #
def test_forbidden_operation_scanner_catches_leakage_and_sources_are_clean():
    assert forbidden_operations(ast.parse("import sacrebleu\n"))
    assert forbidden_operations(ast.parse("from jiwer import wer\n"))
    assert forbidden_operations(ast.parse("Trainer()\n"))
    assert forbidden_operations(ast.parse("model.generate()\n"))
    assert forbidden_operations(ast.parse("run_peak_safe_teacher_inference()\n"))
    assert forbidden_operations(ast.parse("x = 'rq1_' + 'test.csv'\n")) == []
    assert forbidden_operations(ast.parse("x = 'rq1_test.csv'\n"))
    for path in (
        "data/manifests/rq1_test.csv",
        "/tmp/x/rq1_test.parquet",
        "artifacts/rq2/g_test/references.csv",
        "foo/frozen_test/bar.csv",
        "data\\manifests\\RQ1_TEST.csv",
    ):
        assert forbidden_operations(ast.parse(f"x = {path!r}\n")), path
    assert forbidden_operations(ast.parse("gate = 'no_g_test_access'\n")) == []
    assert_nb13_sources_clean([*selection_source_paths(), NB13_NOTEBOOK])


def test_only_publish_same_budget_selection_can_seal():
    import src.rq2_selection as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_commit_selection" not in source
    tree = ast.parse(source)
    trust_keywords = [
        node.arg for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg in {"frozen_input_verified", "nb12_verified", "trusted"}
    ]
    assert trust_keywords == []
    writers = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = [arg.arg for arg in node.args.args] + [arg.arg for arg in node.args.kwonlyargs]
        assert not (set(names) & {"frozen_input_verified", "nb12_verified", "trusted"})
        for child in ast.walk(node):
            func = getattr(child, "func", None)
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if called in {"stage_generation", "finalize_generation"}:
                writers.append(node.name)
    assert writers == ["publish_same_budget_selection", "publish_same_budget_selection"]


# --------------------------------------------------------------------------- #
# Durable path, generation pin, realized budget                               #
# --------------------------------------------------------------------------- #
def test_durable_selection_root_does_not_use_code_checkout(tmp_path):
    project = tmp_path / "code"
    durable = tmp_path / "durable"
    project.mkdir()
    decoy = project / "artifacts" / "rq2" / "pseudo_labels"
    decoy.mkdir(parents=True)
    (decoy / "CURRENT").write_text("code-root-current\n", encoding="utf-8")
    seal_nb12_generation(durable, _pool())
    pseudo = durable / "artifacts" / "rq2" / "pseudo_labels"
    u_clean = durable / "artifacts" / "rq2" / "u_clean"
    resolved = resolve_frozen_u_prime(
        project,
        durable_root=durable,
        pseudo_dir=pseudo,
        u_clean_dir=u_clean,
        generation_id="nb12-fixture-gen",
        expected_generation_id="nb12-fixture-gen",
    )
    assert resolved.generation_id == "nb12-fixture-gen"
    assert resolved.identity["expected_nb12_generation_id"] == "nb12-fixture-gen"
    out = assert_nb13_output_dir(durable / "artifacts" / "rq2" / "selection", project, durable_root=durable)
    assert out == (durable / "artifacts" / "rq2" / "selection").resolve()
    with pytest.raises(SelectionPolicyError, match="NB13 output dir must be"):
        assert_nb13_output_dir(project / "artifacts" / "rq2" / "selection", project, durable_root=durable)


def test_pinned_generation_ignores_a_different_current(tmp_path):
    seal_nb12_generation(tmp_path, _pool())
    pseudo = tmp_path / "artifacts" / "rq2" / "pseudo_labels"
    (pseudo / "CURRENT").write_text("other-current-gen\n", encoding="utf-8")
    resolved = resolve_frozen_u_prime(
        tmp_path,
        durable_root=tmp_path,
        generation_id="nb12-fixture-gen",
        expected_generation_id="nb12-fixture-gen",
    )
    assert resolved.generation_id == "nb12-fixture-gen"
    assert resolved.identity["nb12_current_generation_id"] == "other-current-gen"
    assert resolved.identity["resolved_nb12_generation_id"] == "nb12-fixture-gen"


def test_missing_or_mismatched_pin_fails(tmp_path):
    seal_nb12_generation(tmp_path, _pool())
    with pytest.raises((SelectionIntegrityError, RuntimeError)):
        resolve_frozen_u_prime(
            tmp_path,
            durable_root=tmp_path,
            generation_id="missing-pinned-gen",
            expected_generation_id="missing-pinned-gen",
        )
    with pytest.raises(RuntimeError, match="pinned NB12 generation"):
        resolve_frozen_u_prime(
            tmp_path,
            durable_root=tmp_path,
            generation_id="nb12-fixture-gen",
            expected_generation_id="other-pinned-gen",
        )


def test_realized_gap_within_frozen_tolerance_is_published(tmp_path):
    resolved = _sealed(tmp_path)
    published = _publish(tmp_path, resolved, 0.00005)
    contract = published["contract"]
    gap = abs(contract["random_selected_duration_seconds"] - contract["quality_selected_duration_seconds"])
    assert contract["target_budget_seconds"] == contract["selection_budget_seconds"]
    assert contract["random_selected_duration_seconds"] <= contract["target_budget_seconds"]
    assert contract["quality_selected_duration_seconds"] <= contract["target_budget_seconds"]
    assert contract["realized_duration_gap_seconds"] == gap
    assert contract["same_realized_budget_within_tolerance"] is True
    assert contract["realized_duration_gap_seconds"] <= contract["realized_duration_tolerance_seconds"]
    assert "indivisible-segment tolerance" in contract["same_budget_definition"]
    verified = verify_published_selection(_selection_dir(tmp_path), project_root=tmp_path, durable_root=tmp_path)
    assert verified["contract"]["resolved_nb12_generation_id"] == resolved.generation_id


def test_realized_gap_above_tolerance_leaves_current(tmp_path, monkeypatch):
    rows = [u_prime_row("a", 1, 0.9), u_prime_row("b", 1, 0.1), u_prime_row("c", 1, 0.5)]
    seal_nb12_generation(tmp_path, rows, n_samples=[8 * 16000, 7 * 16000, 5 * 16000, 1600])
    resolved = resolve_frozen_u_prime(tmp_path, durable_root=tmp_path)
    hours = 11.0 / 3600.0
    target = budget_seconds_from_hours(hours)
    gap_seed = None
    gap = 0.0
    for seed in range(1, 30):
        result = select_same_budget(resolved.rows, target_seconds=target, random_seed=seed)
        audit = realized_duration_audit(result, 40.0)
        if (
            audit["realized_duration_gap_seconds"] > 0.0
            and result[ARM_RANDOM]["selected_duration_seconds"] <= target
            and result[ARM_QUALITY]["selected_duration_seconds"] <= target
        ):
            gap_seed = seed
            gap = audit["realized_duration_gap_seconds"]
            break
    assert gap_seed is not None and gap > 0.0
    monkeypatch.setattr(
        "src.rq2_selection_contract.proven_segment_duration_tolerance",
        lambda segmentation_contract_sha256: gap / 2.0,
    )
    out = _selection_dir(tmp_path)
    _keep_current(out)
    with pytest.raises(SelectionIntegrityError, match="same_realized_budget_within_tolerance"):
        _publish(tmp_path, resolved, hours, seed=gap_seed, out=out)
    _assert_current_untouched(out)


def test_nb13_sources_do_not_train_or_open_g_test():
    assert_nb13_sources_clean([*selection_source_paths(), NB13_NOTEBOOK])
    for path in selection_source_paths():
        text = Path(path).read_text(encoding="utf-8")
        assert "Trainer.fit" not in text
        assert ".fit(" not in text
        assert "rq1_test" not in text


# --------------------------------------------------------------------------- #
# I. Notebook wiring                                                           #
# --------------------------------------------------------------------------- #
def _notebook_code_cells():
    notebook = json.loads(NB13_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell.get("source") or []) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def test_notebook_flags_budget_and_cells_compile():
    cells = _notebook_code_cells()
    joined = "\n".join(cells)
    assert "RUN_REAL_SELECTION = False" in joined
    assert "SELECTION_POLICY_FROZEN = False" in joined
    assert "SELECTION_BUDGET_HOURS = None" in joined
    assert "SELECTION_RANDOM_SEED = 42" in joined
    assert 'EXPECTED_NB12_GENERATION_ID = "20261005T095322062456Z-1863e3d9"' in joined
    assert "DURABLE_RQ2_ROOT" in joined
    assert "resolve_rq1_runtime_paths" in joined
    for index, source in enumerate(cells):
        compile(source, f"nb13_cell_{index}", "exec")


def test_notebook_calls_match_nb13_signatures():
    import importlib

    imported = {}
    problems = []
    for index, source in enumerate(_notebook_code_cells()):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in NB13_MODULES:
                module = importlib.import_module(node.module)
                for alias in node.names:
                    obj = getattr(module, alias.name, None)
                    if callable(obj):
                        imported[alias.asname or alias.name] = obj
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = imported.get(node.func.id)
            if target is None:
                continue
            signature = inspect.signature(target)
            if any(kw.arg is None for kw in node.keywords) or any(isinstance(arg, ast.Starred) for arg in node.args):
                continue
            supplied = {kw.arg for kw in node.keywords}
            positional = [
                parameter for parameter in signature.parameters.values()
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            for offset, parameter in enumerate(positional):
                if offset < len(node.args):
                    supplied.add(parameter.name)
            missing = [
                parameter.name for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and parameter.name not in supplied
                and parameter.name != "self"
            ]
            unknown = [
                kw.arg for kw in node.keywords
                if kw.arg is not None and kw.arg not in signature.parameters
                and not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
            ]
            if missing or unknown:
                problems.append(f"cell {index}: {node.func.id} missing {missing} unknown {unknown}")
    assert not problems, "; ".join(problems)
    assert "build_selection_contract" in imported


def test_output_dir_rejects_protected_and_g_test_paths(tmp_path):
    out = tmp_path / "artifacts" / "rq2" / "selection"
    assert assert_nb13_output_dir(out, tmp_path, durable_root=tmp_path) == out.resolve()
    with pytest.raises(SelectionPolicyError):
        assert_nb13_output_dir(tmp_path / "artifacts" / "rq2" / "pseudo_labels", tmp_path, durable_root=tmp_path)
    with pytest.raises(GTestAccessError):
        publish_same_budget_selection(
            tmp_path / "g_test" / "out",
            rows=_pool(),
            identity=selection_identity(_pool()),
            selection_budget_hours=1,
            random_seed=1,
            policy_frozen=True,
            project_root=tmp_path,
        )


def test_offline_self_check_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_offline_synthetic_self_check()
    assert report["wrote_current"] is False
    assert report["d_random_seconds"] <= report["target_budget_seconds"]
    assert report["d_quality_seconds"] <= report["target_budget_seconds"]
    assert not (tmp_path / "CURRENT").exists()
    assert report["d_random_uids"] == run_offline_synthetic_self_check()["d_random_uids"]
