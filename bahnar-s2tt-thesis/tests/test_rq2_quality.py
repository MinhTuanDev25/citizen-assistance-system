"""Notebook 12 eligibility / quality calibration / U' publication tests (synthetic only)."""
from __future__ import annotations

import ast
import inspect
import json
import os
import pickle
import stat
from pathlib import Path

import pandas as pd
import pytest

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import (
    STATUS_FAIL,
    STATUS_READY_FOR_REVIEW,
    STATUS_SUCCESS,
    DataAccessLedger,
    read_current_generation_id,
    resolve_nb11_input,
    verify_generation,
)
from src.rq2_pseudo_label import (
    FORBIDDEN_REFERENCE_FIELDS,
    STATUS_FAILED,
    STATUS_OK,
    InferenceCheckpoint,
    inference_binding,
    items_from_nb11,
    iter_raw_records,
    run_peak_safe_teacher_inference,
)
from src.rq2_quality import (
    ELIGIBLE,
    EXCLUDED_AGREEMENT_UNAVAILABLE,
    EXCLUDED_ASR_EMPTY,
    EXCLUDED_DECODE_FAILURE,
    EXCLUDED_INVALID_ENCODING,
    EXCLUDED_MT_EMPTY,
    EXCLUDED_NONFINITE_CONFIDENCE,
    EXCLUDED_OUTPUT_SANITY,
    GATE_NAMES,
    UPSTREAM_GATE_NAMES,
    NB12_ARTIFACT_FILES,
    U_PRIME_COLUMNS,
    CalibrationError,
    OutputSanityConfig,
    QualityCalibrationConfig,
    QualityContractError,
    ValidationReferences,
    assert_publication_bindings,
    build_quality_score_contract,
    calibration_artifact_sha256,
    calibrate_quality_score,
    derive_status,
    freeze_quality_score_contract,
    hard_eligibility,
    iter_scored,
    load_g_validation,
    nb12_source_has_no_selection,
    publish_u_prime_generation,
    spearman_rho,
    review_identity,
    verify_quality_score_contract,
    weight_grid,
    write_review_bundle,
)
from tests.rq2_nb12_fixtures import build_nb11_generation, fake_contracts, raw_record, synthetic_validation, tree_digest
from tests.test_rq2_pseudo_label import NB12_SOURCES, fake_asr, fake_d0, fake_mt

OFF = OutputSanityConfig(enabled=False)
IDENTITY = {
    "manifest": "rq1_validation.csv", "manifest_sha256": "f" * 64,
    "audio_identity_sha256": "a" * 64, "n_records": 300,
}


def _nb11(generation_id="gen"):
    body = {"contract_version": "rq2_nb11_input_contract_v1", "generation_id": generation_id}
    body["nb11_input_contract_sha256"] = sha256_json(body)
    return body


NB11_FAKE = _nb11()


def _contract(d0_enabled=False, frozen=False, records=None, refs=None, config=None, nb11_sha=None):
    teacher, d0, *_ = fake_contracts(d0_enabled=d0_enabled)
    if records is None:
        records, refs = synthetic_validation(d0_enabled=d0_enabled)
    cal = calibrate_quality_score(records, refs, d0_enabled=d0_enabled, config=config or QualityCalibrationConfig(),
                                  calibration_identity=IDENTITY)
    contract = build_quality_score_contract(
        cal, teacher_contract=teacher, d0_agreement_contract=d0,
        nb11_input_contract_sha256=nb11_sha or NB11_FAKE["nb11_input_contract_sha256"],
    )
    if frozen:
        contract = freeze_quality_score_contract(contract, reviewed_proposal_sha256=contract["quality_score_proposal_sha256"])
    return contract, cal, teacher, d0


def _all_gates(**overrides):
    gates = {g: True for g in GATE_NAMES}
    gates.update(overrides)
    return gates


# --------------------------------------------------------------------------- #
# G. Hard eligibility                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("overrides,reason", [
    ({}, ELIGIBLE),
    ({"asr_status": STATUS_FAILED}, EXCLUDED_DECODE_FAILURE),
    ({"asr_valid_encoding": False}, EXCLUDED_INVALID_ENCODING),
    ({"asr_empty": True, "asr_text_norm": ""}, EXCLUDED_ASR_EMPTY),
    ({"mt_status": STATUS_FAILED}, EXCLUDED_DECODE_FAILURE),
    ({"mt_valid_encoding": False}, EXCLUDED_INVALID_ENCODING),
    ({"mt_empty": True, "pseudo_vi_norm": ""}, EXCLUDED_MT_EMPTY),
    ({"asr_mean_logprob": None}, EXCLUDED_NONFINITE_CONFIDENCE),
    ({"mt_mean_logprob": float("nan")}, EXCLUDED_NONFINITE_CONFIDENCE),
])
def test_hard_eligibility_reason_codes(overrides, reason):
    assert hard_eligibility(raw_record("u", **overrides), d0_enabled=False, sanity=OFF) == reason


def test_hard_eligibility_reason_order_is_fixed():
    rec = raw_record("u", asr_status=STATUS_FAILED, mt_empty=True, mt_mean_logprob=None)
    assert hard_eligibility(rec, d0_enabled=False, sanity=OFF) == EXCLUDED_DECODE_FAILURE
    rec = raw_record("u", asr_empty=True, asr_text_norm="", mt_status="SKIPPED_ASR_EMPTY")
    assert hard_eligibility(rec, d0_enabled=False, sanity=OFF) == EXCLUDED_ASR_EMPTY


def test_d0_agreement_eligibility_only_when_enabled():
    rec = raw_record("u")
    assert hard_eligibility(rec, d0_enabled=False, sanity=OFF) == ELIGIBLE
    assert hard_eligibility(rec, d0_enabled=True, sanity=OFF) == EXCLUDED_AGREEMENT_UNAVAILABLE
    ok = raw_record("u", d0_status=STATUS_OK, d0_valid_encoding=True, teacher_d0_agreement_chrf=0.0)
    assert hard_eligibility(ok, d0_enabled=True, sanity=OFF) == ELIGIBLE


def test_output_sanity_rule_only_when_enabled():
    on = OutputSanityConfig(enabled=True, min_target_source_char_ratio=0.5, max_target_source_char_ratio=2.0)
    wild = raw_record("u", target_source_char_ratio=9.0)
    assert hard_eligibility(wild, d0_enabled=False, sanity=OFF) == ELIGIBLE
    assert hard_eligibility(wild, d0_enabled=False, sanity=on) == EXCLUDED_OUTPUT_SANITY


def test_eligibility_does_not_depend_on_quality_weights():
    records = [raw_record(f"u{i}", mt_mean_logprob=-0.1 * i, asr_mean_logprob=-0.05 * i) for i in range(20)]
    records[3]["mt_empty"] = True
    records[3]["pseudo_vi_norm"] = ""
    a, *_ = _contract(frozen=True)
    b = json.loads(json.dumps(a))
    b["weights"] = {k: (1.0 if k == "asr_confidence" else 0.0) for k in b["weights"]}
    ea = [r["hard_eligible"] for r in iter_scored(records, a)]
    eb = [r["hard_eligible"] for r in iter_scored(records, b)]
    assert ea == eb and ea.count(False) == 1


def test_every_eligible_row_gets_exactly_one_score():
    contract, *_ = _contract(frozen=True)
    records = [raw_record(f"u{i}", mt_mean_logprob=-0.1 * i) for i in range(10)] + [raw_record("bad", asr_status=STATUS_FAILED)]
    scored = list(iter_scored(records, contract))
    assert all(r["quality_score"] is not None for r in scored if r["hard_eligible"])
    assert scored[-1]["quality_score"] is None and scored[-1]["exclusion_reason"] == EXCLUDED_DECODE_FAILURE


# --------------------------------------------------------------------------- #
# F. Calibration (G_validation only)                                           #
# --------------------------------------------------------------------------- #
def test_weight_grid_is_exhaustive_and_ordered():
    grid = weight_grid(3, 0.1)
    assert len(grid) == 66 and all(abs(sum(w) - 1.0) < 1e-9 for w in grid)
    assert grid == weight_grid(3, 0.1)
    with pytest.raises(CalibrationError):
        weight_grid(2, 0.3)


def test_spearman_constant_is_nan():
    assert spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([1, 1, 1], [1, 2, 3]) != spearman_rho([1, 1, 1], [1, 2, 3])


def test_calibration_prefers_the_informative_feature_and_is_deterministic():
    records, refs = synthetic_validation()
    cfg = QualityCalibrationConfig()
    a = calibrate_quality_score(records, refs, d0_enabled=False, config=cfg, calibration_identity=IDENTITY)
    b = calibrate_quality_score(records, refs, d0_enabled=False, config=cfg, calibration_identity=IDENTITY)
    assert a == b
    assert a["selected"]["weights"]["mt_confidence"] >= 0.5 and a["selected"]["spearman_rho"] > 0.5
    assert len(a["candidates"]) == 11 and a["calibration_split"] == "g_validation"
    assert a["single_feature_spearman"]["mt_confidence"] > a["single_feature_spearman"]["asr_confidence"]


def test_calibration_with_d0_uses_three_features():
    records, refs = synthetic_validation(d0_enabled=True)
    cal = calibrate_quality_score(records, refs, d0_enabled=True, config=QualityCalibrationConfig(), calibration_identity=IDENTITY)
    assert [f["name"] for f in cal["features"]] == ["asr_confidence", "mt_confidence", "d0_agreement"]
    assert len(cal["candidates"]) == 66


def test_calibration_tie_break_prefers_uniform_weights():
    records, refs = synthetic_validation()
    for r in records:
        r["asr_mean_logprob"] = r["mt_mean_logprob"]
    cal = calibrate_quality_score(records, refs, d0_enabled=False, config=QualityCalibrationConfig(), calibration_identity=IDENTITY)
    assert cal["selected"]["weights"] == {"asr_confidence": 0.5, "mt_confidence": 0.5}


def test_calibration_refuses_bad_inputs():
    records, refs = synthetic_validation()
    with pytest.raises(CalibrationError, match="ValidationReferences"):
        calibrate_quality_score(records, {"val-0000": "x"}, d0_enabled=False, config=QualityCalibrationConfig(), calibration_identity=IDENTITY)
    with pytest.raises(CalibrationError, match="exactly once"):
        calibrate_quality_score(records[:-1], refs, d0_enabled=False, config=QualityCalibrationConfig(), calibration_identity=IDENTITY)
    with pytest.raises(CalibrationError, match="hard-eligible"):
        calibrate_quality_score(records, refs, d0_enabled=False, config=QualityCalibrationConfig(min_calibration_records=10_000),
                                calibration_identity=IDENTITY)
    flat = [dict(r, asr_mean_logprob=-0.5) for r in records]
    with pytest.raises(CalibrationError, match="degenerate"):
        calibrate_quality_score(flat, refs, d0_enabled=False, config=QualityCalibrationConfig(), calibration_identity=IDENTITY)
    with pytest.raises(CalibrationError, match="penalties"):
        QualityCalibrationConfig(penalties=(("mt_source_truncated", -1.0), ("mt_hit_max_length", 0.0))).payload()


def test_calibration_output_contains_no_reference_text():
    records, refs = synthetic_validation()
    cal = calibrate_quality_score(records, refs, d0_enabled=False, config=QualityCalibrationConfig(), calibration_identity=IDENTITY)
    blob = json.dumps(cal, ensure_ascii=False)
    assert "x0" not in blob and "bahnar bahnar" not in blob
    assert not any(f'"{name}"' in blob for name in FORBIDDEN_REFERENCE_FIELDS)


def test_validation_references_are_not_serialisable():
    _, refs = synthetic_validation(n=3)
    assert "val-" not in repr(refs)
    with pytest.raises(TypeError):
        pickle.dumps(refs)


def _manifests(tmp_path, *, pin_ok=True):
    d = tmp_path / "manifests"
    d.mkdir()
    frame = pd.DataFrame({
        "record_uid": ["v1", "v2"], "record_id": ["r1", "r2"], "parquet_file": ["p", "p"],
        "shard_row_index": ["0", "1"], "duration_seconds": ["2.0", "3.0"], "split": ["validation", "validation"],
        "text_vi": ["gold one", "gold two"], "text_bahnar": ["bah one", "bah two"], "text_en": ["", ""],
    })
    frame.to_csv(d / "rq1_validation.csv", index=False)
    pin = sha256_file(d / "rq1_validation.csv") if pin_ok else "0" * 64
    (d / "split_summary.json").write_text(json.dumps({"manifest_sha256": {"rq1_validation.csv": pin, "rq1_test.csv": "9" * 64}}))
    test_file = d / "rq1_test.csv"
    test_file.write_text("record_uid,text_vi\nt1,never read\n")
    os.chmod(test_file, 0)
    return d, test_file


def test_load_g_validation_reads_only_validation(tmp_path):
    d, test_file = _manifests(tmp_path)
    try:
        ledger = DataAccessLedger()
        ids, refs, identity = load_g_validation(d, ledger=ledger)
        assert list(ids.columns) == ["record_uid", "record_id", "parquet_file", "shard_row_index", "duration_seconds", "split"]
        assert len(refs) == 2 and identity["n_records"] == 2
        assert {e["file"] for e in ledger.entries} == {"split_summary.json", "rq1_validation.csv"}
        assert not ledger.g_test_accessed
    finally:
        os.chmod(test_file, stat.S_IRUSR | stat.S_IWUSR)


def test_load_g_validation_pin_mismatch_refuses(tmp_path):
    d, test_file = _manifests(tmp_path, pin_ok=False)
    try:
        with pytest.raises(CalibrationError, match="pin"):
            load_g_validation(d, ledger=DataAccessLedger())
    finally:
        os.chmod(test_file, stat.S_IRUSR | stat.S_IWUSR)


# --------------------------------------------------------------------------- #
# Quality-score contract + status                                              #
# --------------------------------------------------------------------------- #
def test_frozen_flag_changes_contract_hash_not_proposal_hash():
    draft, *_ = _contract()
    frozen = freeze_quality_score_contract(draft, reviewed_proposal_sha256=draft["quality_score_proposal_sha256"])
    assert draft["frozen"] is False and frozen["frozen"] is True
    assert draft["quality_score_proposal_sha256"] == frozen["quality_score_proposal_sha256"]
    assert draft["quality_score_contract_sha256"] != frozen["quality_score_contract_sha256"]
    verify_quality_score_contract(frozen)
    assert frozen["eligibility"]["uses_quality_score"] is False
    assert frozen["calibration"]["split"] == "g_validation"


def test_freeze_requires_the_reviewed_proposal_hash():
    draft, *_ = _contract()
    with pytest.raises(QualityContractError):
        freeze_quality_score_contract(draft, reviewed_proposal_sha256="")
    with pytest.raises(QualityContractError):
        freeze_quality_score_contract(draft, reviewed_proposal_sha256="0" * 64)


def test_tampered_contract_fails_verification():
    frozen, *_ = _contract(frozen=True)
    bad = json.loads(json.dumps(frozen))
    bad["weights"]["asr_confidence"] = 0.99
    with pytest.raises(QualityContractError):
        verify_quality_score_contract(bad)


def test_config_change_changes_proposal_hash():
    a, *_ = _contract()
    b, *_ = _contract(config=QualityCalibrationConfig(normalization_upper_quantile=0.9))
    assert a["quality_score_proposal_sha256"] != b["quality_score_proposal_sha256"]


def test_status_requires_every_gate_and_frozen_contract():
    assert derive_status(_all_gates()) == STATUS_SUCCESS
    assert derive_status(_all_gates(quality_score_contract_frozen=False)) == STATUS_READY_FOR_REVIEW
    assert derive_status(_all_gates(no_g_test_access=False)) == STATUS_FAIL
    assert derive_status({}) == STATUS_FAIL


# --------------------------------------------------------------------------- #
# I. Publication                                                               #
# --------------------------------------------------------------------------- #
def _publish(out, records, contract, cal, teacher, d0, nb11=None, **gate_overrides):
    return publish_u_prime_generation(
        out, raw_records=records, expected_uids=[r["segment_uid"] for r in records],
        quality_contract=contract, calibration=cal, teacher_contract=teacher, d0_agreement_contract=d0,
        nb11_input_contract=NB11_FAKE if nb11 is None else nb11,
        ledger=DataAccessLedger(), upstream_gates=_all_gates(**gate_overrides),
    )


def test_publish_refuses_unfrozen_contract(tmp_path):
    draft, cal, teacher, d0 = _contract()
    with pytest.raises(QualityContractError, match="unfrozen"):
        _publish(tmp_path / "out", [raw_record("u0")], draft, cal, teacher, d0)
    assert not (tmp_path / "out" / "CURRENT").exists()


def test_publish_success_is_atomic_and_complete(tmp_path):
    import pyarrow.parquet as pq

    contract, cal, teacher, d0 = _contract(frozen=True)
    records = [raw_record(f"u{i}", mt_mean_logprob=-0.1 * i) for i in range(6)] + [raw_record("u6", mt_empty=True, pseudo_vi_norm="")]
    res = _publish(tmp_path / "out", records, contract, cal, teacher, d0)
    gen = read_current_generation_id(tmp_path / "out")
    verified = verify_generation(tmp_path / "out", gen)
    assert sorted(verified["files"]) == sorted(NB12_ARTIFACT_FILES)
    summary = res["summary"]
    assert summary["status"] == STATUS_SUCCESS and summary["n_u_prime"] == 6 and summary["n_excluded"] == 1
    assert summary["exclusions_by_reason"] == {EXCLUDED_MT_EMPTY: 1}
    gdir = tmp_path / "out" / "generations" / gen
    u = pq.read_table(gdir / "u_prime_manifest.parquet").to_pandas()
    assert list(u.columns) == U_PRIME_COLUMNS and u["quality_score"].notna().all()
    full = pq.read_table(gdir / "pseudo_label_manifest.parquet").to_pandas()
    assert len(full) == 7 and not (set(full.columns) & FORBIDDEN_REFERENCE_FIELDS)
    assert len((gdir / "exclusions.jsonl").read_text().splitlines()) == 1
    assert not list((tmp_path / "out" / "generations").glob("*u_prime_random*"))


def test_publish_gate_failure_keeps_previous_current(tmp_path):
    contract, cal, teacher, d0 = _contract(frozen=True)
    records = [raw_record(f"u{i}", mt_mean_logprob=-0.1 * i) for i in range(4)]
    _publish(tmp_path / "out", records, contract, cal, teacher, d0)
    first = read_current_generation_id(tmp_path / "out")
    with pytest.raises(QualityContractError, match="gates failed"):
        _publish(tmp_path / "out", records, contract, cal, teacher, d0, audio_identity_verified=False)
    assert read_current_generation_id(tmp_path / "out") == first


def test_publish_refuses_contract_for_other_teacher(tmp_path):
    contract, cal, _, d0 = _contract(frozen=True)
    other_teacher = fake_contracts(digest="7" * 64)[0]
    with pytest.raises(QualityContractError, match="not the teacher bound"):
        _publish(tmp_path / "out", [raw_record("u0")], contract, cal, other_teacher, d0)


def _notebook_upstream_gates():
    """The gate dict Notebook 12 cell 16 builds: upstream facts only, no pool gates."""
    return {name: True for name in UPSTREAM_GATE_NAMES}


def _review(tmp_path, records, contract, cal, teacher, d0, *, expected_uids=None, upstream=None):
    out = tmp_path / "out"
    out.mkdir(parents=True)
    current = out / "CURRENT"
    current.write_text("sentinel-generation\n")
    gates = _notebook_upstream_gates() if upstream is None else upstream
    assert set(gates) == set(UPSTREAM_GATE_NAMES)
    res = write_review_bundle(
        out, raw_records=records, expected_uids=list(expected_uids if expected_uids is not None else [r["segment_uid"] for r in records]),
        quality_contract=contract, calibration=cal, teacher_contract=teacher, d0_agreement_contract=d0,
        nb11_input_contract=NB11_FAKE, ledger=DataAccessLedger(), upstream_gates=gates,
    )
    assert current.read_text() == "sentinel-generation\n"
    assert not (out / "generations").exists()
    assert not (out / "pseudo_label_manifest.parquet").exists()
    assert not (out / "u_prime_manifest.parquet").exists()
    return res


def test_prefreeze_review_derives_pool_gates_from_streamed_u(tmp_path):
    draft, cal, teacher, d0 = _contract()
    records = [raw_record(f"u{i}", mt_source_truncated=(i == 0)) for i in range(3)]
    res = _review(tmp_path, records, draft, cal, teacher, d0)
    summary = res["summary"]
    assert summary["status"] == STATUS_READY_FOR_REVIEW
    assert summary["published_generation"] is False
    assert set(summary["gates"]) == set(GATE_NAMES)
    assert summary["gates"]["quality_score_contract_frozen"] is False
    assert summary["gates"]["every_u_clean_segment_accounted"] is True
    assert summary["gates"]["no_duplicate_segment_uid"] is True
    assert summary["gates"]["quality_score_for_every_eligible_row"] is True
    assert summary["gates"]["u_prime_non_empty"] is True
    pool = summary["pool"]
    assert pool["n_input_segments"] == 3 and pool["n_hard_eligible"] == 3 and pool["n_u_prime"] == 3
    assert pool["n_excluded"] == 0 and pool["n_eligible_missing_score"] == 0
    assert pool["accounted"] is True and pool["no_duplicate_uid"] is True
    assert pool["mt_source_truncated_count"] == 1 and pool["mt_hit_max_length_count"] == 0
    assert pool["input_audio_hours"] > 0 and pool["prospective_u_prime_audio_hours"] == pool["input_audio_hours"]
    assert set(pool["quality_score_stats"]) == {"min", "max", "mean", "p10", "p50", "p90"}
    assert "d0_agreement_available_rate" not in pool
    blob = json.dumps(summary)
    assert not any(f'"{name}"' in blob for name in FORBIDDEN_REFERENCE_FIELDS)


def test_prefreeze_review_fails_on_real_pool_defects(tmp_path, monkeypatch):
    draft, cal, teacher, d0 = _contract()
    missing = _review(tmp_path / "missing", [raw_record("u0")], draft, cal, teacher, d0, expected_uids=["u0", "u1"])
    assert missing["summary"]["status"] == STATUS_FAIL
    assert missing["summary"]["gates"]["every_u_clean_segment_accounted"] is False

    duplicate = _review(tmp_path / "duplicate", [raw_record("u0"), raw_record("u0")], draft, cal, teacher, d0,
                        expected_uids=["u0", "u0"])
    assert duplicate["summary"]["status"] == STATUS_FAIL
    assert duplicate["summary"]["gates"]["no_duplicate_segment_uid"] is False
    assert duplicate["summary"]["pool"]["accounted"] is True

    empty = _review(
        tmp_path / "empty",
        [raw_record("u0", asr_status=STATUS_FAILED), raw_record("u1", asr_status=STATUS_FAILED)],
        draft, cal, teacher, d0,
    )
    assert empty["summary"]["status"] == STATUS_FAIL
    assert empty["summary"]["pool"]["n_u_prime"] == 0
    assert empty["summary"]["gates"]["u_prime_non_empty"] is False

    monkeypatch.setattr("src.rq2_quality._combine", lambda *_args, **_kwargs: None)
    unscored = _review(tmp_path / "unscored", [raw_record("u0"), raw_record("u1")], draft, cal, teacher, d0)
    assert unscored["summary"]["status"] == STATUS_FAIL
    assert unscored["summary"]["pool"]["n_eligible_missing_score"] == 2
    assert unscored["summary"]["pool"]["n_hard_eligible"] == 2
    assert unscored["summary"]["gates"]["quality_score_for_every_eligible_row"] is False
    assert unscored["summary"]["gates"]["u_prime_non_empty"] is True


def test_prefreeze_review_reports_d0_agreement_availability(tmp_path):
    draft, cal, teacher, d0 = _contract(d0_enabled=True)
    records = [
        raw_record("u0", d0_status=STATUS_OK, d0_valid_encoding=True, d0_vi_norm="xin chào", teacher_d0_agreement_chrf=80.0),
        raw_record("u1", d0_status=STATUS_FAILED, d0_valid_encoding=False, teacher_d0_agreement_chrf=None),
    ]
    res = _review(tmp_path, records, draft, cal, teacher, d0)
    pool = res["summary"]["pool"]
    assert pool["d0_agreement_available_count"] == 1
    assert pool["d0_agreement_available_rate"] == 0.5
    assert pool["n_excluded"] == 1 and pool["n_u_prime"] == 1
    assert res["summary"]["status"] == STATUS_READY_FOR_REVIEW
    assert res["summary"]["gates"]["u_prime_non_empty"] is True


_NB12_MODULES = ("src.rq2_pseudo_contract", "src.rq2_pseudo_label", "src.rq2_quality")


def _notebook_code_cells():
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "12_RQ2_PseudoLabel_Quality_Freeze_UPrime.ipynb").read_text())
    return ["".join(c.get("source") or []) for c in nb["cells"] if c.get("cell_type") == "code"]


def _notebook_calls_to_nb12():
    """Name calls in Notebook 12 that target a callable imported from the three NB12 modules."""
    import importlib

    imported = {}
    calls = []
    for index, source in enumerate(_notebook_code_cells()):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _NB12_MODULES:
                module = importlib.import_module(node.module)
                for alias in node.names:
                    obj = getattr(module, alias.name)
                    if callable(obj):
                        imported[alias.asname or alias.name] = obj
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported:
                calls.append((index, node, imported[node.func.id]))
    return calls


def _missing_required_call_args(sig, node):
    if any(kw.arg is None for kw in node.keywords) or any(isinstance(arg, ast.Starred) for arg in node.args):
        return []
    supplied = {kw.arg for kw in node.keywords}
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    for index, param in enumerate(positional):
        if index < len(node.args):
            supplied.add(param.name)
    return [
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and p.name not in supplied
        and p.name != "self"
    ]


def test_notebook_quality_contract_is_bound_to_nb11_contract_sha256():
    """Compilation cannot see a missing keyword. The call itself must name the frozen NB11 input."""
    found = []
    for source in _notebook_code_cells():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build_quality_score_contract":
                found.append(node)
    assert len(found) == 1
    keywords = {kw.arg: kw.value for kw in found[0].keywords}
    assert "nb11_input_contract_sha256" in keywords
    value = keywords["nb11_input_contract_sha256"]
    assert isinstance(value, ast.Attribute)
    assert isinstance(value.value, ast.Name) and value.value.id == "NB11"
    assert value.attr == "contract_sha256"


def test_notebook_calls_match_nb12_signatures():
    """Required arguments added to the three NB12 modules must show up in the notebook calls."""
    problems = []
    for index, node, target in _notebook_calls_to_nb12():
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        accepted = set(sig.parameters)
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for kw in node.keywords:
            if kw.arg is not None and not takes_kwargs and kw.arg not in accepted:
                problems.append(f"cell {index}: {node.func.id}(..., {kw.arg}=)")
        positional = len([arg for arg in node.args if not isinstance(arg, ast.Starred)])
        max_positional = sum(
            1 for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        has_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if positional > max_positional and not has_varargs:
            problems.append(f"cell {index}: {node.func.id}() got {positional} positional args, max {max_positional}")
        missing = _missing_required_call_args(sig, node)
        if missing:
            problems.append(f"cell {index}: {node.func.id}() missing required {missing}")
    assert not problems, "Notebook 12 calls disagree with NB12 signatures: " + "; ".join(problems)
    assert any(node.func.id == "build_quality_score_contract" for _index, node, _target in _notebook_calls_to_nb12())


def test_notebook_review_cell_uses_the_upstream_gate_shape():
    import json as _json

    nb = _json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "12_RQ2_PseudoLabel_Quality_Freeze_UPrime.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "UPSTREAM_GATES = {" in "".join(c["source"]))
    for name in UPSTREAM_GATE_NAMES:
        assert f'"{name}"' in cell
    assert "raw_records=raw_u" in cell and "expected_uids=NB11.ordered_uids" in cell
    for derived in ("every_u_clean_segment_accounted", "no_duplicate_segment_uid",
                    "quality_score_for_every_eligible_row", "u_prime_non_empty"):
        assert f'"{derived}"' not in cell


def test_no_selection_entry_points_in_nb12_sources():
    assert nb12_source_has_no_selection(NB12_SOURCES)


# --------------------------------------------------------------------------- #
# End-to-end synthetic NB12 run                                                #
# --------------------------------------------------------------------------- #
def test_end_to_end_synthetic_pipeline_leaves_nb11_untouched(tmp_path):
    build_nb11_generation(tmp_path, n_segments=9, n_samples=(1600, 800))
    u_clean = tmp_path / "artifacts" / "rq2" / "u_clean"
    before = tree_digest(u_clean)
    nb11 = resolve_nb11_input(tmp_path, u_clean_dir=tmp_path / "artifacts" / "rq2" / "u_clean")
    contract, cal, teacher, d0 = _contract(frozen=True, nb11_sha=nb11.contract_sha256)
    binding = inference_binding(pool="u_clean", input_identity_sha256=nb11.contract_sha256,
                                teacher_contract_sha256=teacher["teacher_contract_sha256"],
                                d0_agreement_contract_sha256=d0["d0_agreement_contract_sha256"],
                                decoding_config_sha256=teacher["decoding_config_sha256"])
    out = tmp_path / "artifacts" / "rq2" / "pseudo_labels"
    ckpt = InferenceCheckpoint(out / "checkpoint" / "u_clean", binding=binding, ordered_uids=nb11.ordered_uids, shard_size=4)
    run_peak_safe_teacher_inference(ckpt=ckpt, items=items_from_nb11(nb11), decoding=teacher["decoding"],
                                    load_asr=lambda: {"adapter": fake_asr}, load_mt=lambda: {"adapter": fake_mt})
    identity = {"nb11_generation_id": nb11.generation_id, "nb11_input_contract_sha256": nb11.contract_sha256,
                "teacher_contract_sha256": teacher["teacher_contract_sha256"],
                "d0_agreement_contract_sha256": d0["d0_agreement_contract_sha256"]}
    records = iter_raw_records(ckpt, nb11.rows, identity=identity, d0_enabled=False)
    res = publish_u_prime_generation(
        out, raw_records=records, expected_uids=nb11.ordered_uids, quality_contract=contract, calibration=cal,
        teacher_contract=teacher, d0_agreement_contract=d0, nb11_input_contract=nb11.contract,
        ledger=DataAccessLedger(), upstream_gates=_all_gates(),
    )
    assert res["summary"]["n_input_segments"] == 9
    assert res["summary"]["exclusions_by_reason"] == {EXCLUDED_ASR_EMPTY: 1}
    assert tree_digest(u_clean) == before


def _body_sha(payload, hash_key):
    return sha256_json({k: v for k, v in payload.items() if k != hash_key})


def _published_summary(tmp_path):
    contract, cal, teacher, d0 = _contract(frozen=True)
    records = [raw_record("u0")]
    out = tmp_path / "out"
    res = _publish(out, records, contract, cal, teacher, d0)
    gen = read_current_generation_id(out)
    return res["summary"], out / "generations" / gen, contract, cal, teacher, d0


def test_matching_contracts_publish_and_summary_matches_files(tmp_path):
    summary, gdir, contract, cal, teacher, d0 = _published_summary(tmp_path)
    assert_publication_bindings(
        quality_contract=json.loads((gdir / "quality_score_contract.json").read_text(encoding="utf-8")),
        calibration=json.loads((gdir / "quality_calibration.json").read_text(encoding="utf-8")),
        teacher_contract=json.loads((gdir / "teacher_contract.json").read_text(encoding="utf-8")),
        d0_agreement_contract=json.loads((gdir / "d0_agreement_contract.json").read_text(encoding="utf-8")),
        nb11_input_contract=json.loads((gdir / "nb11_input_contract.json").read_text(encoding="utf-8")),
    )
    assert summary["generation_id"] == gdir.name
    assert summary["teacher_contract_sha256"] == _body_sha(json.loads((gdir / "teacher_contract.json").read_text()), "teacher_contract_sha256")
    assert summary["d0_agreement_contract_sha256"] == _body_sha(json.loads((gdir / "d0_agreement_contract.json").read_text()), "d0_agreement_contract_sha256")
    assert summary["nb11_input_contract_sha256"] == _body_sha(json.loads((gdir / "nb11_input_contract.json").read_text()), "nb11_input_contract_sha256")
    assert summary["quality_score_contract_sha256"] == _body_sha(json.loads((gdir / "quality_score_contract.json").read_text()), "quality_score_contract_sha256")
    assert summary["quality_calibration_sha256"] == calibration_artifact_sha256(json.loads((gdir / "quality_calibration.json").read_text()))
    assert summary["quality_calibration_sha256"] == contract["calibration_artifact_sha256"]
    assert summary["u_prime_manifest_sha256"] == sha256_file(gdir / "u_prime_manifest.parquet")
    assert summary["validation_audio_identity_sha256"] == contract["validation_audio_identity_sha256"]
    blob = "\n".join(p.read_text(encoding="utf-8") for p in gdir.glob("*.json"))
    assert str(tmp_path) not in blob and "/Users/" not in blob and "/home/" not in blob


def test_mutated_contract_bodies_fail_before_publication(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "CURRENT").write_text("keep-me\n")
    contract, cal, teacher, d0 = _contract(frozen=True)
    records = [raw_record("u0")]

    bad_teacher = json.loads(json.dumps(teacher))
    bad_teacher["role"] = "mutated-after-hash"
    with pytest.raises(QualityContractError, match="teacher contract"):
        _publish(out, records, contract, cal, bad_teacher, d0)

    bad_d0 = json.loads(json.dumps(d0))
    bad_d0["contract_version"] = "mutated"
    with pytest.raises(QualityContractError, match="D0 agreement"):
        _publish(out, records, contract, cal, teacher, bad_d0)

    bad_nb11 = json.loads(json.dumps(NB11_FAKE))
    bad_nb11["generation_id"] = "other"
    with pytest.raises(QualityContractError, match="NB11 input"):
        _publish(out, records, contract, cal, teacher, d0, nb11=bad_nb11)

    bad_cal = json.loads(json.dumps(cal))
    bad_cal["selected"]["spearman_rho"] = 0.0
    with pytest.raises(QualityContractError, match="calibration artifact"):
        _publish(out, records, contract, bad_cal, teacher, d0)

    proposal = {k: v for k, v in contract.items() if k not in ("quality_score_proposal_sha256", "quality_score_contract_sha256", "frozen")}
    proposal["calibration_artifact_sha256"] = "0" * 64
    payload = {**proposal, "quality_score_proposal_sha256": sha256_json(proposal), "frozen": True}
    payload["quality_score_contract_sha256"] = sha256_json(payload)
    with pytest.raises(QualityContractError, match="calibration artifact"):
        _publish(out, records, payload, cal, teacher, d0)

    assert (out / "CURRENT").read_text(encoding="utf-8") == "keep-me\n"
    assert not list(out.glob("generations/*/COMPLETE.json"))


def test_staged_cross_binding_failure_leaves_current_and_writes_complete_last(tmp_path, monkeypatch):
    import src.rq2_quality as quality

    out = tmp_path / "out"
    out.mkdir()
    (out / "CURRENT").write_text("keep-me\n")
    contract, cal, teacher, d0 = _contract(frozen=True)
    seen = {"entered_finalize": False}
    real_finalize = quality.finalize_generation

    def _finalize(out_dir, staged, relative_files, allow_empty=()):
        gen_dir = Path(staged["staging_dir"])
        assert not (gen_dir / "COMPLETE.json").exists()
        for name in relative_files:
            assert (gen_dir / name).exists()
        seen["entered_finalize"] = True
        return real_finalize(out_dir, staged, relative_files, allow_empty=allow_empty)

    monkeypatch.setattr(quality, "finalize_generation", _finalize)
    _publish(out, [raw_record("u0")], contract, cal, teacher, d0)
    assert seen["entered_finalize"] is True
    gen = read_current_generation_id(out)
    assert (out / "generations" / gen / "COMPLETE.json").is_file()

    calls = {"n": 0}
    real_bind = quality.assert_publication_bindings

    def _bind(**kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise QualityContractError("staged cross-binding failed")
        return real_bind(**kwargs)

    monkeypatch.setattr(quality, "assert_publication_bindings", _bind)
    with pytest.raises(QualityContractError, match="staged cross-binding failed"):
        _publish(out, [raw_record("u0")], contract, cal, teacher, d0)
    assert read_current_generation_id(out) == gen
    assert (out / "generations" / gen / "COMPLETE.json").is_file()
    assert not any(p.name == "COMPLETE.json" and ".partial" in p.parent.name for p in out.rglob("COMPLETE.json"))


def test_review_identity_includes_nb11_and_does_not_overwrite(tmp_path):
    draft, cal, teacher, d0 = _contract()
    other = _nb11("other-generation")
    draft2 = build_quality_score_contract(
        cal, teacher_contract=teacher, d0_agreement_contract=d0,
        nb11_input_contract_sha256=other["nb11_input_contract_sha256"],
    )
    assert review_identity(draft)["review_identity_sha256"] != review_identity(draft2)["review_identity_sha256"]
    assert draft["quality_score_proposal_sha256"] != draft2["quality_score_proposal_sha256"]
    records = [raw_record("u0")]
    first = _review(tmp_path, records, draft, cal, teacher, d0)
    summary_path = Path(first["review_dir"]) / "summary.json"
    raw = summary_path.read_bytes()
    out = tmp_path / "out"
    second = write_review_bundle(
        out, raw_records=records, expected_uids=["u0"], quality_contract=draft2, calibration=cal,
        teacher_contract=teacher, d0_agreement_contract=d0, nb11_input_contract=other,
        ledger=DataAccessLedger(), upstream_gates=_notebook_upstream_gates(),
    )
    assert second["review_dir"] != first["review_dir"]
    assert summary_path.read_bytes() == raw
    assert second["summary"]["nb11_input_contract_sha256"] == other["nb11_input_contract_sha256"]
    assert first["summary"]["nb11_input_contract_sha256"] == NB11_FAKE["nb11_input_contract_sha256"]
    for key in (
        "teacher_contract_sha256", "d0_agreement_contract_sha256", "validation_audio_identity_sha256",
        "quality_calibration_sha256", "quality_score_contract_sha256",
    ):
        assert first["summary"][key]
    with pytest.raises(QualityContractError, match="immutable"):
        write_review_bundle(
            out, raw_records=records, expected_uids=["u0"], quality_contract=draft, calibration=cal,
            teacher_contract=teacher, d0_agreement_contract=d0, nb11_input_contract=NB11_FAKE,
            ledger=DataAccessLedger(), upstream_gates=_notebook_upstream_gates(),
        )
    assert summary_path.read_bytes() == raw
    assert not list(out.glob("generations/*/COMPLETE.json"))
