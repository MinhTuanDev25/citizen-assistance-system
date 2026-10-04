"""RQ2 Notebook 12: hard eligibility, G_validation quality calibration, and U'.

* Hard eligibility is intrinsic validity only (decode failure, invalid
  encoding, empty output, non-finite confidence, missing D0 agreement when D0
  is enabled, and an optional frozen output-sanity rule). It never reads the
  quality score, and the same U' feeds every later RQ2 treatment.
* The quality score is calibrated on G_validation only. The human Vietnamese
  reference lives in :class:`ValidationReferences` and is used only inside
  :func:`calibrate_quality_score`; nothing derived per record from it is
  returned or written.
* ``SUCCESS_RQ2_PSEUDO_POOL_FROZEN`` and a ``CURRENT`` pointer exist only for a
  quality-score contract a reviewer froze by its proposal hash. Before that the
  notebook can only write a review bundle with ``READY_FOR_RQ2_QUALITY_REVIEW``.
* No N-hour selection, Random/Top-Q treatment, training, or G_test evaluation
  happens here.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import (
    PSEUDO_SCHEMA_VERSION,
    STATUS_FAIL,
    STATUS_READY_FOR_REVIEW,
    STATUS_SUCCESS,
    DataAccessLedger,
    assert_not_g_test_path,
    finalize_generation,
    stage_generation,
    write_json,
)
from src.rq2_pseudo_label import (
    FORBIDDEN_REFERENCE_FIELDS,
    RAW_RECORD_COLUMNS,
    STATUS_OK,
    assert_no_reference_fields,
    sentence_chrfpp,
)

QUALITY_SCHEMA_VERSION = "rq2-quality-1.0"
QUALITY_CONTRACT_VERSION = "rq2_quality_score_contract_v1"
ELIGIBILITY_POLICY_VERSION = "rq2_u_prime_hard_eligibility_v1"
CALIBRATION_CRITERION = "spearman_rho(Q, sentence_chrF++(C0 pseudo-label, G_validation reference))"
SCORE_FORMULA = "Q = sum_f w_f * clip((x_f - lo_f) / (hi_f - lo_f), 0, 1) - sum_p penalty_p * flag_p"

ELIGIBLE = "U_PRIME_ELIGIBLE"
EXCLUDED_DECODE_FAILURE = "EXCLUDED_DECODE_FAILURE"
EXCLUDED_INVALID_ENCODING = "EXCLUDED_INVALID_OUTPUT_ENCODING"
EXCLUDED_ASR_EMPTY = "EXCLUDED_ASR_EMPTY"
EXCLUDED_MT_EMPTY = "EXCLUDED_MT_EMPTY"
EXCLUDED_NONFINITE_CONFIDENCE = "EXCLUDED_NONFINITE_CONFIDENCE"
EXCLUDED_AGREEMENT_UNAVAILABLE = "EXCLUDED_D0_AGREEMENT_UNAVAILABLE"
EXCLUDED_OUTPUT_SANITY = "EXCLUDED_OUTPUT_SANITY"
EXCLUSION_REASONS = (
    EXCLUDED_DECODE_FAILURE,
    EXCLUDED_INVALID_ENCODING,
    EXCLUDED_ASR_EMPTY,
    EXCLUDED_MT_EMPTY,
    EXCLUDED_NONFINITE_CONFIDENCE,
    EXCLUDED_AGREEMENT_UNAVAILABLE,
    EXCLUDED_OUTPUT_SANITY,
)

FEATURES = (
    ("asr_confidence", "asr_mean_logprob"),
    ("mt_confidence", "mt_mean_logprob"),
    ("d0_agreement", "teacher_d0_agreement_chrf"),
)
PENALTY_FLAGS = ("mt_source_truncated", "mt_hit_max_length")

SCORED_COLUMNS = [
    "hard_eligible", "eligibility_status", "exclusion_reason",
    "asr_confidence_norm", "mt_confidence_norm", "d0_agreement_norm",
    "quality_penalty", "quality_score", "quality_score_contract_sha256", "quality_score_frozen",
]
PSEUDO_LABEL_MANIFEST_COLUMNS = list(RAW_RECORD_COLUMNS) + SCORED_COLUMNS
U_PRIME_COLUMNS = [
    "segment_uid", "source_id", "source_group_id", "segment_local_path",
    "segment_pcm16_sha256", "segment_wav_sha256", "duration_seconds",
    "asr_text_norm", "pseudo_vi_norm",
    "asr_mean_logprob", "mt_mean_logprob", "teacher_d0_agreement_chrf",
    "asr_confidence_norm", "mt_confidence_norm", "d0_agreement_norm",
    "quality_penalty", "quality_score",
    "nb11_generation_id", "nb11_input_contract_sha256",
    "teacher_contract_sha256", "d0_agreement_contract_sha256", "quality_score_contract_sha256",
]
EXCLUSION_COLUMNS = [
    "segment_uid", "source_id", "duration_seconds", "exclusion_reason",
    "asr_status", "mt_status", "d0_status", "asr_decode_warning", "mt_decode_warning", "d0_decode_warning",
]

_FORBIDDEN_NB12_SYMBOLS = ("select_random", "select_top", "random_n_hours", "top_q", "train_student", "evaluate_g_test")


class CalibrationError(RuntimeError):
    """G_validation calibration cannot produce a valid quality-score proposal."""


class QualityContractError(RuntimeError):
    """A quality-score contract is not frozen, tampered, or does not match."""


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------- #
# Hard eligibility (shared by Random and Quality later; never reads Q)         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OutputSanityConfig:
    """Optional bound on pseudo_vi_chars / asr_chars, derived from G_validation human pairs."""

    enabled: bool = False
    min_target_source_char_ratio: Optional[float] = None
    max_target_source_char_ratio: Optional[float] = None
    lower_quantile: Optional[float] = None
    upper_quantile: Optional[float] = None
    derived_from: str = ""

    def payload(self) -> Dict[str, Any]:
        if self.enabled and (self.min_target_source_char_ratio is None or self.max_target_source_char_ratio is None):
            raise CalibrationError("enabled output sanity needs both bounds")
        return {
            "rule": "target_source_char_ratio within [min, max]",
            "enabled": bool(self.enabled),
            "min_target_source_char_ratio": self.min_target_source_char_ratio,
            "max_target_source_char_ratio": self.max_target_source_char_ratio,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "derived_from": self.derived_from,
        }


def hard_eligibility(record: Mapping[str, Any], *, d0_enabled: bool, sanity: OutputSanityConfig) -> str:
    """Fixed reason order; the first failing check wins."""
    if record.get("asr_status") != STATUS_OK:
        return EXCLUDED_DECODE_FAILURE
    if not record.get("asr_valid_encoding"):
        return EXCLUDED_INVALID_ENCODING
    if record.get("asr_empty") or not record.get("asr_text_norm"):
        return EXCLUDED_ASR_EMPTY
    if record.get("mt_status") != STATUS_OK:
        return EXCLUDED_DECODE_FAILURE
    if not record.get("mt_valid_encoding"):
        return EXCLUDED_INVALID_ENCODING
    if record.get("mt_empty") or not record.get("pseudo_vi_norm"):
        return EXCLUDED_MT_EMPTY
    if _finite(record.get("asr_mean_logprob")) is None or _finite(record.get("mt_mean_logprob")) is None:
        return EXCLUDED_NONFINITE_CONFIDENCE
    if d0_enabled:
        if record.get("d0_status") != STATUS_OK or not record.get("d0_valid_encoding"):
            return EXCLUDED_AGREEMENT_UNAVAILABLE
        if _finite(record.get("teacher_d0_agreement_chrf")) is None:
            return EXCLUDED_AGREEMENT_UNAVAILABLE
    if sanity.enabled:
        ratio = _finite(record.get("target_source_char_ratio"))
        if ratio is None or not (sanity.min_target_source_char_ratio <= ratio <= sanity.max_target_source_char_ratio):
            return EXCLUDED_OUTPUT_SANITY
    return ELIGIBLE


# --------------------------------------------------------------------------- #
# G_validation (calibration only)                                              #
# --------------------------------------------------------------------------- #
VALIDATION_MANIFEST_NAME = "rq1_validation.csv"
SPLIT_SUMMARY_NAME = "split_summary.json"
_VALIDATION_ID_COLUMNS = ("record_uid", "record_id", "parquet_file", "shard_row_index", "duration_seconds", "split")


class ValidationReferences:
    """Human G_validation references, usable only by the calibration code path."""

    __slots__ = ("_vi", "_bahnar")

    def __init__(self, vi: Mapping[str, str], bahnar: Mapping[str, str]):
        self._vi = dict(vi)
        self._bahnar = dict(bahnar)

    def __repr__(self) -> str:
        return f"ValidationReferences(n={len(self._vi)})"

    def __len__(self) -> int:
        return len(self._vi)

    @property
    def uids(self) -> List[str]:
        return list(self._vi)

    def _reference_vi(self, uid: str) -> str:
        return self._vi[uid]

    def _reference_bahnar(self, uid: str) -> str:
        return self._bahnar[uid]

    def __reduce__(self):
        raise TypeError("ValidationReferences must not be serialised")


def load_g_validation(
    manifests_dir: Union[str, Path],
    *,
    ledger: DataAccessLedger,
) -> Tuple[Any, ValidationReferences, Dict[str, Any]]:
    """Read only rq1_validation.csv (sha-pinned by split_summary.json) and split ids from references.

    The returned id frame has no text column. The frozen test manifest is never
    opened; its pinned hash is not even read.
    """
    import pandas as pd

    root = Path(manifests_dir)
    summary_path = ledger.record("g_validation", root / SPLIT_SUMMARY_NAME, "pin for rq1_validation.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pin = str((summary.get("manifest_sha256") or {}).get(VALIDATION_MANIFEST_NAME) or "")
    if len(pin) != 64:
        raise CalibrationError("split_summary.json has no rq1_validation.csv pin")
    path = ledger.record("g_validation", root / VALIDATION_MANIFEST_NAME, "quality-score calibration")
    actual = sha256_file(path)
    if actual != pin:
        raise CalibrationError("rq1_validation.csv differs from its split_summary pin")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(frame["split"].astype(str)) != {"validation"}:
        raise CalibrationError("rq1_validation.csv contains non-validation rows")
    if frame["record_uid"].duplicated().any():
        raise CalibrationError("duplicate record_uid in G_validation")
    uids = frame["record_uid"].astype(str).tolist()
    refs = ValidationReferences(
        dict(zip(uids, frame["text_vi"].astype(str))),
        dict(zip(uids, frame["text_bahnar"].astype(str))),
    )
    ids = frame[list(_VALIDATION_ID_COLUMNS)].copy()
    assert_no_reference_fields(ids.columns, "G_validation id frame")
    identity = {
        "manifest": VALIDATION_MANIFEST_NAME,
        "manifest_sha256": actual,
        "n_records": len(uids),
        "ordered_record_uid_sha256": sha256_json(uids),
    }
    return ids, refs, identity


def derive_output_sanity(
    references: ValidationReferences,
    *,
    enabled: bool,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> OutputSanityConfig:
    """Bounds from human G_validation pairs (|vi| / |bahnar| characters), never from U."""
    from src.data_utils import normalize_bahnar_ctc_v1
    from src.mt_normalize import normalize_mt_text_v1

    ratios = []
    for uid in references.uids:
        src = normalize_bahnar_ctc_v1(references._reference_bahnar(uid))
        tgt = normalize_mt_text_v1(references._reference_vi(uid))
        if src and tgt:
            ratios.append(len(tgt) / len(src))
    if not ratios:
        raise CalibrationError("no usable G_validation pairs for output-sanity bounds")
    arr = np.asarray(ratios, dtype=np.float64)
    return OutputSanityConfig(
        enabled=bool(enabled),
        min_target_source_char_ratio=round(float(np.quantile(arr, lower_quantile)), 6),
        max_target_source_char_ratio=round(float(np.quantile(arr, upper_quantile)), 6),
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        derived_from="g_validation_human_pairs",
    )


# --------------------------------------------------------------------------- #
# Normalisation, Spearman, weight grid                                         #
# --------------------------------------------------------------------------- #
def active_features(d0_enabled: bool) -> List[Tuple[str, str]]:
    return [f for f in FEATURES if d0_enabled or f[0] != "d0_agreement"]


def fit_normalization(records: Sequence[Mapping[str, Any]], features: Sequence[Tuple[str, str]], *, lower_quantile: float, upper_quantile: float) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, column in features:
        values = np.asarray([float(r[column]) for r in records], dtype=np.float64)
        lo = float(np.quantile(values, lower_quantile))
        hi = float(np.quantile(values, upper_quantile))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise CalibrationError(f"feature {name} is degenerate on G_validation (lo={lo}, hi={hi})")
        out[name] = {"column": column, "lo": lo, "hi": hi,
                     "lower_quantile": float(lower_quantile), "upper_quantile": float(upper_quantile)}
    return out


def normalize_feature(value: Any, stats: Mapping[str, Any]) -> Optional[float]:
    x = _finite(value)
    if x is None:
        return None
    return float(min(1.0, max(0.0, (x - float(stats["lo"])) / (float(stats["hi"]) - float(stats["lo"])))))


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Average-rank Spearman; NaN if either side is constant."""
    import pandas as pd

    rx = pd.Series(list(x), dtype="float64").rank(method="average").to_numpy()
    ry = pd.Series(list(y), dtype="float64").rank(method="average").to_numpy()
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def weight_grid(n_features: int, step: float) -> List[Tuple[float, ...]]:
    """Every non-negative weight vector on a ``step`` lattice that sums to 1, in a fixed order."""
    units = int(round(1.0 / float(step)))
    if units < 1 or abs(units * float(step) - 1.0) > 1e-9:
        raise CalibrationError("weight_grid_step must divide 1 exactly")
    out = []
    for combo in itertools.product(range(units + 1), repeat=n_features):
        if sum(combo) == units:
            out.append(tuple(round(c / units, 10) for c in combo))
    return out


def _score_components(record: Mapping[str, Any], normalization: Mapping[str, Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    return {name: normalize_feature(record.get(stats["column"]), stats) for name, stats in normalization.items()}


def _penalty(record: Mapping[str, Any], penalties: Mapping[str, float]) -> float:
    return float(sum(float(v) for flag, v in penalties.items() if bool(record.get(flag))))


def _combine(components: Mapping[str, Optional[float]], weights: Mapping[str, float], penalty: float) -> Optional[float]:
    total = 0.0
    for name, w in weights.items():
        value = components.get(name)
        if value is None:
            return None
        total += float(w) * float(value)
    return float(total - penalty)


@dataclass(frozen=True)
class QualityCalibrationConfig:
    """Pre-declared calibration choices. Changing any of them changes the contract hash."""

    weight_grid_step: float = 0.1
    normalization_lower_quantile: float = 0.05
    normalization_upper_quantile: float = 0.95
    penalties: Tuple[Tuple[str, float], ...] = (("mt_source_truncated", 0.0), ("mt_hit_max_length", 0.0))
    min_calibration_records: int = 200
    output_sanity_enabled: bool = False
    sanity_lower_quantile: float = 0.005
    sanity_upper_quantile: float = 0.995

    def payload(self) -> Dict[str, Any]:
        flags = [p[0] for p in self.penalties]
        if sorted(flags) != sorted(PENALTY_FLAGS) or any(float(p[1]) < 0 for p in self.penalties):
            raise CalibrationError(f"penalties must list exactly {PENALTY_FLAGS} with values >= 0")
        if not (0.0 <= self.normalization_lower_quantile < self.normalization_upper_quantile <= 1.0):
            raise CalibrationError("normalization quantiles must satisfy 0 <= lower < upper <= 1")
        return {
            "weight_grid_step": float(self.weight_grid_step),
            "normalization": "validation_quantile_minmax_clip",
            "normalization_lower_quantile": float(self.normalization_lower_quantile),
            "normalization_upper_quantile": float(self.normalization_upper_quantile),
            "penalties": {k: float(v) for k, v in sorted(self.penalties)},
            "min_calibration_records": int(self.min_calibration_records),
            "output_sanity_enabled": bool(self.output_sanity_enabled),
            "sanity_lower_quantile": float(self.sanity_lower_quantile),
            "sanity_upper_quantile": float(self.sanity_upper_quantile),
            "criterion": CALIBRATION_CRITERION,
            "tie_break": "max rho (12 dp), then min L2 distance to uniform weights, then lexicographic weights",
        }


def _select_candidate(candidates: Sequence[Mapping[str, Any]], n_features: int) -> Mapping[str, Any]:
    uniform = 1.0 / n_features

    def key(c):
        rho = c["spearman_rho"]
        rho_key = -round(rho, 12) if rho is not None else float("inf")
        dist = round(sum((w - uniform) ** 2 for w in c["weight_vector"]), 12)
        return (rho_key, dist, tuple(c["weight_vector"]))

    return sorted(candidates, key=key)[0]


def calibrate_quality_score(
    validation_records: Sequence[Mapping[str, Any]],
    references: ValidationReferences,
    *,
    d0_enabled: bool,
    config: QualityCalibrationConfig,
    calibration_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fit normalisation + weights on G_validation raw C0 outputs. Returns aggregates only."""
    from src.mt_normalize import normalize_mt_text_v1

    if not isinstance(references, ValidationReferences):
        raise CalibrationError("calibration requires G_validation ValidationReferences")
    config_payload = config.payload()
    records = [dict(r) for r in validation_records]
    uids = [str(r["segment_uid"]) for r in records]
    if len(set(uids)) != len(uids) or set(uids) != set(references.uids):
        raise CalibrationError("validation raw records do not cover G_validation exactly once")
    for r in records:
        assert_no_reference_fields(r.keys(), "validation raw record")
    sanity = derive_output_sanity(
        references,
        enabled=config.output_sanity_enabled,
        lower_quantile=config.sanity_lower_quantile,
        upper_quantile=config.sanity_upper_quantile,
    )
    disabled_sanity = OutputSanityConfig(enabled=False)
    reasons: Dict[str, int] = {}
    eligible = []
    would_fail_sanity = 0
    for r in records:
        status = hard_eligibility(r, d0_enabled=d0_enabled, sanity=sanity)
        reasons[status] = reasons.get(status, 0) + 1
        if status == ELIGIBLE:
            eligible.append(r)
        if hard_eligibility(r, d0_enabled=d0_enabled, sanity=disabled_sanity) == ELIGIBLE:
            probe = OutputSanityConfig(
                enabled=True,
                min_target_source_char_ratio=sanity.min_target_source_char_ratio,
                max_target_source_char_ratio=sanity.max_target_source_char_ratio,
            )
            if hard_eligibility(r, d0_enabled=d0_enabled, sanity=probe) == EXCLUDED_OUTPUT_SANITY:
                would_fail_sanity += 1
    if len(eligible) < int(config.min_calibration_records):
        raise CalibrationError(
            f"only {len(eligible)} hard-eligible G_validation records (< {config.min_calibration_records})"
        )
    targets = [sentence_chrfpp(r["pseudo_vi_norm"], normalize_mt_text_v1(references._reference_vi(r["segment_uid"])))
               for r in eligible]
    features = active_features(d0_enabled)
    normalization = fit_normalization(
        eligible, features,
        lower_quantile=config.normalization_lower_quantile,
        upper_quantile=config.normalization_upper_quantile,
    )
    penalties = config_payload["penalties"]
    components = [_score_components(r, normalization) for r in eligible]
    pens = [_penalty(r, penalties) for r in eligible]
    names = [f[0] for f in features]
    candidates = []
    for vec in weight_grid(len(names), config.weight_grid_step):
        weights = dict(zip(names, vec))
        q = [_combine(c, weights, p) for c, p in zip(components, pens)]
        rho = spearman_rho(q, targets)
        candidates.append({"weight_vector": list(vec), "weights": weights,
                           "spearman_rho": rho if math.isfinite(rho) else None})
    best = _select_candidate(candidates, len(names))
    if best["spearman_rho"] is None:
        raise CalibrationError("no weight vector has a finite Spearman rho on G_validation")
    single = {name: spearman_rho([c[name] for c in components], targets) for name in names}
    flag_rates = {flag: float(np.mean([bool(r.get(flag)) for r in eligible])) for flag in PENALTY_FLAGS}
    return {
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "calibration_split": "g_validation",
        "calibration_identity": dict(calibration_identity),
        "config": config_payload,
        "config_sha256": sha256_json(config_payload),
        "d0_enabled": bool(d0_enabled),
        "features": [{"name": n, "column": c} for n, c in features],
        "normalization": normalization,
        "output_sanity": sanity.payload(),
        "output_sanity_would_exclude_if_enabled": int(would_fail_sanity),
        "n_validation_records": len(records),
        "n_hard_eligible": len(eligible),
        "validation_eligibility_counts": dict(sorted(reasons.items())),
        "target_sentence_chrfpp_mean": float(np.mean(targets)),
        "single_feature_spearman": {k: (v if math.isfinite(v) else None) for k, v in single.items()},
        "penalty_flag_rates": flag_rates,
        "candidates": candidates,
        "candidates_sha256": sha256_json(candidates),
        "selected": {"weights": best["weights"], "spearman_rho": best["spearman_rho"]},
    }


# --------------------------------------------------------------------------- #
# Quality-score contract                                                       #
# --------------------------------------------------------------------------- #
def _without_hash(payload: Mapping[str, Any], hash_key: str) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != hash_key}


def assert_self_hashed(payload: Mapping[str, Any], hash_key: str, label: str) -> None:
    """Recompute a contract hash from its body. The stored field is not trusted."""
    if sha256_json(_without_hash(payload, hash_key)) != str(payload.get(hash_key) or ""):
        raise QualityContractError(f"{label} {hash_key} does not recompute from its canonical body")


def calibration_artifact_sha256(calibration: Mapping[str, Any]) -> str:
    return sha256_json(_without_hash(calibration, "calibration_artifact_sha256"))


def build_quality_score_contract(
    calibration: Mapping[str, Any],
    *,
    teacher_contract: Mapping[str, Any],
    d0_agreement_contract: Mapping[str, Any],
    nb11_input_contract_sha256: str,
    frozen: bool = False,
) -> Dict[str, Any]:
    if str(calibration.get("calibration_split")) != "g_validation":
        raise QualityContractError("quality score must be calibrated on g_validation")
    if bool(calibration["d0_enabled"]) != bool(d0_agreement_contract.get("enabled")):
        raise QualityContractError("calibration and D0 agreement contract disagree on D0")
    identity = dict(calibration.get("calibration_identity") or {})
    for key in ("audio_identity_sha256", "manifest_sha256"):
        if len(str(identity.get(key) or "")) != 64:
            raise QualityContractError(f"calibration identity missing {key}")
    if len(str(nb11_input_contract_sha256 or "")) != 64:
        raise QualityContractError("quality contract requires nb11_input_contract_sha256")
    assert_self_hashed(teacher_contract, "teacher_contract_sha256", "teacher contract")
    assert_self_hashed(d0_agreement_contract, "d0_agreement_contract_sha256", "D0 agreement contract")
    proposal = {
        "contract_version": QUALITY_CONTRACT_VERSION,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "score_formula": SCORE_FORMULA,
        "features": list(calibration["features"]),
        "normalization": dict(calibration["normalization"]),
        "weights": dict(calibration["selected"]["weights"]),
        "penalties": dict(calibration["config"]["penalties"]),
        "eligibility": {
            "policy_version": ELIGIBILITY_POLICY_VERSION,
            "reason_order": list(EXCLUSION_REASONS),
            "d0_enabled": bool(calibration["d0_enabled"]),
            "output_sanity": dict(calibration["output_sanity"]),
            "uses_quality_score": False,
        },
        "calibration": {
            "split": "g_validation",
            "criterion": CALIBRATION_CRITERION,
            "identity": dict(calibration["calibration_identity"]),
            "config_sha256": calibration["config_sha256"],
            "candidates_sha256": calibration["candidates_sha256"],
            "n_candidates": len(calibration["candidates"]),
            "n_hard_eligible": int(calibration["n_hard_eligible"]),
            "selected_spearman_rho": calibration["selected"]["spearman_rho"],
        },
        "teacher_contract_sha256": str(teacher_contract["teacher_contract_sha256"]),
        "d0_agreement_contract_sha256": str(d0_agreement_contract["d0_agreement_contract_sha256"]),
        "decoding_config_sha256": str(teacher_contract["decoding_config_sha256"]),
        "nb11_input_contract_sha256": str(nb11_input_contract_sha256),
        "validation_input_contract_sha256": str(identity["manifest_sha256"]),
        "validation_audio_identity_sha256": str(identity["audio_identity_sha256"]),
        "calibration_artifact_sha256": calibration_artifact_sha256(calibration),
    }
    payload = {**proposal, "quality_score_proposal_sha256": sha256_json(proposal), "frozen": bool(frozen)}
    payload["quality_score_contract_sha256"] = sha256_json(payload)
    return payload


def verify_quality_score_contract(contract: Mapping[str, Any]) -> None:
    assert_self_hashed(contract, "quality_score_contract_sha256", "quality score contract")
    body = _without_hash(contract, "quality_score_contract_sha256")
    proposal = {k: v for k, v in body.items() if k not in ("quality_score_proposal_sha256", "frozen")}
    if sha256_json(proposal) != contract.get("quality_score_proposal_sha256"):
        raise QualityContractError("quality_score_proposal_sha256 does not verify")


def review_identity(quality_contract: Mapping[str, Any]) -> Dict[str, str]:
    """Directory identity for one review. A different NB11 input cannot collide."""
    payload = {
        "quality_score_proposal_sha256": str(quality_contract["quality_score_proposal_sha256"]),
        "nb11_input_contract_sha256": str(quality_contract["nb11_input_contract_sha256"]),
        "teacher_contract_sha256": str(quality_contract["teacher_contract_sha256"]),
        "d0_agreement_contract_sha256": str(quality_contract["d0_agreement_contract_sha256"]),
    }
    payload["review_identity_sha256"] = sha256_json(payload)
    return payload


def assert_publication_bindings(
    *,
    quality_contract: Mapping[str, Any],
    calibration: Mapping[str, Any],
    teacher_contract: Mapping[str, Any],
    d0_agreement_contract: Mapping[str, Any],
    nb11_input_contract: Mapping[str, Any],
) -> None:
    """Recompute every body and require the quality contract to name those hashes."""
    verify_quality_score_contract(quality_contract)
    assert_self_hashed(teacher_contract, "teacher_contract_sha256", "teacher contract")
    assert_self_hashed(d0_agreement_contract, "d0_agreement_contract_sha256", "D0 agreement contract")
    assert_self_hashed(nb11_input_contract, "nb11_input_contract_sha256", "NB11 input contract")
    if str(teacher_contract["teacher_contract_sha256"]) != str(quality_contract["teacher_contract_sha256"]):
        raise QualityContractError("teacher contract body is not the teacher bound into the quality contract")
    if str(d0_agreement_contract["d0_agreement_contract_sha256"]) != str(quality_contract["d0_agreement_contract_sha256"]):
        raise QualityContractError("D0 agreement contract body is not the contract bound into the quality contract")
    if str(nb11_input_contract["nb11_input_contract_sha256"]) != str(quality_contract["nb11_input_contract_sha256"]):
        raise QualityContractError("NB11 input contract body is not the input bound into the quality contract")
    if calibration_artifact_sha256(calibration) != str(quality_contract.get("calibration_artifact_sha256") or ""):
        raise QualityContractError("calibration artifact hash does not match the quality contract")
    identity = calibration.get("calibration_identity") or {}
    if str(identity.get("audio_identity_sha256") or "") != str(quality_contract.get("validation_audio_identity_sha256") or ""):
        raise QualityContractError("validation audio identity is not the one named by the quality contract")
    if str(identity.get("manifest_sha256") or "") != str(quality_contract.get("validation_input_contract_sha256") or ""):
        raise QualityContractError("validation manifest hash is not the one named by the quality contract")
    if str(quality_contract.get("quality_schema_version") or "") != QUALITY_SCHEMA_VERSION:
        raise QualityContractError("quality schema version drifted")


def freeze_quality_score_contract(contract: Mapping[str, Any], *, reviewed_proposal_sha256: str) -> Dict[str, Any]:
    """Freeze exactly the proposal a reviewer signed off by hash."""
    verify_quality_score_contract(contract)
    if not reviewed_proposal_sha256 or str(reviewed_proposal_sha256) != contract["quality_score_proposal_sha256"]:
        raise QualityContractError("reviewed proposal hash does not match this quality-score proposal")
    payload = {k: v for k, v in contract.items() if k != "quality_score_contract_sha256"}
    payload["frozen"] = True
    payload["quality_score_contract_sha256"] = sha256_json(payload)
    return payload


def sanity_from_contract(contract: Mapping[str, Any]) -> OutputSanityConfig:
    s = contract["eligibility"]["output_sanity"]
    return OutputSanityConfig(
        enabled=bool(s["enabled"]),
        min_target_source_char_ratio=s["min_target_source_char_ratio"],
        max_target_source_char_ratio=s["max_target_source_char_ratio"],
        lower_quantile=s["lower_quantile"],
        upper_quantile=s["upper_quantile"],
        derived_from=s["derived_from"],
    )


# --------------------------------------------------------------------------- #
# Scoring U_clean -> U'                                                        #
# --------------------------------------------------------------------------- #
def score_record(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    sanity: Optional[OutputSanityConfig] = None,
    allow_missing_score: bool = False,
) -> Dict[str, Any]:
    """Eligibility first (independent of Q), then Q for eligible rows only."""
    assert_no_reference_fields(record.keys())
    d0_enabled = bool(contract["eligibility"]["d0_enabled"])
    status = hard_eligibility(record, d0_enabled=d0_enabled, sanity=sanity or sanity_from_contract(contract))
    out = {c: record.get(c) for c in RAW_RECORD_COLUMNS}
    eligible = status == ELIGIBLE
    comps = _score_components(record, contract["normalization"]) if eligible else {}
    penalty = _penalty(record, contract["penalties"]) if eligible else None
    score = _combine(comps, contract["weights"], penalty) if eligible else None
    if eligible and score is None and not allow_missing_score:
        raise QualityContractError(f"eligible row has no quality score: {record.get('segment_uid')}")
    out.update({
        "hard_eligible": eligible,
        "eligibility_status": status,
        "exclusion_reason": None if eligible else status,
        "asr_confidence_norm": comps.get("asr_confidence"),
        "mt_confidence_norm": comps.get("mt_confidence"),
        "d0_agreement_norm": comps.get("d0_agreement"),
        "quality_penalty": penalty,
        "quality_score": score,
        "quality_score_contract_sha256": contract["quality_score_contract_sha256"],
        "quality_score_frozen": bool(contract["frozen"]),
    })
    return out


def iter_scored(
    raw_records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    allow_missing_score: bool = False,
) -> Iterator[Dict[str, Any]]:
    sanity = sanity_from_contract(contract)
    for rec in raw_records:
        yield score_record(rec, contract, sanity=sanity, allow_missing_score=allow_missing_score)


# --------------------------------------------------------------------------- #
# Status + gates                                                               #
# --------------------------------------------------------------------------- #
GATE_NAMES = (
    "nb11_input_valid",
    "nb11_input_unchanged",
    "teacher_contract_valid",
    "d0_agreement_contract_valid",
    "audio_identity_verified",
    "every_u_clean_segment_accounted",
    "no_duplicate_segment_uid",
    "no_g_test_access",
    "no_reference_text_in_u_artifacts",
    "quality_score_for_every_eligible_row",
    "u_prime_non_empty",
    "no_selection_or_training_in_nb12",
    "quality_score_contract_frozen",
)

# Gates the notebook can know before it streams U. The four pool gates are not
# in this tuple: a review derives them from the scored U records.
UPSTREAM_GATE_NAMES = (
    "nb11_input_valid",
    "nb11_input_unchanged",
    "teacher_contract_valid",
    "d0_agreement_contract_valid",
    "audio_identity_verified",
    "no_g_test_access",
    "no_reference_text_in_u_artifacts",
    "no_selection_or_training_in_nb12",
)
REVIEW_SAMPLE_ROWS = 8
_REVIEW_SAMPLE_FIELDS = (
    "segment_uid", "eligibility_status", "exclusion_reason", "quality_score",
    "mt_source_truncated", "mt_hit_max_length",
)


def derive_status(gates: Mapping[str, bool]) -> str:
    missing = [g for g in GATE_NAMES if g not in gates]
    if missing:
        return STATUS_FAIL
    others = [bool(gates[g]) for g in GATE_NAMES if g != "quality_score_contract_frozen"]
    if not all(others):
        return STATUS_FAIL
    return STATUS_SUCCESS if bool(gates["quality_score_contract_frozen"]) else STATUS_READY_FOR_REVIEW


def nb12_source_has_no_selection(paths: Sequence[Union[str, Path]]) -> bool:
    """Static check: no N-hour selection / treatment training entry points in NB12 sources."""
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_NB12_SYMBOLS:
            if f"def {symbol}" in text:
                return False
    return True


# --------------------------------------------------------------------------- #
# Publication                                                                  #
# --------------------------------------------------------------------------- #
_INT_COLUMNS = {"asr_token_count", "asr_n_frames", "mt_token_count", "asr_char_count", "asr_word_count",
                "pseudo_vi_char_count", "pseudo_vi_word_count"}
_BOOL_COLUMNS = {"asr_valid_encoding", "asr_empty", "mt_valid_encoding", "mt_empty", "mt_source_truncated",
                 "mt_hit_max_length", "d0_valid_encoding", "hard_eligible", "quality_score_frozen"}
_FLOAT_COLUMNS = {"duration_seconds", "vad_speech_fraction", "rms_dbfs", "peak_dbfs", "silence_fraction_energy",
                  "asr_mean_logprob", "asr_confidence_raw", "asr_frame_mean_logprob", "asr_blank_frame_fraction",
                  "mt_mean_logprob", "mt_confidence_raw", "teacher_d0_agreement_chrf", "teacher_d0_agreement_char_ratio",
                  "target_source_char_ratio", "target_source_word_ratio", "asr_confidence_norm", "mt_confidence_norm",
                  "d0_agreement_norm", "quality_penalty", "quality_score"}


def _arrow_schema(columns: Sequence[str]):
    import pyarrow as pa

    def typ(c):
        if c in _INT_COLUMNS:
            return pa.int64()
        if c in _BOOL_COLUMNS:
            return pa.bool_()
        if c in _FLOAT_COLUMNS:
            return pa.float64()
        return pa.string()

    return pa.schema([(c, typ(c)) for c in columns])


def _coerce(column: str, value: Any) -> Any:
    typed = column in _INT_COLUMNS or column in _BOOL_COLUMNS or column in _FLOAT_COLUMNS
    if value is None or (typed and isinstance(value, str) and value == ""):
        return None
    if column in _INT_COLUMNS:
        return int(value)
    if column in _BOOL_COLUMNS:
        return bool(value)
    if column in _FLOAT_COLUMNS:
        return _finite(value)
    return str(value)


class _ParquetSink:
    def __init__(self, path: Path, columns: Sequence[str], chunk: int = 2048):
        import pyarrow.parquet as pq

        self.columns = list(columns)
        self.schema = _arrow_schema(self.columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(str(path), self.schema)
        self.buffer: List[Dict[str, Any]] = []
        self.chunk = int(chunk)
        self.n = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.buffer.append({c: _coerce(c, row.get(c)) for c in self.columns})
        if len(self.buffer) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa

        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.n += len(self.buffer)
            self.buffer = []

    def close(self) -> None:
        self.flush()
        self.writer.close()


def write_pool_artifacts(
    directory: Union[str, Path],
    scored: Iterable[Mapping[str, Any]],
    *,
    expected_uids: Sequence[str],
) -> Dict[str, Any]:
    """Stream scored records into pseudo_label_manifest / u_prime_manifest / exclusions."""
    d = Path(directory)
    all_sink = _ParquetSink(d / "pseudo_label_manifest.parquet", PSEUDO_LABEL_MANIFEST_COLUMNS)
    u_sink = _ParquetSink(d / "u_prime_manifest.parquet", U_PRIME_COLUMNS)
    exclusions_path = d / "exclusions.jsonl"
    seen: List[str] = []
    reasons: Dict[str, int] = {}
    input_seconds = eligible_seconds = 0.0
    scores: List[float] = []
    missing_score = 0
    with open(exclusions_path, "w", encoding="utf-8") as exc_handle:
        try:
            for row in scored:
                assert_no_reference_fields(row.keys(), "published pseudo-label row")
                uid = str(row["segment_uid"])
                seen.append(uid)
                dur = float(row["duration_seconds"] or 0.0)
                input_seconds += dur
                all_sink.add(row)
                if row["hard_eligible"]:
                    if row["quality_score"] is None:
                        missing_score += 1
                    else:
                        scores.append(float(row["quality_score"]))
                    eligible_seconds += dur
                    u_sink.add(row)
                else:
                    reasons[row["exclusion_reason"]] = reasons.get(row["exclusion_reason"], 0) + 1
                    exc_handle.write(json.dumps({c: row.get(c) for c in EXCLUSION_COLUMNS}, ensure_ascii=False, sort_keys=True) + "\n")
        finally:
            all_sink.close()
            u_sink.close()
    arr = np.asarray(scores, dtype=np.float64)
    stats = {}
    if arr.size:
        stats = {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
                 "p10": float(np.quantile(arr, 0.1)), "p50": float(np.quantile(arr, 0.5)), "p90": float(np.quantile(arr, 0.9))}
    return {
        "n_input_segments": len(seen),
        "n_u_prime": int(u_sink.n),
        "n_excluded": int(sum(reasons.values())),
        "exclusions_by_reason": dict(sorted(reasons.items())),
        "input_audio_hours": round(input_seconds / 3600.0, 6),
        "u_prime_audio_hours": round(eligible_seconds / 3600.0, 6),
        "quality_score_stats": stats,
        "n_eligible_missing_score": int(missing_score),
        "accounted": seen == [str(u) for u in expected_uids],
        "no_duplicate_uid": len(set(seen)) == len(seen),
    }


NB12_ARTIFACT_FILES = (
    "pseudo_label_manifest.parquet",
    "u_prime_manifest.parquet",
    "exclusions.jsonl",
    "quality_calibration.json",
    "quality_score_contract.json",
    "teacher_contract.json",
    "d0_agreement_contract.json",
    "nb11_input_contract.json",
    "data_access_ledger.json",
    "contract.json",
    "summary.json",
)


def calibration_public_view(calibration: Mapping[str, Any]) -> Dict[str, Any]:
    """Calibration diagnostics are aggregates only; refuse anything that looks per-record."""
    blob = json.dumps(calibration, ensure_ascii=False, sort_keys=True)
    for name in FORBIDDEN_REFERENCE_FIELDS:
        if f'"{name}"' in blob:
            raise QualityContractError(f"calibration payload carries reference field {name}")
    return dict(calibration)


def publish_u_prime_generation(
    out_dir: Union[str, Path],
    *,
    raw_records: Iterable[Mapping[str, Any]],
    expected_uids: Sequence[str],
    quality_contract: Mapping[str, Any],
    calibration: Mapping[str, Any],
    teacher_contract: Mapping[str, Any],
    d0_agreement_contract: Mapping[str, Any],
    nb11_input_contract: Mapping[str, Any],
    ledger: DataAccessLedger,
    upstream_gates: Mapping[str, bool],
) -> Dict[str, Any]:
    """Publish U' as an atomic generation. Refuses unless every gate (incl. frozen Q) passes."""
    if not quality_contract.get("frozen"):
        raise QualityContractError("refusing to publish U' with an unfrozen quality-score contract")
    assert_publication_bindings(
        quality_contract=quality_contract, calibration=calibration, teacher_contract=teacher_contract,
        d0_agreement_contract=d0_agreement_contract, nb11_input_contract=nb11_input_contract,
    )
    out = Path(out_dir)
    assert_not_g_test_path(out)
    staged = stage_generation(out)
    gen_dir: Path = staged["staging_dir"]
    pool = write_pool_artifacts(gen_dir, iter_scored(raw_records, quality_contract), expected_uids=expected_uids)
    gates = dict(upstream_gates)
    gates.update({
        "every_u_clean_segment_accounted": bool(pool["accounted"]),
        "no_duplicate_segment_uid": bool(pool["no_duplicate_uid"]),
        "no_g_test_access": not ledger.g_test_accessed,
        "quality_score_for_every_eligible_row": pool["n_eligible_missing_score"] == 0,
        "u_prime_non_empty": pool["n_u_prime"] > 0,
        "quality_score_contract_frozen": bool(quality_contract["frozen"]),
    })
    status = derive_status(gates)
    if status != STATUS_SUCCESS:
        raise QualityContractError(f"NB12 gates failed; generation left unpublished: {sorted(g for g in GATE_NAMES if not gates.get(g))}")
    write_json(gen_dir / "quality_calibration.json", calibration_public_view(calibration))
    write_json(gen_dir / "quality_score_contract.json", dict(quality_contract))
    write_json(gen_dir / "teacher_contract.json", dict(teacher_contract))
    write_json(gen_dir / "d0_agreement_contract.json", dict(d0_agreement_contract))
    write_json(gen_dir / "nb11_input_contract.json", dict(nb11_input_contract))
    write_json(gen_dir / "data_access_ledger.json", ledger.as_dict())
    assert_publication_bindings(
        quality_contract=json.loads((gen_dir / "quality_score_contract.json").read_text(encoding="utf-8")),
        calibration=json.loads((gen_dir / "quality_calibration.json").read_text(encoding="utf-8")),
        teacher_contract=json.loads((gen_dir / "teacher_contract.json").read_text(encoding="utf-8")),
        d0_agreement_contract=json.loads((gen_dir / "d0_agreement_contract.json").read_text(encoding="utf-8")),
        nb11_input_contract=json.loads((gen_dir / "nb11_input_contract.json").read_text(encoding="utf-8")),
    )
    contract = {
        "schema_version": PSEUDO_SCHEMA_VERSION,
        "nb11_input_contract_sha256": nb11_input_contract["nb11_input_contract_sha256"],
        "nb11_generation_id": nb11_input_contract["generation_id"],
        "teacher_contract_sha256": teacher_contract["teacher_contract_sha256"],
        "d0_agreement_contract_sha256": d0_agreement_contract["d0_agreement_contract_sha256"],
        "decoding_config_sha256": teacher_contract["decoding_config_sha256"],
        "validation_input_contract_sha256": quality_contract["validation_input_contract_sha256"],
        "validation_audio_identity_sha256": quality_contract["validation_audio_identity_sha256"],
        "calibration_artifact_sha256": quality_contract["calibration_artifact_sha256"],
        "quality_score_contract_sha256": quality_contract["quality_score_contract_sha256"],
        "quality_score_proposal_sha256": quality_contract["quality_score_proposal_sha256"],
        "u_prime_is_common_pool_for_all_rq2_treatments": True,
    }
    contract["nb12_contract_sha256"] = sha256_json(contract)
    write_json(gen_dir / "contract.json", contract)
    summary = {
        "status": status,
        "schema_version": PSEUDO_SCHEMA_VERSION,
        **{k: contract[k] for k in contract},
        "quality_score_frozen": True,
        "d0_agreement_enabled": bool(d0_agreement_contract.get("enabled")),
        "validation_audio_identity_sha256": quality_contract["validation_audio_identity_sha256"],
        "quality_calibration_sha256": quality_contract["calibration_artifact_sha256"],
        "u_prime_manifest_sha256": sha256_file(gen_dir / "u_prime_manifest.parquet"),
        "generation_id": str(staged["generation_id"]),
        **{k: pool[k] for k in ("n_input_segments", "n_u_prime", "n_excluded", "exclusions_by_reason",
                                "input_audio_hours", "u_prime_audio_hours", "quality_score_stats")},
        "gates": {g: bool(gates[g]) for g in GATE_NAMES},
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(gen_dir / "summary.json", summary)
    verified = finalize_generation(out, staged, NB12_ARTIFACT_FILES, allow_empty=("exclusions.jsonl",))
    return {"summary": summary, "generation": verified}


def review_pool_diagnostics(
    scored: Iterable[Mapping[str, Any]],
    expected_uids: Sequence[str],
    *,
    d0_enabled: bool,
) -> Dict[str, Any]:
    """Aggregate a full scored pass over U. No selection and no per-row text."""
    seen: List[str] = []
    reasons: Dict[str, int] = {}
    input_seconds = eligible_seconds = 0.0
    scores: List[float] = []
    missing_score = n_eligible = truncated = hit_max = d0_available = 0
    sample: List[Dict[str, Any]] = []
    for row in scored:
        assert_no_reference_fields(row.keys(), "review scored row")
        uid = str(row["segment_uid"])
        seen.append(uid)
        dur = float(row.get("duration_seconds") or 0.0)
        input_seconds += dur
        if bool(row.get("mt_source_truncated")):
            truncated += 1
        if bool(row.get("mt_hit_max_length")):
            hit_max += 1
        if d0_enabled and row.get("d0_status") == STATUS_OK and _finite(row.get("teacher_d0_agreement_chrf")) is not None:
            d0_available += 1
        if row["hard_eligible"]:
            n_eligible += 1
            if row.get("quality_score") is None:
                missing_score += 1
            else:
                scores.append(float(row["quality_score"]))
            eligible_seconds += dur
        else:
            reason = str(row.get("exclusion_reason") or "")
            reasons[reason] = reasons.get(reason, 0) + 1
        if len(sample) < REVIEW_SAMPLE_ROWS:
            sample.append({field: row.get(field) for field in _REVIEW_SAMPLE_FIELDS})
    n = len(seen)
    arr = np.asarray(scores, dtype=np.float64)
    stats = {}
    if arr.size:
        stats = {
            "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
            "p10": float(np.quantile(arr, 0.1)), "p50": float(np.quantile(arr, 0.5)),
            "p90": float(np.quantile(arr, 0.9)),
        }
    out = {
        "n_input_segments": n,
        "n_hard_eligible": int(n_eligible),
        "n_u_prime": int(n_eligible),
        "n_excluded": int(sum(reasons.values())),
        "exclusions_by_reason": dict(sorted(reasons.items())),
        "input_audio_hours": round(input_seconds / 3600.0, 6),
        "prospective_u_prime_audio_hours": round(eligible_seconds / 3600.0, 6),
        "quality_score_stats": stats,
        "n_eligible_missing_score": int(missing_score),
        "accounted": seen == [str(u) for u in expected_uids],
        "no_duplicate_uid": len(set(seen)) == n,
        "mt_source_truncated_count": int(truncated),
        "mt_source_truncated_rate": (truncated / n) if n else 0.0,
        "mt_hit_max_length_count": int(hit_max),
        "mt_hit_max_length_rate": (hit_max / n) if n else 0.0,
        "diagnostic_sample": sample,
    }
    if d0_enabled:
        out["d0_agreement_available_count"] = int(d0_available)
        out["d0_agreement_available_rate"] = (d0_available / n) if n else 0.0
    return out


def review_gates(upstream_gates: Mapping[str, bool], diagnostics: Mapping[str, Any]) -> Dict[str, bool]:
    """Upstream gates stay as supplied. Pool gates come only from ``diagnostics``."""
    gates = {name: bool(upstream_gates.get(name, False)) for name in UPSTREAM_GATE_NAMES}
    gates["every_u_clean_segment_accounted"] = bool(diagnostics["accounted"])
    gates["no_duplicate_segment_uid"] = bool(diagnostics["no_duplicate_uid"])
    gates["quality_score_for_every_eligible_row"] = diagnostics["n_eligible_missing_score"] == 0
    gates["u_prime_non_empty"] = int(diagnostics["n_hard_eligible"]) > 0
    gates["quality_score_contract_frozen"] = False
    return gates


def write_review_bundle(
    out_dir: Union[str, Path],
    *,
    raw_records: Iterable[Mapping[str, Any]],
    expected_uids: Sequence[str],
    quality_contract: Mapping[str, Any],
    calibration: Mapping[str, Any],
    teacher_contract: Mapping[str, Any],
    d0_agreement_contract: Mapping[str, Any],
    nb11_input_contract: Mapping[str, Any],
    ledger: DataAccessLedger,
    upstream_gates: Mapping[str, bool],
) -> Dict[str, Any]:
    """Score every raw U row with the draft contract and write review diagnostics.

    Does not stage a generation and does not read or write ``CURRENT``. The
    review directory is bound to the proposal and the NB11 input together, and
    a completed review directory is not overwritten.
    """
    if quality_contract.get("frozen"):
        raise QualityContractError("a frozen contract is published with publish_u_prime_generation, not as a review bundle")
    assert_publication_bindings(
        quality_contract=quality_contract, calibration=calibration, teacher_contract=teacher_contract,
        d0_agreement_contract=d0_agreement_contract, nb11_input_contract=nb11_input_contract,
    )
    d0_enabled = bool(quality_contract["eligibility"]["d0_enabled"])
    identity = review_identity(quality_contract)
    review = assert_not_g_test_path(Path(out_dir)) / "review" / identity["review_identity_sha256"]
    if (review / "summary.json").is_file():
        raise QualityContractError("review bundle already exists and is immutable")
    diagnostics = review_pool_diagnostics(
        iter_scored(raw_records, quality_contract, allow_missing_score=True),
        expected_uids,
        d0_enabled=d0_enabled,
    )
    gates = review_gates(upstream_gates, diagnostics)
    if ledger.g_test_accessed:
        gates["no_g_test_access"] = False
    status = derive_status(gates)
    if status == STATUS_SUCCESS:
        raise QualityContractError("a frozen contract is published with publish_u_prime_generation, not as a review bundle")
    summary = {
        "status": status,
        "quality_score_frozen": False,
        "published_generation": False,
        "review_identity_sha256": identity["review_identity_sha256"],
        "quality_score_proposal_sha256": quality_contract["quality_score_proposal_sha256"],
        "nb11_input_contract_sha256": nb11_input_contract["nb11_input_contract_sha256"],
        "teacher_contract_sha256": teacher_contract["teacher_contract_sha256"],
        "d0_agreement_contract_sha256": d0_agreement_contract["d0_agreement_contract_sha256"],
        "validation_input_contract_sha256": quality_contract["validation_input_contract_sha256"],
        "validation_audio_identity_sha256": quality_contract["validation_audio_identity_sha256"],
        "quality_calibration_sha256": quality_contract["calibration_artifact_sha256"],
        "quality_score_contract_sha256": quality_contract["quality_score_contract_sha256"],
        "gates": {name: bool(gates[name]) for name in GATE_NAMES},
        "pool": diagnostics,
    }
    write_json(review / "quality_calibration.json", calibration_public_view(calibration))
    write_json(review / "quality_score_contract_proposal.json", dict(quality_contract))
    write_json(review / "teacher_contract.json", dict(teacher_contract))
    write_json(review / "d0_agreement_contract.json", dict(d0_agreement_contract))
    write_json(review / "nb11_input_contract.json", dict(nb11_input_contract))
    write_json(review / "data_access_ledger.json", ledger.as_dict())
    write_json(review / "pool_diagnostics.json", diagnostics)
    write_json(review / "summary.json", summary)
    return {"review_dir": str(review), "summary": summary}
