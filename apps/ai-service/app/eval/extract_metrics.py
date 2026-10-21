"""Deterministic extract.v1 scoring. Gold labels are authoritative.

STRICT and RELAXED slot metrics are computed separately. Relaxed matching
never overwrites strict fields. Precision/recall are N/A when the
denominator is zero.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value)).strip().lower()
    return " ".join(text.split())


def values_equal_strict(expected: Any, got: Any, slot_type: str) -> bool:
    if slot_type == "boolean":
        return expected is got and isinstance(got, bool)
    if slot_type == "enum":
        return expected == got
    if slot_type == "number":
        try:
            return math.isclose(float(expected), float(got), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return normalize_text(expected) == normalize_text(got)


def values_equal_relaxed(expected: Any, got: Any, slot_type: str) -> bool:
    if slot_type != "string":
        return values_equal_strict(expected, got, slot_type)
    exp = normalize_text(expected)
    got_n = normalize_text(got)
    if not exp or not got_n:
        return False
    return exp in got_n or got_n in exp


def _unique_predicted(predicted: dict[str, Any] | list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    if isinstance(predicted, dict):
        return dict(predicted), 0
    out: dict[str, Any] = {}
    extras = 0
    for item in predicted:
        key = item["key"]
        if key in out:
            extras += 1
            continue
        out[key] = item.get("value")
    return out, extras


def score_intent(expected_code: str | None, predicted_code: str | None) -> bool:
    return expected_code == predicted_code


def score_slots(
    expected: dict[str, Any],
    predicted: dict[str, Any] | list[dict[str, Any]],
    slot_types: dict[str, str],
    *,
    relaxed: bool = False,
) -> dict[str, Any]:
    pred, extra_dupes = _unique_predicted(predicted)
    eq = values_equal_relaxed if relaxed else values_equal_strict
    tp = fp = fn = 0
    details: list[dict[str, Any]] = []

    for key, exp in expected.items():
        stype = slot_types.get(key, "string")
        if key not in pred:
            fn += 1
            details.append({"key": key, "expected": exp, "got": None, "status": "FN"})
            continue
        got = pred[key]
        if eq(exp, got, stype):
            tp += 1
            details.append({"key": key, "expected": exp, "got": got, "status": "TP"})
        else:
            fp += 1
            fn += 1
            details.append({"key": key, "expected": exp, "got": got, "status": "FP+FN"})

    for key, got in pred.items():
        if key in expected:
            continue
        fp += 1
        details.append({"key": key, "expected": None, "got": got, "status": "FP"})

    fp += extra_dupes
    if extra_dupes:
        details.append({"key": "<duplicate>", "expected": None, "got": extra_dupes, "status": "FP"})

    def _rate(num: int, den: int) -> float | None:
        if den == 0:
            return None
        return round(num / den, 4)

    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    elif precision is not None and recall is not None:
        f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": fp == 0 and fn == 0,
        "duplicate_predicted": extra_dupes,
        "details": details,
        "mode": "relaxed" if relaxed else "strict",
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases)
    intent_ok = sum(1 for c in cases if c["intent_ok"])
    schema_ok = sum(1 for c in cases if c.get("schema_valid"))
    strict_exact = sum(1 for c in cases if c["strict"]["exact"] and c["intent_ok"])
    relaxed_exact = sum(1 for c in cases if c["relaxed"]["exact"] and c["intent_ok"])
    tp = sum(c["strict"]["tp"] for c in cases)
    fp = sum(c["strict"]["fp"] for c in cases)
    fn = sum(c["strict"]["fn"] for c in cases)

    def _rate(num: int, den: int) -> float | None:
        if den == 0:
            return None
        return round(num / den, 4)

    prec = _rate(tp, tp + fp)
    rec = _rate(tp, tp + fn)
    f1 = None
    if prec is not None and rec is not None and prec + rec > 0:
        f1 = round(2 * prec * rec / (prec + rec), 4)
    elif prec is not None and rec is not None:
        f1 = 0.0

    return {
        "n": n,
        "intent_correct": intent_ok,
        "intent_incorrect": n - intent_ok,
        "intent_exact_accuracy": _rate(intent_ok, n),
        "schema_valid": schema_ok,
        "schema_validity_rate": _rate(schema_ok, n),
        "strict_slot_micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "exact_case_rate": _rate(sum(1 for c in cases if c["strict"]["exact"]), n),
        },
        "case_pass_strict_intent_and_slots": {
            "numerator": strict_exact,
            "denominator": n,
            "rate": _rate(strict_exact, n),
        },
        "case_pass_relaxed_intent_and_slots": {
            "numerator": relaxed_exact,
            "denominator": n,
            "rate": _rate(relaxed_exact, n),
        },
        "note": "N/A means undefined (zero denominator). Relaxed string matching does not replace strict metrics.",
    }


def score_case(
    expected_code: str | None,
    predicted_code: str | None,
    expected_slots: dict[str, Any],
    predicted_slots: dict[str, Any] | list[dict[str, Any]],
    slot_types: dict[str, str],
    *,
    schema_valid: bool = True,
) -> dict[str, Any]:
    intent_ok = score_intent(expected_code, predicted_code)
    strict = score_slots(expected_slots, predicted_slots, slot_types, relaxed=False)
    relaxed = score_slots(expected_slots, predicted_slots, slot_types, relaxed=True)
    return {
        "intent_ok": intent_ok,
        "schema_valid": schema_valid,
        "strict": strict,
        "relaxed": relaxed,
        "case_pass_strict": bool(intent_ok and strict["exact"] and schema_valid),
        "case_pass_relaxed": bool(intent_ok and relaxed["exact"] and schema_valid),
    }
