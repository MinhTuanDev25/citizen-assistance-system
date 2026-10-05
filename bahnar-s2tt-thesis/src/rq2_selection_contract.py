"""RQ2 Notebook 13 contracts.

NB13 reads the immutable NB12 generation named by
``artifacts/rq2/pseudo_labels/CURRENT`` and selects two subsets of that one
frozen U′. It does not run the teacher, recalibrate quality, open G_test, or
train.

NB12 seals ``u_prime_manifest.parquet`` and ``nb12_contract_sha256``, but it
does not seal a separate ordered-UID hash. NB13 recomputes
``u_prime_ordered_uid_sha256`` from the hash-verified parquet row order and
binds that value into the selection contract.

NB12 also does not forbid two segment UIDs from sharing a PCM hash. NB13
reports that count and does not drop rows for it.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import (
    NB11_RELATIVE_DIR,
    PSEUDO_RELATIVE_DIR,
    PSEUDO_SCHEMA_VERSION,
    STATUS_SUCCESS,
    assert_not_g_test_path,
    portable_relative,
    read_current_generation_id,
    resolve_nb11_generation,
    resolve_nb11_input,
    verify_generation,
)
from src.rq2_quality import U_PRIME_COLUMNS

SELECTION_RELATIVE_DIR = "artifacts/rq2/selection"
SELECTION_SCHEMA_VERSION = "rq2-selection-1.0"
SELECTION_CONTRACT_VERSION = "rq2_same_budget_selection_contract_v1"
SELECTION_CODE_VERSION = "rq2_same_budget_selection_v1"
STATUS_SELECTION_FROZEN = "SUCCESS_RQ2_SAME_BUDGET_SELECTION_FROZEN"

RANDOM_ALGORITHM_VERSION = "rq2_random_sha256_uid_v1"
QUALITY_ALGORITHM_VERSION = "rq2_quality_desc_uid_asc_v1"
DURATION_BUDGET_ALGORITHM_VERSION = "rq2_duration_budget_greedy_v1"
SELECTION_TIE_BREAK = {
    "d_random": ["sha256(seed + newline + segment_uid) ascending", "segment_uid ascending"],
    "d_quality": ["quality_score descending", "segment_uid ascending"],
}
SELECTION_DURATION_RULE = (
    "Walk the arm order. Take a segment when used+duration <= target. "
    "Otherwise leave it out and continue. Do not cut audio and do not reorder to pack tighter."
)
SELECTION_UNDERFILL_RULE = (
    "Unused budget must be strictly smaller than every segment that was not selected, "
    "unless the pool is exhausted. The bound is the next segment's own duration."
)

ARM_RANDOM = "d_random"
ARM_QUALITY = "d_quality"
PSEUDO_LABEL_COLUMN = "pseudo_vi_norm"
DURATION_COLUMN = "duration_seconds"
QUALITY_COLUMN = "quality_score"
UID_COLUMN = "segment_uid"

SELECTION_AUDIT_COLUMNS = (
    "selection_arm",
    "selection_rank",
    "selection_key",
    "cumulative_duration_seconds",
)
MANIFEST_COLUMNS = list(U_PRIME_COLUMNS) + list(SELECTION_AUDIT_COLUMNS)

def _forbidden_path_needles() -> tuple:
    """Built from fragments so this source file contains no complete forbidden literal."""
    return ("rq1_" + "test", "frozen_" + "test", "g_" + "test")
_FORBIDDEN_IMPORT_ROOTS = ("sacrebleu", "jiwer", "transformers", "torch")
_FORBIDDEN_CALLS = frozenset({
    "Trainer",
    "sentence_bleu",
    "corpus_bleu",
    "bleu",
    "chrf",
    "sentence_chrf",
    "corpus_chrf",
    "sentence_chrfpp",
    "cer",
    "wer",
    "run_peak_safe_teacher_inference",
    "run_asr_stage",
    "run_mt_stage",
    "run_d0_stage",
    "calibrate_quality_score",
    "load_g_validation",
    "materialize_validation_audio",
    "generate",
})


class SelectionIntegrityError(RuntimeError):
    """A frozen U′ row or artifact is not usable. The UID is part of the message."""

    def __init__(self, reason: str, uid: Optional[str] = None):
        self.uid = uid
        self.reason = reason
        super().__init__(f"{uid}: {reason}" if uid else reason)


class SelectionPolicyError(RuntimeError):
    """The selection budget or freeze flag is not fixed."""


@dataclass(frozen=True)
class FrozenUPrime:
    """Hash-verified U′ rows. ``identity`` contains no absolute path."""

    generation_id: str
    rows: tuple
    identity: Dict[str, Any]


def selection_source_paths() -> List[Path]:
    root = Path(__file__).resolve().parents[1]
    return [root / "src" / "rq2_selection_contract.py", root / "src" / "rq2_selection.py"]


def ordered_uid_sha256(uids: Sequence[str]) -> str:
    return sha256_json([str(uid) for uid in uids])


def budget_seconds_from_hours(hours: Any) -> float:
    """Convert an explicit hour budget to seconds. ``None`` is refused."""
    if hours is None or isinstance(hours, bool):
        raise SelectionPolicyError("SELECTION_BUDGET_HOURS is not fixed")
    try:
        seconds = float(hours) * 3600.0
    except (TypeError, ValueError) as exc:
        raise SelectionPolicyError("SELECTION_BUDGET_HOURS is not a number") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise SelectionPolicyError("selection budget must be finite and > 0 hours")
    return seconds


def assert_nb13_output_dir(out_dir: Union[str, Path], project_root: Union[str, Path]) -> Path:
    """NB13 writes only ``artifacts/rq2/selection``, never into NB11, NB12, or RQ1."""
    root = Path(project_root).resolve()
    out = Path(out_dir).resolve()
    expected = (root / SELECTION_RELATIVE_DIR).resolve()
    if out != expected:
        raise SelectionPolicyError(f"NB13 output dir must be {SELECTION_RELATIVE_DIR}")
    for protected in (
        root / NB11_RELATIVE_DIR,
        root / PSEUDO_RELATIVE_DIR,
        root / "artifacts" / "rq1",
        root / "data" / "manifests",
    ):
        try:
            out.relative_to(protected.resolve())
        except ValueError:
            continue
        raise SelectionPolicyError(f"NB13 output dir is inside a protected tree: {protected}")
    assert_not_g_test_path(out)
    return out


def _without(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != key}


def verify_nb12_contract_hash(contract: Mapping[str, Any]) -> None:
    if sha256_json(_without(contract, "nb12_contract_sha256")) != str(contract.get("nb12_contract_sha256") or ""):
        raise SelectionIntegrityError("NB12 contract hash does not recompute from its body")


def verify_stored_hash(payload: Mapping[str, Any], key: str, label: str) -> None:
    if sha256_json(_without(payload, key)) != str(payload.get(key) or ""):
        raise SelectionIntegrityError(f"{label} hash does not recompute from its body")


def build_selection_contract(
    identity: Mapping[str, Any],
    *,
    selection_budget_hours: float,
    random_seed: int,
    random_algorithm: str = RANDOM_ALGORITHM_VERSION,
    quality_algorithm: str = QUALITY_ALGORITHM_VERSION,
    duration_budget_algorithm: str = DURATION_BUDGET_ALGORITHM_VERSION,
) -> Dict[str, Any]:
    """Canonical selection contract. Absolute paths are not part of the hash."""
    seconds = budget_seconds_from_hours(selection_budget_hours)
    if type(random_seed) is not int:
        raise SelectionPolicyError("random_seed must be an int")
    for key in (
        "nb11_input_contract_sha256",
        "nb12_contract_sha256",
        "u_prime_manifest_sha256",
        "u_prime_ordered_uid_sha256",
    ):
        if len(str(identity.get(key) or "")) != 64:
            raise SelectionIntegrityError(f"selection contract is missing {key}")
    proposal = {
        "contract_version": SELECTION_CONTRACT_VERSION,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "code_version": SELECTION_CODE_VERSION,
        "nb11_generation_id": str(identity["nb11_generation_id"]),
        "nb11_input_contract_sha256": str(identity["nb11_input_contract_sha256"]),
        "nb12_generation_id": str(identity["nb12_generation_id"]),
        "nb12_contract_sha256": str(identity["nb12_contract_sha256"]),
        "nb12_schema_version": PSEUDO_SCHEMA_VERSION,
        "u_prime_manifest_sha256": str(identity["u_prime_manifest_sha256"]),
        "u_prime_ordered_uid_sha256": str(identity["u_prime_ordered_uid_sha256"]),
        "u_prime_columns": list(U_PRIME_COLUMNS),
        "pseudo_label_column": PSEUDO_LABEL_COLUMN,
        "selection_budget_hours": float(selection_budget_hours),
        "selection_budget_seconds": seconds,
        "random_seed": int(random_seed),
        "random_algorithm": str(random_algorithm),
        "quality_algorithm": str(quality_algorithm),
        "duration_budget_algorithm": str(duration_budget_algorithm),
        "tie_break": {
            "d_random": list(SELECTION_TIE_BREAK["d_random"]),
            "d_quality": list(SELECTION_TIE_BREAK["d_quality"]),
        },
        "duration_rule": SELECTION_DURATION_RULE,
        "underfill_rule": SELECTION_UNDERFILL_RULE,
        "same_budget_both_arms": True,
        "quality_score_used_by_random": False,
        "audio_cut_allowed": False,
    }
    proposal["selection_contract_sha256"] = sha256_json(proposal)
    return proposal


def verify_selection_contract(contract: Mapping[str, Any]) -> None:
    """Self-hash, then the frozen algorithm versions and scientific invariants."""
    verify_stored_hash(contract, "selection_contract_sha256", "NB13 selection contract")
    expected = {
        "contract_version": SELECTION_CONTRACT_VERSION,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "code_version": SELECTION_CODE_VERSION,
        "random_algorithm": RANDOM_ALGORITHM_VERSION,
        "quality_algorithm": QUALITY_ALGORITHM_VERSION,
        "duration_budget_algorithm": DURATION_BUDGET_ALGORITHM_VERSION,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise SelectionIntegrityError(f"selection contract {key} is {contract.get(key)!r}, not {value!r}")
    if contract.get("same_budget_both_arms") is not True:
        raise SelectionIntegrityError("selection contract same_budget_both_arms is not true")
    if contract.get("quality_score_used_by_random") is not False:
        raise SelectionIntegrityError("selection contract quality_score_used_by_random is not false")
    if contract.get("audio_cut_allowed") is not False:
        raise SelectionIntegrityError("selection contract audio_cut_allowed is not false")
    for key in ("nb11_generation_id", "nb12_generation_id"):
        if not str(contract.get(key) or "").strip():
            raise SelectionIntegrityError(f"selection contract is missing {key}")
    for key in (
        "nb11_input_contract_sha256",
        "nb12_contract_sha256",
        "u_prime_manifest_sha256",
        "u_prime_ordered_uid_sha256",
    ):
        if len(str(contract.get(key) or "")) != 64:
            raise SelectionIntegrityError(f"selection contract is missing {key}")
    if type(contract.get("random_seed")) is not int:
        raise SelectionIntegrityError("selection contract random_seed is not an int")
    if contract.get("nb12_schema_version") != PSEUDO_SCHEMA_VERSION:
        raise SelectionIntegrityError(
            f"selection contract nb12_schema_version is {contract.get('nb12_schema_version')!r}"
        )
    if list(contract.get("u_prime_columns") or []) != list(U_PRIME_COLUMNS):
        raise SelectionIntegrityError("selection contract u_prime_columns does not match the NB12 schema")
    if contract.get("pseudo_label_column") != PSEUDO_LABEL_COLUMN:
        raise SelectionIntegrityError(
            f"selection contract pseudo_label_column is {contract.get('pseudo_label_column')!r}"
        )
    if contract.get("tie_break") != SELECTION_TIE_BREAK:
        raise SelectionIntegrityError("selection contract tie_break does not match the frozen policy")
    if contract.get("duration_rule") != SELECTION_DURATION_RULE:
        raise SelectionIntegrityError("selection contract duration_rule does not match the frozen policy")
    if contract.get("underfill_rule") != SELECTION_UNDERFILL_RULE:
        raise SelectionIntegrityError("selection contract underfill_rule does not match the frozen policy")
    hours = contract.get("selection_budget_hours")
    seconds = contract.get("selection_budget_seconds")
    if isinstance(hours, bool) or _finite_float(hours) is None or _finite_float(hours) <= 0.0:
        raise SelectionIntegrityError("selection contract selection_budget_hours must be finite and > 0")
    if isinstance(seconds, bool) or _finite_float(seconds) is None or _finite_float(seconds) <= 0.0:
        raise SelectionIntegrityError("selection contract selection_budget_seconds must be finite and > 0")
    expected_seconds = budget_seconds_from_hours(hours)
    if float(seconds) != expected_seconds:
        raise SelectionIntegrityError(
            "selection contract selection_budget_seconds does not match selection_budget_hours"
        )


def _finite_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_u_prime_row(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    """Coerce one parquet row. Malformed values become an integrity error naming the UID."""
    missing = [column for column in U_PRIME_COLUMNS if column not in raw]
    uid = str(raw.get(UID_COLUMN) or "")
    if missing:
        raise SelectionIntegrityError(f"missing columns {missing}", uid or f"row-{index}")
    if not uid:
        raise SelectionIntegrityError("empty segment_uid", f"row-{index}")
    row: Dict[str, Any] = {}
    for column in U_PRIME_COLUMNS:
        value = raw.get(column)
        if column in _FLOAT_U_PRIME_COLUMNS:
            row[column] = None if value is None else _finite_float(value)
            if value is not None and row[column] is None:
                raise SelectionIntegrityError(f"{column} is not finite", uid)
        else:
            row[column] = None if value is None else str(value)
    duration = row[DURATION_COLUMN]
    if duration is None or duration <= 0.0:
        raise SelectionIntegrityError("duration_seconds must be finite and > 0", uid)
    if row[QUALITY_COLUMN] is None:
        raise SelectionIntegrityError("quality_score is not finite", uid)
    text = row.get(PSEUDO_LABEL_COLUMN)
    if text is None or str(text).strip() == "":
        raise SelectionIntegrityError("pseudo_vi_norm is missing or empty", uid)
    row[PSEUDO_LABEL_COLUMN] = str(text)
    if not portable_relative(str(row.get("segment_local_path") or "")):
        raise SelectionIntegrityError("segment_local_path is not a portable project-relative path", uid)
    for hash_column in ("segment_pcm16_sha256", "segment_wav_sha256"):
        if len(str(row.get(hash_column) or "")) != 64:
            raise SelectionIntegrityError(f"{hash_column} is not a sha256", uid)
    return row


_FLOAT_U_PRIME_COLUMNS = frozenset({
    "duration_seconds",
    "asr_mean_logprob",
    "mt_mean_logprob",
    "teacher_d0_agreement_chrf",
    "asr_confidence_norm",
    "mt_confidence_norm",
    "d0_agreement_norm",
    "quality_penalty",
    "quality_score",
})


def validate_u_prime_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Fail closed on the first malformed row. Nothing is dropped."""
    if not rows:
        raise SelectionIntegrityError("U′ is empty")
    normalized = [normalize_u_prime_row(row, index) for index, row in enumerate(rows)]
    seen = set()
    for row in normalized:
        uid = row[UID_COLUMN]
        if uid in seen:
            raise SelectionIntegrityError("duplicate segment_uid", uid)
        seen.add(uid)
    return normalized


def load_u_prime_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_path = assert_not_g_test_path(path)
    table = pq.read_table(parquet_path)
    if list(table.column_names) != list(U_PRIME_COLUMNS):
        raise SelectionIntegrityError(
            f"U′ columns {list(table.column_names)} do not match the NB12 schema"
        )
    return validate_u_prime_rows(table.to_pylist())


def resolve_frozen_u_prime(
    project_root: Union[str, Path],
    *,
    generation_id: Optional[str] = None,
) -> FrozenUPrime:
    """Lock NB11 and one NB12 generation. Fail closed on any mismatch.

    ``generation_id`` omitted uses NB12 ``CURRENT`` and NB11 ``CURRENT``. A
    caller that already has a selection contract passes the pinned NB12 id.
    That path then opens the NB11 generation named by the NB12 contract, not
    whichever generation NB11 ``CURRENT`` names now.
    """
    root = Path(project_root)
    pseudo_dir = assert_not_g_test_path(root / PSEUDO_RELATIVE_DIR)
    verified = verify_generation(pseudo_dir, generation_id)
    gen_id = str(verified["generation_id"])
    gen_dir = pseudo_dir / "generations" / gen_id
    contract = _read_json(gen_dir / "contract.json")
    summary = _read_json(gen_dir / "summary.json")
    pinned_nb11_id = str(contract.get("nb11_generation_id") or "").strip()
    if not pinned_nb11_id:
        raise SelectionIntegrityError("NB12 contract has no nb11_generation_id")
    if generation_id is None:
        nb11 = resolve_nb11_input(root, u_clean_dir=root / NB11_RELATIVE_DIR)
    else:
        nb11 = resolve_nb11_generation(root, pinned_nb11_id, u_clean_dir=root / NB11_RELATIVE_DIR)
    verify_nb12_contract_hash(contract)
    if summary.get("status") != STATUS_SUCCESS:
        raise SelectionIntegrityError(f"NB12 status is {summary.get('status')!r}, not {STATUS_SUCCESS}")
    if summary.get("quality_score_frozen") is not True:
        raise SelectionIntegrityError("NB12 quality score is not frozen")
    if str(summary.get("nb12_contract_sha256") or "") != str(contract["nb12_contract_sha256"]):
        raise SelectionIntegrityError("NB12 summary contract hash does not match contract.json")
    if str(summary.get("schema_version") or "") != PSEUDO_SCHEMA_VERSION:
        raise SelectionIntegrityError("NB12 summary schema drift")
    parquet_path = gen_dir / "u_prime_manifest.parquet"
    manifest_sha = sha256_file(parquet_path)
    if manifest_sha != str(summary.get("u_prime_manifest_sha256") or ""):
        raise SelectionIntegrityError("U′ parquet hash does not match the NB12 summary")
    nb11_payload = _read_json(gen_dir / "nb11_input_contract.json")
    verify_stored_hash(nb11_payload, "nb11_input_contract_sha256", "NB11 input contract")
    if str(nb11_payload["nb11_input_contract_sha256"]) != str(contract.get("nb11_input_contract_sha256") or ""):
        raise SelectionIntegrityError("NB12 contract is not bound to the stored NB11 input contract")
    if str(nb11_payload.get("generation_id") or "") != pinned_nb11_id:
        raise SelectionIntegrityError("stored NB11 input contract is not the NB12 pinned generation")
    if nb11.generation_id != pinned_nb11_id:
        raise SelectionIntegrityError("NB12 contract nb11_generation_id does not match the immutable NB11 generation")
    if nb11.contract_sha256 != str(contract.get("nb11_input_contract_sha256") or ""):
        raise SelectionIntegrityError("NB12 contract is not bound to the immutable NB11 input contract")
    rows = load_u_prime_parquet(parquet_path)
    nb11_by_uid = {str(row["segment_uid"]): row for row in nb11.rows}
    for row in rows:
        uid = row[UID_COLUMN]
        source = nb11_by_uid.get(uid)
        if source is None:
            raise SelectionIntegrityError("segment_uid is not in the locked NB11 U_clean", uid)
        if str(row["segment_pcm16_sha256"]) != str(source["segment_pcm16_sha256"]):
            raise SelectionIntegrityError("PCM hash differs from NB11", uid)
        if str(row["segment_wav_sha256"]) != str(source["segment_wav_sha256"]):
            raise SelectionIntegrityError("WAV hash differs from NB11", uid)
        if abs(float(row[DURATION_COLUMN]) - float(source["duration_seconds"])) > 1e-6:
            raise SelectionIntegrityError("duration differs from NB11", uid)
        if str(row.get("nb11_input_contract_sha256") or "") != nb11.contract_sha256:
            raise SelectionIntegrityError("row is not bound to the locked NB11 contract", uid)
    identity = {
        "nb11_generation_id": nb11.generation_id,
        "nb11_input_contract_sha256": nb11.contract_sha256,
        "nb12_generation_id": gen_id,
        "nb12_contract_sha256": str(contract["nb12_contract_sha256"]),
        "u_prime_manifest_sha256": manifest_sha,
        "u_prime_ordered_uid_sha256": ordered_uid_sha256([row[UID_COLUMN] for row in rows]),
        "quality_score_contract_sha256": str(contract.get("quality_score_contract_sha256") or ""),
        "n_u_prime": len(rows),
        "total_duration_seconds": round(sum(float(row[DURATION_COLUMN]) for row in rows), 6),
    }
    if len(identity["quality_score_contract_sha256"]) != 64:
        raise SelectionIntegrityError("NB12 contract has no quality_score_contract_sha256")
    for row in rows:
        if str(row.get("quality_score_contract_sha256") or "") != identity["quality_score_contract_sha256"]:
            raise SelectionIntegrityError("row quality contract hash differs from NB12", row[UID_COLUMN])
    return FrozenUPrime(generation_id=gen_id, rows=tuple(rows), identity=identity)


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    if not path.is_file():
        raise SelectionIntegrityError(f"missing required JSON: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SelectionIntegrityError(f"unreadable JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise SelectionIntegrityError(f"JSON object required: {path.name}")
    return payload


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _docstring_constants(tree: ast.AST) -> set:
    """Docstrings are constants, but they are not executable path references."""
    found = set()

    def mark(body: Sequence[ast.stmt]) -> None:
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            found.add(id(first.value))

    if isinstance(tree, ast.Module):
        mark(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark(node.body)
    return found


def _constant_references_forbidden_path(value: str) -> str:
    text = str(value).replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    for needle in _forbidden_path_needles():
        if any(part == needle or part.startswith(needle + ".") or part.startswith(needle + "_") for part in parts):
            return needle
        if needle in text and ("/" in text or "." in text):
            return needle
    return ""


def forbidden_operations(tree: ast.AST) -> List[str]:
    """Calls, imports, and frozen-test path strings. Comments and docstrings are ignored."""
    docstrings = _docstring_constants(tree)
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".")[0]
                if root_name in _FORBIDDEN_IMPORT_ROOTS or alias.name in _FORBIDDEN_CALLS:
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root_name = module.split(".")[0]
            if root_name in _FORBIDDEN_IMPORT_ROOTS:
                found.append(f"from {module}")
            for alias in node.names:
                if alias.name in _FORBIDDEN_CALLS:
                    found.append(f"from {module} import {alias.name}")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _FORBIDDEN_CALLS or name == "train":
                found.append(f"call {name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            needle = _constant_references_forbidden_path(node.value)
            if needle:
                found.append(f"path {node.value}")
    return found


def _trees_for_source(path: Path) -> List[ast.AST]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        import json

        notebook = json.loads(text)
        trees = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            trees.append(ast.parse("".join(cell.get("source") or [])))
        return trees
    return [ast.parse(text)]


def assert_nb13_sources_clean(paths: Sequence[Union[str, Path]]) -> None:
    problems = []
    for path in paths:
        for tree in _trees_for_source(Path(path)):
            problems.extend(f"{Path(path).name}: {item}" for item in forbidden_operations(tree))
    if problems:
        raise SelectionIntegrityError("NB13 source is not selection-only: " + "; ".join(problems))


def read_selection_generation_id(out_dir: Union[str, Path]) -> str:
    return read_current_generation_id(out_dir, error=SelectionIntegrityError)
