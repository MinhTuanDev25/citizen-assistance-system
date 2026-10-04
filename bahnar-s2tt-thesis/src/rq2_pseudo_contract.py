"""RQ2 Notebook 12 contracts.

* NB11 input lock: NB12 reads only the published NB11 generation that
  ``artifacts/rq2/u_clean/CURRENT`` names, and only when that generation has
  ``COMPLETE.json`` and status ``SUCCESS_RQ2_U_CLEAN_FROZEN``. Checkpoint state
  under ``u_clean/checkpoint`` is never read.
* Fixed C0 teacher: resolved from the RQ1 final contract chain that Notebook 06
  wrote (``rq1_final_contract.json`` + ``checkpoint_proof``). No checkpoint is
  chosen by mtime, filename order, "latest", or test score.
* D0 agreement contract: D0 only produces an independent hypothesis for an
  agreement feature. It is not a second teacher.
* G_test guard: NB12 never opens the frozen RQ1 test manifest. Every manifest
  read goes through :class:`DataAccessLedger`.
* Publication: a generation is staged, hashed, sealed with ``COMPLETE.json``
  last, renamed into place, reverified, and only then named by ``CURRENT``.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.rq1_contract import sha256_file, sha256_json

PSEUDO_SCHEMA_VERSION = "rq2-pseudo-1.0"
NB11_INPUT_CONTRACT_VERSION = "rq2_nb11_input_contract_v1"
TEACHER_CONTRACT_VERSION = "rq2_fixed_c0_teacher_contract_v1"
D0_AGREEMENT_CONTRACT_VERSION = "rq2_d0_agreement_contract_v1"
DECODING_CONFIG_VERSION = "rq2_teacher_decoding_v1"

NB11_SUCCESS_STATUS = "SUCCESS_RQ2_U_CLEAN_FROZEN"
EXPECTED_NB11_SCHEMA_VERSION = "rq2-uclean-1.1"
EXPECTED_SEGMENTATION_CONTRACT_SHA256 = "0ce196cc20aca45bc707c4347f1eaa602cd04eacb0ced64c4665f2870a8cefab"
EXPECTED_OVERLAP_CONTRACT_SHA256 = "d5830cfa285b6cf5cf82f817262c612e254c2cd03ac13b33078c06f31c6afb2b"
NB11_RETAINED_STATUS = "U_CLEAN_RETAINED"
NB11_REQUIRED_FILES = ("u_clean_manifest.jsonl", "u_clean_manifest.csv", "contract.json", "summary.json")
NB11_RELATIVE_DIR = "artifacts/rq2/u_clean"
PSEUDO_RELATIVE_DIR = "artifacts/rq2/pseudo_labels"
SAMPLE_RATE = 16000

RQ1_FINAL_STATUS = "SUCCESS_RQ1_FINAL"
RQ1_FINAL_CONTRACT_FILENAME = "rq1_final_contract.json"
RQ1_FINAL_SUMMARY_FILENAME = "rq1_final_summary.json"
ASR_NORMALIZATION = "normalize_bahnar_ctc_v1"

STATUS_SUCCESS = "SUCCESS_RQ2_PSEUDO_POOL_FROZEN"
STATUS_READY_FOR_REVIEW = "READY_FOR_RQ2_QUALITY_REVIEW"
STATUS_FAIL = "FAIL_RQ2_PSEUDO_POOL"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID = re.compile(r"^[0-9A-Za-z._-]+$")


class Nb11InputError(RuntimeError):
    """The NB11 generation is missing, incomplete, or does not match its lock."""


class TeacherContractError(RuntimeError):
    """The fixed RQ1 teacher chain cannot be proven."""


class GTestAccessError(RuntimeError):
    """NB12 tried to touch frozen RQ1 test data."""


def _is_sha256(value: object) -> bool:
    return bool(_HEX64.match(str(value or "").strip().lower()))


def _read_json(path: Path, error=RuntimeError) -> Dict[str, Any]:
    if not path.is_file():
        raise error(f"missing required JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise error(f"unreadable JSON: {path}") from exc


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Union[str, Path], obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def portable_relative(value: str) -> bool:
    text = str(value or "")
    if not text or Path(text).is_absolute():
        return False
    return ".." not in Path(text).parts and not any(m in text for m in ("/Users/", "/home/", "/workspace/", "\\"))


# --------------------------------------------------------------------------- #
# G_test guard                                                                 #
# --------------------------------------------------------------------------- #
_G_TEST_MARKERS = ("rq1_" + "test", "frozen_" + "test", "g_" + "test")


def _names_g_test(part: str) -> bool:
    lowered = str(part).lower()
    return (
        lowered.startswith(_G_TEST_MARKERS[0])
        or _G_TEST_MARKERS[1] in lowered
        or lowered == _G_TEST_MARKERS[2]
        or lowered.startswith(_G_TEST_MARKERS[2] + "_")
        or lowered.startswith(_G_TEST_MARKERS[2] + ".")
    )


def assert_not_g_test_path(path: Union[str, Path]) -> Path:
    """Refuse any path whose components name frozen RQ1 test data, before it is opened."""
    p = Path(path)
    if any(_names_g_test(part) for part in Path(str(p).replace("\\", "/")).parts):
        raise GTestAccessError(f"NB12 refuses frozen G_test data: {p.name}")
    return p


@dataclass
class DataAccessLedger:
    """Every RQ1 manifest NB12 opens is recorded here. Only G_validation is allowed."""

    allowed_splits: tuple = ("g_validation",)
    entries: List[Dict[str, str]] = field(default_factory=list)

    def record(self, split: str, path: Union[str, Path], purpose: str) -> Path:
        p = assert_not_g_test_path(path)
        if split not in self.allowed_splits:
            raise GTestAccessError(f"NB12 may not read split {split!r}")
        self.entries.append({"split": split, "file": p.name, "purpose": str(purpose)})
        return p

    @property
    def g_test_accessed(self) -> bool:
        return any(_names_g_test(e["file"]) for e in self.entries)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed_splits": list(self.allowed_splits),
            "entries": list(self.entries),
            "g_test_accessed": self.g_test_accessed,
        }


# --------------------------------------------------------------------------- #
# NB11 input lock                                                              #
# --------------------------------------------------------------------------- #
def nb11_manifest_columns() -> List[str]:
    from src.rq2_u_clean import U_CLEAN_MANIFEST_COLUMNS

    return list(U_CLEAN_MANIFEST_COLUMNS)


@dataclass(frozen=True)
class Nb11Input:
    project_root: Path
    generation_id: str
    generation_dir: Path
    rows: tuple
    contract: Dict[str, Any]

    @property
    def contract_sha256(self) -> str:
        return str(self.contract["nb11_input_contract_sha256"])

    @property
    def ordered_uids(self) -> List[str]:
        return [str(row["segment_uid"]) for row in self.rows]


def read_current_generation_id(out_dir: Union[str, Path], error=RuntimeError) -> str:
    current = Path(out_dir) / "CURRENT"
    if not current.is_file():
        raise error(f"no published generation: {current} is missing")
    gen_id = current.read_text(encoding="utf-8").strip()
    if not gen_id or not _GENERATION_ID.match(gen_id) or gen_id.endswith(".partial"):
        raise error(f"CURRENT names an invalid generation: {gen_id!r}")
    return gen_id


def _parse_jsonl(path: Path) -> List[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise Nb11InputError(f"u_clean_manifest.jsonl line {number} is not JSON") from exc
    return rows


def resolve_nb11_generation(
    project_root: Union[str, Path],
    generation_id: str,
    *,
    u_clean_dir: Optional[Union[str, Path]] = None,
    require_audio_files: bool = True,
) -> Nb11Input:
    """Verify one immutable NB11 generation. This does not read CURRENT."""
    if generation_id is None or str(generation_id).strip() == "":
        raise Nb11InputError("NB11 generation_id is required")
    return resolve_nb11_input(
        project_root,
        u_clean_dir=u_clean_dir,
        require_audio_files=require_audio_files,
        generation_id=str(generation_id),
    )


def resolve_nb11_input(
    project_root: Union[str, Path],
    *,
    u_clean_dir: Optional[Union[str, Path]] = None,
    require_audio_files: bool = True,
    generation_id: Optional[str] = None,
) -> Nb11Input:
    """Lock one NB11 generation. Omit ``generation_id`` to use CURRENT."""
    import pandas as pd

    root = Path(project_root)
    out_dir = Path(u_clean_dir) if u_clean_dir is not None else root / NB11_RELATIVE_DIR
    if generation_id is None:
        gen_id = read_current_generation_id(out_dir, error=Nb11InputError)
        pinned = False
    else:
        gen_id = str(generation_id).strip()
        if not gen_id or not _GENERATION_ID.match(gen_id) or gen_id.endswith(".partial"):
            raise Nb11InputError(f"NB11 generation id is invalid: {gen_id!r}")
        pinned = True
    gen_dir = out_dir / "generations" / gen_id
    if not gen_dir.is_dir():
        if pinned:
            raise Nb11InputError(f"pinned NB11 generation is missing: {gen_id}")
        raise Nb11InputError(f"CURRENT points at a missing generation: {gen_id}")
    complete_path = gen_dir / "COMPLETE.json"
    if not complete_path.is_file():
        if pinned:
            raise Nb11InputError(f"pinned NB11 generation is incomplete (no COMPLETE.json): {gen_id}")
        raise Nb11InputError(f"CURRENT points at an incomplete generation (no COMPLETE.json): {gen_id}")
    complete = _read_json(complete_path, Nb11InputError)
    if str(complete.get("generation") or "") != gen_id:
        if pinned:
            raise Nb11InputError("COMPLETE.json generation does not match the pinned NB11 generation")
        raise Nb11InputError("COMPLETE.json generation does not match CURRENT")
    files = [str(f) for f in (complete.get("files") or [])]
    missing_required = [f for f in NB11_REQUIRED_FILES if f not in files]
    if missing_required:
        raise Nb11InputError(f"NB11 generation does not list required files: {missing_required}")
    file_sha256: Dict[str, str] = {}
    for rel in sorted(files):
        path = gen_dir / rel
        if not path.is_file() or path.stat().st_size <= 0:
            raise Nb11InputError(f"NB11 generation file missing or empty: {rel}")
        file_sha256[rel] = sha256_file(path)

    summary = _read_json(gen_dir / "summary.json", Nb11InputError)
    contract = _read_json(gen_dir / "contract.json", Nb11InputError)
    if summary.get("status") != NB11_SUCCESS_STATUS:
        raise Nb11InputError(f"NB11 status is {summary.get('status')!r}, not {NB11_SUCCESS_STATUS}")
    for label, payload in (("summary", summary), ("contract", contract)):
        if str(payload.get("schema_version") or "") != EXPECTED_NB11_SCHEMA_VERSION:
            raise Nb11InputError(f"NB11 {label} schema drift: {payload.get('schema_version')!r}")
        if str(payload.get("segmentation_contract_sha256") or "") != EXPECTED_SEGMENTATION_CONTRACT_SHA256:
            raise Nb11InputError(f"NB11 {label} segmentation_contract_sha256 differs from the frozen pin")
        if str(payload.get("overlap_contract_sha256") or "") != EXPECTED_OVERLAP_CONTRACT_SHA256:
            raise Nb11InputError(f"NB11 {label} overlap_contract_sha256 differs from the frozen pin")
    if not (summary.get("segmentation_config_frozen") and summary.get("overlap_config_frozen")):
        raise Nb11InputError("NB11 summary does not record frozen segmentation and overlap configs")
    gates = summary.get("gates") or {}
    if not gates or not all(bool(v) for v in gates.values()):
        raise Nb11InputError("NB11 summary gates are not all true")

    columns = nb11_manifest_columns()
    rows = _parse_jsonl(gen_dir / "u_clean_manifest.jsonl")
    if not rows:
        raise Nb11InputError("NB11 U_clean manifest is empty")
    seen = set()
    total_seconds = 0.0
    for row in rows:
        if list(row.keys()) != columns:
            raise Nb11InputError(f"NB11 manifest schema drift for {row.get('segment_uid')!r}")
        uid = str(row["segment_uid"] or "")
        if not uid:
            raise Nb11InputError("NB11 manifest row has an empty segment_uid")
        if uid in seen:
            raise Nb11InputError(f"duplicate segment_uid in NB11 U_clean: {uid}")
        seen.add(uid)
        if row["u_clean_status"] != NB11_RETAINED_STATUS:
            raise Nb11InputError(f"NB11 manifest row is not retained: {uid}")
        if not _is_sha256(row["segment_pcm16_sha256"]) or not _is_sha256(row["segment_wav_sha256"]):
            raise Nb11InputError(f"NB11 manifest row has no valid audio hash: {uid}")
        if not portable_relative(row["segment_local_path"]):
            raise Nb11InputError(f"NB11 manifest row has a non-portable segment path: {uid}")
        n_samples = int(row["end_sample"]) - int(row["start_sample"])
        if n_samples <= 0 or abs(n_samples / float(SAMPLE_RATE) - float(row["duration_seconds"])) > 1e-6:
            raise Nb11InputError(f"NB11 manifest duration is inconsistent: {uid}")
        total_seconds += float(row["duration_seconds"])
        if require_audio_files and not (root / row["segment_local_path"]).is_file():
            raise Nb11InputError(f"NB11 manifest row references missing audio: {uid}")

    csv = pd.read_csv(gen_dir / "u_clean_manifest.csv", dtype=str, keep_default_na=False)
    if list(csv.columns) != columns:
        raise Nb11InputError("NB11 u_clean_manifest.csv schema drift")
    if csv["segment_uid"].astype(str).tolist() != [str(r["segment_uid"]) for r in rows]:
        raise Nb11InputError("NB11 manifest CSV and JSONL disagree on segment order")
    if int(summary.get("n_u_clean", -1)) != len(rows):
        raise Nb11InputError(f"NB11 summary n_u_clean={summary.get('n_u_clean')} != manifest rows {len(rows)}")
    if abs(float(summary.get("retained_audio_seconds", -1.0)) - round(total_seconds, 3)) > 1e-3 + 1e-9:
        raise Nb11InputError("NB11 summary retained_audio_seconds does not match the manifest")

    ordered = [str(r["segment_uid"]) for r in rows]
    payload = {
        "contract_version": NB11_INPUT_CONTRACT_VERSION,
        "nb11_status": NB11_SUCCESS_STATUS,
        "nb11_schema_version": EXPECTED_NB11_SCHEMA_VERSION,
        "u_clean_relative_dir": NB11_RELATIVE_DIR if u_clean_dir is None else "",
        "generation_id": gen_id,
        "complete_json_sha256": sha256_file(complete_path),
        "file_sha256": file_sha256,
        "summary_sha256": file_sha256["summary.json"],
        "manifest_jsonl_sha256": file_sha256["u_clean_manifest.jsonl"],
        "manifest_csv_sha256": file_sha256["u_clean_manifest.csv"],
        "contract_sha256": file_sha256["contract.json"],
        "segmentation_contract_sha256": EXPECTED_SEGMENTATION_CONTRACT_SHA256,
        "overlap_contract_sha256": EXPECTED_OVERLAP_CONTRACT_SHA256,
        "n_segments": len(rows),
        "total_duration_seconds": round(total_seconds, 6),
        "ordered_segment_uid_sha256": sha256_json(ordered),
        "segment_audio_identity_sha256": sha256_json(
            [[r["segment_uid"], r["segment_pcm16_sha256"], r["segment_wav_sha256"]] for r in rows]
        ),
    }
    payload["nb11_input_contract_sha256"] = sha256_json(payload)
    return Nb11Input(
        project_root=root,
        generation_id=gen_id,
        generation_dir=gen_dir,
        rows=tuple(rows),
        contract=payload,
    )


def assert_nb11_input_unchanged(locked: Mapping[str, Any], current: Nb11Input) -> None:
    if str(locked.get("nb11_input_contract_sha256") or "") != current.contract_sha256:
        raise Nb11InputError("NB11 input changed since it was locked (fail closed)")


def verify_nb11_segment_audio(nb11: Nb11Input, *, progress_every: int = 0) -> Dict[str, Any]:
    """Re-read every U_clean WAV once and compare its file and PCM hashes."""
    from src.rq2_u_clean import read_wav_pcm16

    for index, row in enumerate(nb11.rows, 1):
        uid = str(row["segment_uid"])
        path = nb11.project_root / row["segment_local_path"]
        if not path.is_file():
            raise Nb11InputError(f"U_clean audio missing: {uid}")
        info = read_wav_pcm16(path)
        if info["wav_sha256"] != row["segment_wav_sha256"]:
            raise Nb11InputError(f"U_clean WAV hash differs from NB11 manifest: {uid}")
        if info["pcm16_sha256"] != row["segment_pcm16_sha256"]:
            raise Nb11InputError(f"U_clean PCM hash differs from NB11 manifest: {uid}")
        if info["n_samples"] != int(row["end_sample"]) - int(row["start_sample"]):
            raise Nb11InputError(f"U_clean sample count differs from NB11 manifest: {uid}")
        if progress_every and index % progress_every == 0:
            print(f"verified {index}/{len(nb11.rows)} U_clean WAVs")
    return {"n_verified": len(nb11.rows), "nb11_input_contract_sha256": nb11.contract_sha256}


# --------------------------------------------------------------------------- #
# Fixed C0 teacher + D0 agreement                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeacherDecodingConfig:
    """Deterministic decoding. MT/D0 beam and length settings come from the RQ1 contracts.

    Batch sizes are operational, but padding can change logits, so they are part
    of this hash and a resumed run cannot mix outputs from different batching.
    """

    asr_batch_size: int = 1
    mt_batch_size: int = 8
    d0_batch_size: int = 1
    precision: str = "fp32"

    def payload(self, mt_contract: Mapping[str, Any], direct_contract: Optional[Mapping[str, Any]]) -> dict:
        for name in ("asr_batch_size", "mt_batch_size", "d0_batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.precision != "fp32":
            raise ValueError("NB12 teacher inference runs in fp32 to match RQ1")
        out = {
            "decoding_config_version": DECODING_CONFIG_VERSION,
            "precision": self.precision,
            "sampling": False,
            "asr": {
                "decoder": "ctc_greedy_argmax",
                "batch_size": int(self.asr_batch_size),
                "text_normalization": ASR_NORMALIZATION,
                "confidence": "ctc_greedy_token_run_mean_logprob_v1",
            },
            "mt": {
                "source_normalization": ASR_NORMALIZATION,
                "target_normalization": _mt_normalization_version(),
                "max_source_length": int(mt_contract["max_source_length"]),
                "generation_max_length": int(mt_contract["generation_max_length"]),
                "num_beams": int(mt_contract["num_beams"]),
                "batch_size": int(self.mt_batch_size),
                "skip_special_tokens": True,
                "confidence": "teacher_forced_mean_token_logprob_v1",
            },
        }
        if direct_contract is not None:
            out["d0"] = {
                "target_lang": str(direct_contract["target_lang"]),
                "generation_max_length": int(direct_contract["generation_max_length"]),
                "num_beams": int(direct_contract["num_beams"]),
                "batch_size": int(self.d0_batch_size),
                "target_normalization": _mt_normalization_version(),
            }
        return out


def decoding_config_from_rq1(
    mt_contract: Mapping[str, Any],
    direct_contract: Optional[Mapping[str, Any]] = None,
) -> TeacherDecodingConfig:
    """The batch sizes Notebook 06 used for C0/D0 on the frozen test set."""
    return TeacherDecodingConfig(
        asr_batch_size=1,
        mt_batch_size=int(mt_contract.get("per_device_eval_batch_size") or 8),
        d0_batch_size=int((direct_contract or {}).get("per_device_eval_batch_size") or 1),
    )


def assert_nb12_output_dir(out_dir: Union[str, Path], project_root: Union[str, Path]) -> Path:
    """NB12 writes only under artifacts/rq2/pseudo_labels-like dirs, never into NB11 or RQ1 trees."""
    out = Path(out_dir).resolve()
    root = Path(project_root).resolve()
    for protected in (root / NB11_RELATIVE_DIR, root / "artifacts" / "rq1", root / "data" / "manifests"):
        try:
            out.relative_to(protected.resolve())
        except ValueError:
            continue
        raise RuntimeError(f"NB12 output dir is inside a protected tree: {protected}")
    if any(_names_g_test(part) for part in out.parts):
        raise GTestAccessError("NB12 output dir names frozen G_test data")
    return out


def _mt_normalization_version() -> str:
    from src.mt_runtime_paths import MT_NORMALIZATION_VERSION

    return MT_NORMALIZATION_VERSION


def _final_contract_self_hash_ok(final_contract: Mapping[str, Any]) -> bool:
    body = {k: v for k, v in final_contract.items() if k != "rq1_final_contract_hash"}
    return sha256_json(body) == str(final_contract.get("rq1_final_contract_hash") or "")


def project_rq1_final_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only identity fields. RQ1 test metrics never leave this function."""
    return {
        "status": summary.get("status"),
        "ready_rq1_final": bool(summary.get("ready_rq1_final")),
        "rq1_final_contract_hash": str(summary.get("rq1_final_contract_hash") or ""),
        "rq1_test_contract_hash": str(summary.get("rq1_test_contract_hash") or ""),
    }


def _checkpoint_entry(final_contract: Mapping[str, Any], system: str, identity: Mapping[str, Any]) -> Dict[str, Any]:
    entry = dict((final_contract.get("checkpoint_proof") or {}).get(system) or {})
    if not entry.get("verified") or not entry.get("loaded"):
        raise TeacherContractError(f"RQ1 checkpoint proof for {system} is not verified and loaded")
    if not _is_sha256(entry.get("checkpoint_content_digest")):
        raise TeacherContractError(f"RQ1 checkpoint proof for {system} has no content digest")
    for key in ("experiment_id", "contract_hash", "best_checkpoint_name"):
        if str(entry.get(key) or "") != str(identity.get(key) or ""):
            raise TeacherContractError(f"RQ1 checkpoint proof {system}.{key} differs from the handoff")
    return {
        "experiment_id": str(entry["experiment_id"]),
        "contract_hash": str(entry["contract_hash"]),
        "best_checkpoint_name": str(entry["best_checkpoint_name"]),
        "checkpoint_content_digest": str(entry["checkpoint_content_digest"]),
        "checkpoint_n_files": int(entry.get("checkpoint_n_files") or 0),
    }


def build_teacher_contract(
    *,
    final_contract: Mapping[str, Any],
    handoff: Mapping[str, Any],
    decoding: TeacherDecodingConfig,
) -> Dict[str, Any]:
    """Pure builder. The same RQ1 chain and decoding config always give the same hash."""
    for key in ("asr_handoff_hash", "mt_handoff_hash"):
        if str(handoff.get(key) or "") != str(final_contract.get(key) or ""):
            raise TeacherContractError(f"current RQ1 {key} differs from rq1_final_contract")
    asr_id = handoff["asr"]["identity"]
    mt_id = handoff["mt"]["identity"]
    mt_contract = handoff["mt"]["training_contract"]
    if not str(mt_contract.get("tokenizer_fingerprint") or ""):
        raise TeacherContractError("RQ1 MT contract has no tokenizer_fingerprint")
    decoding_payload = decoding.payload(mt_contract, None)
    payload = {
        "contract_version": TEACHER_CONTRACT_VERSION,
        "role": "fixed_rq1_c0_teacher",
        "rq1_final_contract_hash": str(final_contract["rq1_final_contract_hash"]),
        "rq1_test_contract_hash_provenance_only": str(final_contract.get("rq1_test_contract_hash") or ""),
        "rq1_source_fingerprint_sha256": str(final_contract.get("source_fingerprint_sha256") or ""),
        "runtime_versions": dict(final_contract.get("runtime_versions") or {}),
        "asr": {
            "handoff_hash": str(handoff["asr_handoff_hash"]),
            "model_family": "xls_r_ctc",
            "model_id": str(asr_id.get("model_id") or ""),
            "model_revision": str(asr_id.get("model_revision") or ""),
            "processor": "Wav2Vec2Processor saved in the best checkpoint (inside checkpoint_content_digest)",
            "checkpoint": _checkpoint_entry(final_contract, "asr", asr_id),
            "normalization": ASR_NORMALIZATION,
        },
        "mt": {
            "handoff_hash": str(handoff["mt_handoff_hash"]),
            "model_family": "bartpho_seq2seq",
            "model_id": str(mt_contract.get("model_id") or ""),
            "model_revision": str(mt_contract.get("model_revision") or ""),
            "tokenizer_fingerprint": str(mt_contract["tokenizer_fingerprint"]),
            "checkpoint": _checkpoint_entry(final_contract, "mt", mt_id),
            "normalization": _mt_normalization_version(),
        },
        "decoding": decoding_payload,
        "decoding_config_sha256": sha256_json(decoding_payload),
    }
    payload["teacher_contract_sha256"] = sha256_json(payload)
    return payload


def build_d0_agreement_contract(
    *,
    enabled: bool,
    final_contract: Optional[Mapping[str, Any]] = None,
    handoff: Optional[Mapping[str, Any]] = None,
    decoding: Optional[TeacherDecodingConfig] = None,
) -> Dict[str, Any]:
    if not enabled:
        payload: Dict[str, Any] = {"contract_version": D0_AGREEMENT_CONTRACT_VERSION, "enabled": False}
        payload["d0_agreement_contract_sha256"] = sha256_json(payload)
        return payload
    if final_contract is None or handoff is None or decoding is None:
        raise TeacherContractError("D0 agreement needs the RQ1 final contract, handoff, and decoding config")
    if str(handoff.get("direct_handoff_hash") or "") != str(final_contract.get("direct_handoff_hash") or ""):
        raise TeacherContractError("current RQ1 direct_handoff_hash differs from rq1_final_contract")
    direct_id = handoff["direct"]["identity"]
    contract = handoff["direct"]["training_contract"]
    if not str(contract.get("tokenizer_fingerprint") or ""):
        raise TeacherContractError("RQ1 Direct contract has no tokenizer_fingerprint")
    d0_decoding = decoding.payload(handoff["mt"]["training_contract"], contract)["d0"]
    payload = {
        "contract_version": D0_AGREEMENT_CONTRACT_VERSION,
        "enabled": True,
        "role": "agreement_feature_only_not_a_teacher",
        "rq1_final_contract_hash": str(final_contract["rq1_final_contract_hash"]),
        "direct_handoff_hash": str(handoff["direct_handoff_hash"]),
        "encoder_id": str(contract.get("encoder_id") or ""),
        "encoder_revision": str(contract.get("encoder_revision") or ""),
        "decoder_id": str(contract.get("decoder_id") or ""),
        "decoder_revision": str(contract.get("decoder_revision") or ""),
        "tokenizer_fingerprint": str(contract["tokenizer_fingerprint"]),
        "checkpoint": _checkpoint_entry(final_contract, "direct", direct_id),
        "decoding": d0_decoding,
        "agreement_feature": "teacher_d0_agreement_chrf = mean(sentence chrF++(c0|d0), sentence chrF++(d0|c0))",
        "applies_to": "every U_clean segment and every G_validation calibration record",
    }
    payload["d0_agreement_contract_sha256"] = sha256_json(payload)
    return payload


def resolve_fixed_teacher(
    *,
    project_root: Union[str, Path],
    rq1_final_state_dir: Union[str, Path],
    asr_state_dir: Union[str, Path],
    mt_state_dir: Union[str, Path],
    direct_state_dir: Union[str, Path],
    d0_agreement_enabled: bool,
    decoding: Optional[TeacherDecodingConfig] = None,
    actual_runtime: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Re-prove the RQ1 C0 chain from explicit state directories. Never reads rq1_test.*.

    Without an explicit ``decoding``, batch sizes are the ones Notebook 06 used.
    """
    from src.rq1_contract import assert_locked_runtime, compute_rq1_source_fingerprint
    from src.rq1_evaluation import verify_upstream_handoffs

    state = Path(rq1_final_state_dir)
    final_contract = _read_json(state / RQ1_FINAL_CONTRACT_FILENAME, TeacherContractError)
    if not _final_contract_self_hash_ok(final_contract):
        raise TeacherContractError("rq1_final_contract.json self-hash does not verify")
    final_hash = str(final_contract["rq1_final_contract_hash"])
    if state.name != f"contract_{final_hash[:16]}":
        raise TeacherContractError("RQ1 final state dir name does not match rq1_final_contract_hash")
    summary = project_rq1_final_summary(_read_json(state / RQ1_FINAL_SUMMARY_FILENAME, TeacherContractError))
    if summary["status"] != RQ1_FINAL_STATUS or not summary["ready_rq1_final"]:
        raise TeacherContractError("RQ1 is not SUCCESS_RQ1_FINAL")
    if summary["rq1_final_contract_hash"] != final_hash:
        raise TeacherContractError("rq1_final_summary hash differs from rq1_final_contract")
    source_fp = compute_rq1_source_fingerprint(project_root)["aggregate_sha256"]
    if source_fp != str(final_contract.get("source_fingerprint_sha256") or ""):
        raise TeacherContractError("RQ1 inference source code changed since RQ1 final (fail closed)")
    if actual_runtime is not None:
        assert_locked_runtime(final_contract.get("runtime_versions") or {}, actual_runtime)
    handoff = verify_upstream_handoffs(
        asr_state_dir=asr_state_dir,
        mt_state_dir=mt_state_dir,
        direct_state_dir=direct_state_dir,
    )
    if decoding is None:
        decoding = decoding_config_from_rq1(handoff["mt"]["training_contract"], handoff["direct"]["training_contract"])
    teacher = build_teacher_contract(final_contract=final_contract, handoff=handoff, decoding=decoding)
    d0 = build_d0_agreement_contract(
        enabled=d0_agreement_enabled,
        final_contract=final_contract,
        handoff=handoff,
        decoding=decoding,
    )
    return {
        "teacher_contract": teacher,
        "d0_agreement_contract": d0,
        "checkpoint_proof": dict(final_contract.get("checkpoint_proof") or {}),
        "mt_training_contract": dict(handoff["mt"]["training_contract"]),
        "direct_training_contract": dict(handoff["direct"]["training_contract"]),
        "rq1_final_summary": summary,
        "decoding": decoding,
    }


# --------------------------------------------------------------------------- #
# Atomic generation publication                                                #
# --------------------------------------------------------------------------- #
ARTIFACT_MANIFEST = "artifact_manifest.json"
COMPLETE_MARKER = "COMPLETE.json"


def new_generation_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]


def stage_generation(out_dir: Union[str, Path], gen_id: Optional[str] = None) -> Dict[str, Any]:
    gen = gen_id or new_generation_id()
    generations = Path(out_dir) / "generations"
    final = generations / gen
    partial = generations / (gen + ".partial")
    if final.exists() or partial.exists():
        raise RuntimeError(f"generation already exists: {gen}")
    partial.mkdir(parents=True)
    return {"generation_id": gen, "staging_dir": partial, "final_dir": final}


def _hash_files(directory: Path, relative_files: Sequence[str], allow_empty: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    out = {}
    for rel in sorted(relative_files):
        path = directory / rel
        if not path.is_file() or (path.stat().st_size <= 0 and rel not in allow_empty):
            raise RuntimeError(f"staged artifact missing or empty: {rel}")
        out[rel] = {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
    return out


def verify_generation(out_dir: Union[str, Path], gen_id: Optional[str] = None) -> Dict[str, Any]:
    """Rehash every file of a published generation against its sealed manifest."""
    out = Path(out_dir)
    gen = gen_id or read_current_generation_id(out)
    directory = out / "generations" / gen
    complete = _read_json(directory / COMPLETE_MARKER)
    if str(complete.get("generation_id") or "") != gen:
        raise RuntimeError("COMPLETE.json generation_id mismatch")
    manifest_path = directory / ARTIFACT_MANIFEST
    if sha256_file(manifest_path) != str(complete.get("artifact_manifest_sha256") or ""):
        raise RuntimeError("artifact_manifest.json hash differs from COMPLETE.json")
    manifest = _read_json(manifest_path)
    files = manifest.get("files") or {}
    actual = _hash_files(directory, list(files), allow_empty=[rel for rel, meta in files.items() if not meta.get("bytes")])
    if actual != files:
        bad = sorted(rel for rel in files if actual.get(rel) != files[rel])
        raise RuntimeError(f"published artifact hash mismatch: {bad}")
    return {"generation_id": gen, "files": files, "artifact_manifest_sha256": complete["artifact_manifest_sha256"]}


def finalize_generation(
    out_dir: Union[str, Path],
    staged: Mapping[str, Any],
    relative_files: Sequence[str],
    allow_empty: Sequence[str] = (),
) -> Dict[str, Any]:
    """Seal, rename, reverify, then move CURRENT. A failure leaves CURRENT unchanged."""
    partial = Path(staged["staging_dir"])
    final = Path(staged["final_dir"])
    gen = str(staged["generation_id"])
    files = _hash_files(partial, relative_files, allow_empty=allow_empty)
    write_json(partial / ARTIFACT_MANIFEST, {"generation_id": gen, "files": files})
    write_json(
        partial / COMPLETE_MARKER,
        {
            "generation_id": gen,
            "files": sorted(files),
            "artifact_manifest_sha256": sha256_file(partial / ARTIFACT_MANIFEST),
            "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    if final.exists():
        raise RuntimeError(f"refuse to overwrite completed generation: {gen}")
    os.rename(partial, final)
    verified = verify_generation(out_dir, gen)
    atomic_write_text(Path(out_dir) / "CURRENT", gen + "\n")
    if read_current_generation_id(out_dir) != gen:
        raise RuntimeError("CURRENT did not move to the verified generation")
    return verified
