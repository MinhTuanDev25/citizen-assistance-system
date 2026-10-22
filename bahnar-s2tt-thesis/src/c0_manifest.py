"""Inference artifact manifest for the C0 cascade.

Pretrained Hub identity, fine-tuned checkpoint identity, and runtime pins are
separate fields. SHA-256 is recorded only when the file exists and hashing is
requested. Required dependency pins are not claimed as installed versions.
"""
from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from src.c0_cascaded import (
    ASR_PRETRAINED_ID,
    ASR_PRETRAINED_REVISION,
    ASR_REQUIRED_FILES,
    DEFAULT_MAX_AUDIO_BYTES,
    DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    MT_REQUIRED_FILES,
    RQ1_GENERATION_MAX_LENGTH,
    RQ1_MAX_SOURCE_LENGTH,
    RQ1_NUM_BEAMS,
    SAMPLE_RATE,
    TRAINING_STATE_FILES,
    VERIFIED_ASR_CHECKPOINT_STEP,
    VERIFIED_MT_CHECKPOINT_STEP,
    C0ConfigError,
    assess_checkpoint,
    configured_path,
    generation_matches_rq1,
    load_c0_config,
    load_research_pins,
    optional_sha256,
    pair_checkpoint_identity,
    pair_research_provenance,
    _generation_ints,
)
from src.c0_tokenizer import LOCKED_MODEL_ID, LOCKED_MODEL_REVISION, PARITY_NOT_VERIFIED, TOKENIZER_SOURCE_HUB_PINNED_RQ1
from src.rq1_contract import sha256_file

REQUIRED_DEPENDENCIES = {
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "accelerate": "1.10.1",
}


def _installed_version(name: str) -> Optional[str]:
    try:
        return im.version(name)
    except im.PackageNotFoundError:
        return None


def _artifact_rows(root: Path, names: tuple[str, ...], *, compute_sha256: bool) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        path = Path(root) / name
        present = path.is_file()
        row: dict[str, Any] = {"name": name, "present": present, "sha256": None, "sha256_status": "NOT_COMPUTED"}
        if present and compute_sha256:
            row["sha256"] = sha256_file(path)
            row["sha256_status"] = "COMPUTED"
        rows.append(row)
    return rows


def _trainer_state_step(root: Path) -> dict[str, Any]:
    state_path = Path(root) / "trainer_state.json"
    if not state_path.is_file():
        return {"present": False, "global_step": None, "best_metric": None}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"present": True, "global_step": None, "best_metric": None, "parse": "failed"}
    best = raw.get("best_model_checkpoint")
    best_name = Path(str(best)).name if best else None
    return {
        "present": True,
        "global_step": raw.get("global_step"),
        "best_metric": raw.get("best_metric"),
        "best_checkpoint_basename": best_name,
        "path_note": "best_model_checkpoint absolute paths from training runtime are ignored",
    }


def _fine_tuned_block(
    root: Optional[Path],
    required: tuple[str, ...],
    *,
    expected_step: int,
    compute_sha256: bool,
    expected_sha256: Optional[str] = None,
    research_sha256: Optional[str] = None,
) -> dict[str, Any]:
    assessed = assess_checkpoint(
        root,
        required,
        expected_step=expected_step,
        configured_sha256=expected_sha256,
        research_sha256=research_sha256,
    )
    files = _artifact_rows(root, required, compute_sha256=compute_sha256) if root is not None else []
    weights = next((row for row in files if row["name"] == "model.safetensors"), None)
    sha = assessed["weights_sha256"]
    sha_status = assessed["weights_sha256_status"]
    if weights is not None and weights["sha256"] is not None:
        sha = weights["sha256"]
        sha_status = weights["sha256_status"]
    return {
        "checkpoint": assessed["checkpoint"],
        "checkpoint_step": assessed["checkpoint_step"],
        "expected_checkpoint_step": expected_step,
        "step_name_match": assessed["step_name_match"],
        "artifacts_present": assessed["artifacts_present"],
        "experiment_id": None,
        "identity": assessed["identity"],
        "weights_filename": "model.safetensors",
        "weights_sha256": sha,
        "weights_sha256_status": sha_status,
        "configured_hash_match": assessed["configured_hash_match"],
        "research_hash_match": assessed["research_hash_match"],
        "research_provenance": assessed["research_provenance"],
        "missing_files": assessed["missing_files"],
        "files": files,
        "trainer_state": _trainer_state_step(root) if root is not None else {"present": False},
    }


def build_manifest(config: dict[str, Any], *, compute_sha256: bool = False) -> dict[str, Any]:
    max_source, max_length, beams = _generation_ints(config)
    rq1_gen = generation_matches_rq1(max_source, max_length, beams)
    asr = configured_path(config.get("asr_checkpoint"))
    mt = configured_path(config.get("mt_checkpoint"))
    asr_pin = optional_sha256(config, "asr_weights_sha256")
    mt_pin = optional_sha256(config, "mt_weights_sha256")
    research = load_research_pins(config)
    asr_block = _fine_tuned_block(
        asr,
        ASR_REQUIRED_FILES,
        expected_step=VERIFIED_ASR_CHECKPOINT_STEP,
        compute_sha256=compute_sha256,
        expected_sha256=asr_pin,
        research_sha256=research.get("asr"),
    )
    mt_block = _fine_tuned_block(
        mt,
        MT_REQUIRED_FILES,
        expected_step=VERIFIED_MT_CHECKPOINT_STEP,
        compute_sha256=compute_sha256,
        expected_sha256=mt_pin,
        research_sha256=research.get("mt"),
    )
    identity = pair_checkpoint_identity(asr_block, mt_block)
    installed = {name: _installed_version(name) for name in REQUIRED_DEPENDENCIES}
    return {
        "model_id": "c0_cascaded",
        "checkpoint_identity": identity,
        "research_provenance": pair_research_provenance(asr_block, mt_block),
        "research_pins_source": research.get("source"),
        "pretrained": {
            "asr": {
                "architecture": "transformers.Wav2Vec2ForCTC",
                "processor": "transformers.Wav2Vec2Processor",
                "pretrained_model_id": ASR_PRETRAINED_ID,
                "pretrained_revision": ASR_PRETRAINED_REVISION,
                "note": "Base model pin from Notebook 03. Not the fine-tuned ASR identity.",
            },
            "mt": {
                "architecture": "transformers.AutoModelForSeq2SeqLM",
                "pretrained_model_id": LOCKED_MODEL_ID,
                "pretrained_revision": LOCKED_MODEL_REVISION,
                "note": "BARTpho syllable pin from Notebook 04. Not the fine-tuned MT weight identity.",
            },
        },
        "tokenizer": {
            "tokenizer_model_id": LOCKED_MODEL_ID,
            "tokenizer_revision": LOCKED_MODEL_REVISION,
            "tokenizer_source": TOKENIZER_SOURCE_HUB_PINNED_RQ1,
            "tokenizer_parity": PARITY_NOT_VERIFIED,
            "tokenizer_parity_note": "VERIFIED only after load_mt_tokenizer fingerprint matches mt_training_contract.json",
            "pad_fallback": "pad_token = eos_token when pad_token is missing (load_mt_tokenizer)",
        },
        "fine_tuned": {
            "asr": asr_block,
            "mt": mt_block,
        },
        "decoding": {
            "asr": "argmax over CTC logits, processor.batch_decode, normalize_bahnar_ctc_v1",
            "mt_input": "normalize_bahnar_ctc_v1",
            "mt_output": "normalize_mt_text_v1",
            "generation": {
                "max_source_length": max_source,
                "generation_max_length": max_length,
                "num_beams": beams,
                "matches_rq1": rq1_gen,
                "source": "configs/mt.yaml locked Notebook 04 contract"
                if rq1_gen
                else "operator override; not the locked RQ1 generation contract",
            },
            "dtype": "from_pretrained default; evaluation does not cast dtype",
        },
        "audio": {
            "research": {
                "sample_rate_hz": SAMPLE_RATE,
                "channels": "mono mean if the file has more than one channel",
                "dtype": "float32",
                "resample": False,
                "constraint": "soundfile read must already be 16000 Hz; other rates are rejected",
            },
            "serving": {
                "max_audio_duration_seconds": config.get("max_audio_duration_seconds", DEFAULT_MAX_AUDIO_DURATION_SECONDS),
                "max_audio_bytes": config.get("max_audio_bytes", DEFAULT_MAX_AUDIO_BYTES),
                "note": "Duration and size caps are CLI serving policy, not an RQ1 metric contract.",
            },
        },
        "runtime": {
            "device": str(config.get("device") or "cpu"),
            "required_dependencies": dict(REQUIRED_DEPENDENCIES),
            "installed_dependencies": installed,
            "installed_match_required": {
                name: (installed[name] == pin) if installed[name] is not None else False
                for name, pin in REQUIRED_DEPENDENCIES.items()
            },
        },
        "excluded_training_files": list(TRAINING_STATE_FILES),
    }


def _dump_error(exc: BaseException, *, error_code: str) -> int:
    json.dump(
        {"error": str(exc) if isinstance(exc, C0ConfigError) else f"{type(exc).__name__}: {exc}", "error_code": error_code},
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write the C0 inference artifact manifest")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compute-sha256", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(load_c0_config(args.config), compute_sha256=args.compute_sha256)
        text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except C0ConfigError as exc:
        return _dump_error(exc, error_code=getattr(exc, "error_code", "config_error"))
    except (yaml.YAMLError, ValueError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _dump_error(exc, error_code="config_error")
    except OSError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _dump_error(exc, error_code="output_error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
