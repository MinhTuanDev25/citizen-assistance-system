"""Synthetic fixtures for Notebook 12 tests. Nothing here touches real artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import (
    EXPECTED_NB11_SCHEMA_VERSION,
    EXPECTED_OVERLAP_CONTRACT_SHA256,
    EXPECTED_SEGMENTATION_CONTRACT_SHA256,
    NB11_SUCCESS_STATUS,
    build_d0_agreement_contract,
    build_teacher_contract,
    decoding_config_from_rq1,
)
from src.rq2_pseudo_label import RAW_RECORD_COLUMNS, RAW_SCHEMA_VERSION, STATUS_DISABLED, STATUS_OK
from src.rq2_u_clean import U_CLEAN_MANIFEST_COLUMNS, write_wav_pcm16_atomic


def tree_digest(root: Path) -> Dict[str, str]:
    root = Path(root)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def segment_pcm(index: int, n_samples: int) -> np.ndarray:
    """Piecewise-constant levels so fake CTC models emit several token runs."""
    levels = np.asarray([(index + j) % 4 for j in range(8)], dtype=np.float64)
    block = max(1, n_samples // len(levels))
    wave = np.repeat(levels, block)[:n_samples]
    if wave.size < n_samples:
        wave = np.pad(wave, (0, n_samples - wave.size), constant_values=levels[-1])
    return np.round(wave / 50.0 * 32767.0).astype("<i2")


def build_nb11_generation(
    project_root: Path,
    *,
    n_segments: int = 7,
    n_samples: Sequence[int] = (),
    gen_id: str = "20261001T000000000000Z-abcd1234",
    status: str = NB11_SUCCESS_STATUS,
    segmentation_sha: str = EXPECTED_SEGMENTATION_CONTRACT_SHA256,
    duplicate_uid: bool = False,
    write_complete: bool = True,
    write_current: bool = True,
    extra_column: bool = False,
) -> Dict[str, Any]:
    root = Path(project_root)
    out = root / "artifacts" / "rq2" / "u_clean"
    seg_dir = out / "segments" / "src-a"
    rows: List[Dict[str, Any]] = []
    total = 0.0
    for i in range(n_segments):
        uid = f"seg-{i:04d}"
        if duplicate_uid and i == n_segments - 1:
            uid = "seg-0000"
        ns = int(n_samples[i]) if i < len(n_samples) else 1600 + 160 * i
        pcm = segment_pcm(i, ns)
        rel = f"segments/src-a/{uid}.wav"
        wav_sha = write_wav_pcm16_atomic(out / rel, pcm)
        dur = ns / 16000.0
        total += dur
        row = {c: "" for c in U_CLEAN_MANIFEST_COLUMNS}
        row.update({
            "segment_uid": uid, "canonical_segment_uid": uid, "source_id": "src-a", "source_group_id": "grp-a",
            "segment_local_path": rel, "segment_wav_sha256": wav_sha,
            "segment_pcm16_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "start_sample": 0, "end_sample": ns, "start_seconds": 0.0, "end_seconds": dur, "duration_seconds": dur,
            "vad_speech_fraction": 0.9, "rms_dbfs": -20.0, "peak_dbfs": -3.0, "silence_fraction_energy": 0.1,
            "u_clean_status": "U_CLEAN_RETAINED",
        })
        if extra_column:
            row["text_vi"] = "leak"
        rows.append(row)
    gen_dir = out / "generations" / gen_id
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "u_clean_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    pd.DataFrame(rows, columns=list(rows[0].keys())).to_csv(gen_dir / "u_clean_manifest.csv", index=False)
    pins = {"schema_version": EXPECTED_NB11_SCHEMA_VERSION, "segmentation_contract_sha256": segmentation_sha,
            "overlap_contract_sha256": EXPECTED_OVERLAP_CONTRACT_SHA256}
    (gen_dir / "contract.json").write_text(json.dumps(pins), encoding="utf-8")
    summary = {**pins, "status": status, "segmentation_config_frozen": True, "overlap_config_frozen": True,
               "gates": {"all_ok": True}, "n_u_clean": len(rows), "retained_audio_seconds": round(total, 3)}
    (gen_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    files = ["contract.json", "summary.json", "u_clean_manifest.csv", "u_clean_manifest.jsonl"]
    if write_complete:
        (gen_dir / "COMPLETE.json").write_text(json.dumps({"generation": gen_id, "files": files}), encoding="utf-8")
    if write_current:
        (out / "CURRENT").write_text(gen_id + "\n", encoding="utf-8")
    return {"out_dir": out, "gen_dir": gen_dir, "rows": rows, "gen_id": gen_id}


def raw_record(uid: str, **overrides: Any) -> Dict[str, Any]:
    rec = {c: None for c in RAW_RECORD_COLUMNS}
    rec.update({
        "segment_uid": uid, "source_id": "src-a", "source_group_id": "grp-a",
        "segment_local_path": f"segments/src-a/{uid}.wav",
        "segment_pcm16_sha256": "a" * 64, "segment_wav_sha256": "b" * 64, "duration_seconds": 6.0,
        "nb11_generation_id": "gen", "nb11_input_contract_sha256": "c" * 64,
        "teacher_contract_sha256": "d" * 64, "d0_agreement_contract_sha256": "e" * 64,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "asr_status": STATUS_OK, "asr_text_raw": "bah nar", "asr_text_norm": "bah nar",
        "asr_valid_encoding": True, "asr_empty": False, "asr_token_count": 7,
        "asr_mean_logprob": -0.2, "asr_confidence_raw": 0.8, "asr_frame_mean_logprob": -0.1,
        "asr_blank_frame_fraction": 0.5, "asr_n_frames": 30, "asr_decode_warning": "",
        "mt_status": STATUS_OK, "pseudo_vi_raw": "xin chào", "pseudo_vi_norm": "xin chào",
        "mt_valid_encoding": True, "mt_empty": False, "mt_token_count": 3,
        "mt_mean_logprob": -0.4, "mt_confidence_raw": 0.67, "mt_source_truncated": False,
        "mt_hit_max_length": False, "mt_decode_warning": "",
        "d0_status": STATUS_DISABLED, "d0_vi_raw": None, "d0_vi_norm": None, "d0_valid_encoding": None,
        "d0_decode_warning": "", "teacher_d0_agreement_chrf": None, "teacher_d0_agreement_char_ratio": None,
        "asr_char_count": 7, "asr_word_count": 2, "pseudo_vi_char_count": 8, "pseudo_vi_word_count": 2,
        "target_source_char_ratio": 8 / 7, "target_source_word_ratio": 1.0,
    })
    rec.update(overrides)
    return rec


def fake_rq1_chain(digest: str = "1" * 64, mt_batch: int = 16):
    def ident(name):
        return {"experiment_id": f"{name}-exp", "contract_hash": f"{name}-contract",
                "best_checkpoint_name": "checkpoint-100", "model_id": f"{name}-model", "model_revision": "rev"}

    def proof(name, d):
        return {"verified": True, "loaded": True, "experiment_id": f"{name}-exp", "contract_hash": f"{name}-contract",
                "best_checkpoint_name": "checkpoint-100", "checkpoint_content_digest": d, "checkpoint_n_files": 5}

    body = {
        "asr_handoff_hash": "a" * 64, "mt_handoff_hash": "b" * 64, "direct_handoff_hash": "c" * 64,
        "source_fingerprint_sha256": "f" * 64, "rq1_test_contract_hash": "9" * 64,
        "runtime_versions": {"torch": "2.8.0", "transformers": "4.57.6", "accelerate": "1.10.1"},
        "checkpoint_proof": {"asr": proof("asr", digest), "mt": proof("mt", "2" * 64), "direct": proof("direct", "3" * 64)},
    }
    final_contract = {**body, "rq1_final_contract_hash": sha256_json(body)}
    mt_contract = {"tokenizer_fingerprint": "tok-mt", "model_id": "bartpho", "model_revision": "r",
                   "max_source_length": 128, "generation_max_length": 128, "num_beams": 4,
                   "per_device_eval_batch_size": mt_batch}
    direct_contract = {"tokenizer_fingerprint": "tok-d0", "target_lang": "vi_VN", "generation_max_length": 128,
                       "num_beams": 4, "per_device_eval_batch_size": 2, "encoder_id": "xlsr", "encoder_revision": "r",
                       "decoder_id": "mbart", "decoder_revision": "r"}
    handoff = {
        "asr_handoff_hash": "a" * 64, "mt_handoff_hash": "b" * 64, "direct_handoff_hash": "c" * 64,
        "asr": {"identity": ident("asr")},
        "mt": {"identity": ident("mt"), "training_contract": mt_contract},
        "direct": {"identity": ident("direct"), "training_contract": direct_contract},
    }
    return final_contract, handoff


def fake_contracts(d0_enabled: bool = False, digest: str = "1" * 64):
    final_contract, handoff = fake_rq1_chain(digest)
    decoding = decoding_config_from_rq1(handoff["mt"]["training_contract"], handoff["direct"]["training_contract"])
    teacher = build_teacher_contract(final_contract=final_contract, handoff=handoff, decoding=decoding)
    d0 = build_d0_agreement_contract(enabled=d0_enabled, final_contract=final_contract, handoff=handoff, decoding=decoding)
    return teacher, d0, decoding, final_contract, handoff


def synthetic_validation(n: int = 300, *, d0_enabled: bool = False, seed: int = 7):
    """Raw C0 records + references where MT confidence tracks reference chrF++."""
    from src.rq2_quality import ValidationReferences

    rng = np.random.default_rng(seed)
    words = [f"w{j}" for j in range(10)]
    pseudo = " ".join(words)
    records, vi, bah = [], {}, {}
    for i in range(n):
        k = i % 10
        uid = f"val-{i:04d}"
        vi[uid] = " ".join(words[: 10 - k] + [f"x{j}" for j in range(k)])
        bah[uid] = "bahnar " * (1 + i % 3)
        over = {
            "asr_mean_logprob": float(-rng.uniform(0.0, 1.0)),
            "mt_mean_logprob": float(-0.1 * k - rng.uniform(0.0, 0.01)),
            "pseudo_vi_raw": pseudo, "pseudo_vi_norm": pseudo,
            "asr_text_norm": "bahnar text", "asr_char_count": 11,
            "pseudo_vi_char_count": len(pseudo), "target_source_char_ratio": len(pseudo) / 11.0,
        }
        if d0_enabled:
            over.update({"d0_status": STATUS_OK, "d0_vi_norm": pseudo, "d0_valid_encoding": True,
                         "teacher_d0_agreement_chrf": float(rng.uniform(20.0, 90.0))})
        records.append(raw_record(uid, **over))
    return records, ValidationReferences(vi, bah)
