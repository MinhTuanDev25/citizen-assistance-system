"""Tokenizer fingerprint + compatibility audit for Notebook 04 MT."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.mt_normalize import normalize_mt_text_v1

# Documented hard gates (locked for Notebook 04). Empirical rates always reported.
HARD_MAX_EMPTY_RATE = 0.0
HARD_MAX_UNK_RATE = 0.05
HARD_MAX_TRUNCATION_RATE = 0.05
MT_MONITOR_SIZE_DEFAULT = 512


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """
    Content-oriented fingerprint: class/specials + full vocab digest + config digest.

    Does not store the full vocab; streams a deterministic SHA over sorted items.
    """
    h = hashlib.sha256()

    def _upd(obj: Any) -> None:
        h.update(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        h.update(b"\0")

    _upd(
        {
            "class": type(tokenizer).__name__,
            "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
            "model_max_length": int(getattr(tokenizer, "model_max_length", -1) or -1),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        }
    )
    # Special-token map when available
    try:
        sp = getattr(tokenizer, "special_tokens_map", None)
        if sp:
            _upd({"special_tokens_map": dict(sp)})
    except Exception:
        pass
    # Tokenizer config JSON if present
    try:
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            # Drop huge/non-stable fields
            slim = {
                k: v
                for k, v in init_kwargs.items()
                if k
                in {
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "pad_token",
                    "cls_token",
                    "sep_token",
                    "mask_token",
                    "model_max_length",
                    "tokenizer_class",
                }
            }
            _upd({"init_kwargs_slim": slim})
    except Exception:
        pass
    # Full vocab digest (streamed; not stored)
    try:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab = get_vocab()
            _upd({"vocab_len": len(vocab)})
            for k, v in sorted((str(k), int(v)) for k, v in vocab.items()):
                h.update(k.encode("utf-8"))
                h.update(b"=")
                h.update(str(v).encode("utf-8"))
                h.update(b"\n")
    except Exception:
        pass
    # SentencePiece / backend model bytes digest when available on disk
    try:
        sp_model = getattr(tokenizer, "sp_model", None) or getattr(tokenizer, "vocab_file", None)
        # Some tokenizers expose vocab_file path
        vocab_file = getattr(tokenizer, "vocab_file", None)
        if isinstance(vocab_file, str) and vocab_file:
            from pathlib import Path

            p = Path(vocab_file)
            if p.is_file():
                hh = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hh.update(chunk)
                _upd({"vocab_file_sha256": hh.hexdigest(), "vocab_file_name": p.name})
        elif sp_model is not None and hasattr(sp_model, "serialized_model_proto"):
            proto = sp_model.serialized_model_proto()
            _upd({"spm_proto_sha256": hashlib.sha256(proto).hexdigest()})
    except Exception:
        pass
    return h.hexdigest()


def _percentile(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _encode_len(tokenizer: Any, text: str, *, max_length: int) -> Dict[str, Any]:
    # Content emptiness without specials so BOS/EOS-only encodings fail the gate.
    enc_content = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    content_ids = list(enc_content["input_ids"])
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    ids = list(enc["input_ids"])
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk = sum(1 for i in ids if unk_id is not None and i == unk_id)
    return {
        "token_len": len(ids),
        "unk": int(unk),
        "empty": len(content_ids) == 0,
        "truncated": len(ids) > int(max_length),
    }


def _side_report(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(len(stats), 1)
    lengths = sorted(int(s["token_len"]) for s in stats)
    toks = sum(lengths)
    unks = sum(int(s["unk"]) for s in stats)
    trunc = sum(1 for s in stats if s["truncated"])
    empty = sum(1 for s in stats if s["empty"])
    return {
        "n": len(stats),
        "token_count": int(toks),
        "mean": float(toks / n) if stats else 0.0,
        "p50": _percentile(lengths, 50),
        "p95": _percentile(lengths, 95),
        "p99": _percentile(lengths, 99),
        "max": int(lengths[-1]) if lengths else 0,
        "unk_count": int(unks),
        "unk_rate": float(unks / toks) if toks else 0.0,
        "truncation_count": int(trunc),
        "truncation_rate": float(trunc / n) if stats else 0.0,
        "empty_encoding_count": int(empty),
        "empty_rate": float(empty / n) if stats else 0.0,
    }


def audit_tokenizer_compatibility(
    tokenizer: Any,
    *,
    sources: Sequence[str],
    targets: Sequence[str],
    max_source_length: int,
    max_target_length: int,
    max_unk_rate: float = HARD_MAX_UNK_RATE,
    max_truncation_rate: float = HARD_MAX_TRUNCATION_RATE,
    max_empty_rate: float = HARD_MAX_EMPTY_RATE,
) -> Dict[str, Any]:
    """
    Compatibility audit with full empirical report + documented hard gates.

    Hard fail when empty encodings appear, or UNK/truncation rates exceed the
    documented ceilings (default HARD_MAX_UNK_RATE=0.05 /
    HARD_MAX_TRUNCATION_RATE=0.05). Empirical rates are always reported.
    """
    if len(sources) != len(targets):
        raise ValueError("sources/targets length mismatch")
    src_stats = [
        _encode_len(tokenizer, normalize_mt_text_v1(s), max_length=max_source_length) for s in sources
    ]
    tgt_stats = [
        _encode_len(tokenizer, normalize_mt_text_v1(t), max_length=max_target_length) for t in targets
    ]
    report: Dict[str, Any] = {
        "n": len(sources),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "source": _side_report(src_stats),
        "target": _side_report(tgt_stats),
        "gate_policy": {
            "description": (
                "Fail-closed if empty_rate>0 or unk_rate/truncation_rate exceed "
                "documented hard ceilings (default 0.05). Empirical rates always reported."
            ),
            "max_unk_rate": max_unk_rate,
            "max_truncation_rate": max_truncation_rate,
            "max_empty_rate": max_empty_rate,
        },
    }
    failed = []
    for side in ("source", "target"):
        if report[side]["empty_rate"] > max_empty_rate:
            failed.append(f"{side}_empty_rate")
        if report[side]["unk_rate"] > max_unk_rate:
            failed.append(f"{side}_unk_rate")
        if report[side]["truncation_rate"] > max_truncation_rate:
            failed.append(f"{side}_truncation_rate")
    report["failed_checks"] = failed
    report["passed"] = len(failed) == 0
    return report


def assert_tokenizer_compatible(report: Mapping[str, Any]) -> None:
    if not report.get("passed"):
        raise RuntimeError(
            f"MT tokenizer compatibility gate failed: {report.get('failed_checks')}"
        )


def load_mt_tokenizer(model_id: str, model_revision: str) -> Any:
    from src.mt_contract import assert_model_revision_pinned
    from transformers import AutoTokenizer

    assert_model_revision_pinned(model_id, model_revision)
    tok = AutoTokenizer.from_pretrained(model_id, revision=str(model_revision).strip())
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return tok
