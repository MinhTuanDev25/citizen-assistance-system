"""Model/tokenizer construction and compatibility QA for Direct S2TT."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

from src.direct_contract import assert_locked_direct_baseline
from src.mt_normalize import normalize_mt_text_v1
from src.mt_tokenize import tokenizer_fingerprint


def load_direct_processors(
    *, encoder_id: str, encoder_revision: str, decoder_id: str, decoder_revision: str, target_lang: str
):
    assert_locked_direct_baseline(encoder_id, encoder_revision, decoder_id, decoder_revision, target_lang)
    from transformers import AutoFeatureExtractor, MBart50TokenizerFast

    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id, revision=encoder_revision)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        decoder_id,
        revision=decoder_revision,
        src_lang=target_lang,  # source text is unused; keeps tokenizer fully initialized
        tgt_lang=target_lang,
    )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("mBART-50 tokenizer must provide pad_token_id")
    if target_lang not in tokenizer.lang_code_to_id:
        raise RuntimeError(f"Target language {target_lang!r} not in mBART-50 tokenizer")
    return feature_extractor, tokenizer


def load_direct_model(
    *,
    encoder_id: str,
    encoder_revision: str,
    decoder_id: str,
    decoder_revision: str,
    tokenizer: Any,
    target_lang: str,
    freeze_feature_encoder: bool = True,
):
    """Warm-start only from public pretrained backbones; never NB03/NB04 fine-tuned checkpoints."""
    assert_locked_direct_baseline(encoder_id, encoder_revision, decoder_id, decoder_revision, target_lang)
    from transformers import SpeechEncoderDecoderModel

    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_id,
        decoder_id,
        encoder_revision=encoder_revision,
        decoder_revision=decoder_revision,
    )
    forced_bos = int(tokenizer.lang_code_to_id[target_lang])
    model.config.decoder_start_token_id = int(tokenizer.eos_token_id)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.config.eos_token_id = int(tokenizer.eos_token_id)
    model.config.vocab_size = int(model.config.decoder.vocab_size)
    model.generation_config.decoder_start_token_id = int(tokenizer.eos_token_id)
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    model.generation_config.eos_token_id = int(tokenizer.eos_token_id)
    model.generation_config.forced_bos_token_id = forced_bos

    if freeze_feature_encoder:
        enc = model.encoder
        if hasattr(enc, "freeze_feature_encoder"):
            enc.freeze_feature_encoder()
        elif hasattr(enc, "feature_extractor") and hasattr(enc.feature_extractor, "_freeze_parameters"):
            enc.feature_extractor._freeze_parameters()
        else:
            raise RuntimeError("Cannot locate XLS-R feature encoder freeze method")
    return model


def parameter_report(model: Any) -> Dict[str, Any]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": float(trainable / total) if total else 0.0,
        "approx_parameter_bytes_fp32": int(total * 4),
    }


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    if len(x) == 1:
        return float(x[0])
    k = (len(x) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(x[lo])
    return float(x[lo] * (hi-k) + x[hi] * (k-lo))


def audit_target_tokenizer(
    tokenizer: Any,
    targets: Sequence[str],
    *,
    max_target_length: int,
    max_unk_rate: float = 0.05,
    max_truncation_rate: float = 0.05,
    batch_size: int = 256,
    split: str = "target",
    require_zero_truncation: bool = False,
) -> Dict[str, Any]:
    """Target-only mBART compatibility audit for Vietnamese labels."""
    norm = [normalize_mt_text_v1(t) for t in targets]
    if any(not t for t in norm):
        raise RuntimeError("Empty Vietnamese target reached tokenizer audit")
    lengths: List[int] = []
    unks = 0
    token_count = 0
    unk_id = tokenizer.unk_token_id
    for start in range(0, len(norm), int(batch_size)):
        batch = norm[start:start+int(batch_size)]
        enc = tokenizer(text_target=batch, add_special_tokens=True, truncation=False, return_attention_mask=False)
        for ids in enc["input_ids"]:
            ids = list(ids)
            lengths.append(len(ids))
            token_count += len(ids)
            if unk_id is not None:
                unks += sum(1 for x in ids if int(x) == int(unk_id))
    trunc = sum(1 for n in lengths if n > int(max_target_length))
    report = {
        "n": len(norm),
        "split": str(split),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "target": {
            "token_count": int(token_count),
            "mean": float(token_count / len(norm)) if norm else 0.0,
            "p50": _percentile(lengths, 50),
            "p95": _percentile(lengths, 95),
            "p99": _percentile(lengths, 99),
            "max": max(lengths) if lengths else 0,
            "unk_count": int(unks),
            "unk_rate": float(unks / token_count) if token_count else 0.0,
            "truncation_count": int(trunc),
            "truncation_rate": float(trunc / len(norm)) if norm else 0.0,
        },
        "gates": {"max_unk_rate": float(max_unk_rate), "max_truncation_rate": float(max_truncation_rate)},
    }
    failed = []
    if report["target"]["unk_rate"] > float(max_unk_rate):
        failed.append(f"{split}_unk_rate")
    trunc_rate = report["target"]["truncation_rate"]
    if require_zero_truncation:
        if int(report["target"]["truncation_count"]) != 0 or trunc_rate != 0.0:
            failed.append(f"{split}_truncation_must_be_zero")
    elif trunc_rate > float(max_truncation_rate):
        failed.append(f"{split}_truncation_rate")
    report["failed_checks"] = failed
    report["passed"] = not failed
    report["require_zero_truncation"] = bool(require_zero_truncation)
    return report


def audit_direct_targets_by_split(
    tokenizer: Any,
    *,
    train_targets: Sequence[str],
    validation_targets: Sequence[str],
    max_target_length: int,
    max_unk_rate: float = 0.05,
    max_truncation_rate: float = 0.05,
) -> Dict[str, Any]:
    """Train may truncate under policy; validation truncation must be exactly 0."""
    train = audit_target_tokenizer(
        tokenizer, train_targets, max_target_length=max_target_length,
        max_unk_rate=max_unk_rate, max_truncation_rate=max_truncation_rate, split="train",
        require_zero_truncation=False,
    )
    validation = audit_target_tokenizer(
        tokenizer, validation_targets, max_target_length=max_target_length,
        max_unk_rate=max_unk_rate, max_truncation_rate=0.0, split="validation",
        require_zero_truncation=True,
    )
    failed = list(train.get("failed_checks") or []) + list(validation.get("failed_checks") or [])
    return {
        "train": train,
        "validation": validation,
        "failed_checks": failed,
        "passed": not failed,
        "tokenizer_fingerprint": train.get("tokenizer_fingerprint"),
    }
