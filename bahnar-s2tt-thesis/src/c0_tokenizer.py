"""C0 serving tokenizer: same Hub pin and fingerprint as RQ1 restore.

RQ1 ``restore_mt_for_rq1`` / ``_default_load_mt`` call ``load_mt_tokenizer``
then ``tokenizer_fingerprint``. That fingerprint includes ``name_or_path``,
so a checkpoint-local tokenizer must not be compared with the stored contract
hash. Serving therefore loads the Hub tokenizer. Offline local-vs-Hub parity
uses a separate canonical fingerprint that omits ``name_or_path`` and is never
compared with the RQ1 contract hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from src.mt_contract import LOCKED_MODEL_ID, LOCKED_MODEL_REVISION, assert_locked_bartpho_baseline
from src.mt_tokenize import load_mt_tokenizer, tokenizer_fingerprint

TOKENIZER_SOURCE_HUB_PINNED_RQ1 = "hub_pinned_rq1"
PARITY_VERIFIED = "VERIFIED"
PARITY_NOT_VERIFIED = "NOT_VERIFIED"
PARITY_MISMATCH = "MISMATCH"


class C0TokenizerError(RuntimeError):
    """Tokenizer identity is missing, unusable, or does not match the contract."""

    error_code = "tokenizer_mismatch"


def canonical_tokenizer_fingerprint(tokenizer: Any) -> str:
    """Same digest as ``tokenizer_fingerprint`` except ``name_or_path`` is omitted.

    Do not compare this value with the SHA stored in an RQ1 training contract.
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
        }
    )
    try:
        sp = getattr(tokenizer, "special_tokens_map", None)
        if sp:
            _upd({"special_tokens_map": dict(sp)})
    except Exception:
        pass
    try:
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
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
    try:
        vocab_file = getattr(tokenizer, "vocab_file", None)
        if isinstance(vocab_file, str) and vocab_file:
            p = Path(vocab_file)
            if p.is_file():
                hh = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hh.update(chunk)
                _upd({"vocab_file_sha256": hh.hexdigest(), "vocab_file_name": p.name})
        else:
            sp_model = getattr(tokenizer, "sp_model", None)
            if sp_model is not None and hasattr(sp_model, "serialized_model_proto"):
                proto = sp_model.serialized_model_proto()
                _upd({"spm_proto_sha256": hashlib.sha256(proto).hexdigest()})
    except Exception:
        pass
    return h.hexdigest()


def load_contract_json(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C0TokenizerError(f"Training contract must be a JSON object: {path}")
    return raw


def resolve_tokenizer_pins(config: Mapping[str, Any]) -> tuple[str, str]:
    model_id = str(config.get("tokenizer_model_id") or LOCKED_MODEL_ID).strip()
    revision = str(config.get("tokenizer_revision") or LOCKED_MODEL_REVISION).strip()
    try:
        assert_locked_bartpho_baseline(model_id, revision)
    except RuntimeError as exc:
        raise C0TokenizerError(str(exc)) from exc
    return model_id, revision


def verify_hub_tokenizer_against_contract(
    tokenizer: Any,
    contract: Mapping[str, Any],
    *,
    fingerprint_fn: Callable[[Any], str] = tokenizer_fingerprint,
) -> dict[str, Any]:
    """Compare the original RQ1 fingerprint of a Hub tokenizer with the contract."""
    expected = str(contract.get("tokenizer_fingerprint") or "").strip()
    contract_id = str(contract.get("model_id") or "").strip()
    contract_rev = str(contract.get("model_revision") or "").strip()
    if contract_id and contract_id != LOCKED_MODEL_ID:
        raise C0TokenizerError(
            f"Training contract model_id {contract_id!r} is not {LOCKED_MODEL_ID!r}"
        )
    if contract_rev and contract_rev != LOCKED_MODEL_REVISION:
        raise C0TokenizerError(
            f"Training contract model_revision {contract_rev!r} is not {LOCKED_MODEL_REVISION!r}"
        )
    if not expected:
        return {
            "tokenizer_parity": PARITY_NOT_VERIFIED,
            "tokenizer_source": TOKENIZER_SOURCE_HUB_PINNED_RQ1,
            "reason": "training contract has no tokenizer_fingerprint",
        }
    got = fingerprint_fn(tokenizer)
    if got != expected:
        raise C0TokenizerError(
            "MT tokenizer fingerprint mismatch with RQ1 training contract"
        )
    return {
        "tokenizer_parity": PARITY_VERIFIED,
        "tokenizer_source": TOKENIZER_SOURCE_HUB_PINNED_RQ1,
        "tokenizer_fingerprint": got,
    }


def load_serving_mt_tokenizer(
    config: Mapping[str, Any],
    *,
    loader: Callable[[str, str], Any] = load_mt_tokenizer,
    fingerprint_fn: Callable[[Any], str] = tokenizer_fingerprint,
    contract_reader: Callable[[Path], Mapping[str, Any]] = load_contract_json,
) -> dict[str, Any]:
    """Load the RQ1 Hub tokenizer. Fail closed on fingerprint mismatch."""
    model_id, revision = resolve_tokenizer_pins(config)
    tokenizer = loader(model_id, revision)
    identity: dict[str, Any] = {
        "tokenizer": tokenizer,
        "tokenizer_model_id": model_id,
        "tokenizer_revision": revision,
        "tokenizer_source": TOKENIZER_SOURCE_HUB_PINNED_RQ1,
        "tokenizer_parity": PARITY_NOT_VERIFIED,
    }
    contract_path = str(config.get("mt_training_contract") or "").strip()
    if not contract_path:
        identity["reason"] = "mt_training_contract was not provided"
        return identity
    path = Path(contract_path)
    if not path.is_file():
        raise C0TokenizerError(f"MT training contract not found: {path}")
    checked = verify_hub_tokenizer_against_contract(
        tokenizer, contract_reader(path), fingerprint_fn=fingerprint_fn
    )
    identity.update(checked)
    return identity


def special_token_ids(tokenizer: Any) -> dict[str, Any]:
    return {
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }


def encode_decode_parity(
    left: Any,
    right: Any,
    texts: list[str],
    *,
    max_source_length: int,
) -> dict[str, Any]:
    """Compare two already-loaded tokenizers. Not an RQ1 contract check."""
    mismatches: list[str] = []
    if special_token_ids(left) != special_token_ids(right):
        mismatches.append("special_token_ids")
    left_vocab = left.get_vocab() if callable(getattr(left, "get_vocab", None)) else None
    right_vocab = right.get_vocab() if callable(getattr(right, "get_vocab", None)) else None
    if left_vocab != right_vocab:
        mismatches.append("vocab")
    rows = []
    for text in texts:
        l_enc = left(text, padding=True, truncation=True, max_length=int(max_source_length))
        r_enc = right(text, padding=True, truncation=True, max_length=int(max_source_length))
        same_ids = list(l_enc["input_ids"]) == list(r_enc["input_ids"])
        same_mask = True
        if "attention_mask" in l_enc or "attention_mask" in r_enc:
            same_mask = list(l_enc.get("attention_mask") or []) == list(r_enc.get("attention_mask") or [])
        l_dec = left.decode(l_enc["input_ids"], skip_special_tokens=True)
        r_dec = right.decode(r_enc["input_ids"], skip_special_tokens=True)
        row = {
            "text": text,
            "input_ids_match": same_ids,
            "attention_mask_match": same_mask,
            "decode_match": l_dec == r_dec,
        }
        rows.append(row)
        if not (same_ids and same_mask and l_dec == r_dec):
            mismatches.append(text)
    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "canonical_fingerprint_left": canonical_tokenizer_fingerprint(left),
        "canonical_fingerprint_right": canonical_tokenizer_fingerprint(right),
        "canonical_match": canonical_tokenizer_fingerprint(left) == canonical_tokenizer_fingerprint(right),
        "utterances": rows,
    }


def verification_exit_code(payload: Mapping[str, Any]) -> int:
    local = payload.get("local_canonical_parity")
    if isinstance(local, Mapping) and not local.get("matched"):
        return 1
    if payload.get("error"):
        return 1
    return 0 if payload.get("tokenizer_parity") == PARITY_VERIFIED else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the C0 serving tokenizer against an RQ1 MT contract")
    parser.add_argument("--contract", required=True, type=Path, help="mt_training_contract.json")
    parser.add_argument(
        "--local-tokenizer",
        type=Path,
        default=None,
        help="Optional checkpoint tokenizer dir for canonical/local parity. Never compared with the RQ1 hash.",
    )
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {
        "tokenizer_model_id": LOCKED_MODEL_ID,
        "tokenizer_revision": LOCKED_MODEL_REVISION,
        "tokenizer_source": TOKENIZER_SOURCE_HUB_PINNED_RQ1,
        "tokenizer_parity": PARITY_NOT_VERIFIED,
    }
    try:
        loaded = load_serving_mt_tokenizer({"mt_training_contract": str(args.contract)})
        payload.update({k: v for k, v in loaded.items() if k != "tokenizer"})
        if args.local_tokenizer is not None:
            from transformers import AutoTokenizer

            local = AutoTokenizer.from_pretrained(str(args.local_tokenizer))
            if getattr(local, "pad_token", None) is None and getattr(local, "eos_token", None) is not None:
                local.pad_token = local.eos_token
            from src.data_utils import normalize_bahnar_ctc_v1

            samples = [
                normalize_bahnar_ctc_v1("Pơlei Bahnar"),
                normalize_bahnar_ctc_v1("Inh jua"),
                normalize_bahnar_ctc_v1("Kơthơ inh chih ăn kơ iĕm ki"),
            ]
            payload["local_canonical_parity"] = encode_decode_parity(
                loaded["tokenizer"], local, samples, max_source_length=256
            )
            if not payload["local_canonical_parity"].get("matched"):
                payload["error"] = "local tokenizer canonical/encode-decode parity failed"
                payload["error_code"] = "local_tokenizer_parity_failed"
    except C0TokenizerError as exc:
        payload["tokenizer_parity"] = PARITY_MISMATCH
        payload["error"] = str(exc)
        payload["error_code"] = getattr(exc, "error_code", "tokenizer_mismatch")
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 1
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: tokenizer could not be verified"
        payload["error_code"] = "tokenizer_error"
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return verification_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
