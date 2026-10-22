"""Tokenizer identity tests. These do not download Hub weights or load MT checkpoints."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.c0_tokenizer import (
    LOCKED_MODEL_ID,
    LOCKED_MODEL_REVISION,
    PARITY_NOT_VERIFIED,
    PARITY_VERIFIED,
    C0TokenizerError,
    canonical_tokenizer_fingerprint,
    encode_decode_parity,
    load_serving_mt_tokenizer,
    special_token_ids,
    tokenizer_fingerprint,
    verification_exit_code,
    verify_hub_tokenizer_against_contract,
)
from src.data_utils import normalize_bahnar_ctc_v1

RESEARCH_BAHNAR = (
    "Pơlei Bahnar",
    "Inh jua",
    "Kơthơ inh chih ăn kơ iĕm ki, inh khă kuă ăn iĕm tơbôl păng bơngai pơm kư̆ kă.",
)


class FakeTokenizer:
    def __init__(self, name_or_path: str, vocab: dict[str, int]):
        self.name_or_path = name_or_path
        self.vocab_size = len(vocab)
        self.model_max_length = 256
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 1
        self.unk_token_id = 2
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "</s>"
        self.unk_token = "<unk>"
        self.special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "</s>", "unk_token": "<unk>"}
        self.init_kwargs = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "model_max_length": 256,
            "tokenizer_class": "FakeTokenizer",
        }
        self._vocab = dict(vocab)
        self._id_to_token = {i: t for t, i in vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, padding=True, truncation=True, max_length=256, add_special_tokens=True, return_attention_mask=True):
        tokens = str(text).split()
        ids = [self._vocab.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if truncation:
            ids = ids[: int(max_length)]
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = [1] * len(ids)
        return out

    def decode(self, ids, skip_special_tokens=True):
        specials = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        pieces = []
        for i in ids:
            if skip_special_tokens and i in specials:
                continue
            pieces.append(self._id_to_token.get(int(i), "<unk>"))
        return " ".join(pieces)


VOCAB = {"<s>": 0, "</s>": 1, "<unk>": 2, "pơlei": 3, "bahnar": 4, "inh": 5, "jua": 6, "kơthơ": 7}


def test_canonical_fingerprint_ignores_name_or_path_and_is_not_the_rq1_hash():
    hub = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    local = FakeTokenizer("/some/checkpoint-17500", VOCAB)
    assert tokenizer_fingerprint(hub) != tokenizer_fingerprint(local)
    assert canonical_tokenizer_fingerprint(hub) == canonical_tokenizer_fingerprint(local)
    # Stored RQ1 hashes cannot have name_or_path stripped after the fact.
    assert tokenizer_fingerprint(hub) != canonical_tokenizer_fingerprint(hub)


def test_hub_fingerprint_match_is_verified():
    tok = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    contract = {
        "model_id": LOCKED_MODEL_ID,
        "model_revision": LOCKED_MODEL_REVISION,
        "tokenizer_fingerprint": tokenizer_fingerprint(tok),
    }
    result = verify_hub_tokenizer_against_contract(tok, contract)
    assert result["tokenizer_parity"] == PARITY_VERIFIED
    assert result["tokenizer_source"] == "hub_pinned_rq1"


def test_hub_fingerprint_mismatch_fails_closed():
    tok = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    contract = {
        "model_id": LOCKED_MODEL_ID,
        "model_revision": LOCKED_MODEL_REVISION,
        "tokenizer_fingerprint": "0" * 64,
    }
    with pytest.raises(C0TokenizerError, match="fingerprint mismatch"):
        verify_hub_tokenizer_against_contract(tok, contract)


def test_missing_contract_fingerprint_is_not_verified():
    tok = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    result = verify_hub_tokenizer_against_contract(
        tok, {"model_id": LOCKED_MODEL_ID, "model_revision": LOCKED_MODEL_REVISION}
    )
    assert result["tokenizer_parity"] == PARITY_NOT_VERIFIED


def test_wrong_contract_model_id_fails_closed():
    tok = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    with pytest.raises(C0TokenizerError, match="model_id"):
        verify_hub_tokenizer_against_contract(
            tok,
            {
                "model_id": "some-other-model",
                "model_revision": LOCKED_MODEL_REVISION,
                "tokenizer_fingerprint": tokenizer_fingerprint(tok),
            },
        )


def test_load_serving_tokenizer_without_contract_is_not_verified():
    def loader(model_id, revision):
        assert model_id == LOCKED_MODEL_ID
        assert revision == LOCKED_MODEL_REVISION
        return FakeTokenizer(model_id, VOCAB)

    loaded = load_serving_mt_tokenizer({}, loader=loader)
    assert loaded["tokenizer_parity"] == PARITY_NOT_VERIFIED
    assert loaded["tokenizer_source"] == "hub_pinned_rq1"


def test_load_serving_tokenizer_rejects_unlocked_pin():
    with pytest.raises(C0TokenizerError):
        load_serving_mt_tokenizer(
            {"tokenizer_model_id": "vinai/bartpho-syllable", "tokenizer_revision": "main"},
            loader=lambda *_a, **_k: FakeTokenizer("x", VOCAB),
        )


def test_load_serving_tokenizer_missing_contract_file_fails(tmp_path: Path):
    with pytest.raises(C0TokenizerError, match="not found"):
        load_serving_mt_tokenizer(
            {"mt_training_contract": str(tmp_path / "missing.json")},
            loader=lambda *_a, **_k: FakeTokenizer("vinai/bartpho-syllable", VOCAB),
        )


def test_encode_decode_parity_on_research_bahnar():
    hub = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    local = FakeTokenizer("/checkpoint-17500", VOCAB)
    texts = [normalize_bahnar_ctc_v1(t) for t in RESEARCH_BAHNAR]
    report = encode_decode_parity(hub, local, texts, max_source_length=256)
    assert report["matched"] is True
    assert report["canonical_match"] is True
    assert special_token_ids(hub) == special_token_ids(local)
    for row in report["utterances"]:
        assert row["input_ids_match"]
        assert row["attention_mask_match"]
        assert row["decode_match"]


def test_encode_decode_parity_detects_vocab_drift():
    hub = FakeTokenizer("vinai/bartpho-syllable", VOCAB)
    drifted = dict(VOCAB)
    drifted["bahnar"] = 99
    local = FakeTokenizer("/checkpoint-17500", drifted)
    report = encode_decode_parity(hub, local, [normalize_bahnar_ctc_v1("Pơlei Bahnar")], max_source_length=256)
    assert report["matched"] is False
    assert "vocab" in report["mismatches"]
    assert verification_exit_code({"tokenizer_parity": PARITY_VERIFIED, "local_canonical_parity": report}) == 1


def test_verification_exit_code_requires_local_match():
    assert verification_exit_code({"tokenizer_parity": PARITY_VERIFIED}) == 0
    assert verification_exit_code({"tokenizer_parity": PARITY_NOT_VERIFIED}) == 1
    assert verification_exit_code({"tokenizer_parity": PARITY_VERIFIED, "local_canonical_parity": {"matched": True}}) == 0
    assert verification_exit_code({"tokenizer_parity": PARITY_VERIFIED, "local_canonical_parity": {"matched": False}}) == 1
