"""RQ2 Notebook 12: raw pseudo-label inference with the fixed RQ1 C0 teacher.

Inference only. Every model call runs under ``model.eval()`` and
``torch.inference_mode()`` with greedy CTC / deterministic beam search; nothing
here trains, samples, or reads a Vietnamese reference.

Pipeline (peak-safe, one model resident at a time, like Notebook 06):

1. ``asr`` stage: C0 ASR over every item -> Bahnar transcript + CTC confidence.
2. ``mt`` stage: C0 MT over the ASR shard -> Vietnamese pseudo-label + MT confidence.
3. ``d0`` stage (only if the D0 agreement contract is enabled): D0 over every item.

Each stage writes fixed-membership shards (``shard_size`` items in input order)
under a checkpoint whose binding covers the NB11 input, teacher, D0, decoding
config, schema and code version. A corrupt shard is quarantined and recomputed
alone; a binding mismatch fails closed.

Confidence definitions
----------------------
ASR (``ctc_greedy_token_run_mean_logprob_v1``): take the greedy argmax path over
the valid (unpadded) frames. Consecutive frames with the same id form one run,
exactly as CTC collapsing does. Runs of the blank (= tokenizer pad) id and of
special ids other than ``unk`` emit nothing and are dropped. Each remaining run
is one emitted token whose score is the mean frame log-softmax of its id over
the run. ``asr_mean_logprob`` is the mean over emitted tokens (length
normalised); ``asr_confidence_raw = exp(asr_mean_logprob)``. The word delimiter
is an ordinary vocabulary id and counts as a token. Zero emitted tokens gives
NaN (the row is later excluded as empty / non-finite).

MT (``teacher_forced_mean_token_logprob_v1``): after deterministic generation,
the generated sequence is scored with one teacher-forced forward pass.
``mt_mean_logprob`` is the mean log-softmax of each generated token, over target
positions whose id is not a special token (decoder start, bos, eos, pad,
language codes are all excluded). This is the model probability of its own
output, not the beam score (no length penalty).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

import numpy as np

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import (
    SAMPLE_RATE,
    Nb11Input,
    atomic_write_text,
    write_json,
)

RAW_SCHEMA_VERSION = "rq2-pseudo-raw-1.0"
INFERENCE_CODE_VERSION = "rq2_pseudo_inference_v1"
STAGES = ("asr", "mt", "d0")
DEFAULT_SHARD_SIZE = 256

STATUS_OK = "OK"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED_ASR_EMPTY = "SKIPPED_ASR_EMPTY"
STATUS_SKIPPED_ASR_FAILED = "SKIPPED_ASR_FAILED"
STATUS_DISABLED = "DISABLED"

FORBIDDEN_REFERENCE_FIELDS = frozenset({
    "text_vi", "text_bahnar", "text_en", "pair_key",
    "reference_vi", "reference_text", "gold_vi", "gold_text", "vi_reference",
})

CARRIED_SOURCE_FIELDS = (
    "segment_uid", "source_id", "source_group_id", "segment_local_path",
    "segment_pcm16_sha256", "segment_wav_sha256", "duration_seconds",
    "vad_speech_fraction", "rms_dbfs", "peak_dbfs", "silence_fraction_energy",
)

RAW_RECORD_COLUMNS = list(CARRIED_SOURCE_FIELDS) + [
    "nb11_generation_id", "nb11_input_contract_sha256",
    "teacher_contract_sha256", "d0_agreement_contract_sha256", "raw_schema_version",
    "asr_status", "asr_text_raw", "asr_text_norm", "asr_valid_encoding", "asr_empty",
    "asr_token_count", "asr_mean_logprob", "asr_confidence_raw",
    "asr_frame_mean_logprob", "asr_blank_frame_fraction", "asr_n_frames", "asr_decode_warning",
    "mt_status", "pseudo_vi_raw", "pseudo_vi_norm", "mt_valid_encoding", "mt_empty",
    "mt_token_count", "mt_mean_logprob", "mt_confidence_raw",
    "mt_source_truncated", "mt_hit_max_length", "mt_decode_warning",
    "d0_status", "d0_vi_raw", "d0_vi_norm", "d0_valid_encoding", "d0_decode_warning",
    "teacher_d0_agreement_chrf", "teacher_d0_agreement_char_ratio",
    "asr_char_count", "asr_word_count", "pseudo_vi_char_count", "pseudo_vi_word_count",
    "target_source_char_ratio", "target_source_word_ratio",
]


class AudioIdentityError(RuntimeError):
    """Audio on disk differs from the locked identity (fatal, never an exclusion)."""


class ItemInferenceError(RuntimeError):
    """One segment failed in a way the teacher can isolate.

    Only this exception may become a ``FAILED`` row. Tokenizer, shape, config,
    and any other programming or runtime error aborts the stage instead of
    being written into U' as an exclusion.
    """


class CheckpointBindingError(RuntimeError):
    """A resume checkpoint belongs to a different contract (fail closed)."""


class ReferenceLeakError(RuntimeError):
    """A Vietnamese/Bahnar reference field reached the pseudo-label path."""


def assert_no_reference_fields(keys: Iterable[str], where: str = "pseudo-label record") -> None:
    bad = sorted(set(map(str, keys)) & FORBIDDEN_REFERENCE_FIELDS)
    if bad:
        raise ReferenceLeakError(f"reference field(s) {bad} must never enter the {where}")


# --------------------------------------------------------------------------- #
# Items + verified audio                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AudioItem:
    uid: str
    audio_path: Path
    pcm16_sha256: str
    n_samples: int


def items_from_nb11(nb11: Nb11Input) -> List[AudioItem]:
    return [
        AudioItem(
            uid=str(row["segment_uid"]),
            audio_path=nb11.project_root / str(row["segment_local_path"]),
            pcm16_sha256=str(row["segment_pcm16_sha256"]),
            n_samples=int(row["end_sample"]) - int(row["start_sample"]),
        )
        for row in nb11.rows
    ]


def items_from_audio_frame(audio_frame: Any) -> List[AudioItem]:
    """Items from a materialised G_validation audio frame (no text columns allowed)."""
    assert_no_reference_fields(audio_frame.columns, "validation inference items")
    return [
        AudioItem(
            uid=str(r["record_uid"]),
            audio_path=Path(r["audio_path"]),
            pcm16_sha256=str(r["sha256_pcm"]),
            n_samples=int(r["n_samples"]),
        )
        for _, r in audio_frame.iterrows()
    ]


def load_verified_waveform(item: AudioItem) -> np.ndarray:
    """Read a canonical 16 kHz mono PCM16 WAV and verify it before any model sees it.

    ``int16 / 32768`` is bit-identical to ``soundfile.read(dtype="float32")``,
    which is what RQ1 fed the models.
    """
    import soundfile as sf

    path = Path(item.audio_path)
    if not path.is_file():
        raise AudioIdentityError(f"audio missing for {item.uid}")
    with sf.SoundFile(str(path)) as handle:
        if int(handle.samplerate) != SAMPLE_RATE or int(handle.channels) != 1:
            raise AudioIdentityError(f"audio is not 16 kHz mono for {item.uid}")
        pcm = np.asarray(handle.read(dtype="int16", always_2d=False), dtype="<i2")
    if pcm.size != int(item.n_samples):
        raise AudioIdentityError(f"sample count differs from locked identity for {item.uid}")
    if hashlib.sha256(pcm.tobytes()).hexdigest() != item.pcm16_sha256:
        raise AudioIdentityError(f"PCM hash differs from locked identity for {item.uid}")
    return pcm.astype(np.float32) / np.float32(32768.0)


# --------------------------------------------------------------------------- #
# Confidence + text features (pure)                                            #
# --------------------------------------------------------------------------- #
def _finite_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def ctc_greedy_confidence(
    log_probs: np.ndarray,
    greedy_ids: Sequence[int],
    *,
    blank_id: int,
    ignore_ids: Iterable[int] = (),
) -> Dict[str, Any]:
    lp = np.asarray(log_probs, dtype=np.float64)
    ids = np.asarray(greedy_ids, dtype=np.int64)
    n_frames = int(ids.shape[0])
    if lp.ndim != 2 or lp.shape[0] != n_frames:
        raise ValueError("log_probs must be [frames, vocab] aligned with greedy_ids")
    if n_frames == 0:
        return {
            "asr_token_count": 0, "asr_mean_logprob": None, "asr_confidence_raw": None,
            "asr_frame_mean_logprob": None, "asr_blank_frame_fraction": None, "asr_n_frames": 0,
        }
    ignored = set(int(i) for i in ignore_ids) | {int(blank_id)}
    best = lp[np.arange(n_frames), ids]
    token_scores: List[float] = []
    start = 0
    for t in range(1, n_frames + 1):
        if t == n_frames or ids[t] != ids[start]:
            if int(ids[start]) not in ignored:
                token_scores.append(float(best[start:t].mean()))
            start = t
    mean = float(np.mean(token_scores)) if token_scores else float("nan")
    return {
        "asr_token_count": len(token_scores),
        "asr_mean_logprob": _finite_or_none(mean),
        "asr_confidence_raw": _finite_or_none(math.exp(mean)) if math.isfinite(mean) else None,
        "asr_frame_mean_logprob": _finite_or_none(best.mean()),
        "asr_blank_frame_fraction": float(np.mean(ids == int(blank_id))),
        "asr_n_frames": n_frames,
    }


def seq2seq_mean_logprob(
    token_logprobs: Sequence[float],
    target_ids: Sequence[int],
    *,
    ignore_ids: Iterable[int],
) -> Dict[str, Any]:
    ignored = set(int(i) for i in ignore_ids)
    kept = [float(lp) for lp, tok in zip(token_logprobs, target_ids) if int(tok) not in ignored]
    mean = float(np.mean(kept)) if kept else float("nan")
    return {
        "mt_token_count": len(kept),
        "mt_mean_logprob": _finite_or_none(mean),
        "mt_confidence_raw": _finite_or_none(math.exp(mean)) if math.isfinite(mean) else None,
    }


def text_is_valid(text: Any) -> bool:
    """False for non-strings, U+FFFD, lone surrogates, or control characters other than whitespace."""
    if not isinstance(text, str):
        return False
    for ch in text:
        if ch == "\ufffd" or 0xD800 <= ord(ch) <= 0xDFFF:
            return False
        if unicodedata.category(ch) == "Cc" and not ch.isspace():
            return False
    return True


def sentence_chrfpp(hyp: str, ref: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_chrf(str(hyp), [str(ref)], word_order=2).score)


def agreement_features(c0_vi: Optional[str], d0_vi: Optional[str]) -> Dict[str, Optional[float]]:
    """Symmetric sentence chrF++ between C0 and D0 plus a char-ratio diagnostic.

    No Vietnamese reference is involved. Both empty -> undefined (None); one
    empty -> 0.0.
    """
    import difflib

    a = str(c0_vi or "")
    b = str(d0_vi or "")
    if c0_vi is None or d0_vi is None or (not a and not b):
        return {"teacher_d0_agreement_chrf": None, "teacher_d0_agreement_char_ratio": None}
    if not a or not b:
        return {"teacher_d0_agreement_chrf": 0.0, "teacher_d0_agreement_char_ratio": 0.0}
    chrf = 0.5 * (sentence_chrfpp(a, b) + sentence_chrfpp(b, a))
    return {
        "teacher_d0_agreement_chrf": float(chrf),
        "teacher_d0_agreement_char_ratio": float(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()),
    }


def text_length_features(asr_norm: Optional[str], vi_norm: Optional[str]) -> Dict[str, Any]:
    src = str(asr_norm or "")
    tgt = str(vi_norm or "")
    src_chars = len(src)
    tgt_chars = len(tgt)
    src_words = len(src.split())
    tgt_words = len(tgt.split())
    return {
        "asr_char_count": src_chars,
        "asr_word_count": src_words,
        "pseudo_vi_char_count": tgt_chars,
        "pseudo_vi_word_count": tgt_words,
        "target_source_char_ratio": (tgt_chars / src_chars) if src_chars else None,
        "target_source_word_ratio": (tgt_words / src_words) if src_words else None,
    }


# --------------------------------------------------------------------------- #
# Hugging Face adapters (mirror src/rq1_inference.py call-for-call)            #
# --------------------------------------------------------------------------- #
def configure_deterministic_torch(seed: int = 42) -> Dict[str, Any]:
    import torch

    torch.manual_seed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"seed": int(seed), "cudnn_benchmark": False, "cudnn_deterministic": True}


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _ignored_special_ids(tokenizer: Any, *, keep_unk: bool) -> List[int]:
    ids = set(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            ids.add(int(value))
    unk = getattr(tokenizer, "unk_token_id", None)
    if keep_unk and unk is not None:
        ids.discard(int(unk))
    return sorted(ids)


class HfCtcAsrAdapter:
    """Greedy CTC exactly as ``run_asr_predictions`` plus per-token run confidence."""

    def __init__(self, model: Any, processor: Any):
        self.model = model
        self.processor = processor
        tok = processor.tokenizer
        self.blank_id = int(tok.pad_token_id)
        self.ignore_ids = _ignored_special_ids(tok, keep_unk=True)

    def _valid_frames(self, logits: Any, attention_mask: Any) -> List[int]:
        n = int(logits.shape[1])
        if attention_mask is None or not hasattr(self.model, "_get_feat_extract_output_lengths"):
            return [n] * int(logits.shape[0])
        lengths = self.model._get_feat_extract_output_lengths(attention_mask.sum(-1))
        return [min(n, int(x)) for x in lengths.tolist()]

    def __call__(self, waveforms: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        import torch

        self.model.eval()
        batch = self.processor(list(waveforms), sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
        device = _model_device(self.model)
        kwargs = {"input_values": batch.input_values.to(device)}
        am = getattr(batch, "attention_mask", None)
        if am is not None:
            kwargs["attention_mask"] = am.to(device)
        with torch.inference_mode():
            logits = self.model(**kwargs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            texts = self.processor.batch_decode(pred_ids)
            log_probs = torch.log_softmax(logits.double(), dim=-1).cpu().numpy()
        valid = self._valid_frames(logits, kwargs.get("attention_mask"))
        ids_np = pred_ids.cpu().numpy()
        out = []
        for i, text in enumerate(texts):
            n = valid[i]
            conf = ctc_greedy_confidence(log_probs[i, :n], ids_np[i, :n], blank_id=self.blank_id, ignore_ids=self.ignore_ids)
            conf["text_raw"] = str(text)
            out.append(conf)
        return out


class HfSeq2SeqMtAdapter:
    """C0 MT exactly as ``run_mt_from_asr`` plus teacher-forced confidence."""

    def __init__(self, model: Any, tokenizer: Any, *, max_source_length: int, generation_max_length: int, num_beams: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = int(max_source_length)
        self.generation_max_length = int(generation_max_length)
        self.num_beams = int(num_beams)
        self.ignore_ids = _ignored_special_ids(tokenizer, keep_unk=False)

    def __call__(self, sources: Sequence[str]) -> List[Dict[str, Any]]:
        import torch
        from src.data_utils import normalize_bahnar_ctc_v1

        self.model.eval()
        src = [normalize_bahnar_ctc_v1(x) for x in sources]
        untruncated = [
            len(self.tokenizer(s, truncation=False, add_special_tokens=True, return_attention_mask=False)["input_ids"])
            for s in src
        ]
        device = _model_device(self.model)
        enc = self.tokenizer(src, padding=True, truncation=True, max_length=self.max_source_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outs = self.model.generate(**enc, max_length=self.generation_max_length, num_beams=self.num_beams)
            texts = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            decoder_in = outs[:, :-1]
            labels = outs[:, 1:]
            forward_kwargs = {"input_ids": enc["input_ids"], "decoder_input_ids": decoder_in}
            if "attention_mask" in enc:
                forward_kwargs["attention_mask"] = enc["attention_mask"]
            logits = self.model(**forward_kwargs).logits
            token_lp = torch.log_softmax(logits.double(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_lp_np = token_lp.cpu().numpy()
        labels_np = labels.cpu().numpy()
        pad = getattr(self.tokenizer, "pad_token_id", None)
        out = []
        for i, text in enumerate(texts):
            conf = seq2seq_mean_logprob(token_lp_np[i], labels_np[i], ignore_ids=self.ignore_ids)
            n_generated = int(sum(1 for t in outs[i].tolist() if pad is None or int(t) != int(pad)))
            conf.update({
                "text_raw": str(text),
                "source_truncated": bool(untruncated[i] > self.max_source_length),
                "hit_max_length": bool(n_generated >= self.generation_max_length),
            })
            out.append(conf)
        return out


class HfDirectD0Adapter:
    """D0 exactly as ``run_direct_predictions``. Its output is only an agreement feature."""

    def __init__(self, model: Any, feature_extractor: Any, tokenizer: Any, *, target_lang: str, generation_max_length: int, num_beams: int):
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.forced_bos = int(tokenizer.lang_code_to_id[target_lang])
        self.generation_max_length = int(generation_max_length)
        self.num_beams = int(num_beams)

    def __call__(self, waveforms: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        import torch

        self.model.eval()
        feat = self.feature_extractor(
            list(waveforms), sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt", return_attention_mask=True,
        )
        device = _model_device(self.model)
        kwargs = {"input_values": feat["input_values"].to(device)}
        if "attention_mask" in feat:
            kwargs["attention_mask"] = feat["attention_mask"].to(device)
        with torch.inference_mode():
            outs = self.model.generate(
                **kwargs,
                max_length=self.generation_max_length,
                num_beams=self.num_beams,
                forced_bos_token_id=self.forced_bos,
            )
            texts = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        return [{"text_raw": str(t)} for t in texts]


# --------------------------------------------------------------------------- #
# Resumable shard checkpoint                                                   #
# --------------------------------------------------------------------------- #
def inference_binding(
    *,
    pool: str,
    input_identity_sha256: str,
    teacher_contract_sha256: str,
    d0_agreement_contract_sha256: str,
    decoding_config_sha256: str,
) -> Dict[str, Any]:
    return {
        "pool": str(pool),
        "input_identity_sha256": str(input_identity_sha256),
        "teacher_contract_sha256": str(teacher_contract_sha256),
        "d0_agreement_contract_sha256": str(d0_agreement_contract_sha256),
        "decoding_config_sha256": str(decoding_config_sha256),
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "inference_code_version": INFERENCE_CODE_VERSION,
    }


def _fatal(exc: BaseException) -> bool:
    if isinstance(exc, (AudioIdentityError, MemoryError, CheckpointBindingError, ReferenceLeakError)):
        return True
    return "out of memory" in str(exc).lower()


class InferenceCheckpoint:
    """Fixed-membership shards. Shard ``k`` always holds ``uids[k*S:(k+1)*S]``."""

    def __init__(self, root: Union[str, Path], *, binding: Mapping[str, Any], ordered_uids: Sequence[str], shard_size: int = DEFAULT_SHARD_SIZE):
        if int(shard_size) < 1:
            raise ValueError("shard_size must be >= 1")
        uids = [str(u) for u in ordered_uids]
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate uid in inference pool")
        self.root = Path(root)
        self.uids = uids
        self.shard_size = int(shard_size)
        self.state = {
            "binding": dict(binding),
            "shard_size": self.shard_size,
            "n_items": len(uids),
            "ordered_uid_sha256": sha256_json(uids),
        }
        self.binding_sha256 = sha256_json(self.state)
        self.quarantined: List[str] = []
        self._open()

    def _open(self) -> None:
        state_path = self.root / "state.json"
        if state_path.is_file():
            try:
                existing = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise CheckpointBindingError(f"unreadable checkpoint state: {state_path}") from exc
            if existing.get("binding_sha256") != self.binding_sha256:
                raise CheckpointBindingError(
                    "pseudo-label checkpoint belongs to a different NB11 input / teacher / D0 / decoding / "
                    "schema contract; refusing to reuse it (archive it explicitly with archive_checkpoint)"
                )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(state_path, {**self.state, "binding_sha256": self.binding_sha256})

    @property
    def n_shards(self) -> int:
        return (len(self.uids) + self.shard_size - 1) // self.shard_size

    def shard_uids(self, k: int) -> List[str]:
        return self.uids[k * self.shard_size:(k + 1) * self.shard_size]

    def _paths(self, stage: str, k: int):
        base = self.root / stage / f"shard-{k:05d}"
        return base.with_suffix(".jsonl"), base.with_suffix(".json")

    def _quarantine(self, stage: str, k: int, reason: str) -> None:
        qdir = self.root / "quarantine" / stage
        qdir.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        for p in self._paths(stage, k):
            if p.exists():
                shutil.move(str(p), str(qdir / f"{p.name}.{tag}"))
        self.quarantined.append(f"{stage}/{k}:{reason}")

    def load_shard(self, stage: str, k: int, *, depends_on: Optional[Mapping[str, str]] = None) -> Optional[List[dict]]:
        data_path, meta_path = self._paths(stage, k)
        if not meta_path.is_file():
            if data_path.exists():
                self._quarantine(stage, k, "no_meta")
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self._quarantine(stage, k, "bad_meta")
            return None
        if meta.get("binding_sha256") != self.binding_sha256:
            raise CheckpointBindingError(f"{stage} shard {k} was written under a different binding")
        expected_uids = self.shard_uids(k)
        if meta.get("stage") != stage:
            self._quarantine(stage, k, "bad_stage")
            return None
        if meta.get("shard_index") != int(k):
            self._quarantine(stage, k, "bad_shard_index")
            return None
        if meta.get("uids_sha256") != sha256_json(expected_uids):
            self._quarantine(stage, k, "bad_uids_sha256")
            return None
        if dict(meta.get("depends_on") or {}) != dict(depends_on or {}):
            self._quarantine(stage, k, "stale_dependency")
            return None
        if not data_path.is_file() or sha256_file(data_path) != meta.get("rows_sha256"):
            self._quarantine(stage, k, "hash_mismatch")
            return None
        try:
            rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:  # noqa: BLE001
            self._quarantine(stage, k, "bad_rows")
            return None
        if meta.get("n_rows") != len(rows) or len(rows) != len(expected_uids):
            self._quarantine(stage, k, "bad_n_rows")
            return None
        if [str(r.get("uid")) for r in rows] != expected_uids:
            self._quarantine(stage, k, "membership_mismatch")
            return None
        return rows

    def shard_sha256(self, stage: str, k: int) -> str:
        meta = json.loads(self._paths(stage, k)[1].read_text(encoding="utf-8"))
        return str(meta["rows_sha256"])

    def write_shard(self, stage: str, k: int, rows: Sequence[Mapping[str, Any]], *, depends_on: Optional[Mapping[str, str]] = None) -> None:
        if [str(r.get("uid")) for r in rows] != self.shard_uids(k):
            raise RuntimeError(f"{stage} shard {k} rows do not match its fixed membership")
        for r in rows:
            assert_no_reference_fields(r.keys(), f"{stage} checkpoint shard")
        data_path, meta_path = self._paths(stage, k)
        text = "".join(json.dumps(dict(r), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        atomic_write_text(data_path, text)
        write_json(meta_path, {
            "stage": stage,
            "shard_index": int(k),
            "binding_sha256": self.binding_sha256,
            "uids_sha256": sha256_json(self.shard_uids(k)),
            "rows_sha256": sha256_file(data_path),
            "n_rows": len(rows),
            "depends_on": dict(depends_on or {}),
        })


def archive_checkpoint(root: Union[str, Path]) -> Optional[Path]:
    """Move an incompatible checkpoint aside (never deletes)."""
    src = Path(root)
    if not src.exists():
        return None
    dest = src.with_name(src.name + f".stale-{uuid.uuid4().hex[:8]}")
    shutil.move(str(src), str(dest))
    return dest


def _recover_item(exc: BaseException) -> Dict[str, str]:
    """Return a FAILED marker only for a declared per-item teacher failure."""
    if _fatal(exc) or not isinstance(exc, ItemInferenceError):
        raise exc
    return {"_failed": f"ItemInferenceError: {str(exc)[:200]}"}


def _call_with_item_fallback(fn: Callable[[List[Any]], List[Dict[str, Any]]], inputs: List[Any]) -> List[Dict[str, Any]]:
    """Run a batch. Only :class:`ItemInferenceError` is retried per item.

    A batch-level recoverable failure is retried one segment at a time, and
    only the segments that raise :class:`ItemInferenceError` become FAILED
    rows. Any other exception, including a wrong output count, aborts the stage.
    """
    try:
        out = list(fn(list(inputs)))
    except Exception as exc:  # noqa: BLE001
        if len(inputs) == 1:
            return [_recover_item(exc)]
        _recover_item(exc)
    else:
        if len(out) != len(inputs):
            raise RuntimeError("adapter returned a different number of outputs")
        return out
    results = []
    for x in inputs:
        try:
            res = list(fn([x]))
        except Exception as exc:  # noqa: BLE001
            results.append(_recover_item(exc))
            continue
        if len(res) != 1:
            raise RuntimeError("adapter returned a different number of outputs")
        results.append(res[0])
    return results


def _chunks(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), int(n)):
        yield seq[i:i + int(n)]


def _check_items(ckpt: InferenceCheckpoint, items: Sequence[AudioItem]) -> Dict[str, AudioItem]:
    if [it.uid for it in items] != ckpt.uids:
        raise CheckpointBindingError("inference items do not match the checkpoint's ordered uid list")
    return {it.uid: it for it in items}


def _asr_row(uid: str, out: Mapping[str, Any]) -> Dict[str, Any]:
    from src.data_utils import normalize_bahnar_ctc_v1

    if "_failed" in out:
        return {"uid": uid, "asr_status": STATUS_FAILED, "asr_decode_warning": str(out["_failed"]),
                "asr_text_raw": None, "asr_text_norm": None, "asr_valid_encoding": False, "asr_empty": True,
                "asr_token_count": 0, "asr_mean_logprob": None, "asr_confidence_raw": None,
                "asr_frame_mean_logprob": None, "asr_blank_frame_fraction": None, "asr_n_frames": 0}
    raw = out.get("text_raw")
    valid = text_is_valid(raw)
    norm = normalize_bahnar_ctc_v1(raw) if valid else None
    return {
        "uid": uid,
        "asr_status": STATUS_OK,
        "asr_decode_warning": "",
        "asr_text_raw": raw if valid else None,
        "asr_text_norm": norm,
        "asr_valid_encoding": bool(valid),
        "asr_empty": not bool(norm),
        "asr_token_count": int(out.get("asr_token_count") or 0),
        "asr_mean_logprob": _finite_or_none(out.get("asr_mean_logprob")),
        "asr_confidence_raw": _finite_or_none(out.get("asr_confidence_raw")),
        "asr_frame_mean_logprob": _finite_or_none(out.get("asr_frame_mean_logprob")),
        "asr_blank_frame_fraction": _finite_or_none(out.get("asr_blank_frame_fraction")),
        "asr_n_frames": int(out.get("asr_n_frames") or 0),
    }


def run_asr_stage(
    ckpt: InferenceCheckpoint,
    items: Sequence[AudioItem],
    asr_fn: Callable[[List[np.ndarray]], List[Dict[str, Any]]],
    *,
    batch_size: int,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    by_uid = _check_items(ckpt, items)
    computed = reused = 0
    for k in range(ckpt.n_shards):
        if ckpt.load_shard("asr", k) is not None:
            reused += 1
            continue
        rows: List[Dict[str, Any]] = []
        for part in _chunks(ckpt.shard_uids(k), batch_size):
            waves = [load_verified_waveform(by_uid[u]) for u in part]
            outs = _call_with_item_fallback(asr_fn, waves)
            rows.extend(_asr_row(u, o) for u, o in zip(part, outs))
        ckpt.write_shard("asr", k, rows)
        computed += 1
        if progress:
            progress(f"asr shard {k + 1}/{ckpt.n_shards}")
    return {"stage": "asr", "computed_shards": computed, "reused_shards": reused}


def _mt_row(uid: str, out: Optional[Mapping[str, Any]], skipped: Optional[str]) -> Dict[str, Any]:
    from src.mt_normalize import normalize_mt_text_v1

    empty = {"uid": uid, "pseudo_vi_raw": None, "pseudo_vi_norm": None, "mt_valid_encoding": False, "mt_empty": True,
             "mt_token_count": 0, "mt_mean_logprob": None, "mt_confidence_raw": None,
             "mt_source_truncated": False, "mt_hit_max_length": False}
    if skipped:
        return {**empty, "mt_status": skipped, "mt_decode_warning": ""}
    if out is None or "_failed" in out:
        return {**empty, "mt_status": STATUS_FAILED, "mt_decode_warning": str((out or {}).get("_failed", "missing"))}
    raw = out.get("text_raw")
    valid = text_is_valid(raw)
    norm = normalize_mt_text_v1(raw) if valid else None
    return {
        "uid": uid,
        "mt_status": STATUS_OK,
        "mt_decode_warning": "",
        "pseudo_vi_raw": raw if valid else None,
        "pseudo_vi_norm": norm,
        "mt_valid_encoding": bool(valid),
        "mt_empty": not bool(norm),
        "mt_token_count": int(out.get("mt_token_count") or 0),
        "mt_mean_logprob": _finite_or_none(out.get("mt_mean_logprob")),
        "mt_confidence_raw": _finite_or_none(out.get("mt_confidence_raw")),
        "mt_source_truncated": bool(out.get("source_truncated")),
        "mt_hit_max_length": bool(out.get("hit_max_length")),
    }


def run_mt_stage(
    ckpt: InferenceCheckpoint,
    mt_fn: Callable[[List[str]], List[Dict[str, Any]]],
    *,
    batch_size: int,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Translate only non-empty, valid, successful ASR outputs; MT shard k depends on ASR shard k."""
    computed = reused = 0
    for k in range(ckpt.n_shards):
        asr_rows = ckpt.load_shard("asr", k)
        if asr_rows is None:
            raise RuntimeError(f"ASR stage incomplete: shard {k} missing")
        dep = {"asr": ckpt.shard_sha256("asr", k)}
        if ckpt.load_shard("mt", k, depends_on=dep) is not None:
            reused += 1
            continue
        todo = [r for r in asr_rows if r["asr_status"] == STATUS_OK and r["asr_valid_encoding"] and not r["asr_empty"]]
        outputs: Dict[str, Dict[str, Any]] = {}
        for part in _chunks(todo, batch_size):
            outs = _call_with_item_fallback(mt_fn, [r["asr_text_norm"] for r in part])
            outputs.update({r["uid"]: o for r, o in zip(part, outs)})
        rows = []
        for r in asr_rows:
            if r["asr_status"] != STATUS_OK or not r["asr_valid_encoding"]:
                rows.append(_mt_row(r["uid"], None, STATUS_SKIPPED_ASR_FAILED))
            elif r["asr_empty"]:
                rows.append(_mt_row(r["uid"], None, STATUS_SKIPPED_ASR_EMPTY))
            else:
                rows.append(_mt_row(r["uid"], outputs.get(r["uid"]), None))
        ckpt.write_shard("mt", k, rows, depends_on=dep)
        computed += 1
        if progress:
            progress(f"mt shard {k + 1}/{ckpt.n_shards}")
    return {"stage": "mt", "computed_shards": computed, "reused_shards": reused}


def _d0_row(uid: str, out: Mapping[str, Any]) -> Dict[str, Any]:
    from src.mt_normalize import normalize_mt_text_v1

    if "_failed" in out:
        return {"uid": uid, "d0_status": STATUS_FAILED, "d0_vi_raw": None, "d0_vi_norm": None,
                "d0_valid_encoding": False, "d0_decode_warning": str(out["_failed"])}
    raw = out.get("text_raw")
    valid = text_is_valid(raw)
    return {
        "uid": uid,
        "d0_status": STATUS_OK,
        "d0_vi_raw": raw if valid else None,
        "d0_vi_norm": normalize_mt_text_v1(raw) if valid else None,
        "d0_valid_encoding": bool(valid),
        "d0_decode_warning": "",
    }


def run_d0_stage(
    ckpt: InferenceCheckpoint,
    items: Sequence[AudioItem],
    d0_fn: Callable[[List[np.ndarray]], List[Dict[str, Any]]],
    *,
    batch_size: int,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    by_uid = _check_items(ckpt, items)
    computed = reused = 0
    for k in range(ckpt.n_shards):
        if ckpt.load_shard("d0", k) is not None:
            reused += 1
            continue
        rows: List[Dict[str, Any]] = []
        for part in _chunks(ckpt.shard_uids(k), batch_size):
            waves = [load_verified_waveform(by_uid[u]) for u in part]
            outs = _call_with_item_fallback(d0_fn, waves)
            rows.extend(_d0_row(u, o) for u, o in zip(part, outs))
        ckpt.write_shard("d0", k, rows)
        computed += 1
        if progress:
            progress(f"d0 shard {k + 1}/{ckpt.n_shards}")
    return {"stage": "d0", "computed_shards": computed, "reused_shards": reused}


def stage_pending_shards(ckpt: InferenceCheckpoint, stage: str) -> List[int]:
    pending = []
    for k in range(ckpt.n_shards):
        dep = None
        if stage == "mt":
            if ckpt.load_shard("asr", k) is None:
                pending.append(k)
                continue
            dep = {"asr": ckpt.shard_sha256("asr", k)}
        if ckpt.load_shard(stage, k, depends_on=dep) is None:
            pending.append(k)
    return pending


def _drop_stage_residency() -> None:
    """Collect the stage that just finished, then release cached CUDA blocks."""
    import gc

    gc.collect()
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _execute_one_stage(
    stage: str,
    loader: Callable[[], Mapping[str, Any]],
    ckpt: InferenceCheckpoint,
    items: Sequence[AudioItem],
    decoding: Mapping[str, Any],
    progress: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Load, run, and drop one teacher stage before the caller can load the next.

    ``handle`` and ``adapter`` live only in this frame. The frame ends, and a
    collection pass runs, before ``run_peak_safe_teacher_inference`` calls the
    next loader.
    """
    handle: Optional[dict] = dict(loader())
    adapter = None
    release = None
    try:
        adapter = handle["adapter"]
        if stage == "asr":
            return run_asr_stage(ckpt, items, adapter, batch_size=int(decoding["asr"]["batch_size"]), progress=progress)
        if stage == "mt":
            return run_mt_stage(ckpt, adapter, batch_size=int(decoding["mt"]["batch_size"]), progress=progress)
        return run_d0_stage(ckpt, items, adapter, batch_size=int(decoding["d0"]["batch_size"]), progress=progress)
    finally:
        try:
            if handle is not None:
                release = handle.pop("release", None)
                handle.clear()
        finally:
            handle = None
            adapter = None
            try:
                if release is not None:
                    release()
            finally:
                release = None
                _drop_stage_residency()


def run_peak_safe_teacher_inference(
    *,
    ckpt: InferenceCheckpoint,
    items: Sequence[AudioItem],
    decoding: Mapping[str, Any],
    load_asr: Callable[[], Mapping[str, Any]],
    load_mt: Callable[[], Mapping[str, Any]],
    load_d0: Optional[Callable[[], Mapping[str, Any]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """ASR -> MT -> (D0), one model resident at a time.

    Each loader returns ``{"adapter": callable, "release": callable}``. The
    loader for a stage is not called until the previous stage's handle has
    left scope and residency cleanup has run. A stage whose shards are all
    valid is not loaded. ``release`` runs even when the stage raises.
    """
    report: Dict[str, Any] = {"stages": []}
    plan = [("asr", load_asr), ("mt", load_mt)]
    if "d0" in decoding:
        if load_d0 is None:
            raise RuntimeError("D0 agreement is enabled but no D0 loader was given")
        plan.append(("d0", load_d0))
    elif load_d0 is not None:
        raise RuntimeError("a D0 loader was given but the D0 agreement contract is disabled")
    for stage, loader in plan:
        if not stage_pending_shards(ckpt, stage):
            report["stages"].append({"stage": stage, "computed_shards": 0, "reused_shards": ckpt.n_shards})
            continue
        report["stages"].append(_execute_one_stage(stage, loader, ckpt, items, decoding, progress))
    report["quarantined_shards"] = list(ckpt.quarantined)
    return report


_D0_DISABLED = {"d0_status": STATUS_DISABLED, "d0_vi_raw": None, "d0_vi_norm": None,
                "d0_valid_encoding": None, "d0_decode_warning": ""}


def iter_raw_records(
    ckpt: InferenceCheckpoint,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, str],
    d0_enabled: bool,
) -> Iterator[Dict[str, Any]]:
    """Merge completed stage shards into raw records in input order (streamed shard by shard)."""
    assert_no_reference_fields(identity.keys(), "raw record identity")
    if [str(r.get("segment_uid")) for r in source_rows] != ckpt.uids:
        raise CheckpointBindingError("source rows do not match the checkpoint's ordered uid list")
    for k in range(ckpt.n_shards):
        asr_rows = ckpt.load_shard("asr", k)
        if asr_rows is None:
            raise RuntimeError(f"ASR stage incomplete: shard {k}")
        mt_rows = ckpt.load_shard("mt", k, depends_on={"asr": ckpt.shard_sha256("asr", k)})
        if mt_rows is None:
            raise RuntimeError(f"MT stage incomplete: shard {k}")
        d0_rows = ckpt.load_shard("d0", k) if d0_enabled else None
        if d0_enabled and d0_rows is None:
            raise RuntimeError(f"D0 stage incomplete: shard {k}")
        base = k * ckpt.shard_size
        for i, (a, m) in enumerate(zip(asr_rows, mt_rows)):
            src = source_rows[base + i]
            rec: Dict[str, Any] = {c: src.get(c) for c in CARRIED_SOURCE_FIELDS}
            rec.update({
                "nb11_generation_id": identity.get("nb11_generation_id"),
                "nb11_input_contract_sha256": identity.get("nb11_input_contract_sha256"),
                "teacher_contract_sha256": identity["teacher_contract_sha256"],
                "d0_agreement_contract_sha256": identity["d0_agreement_contract_sha256"],
                "raw_schema_version": RAW_SCHEMA_VERSION,
            })
            rec.update({key: value for key, value in a.items() if key != "uid"})
            rec.update({key: value for key, value in m.items() if key != "uid"})
            if d0_enabled:
                d = d0_rows[i]
                rec.update({key: value for key, value in d.items() if key != "uid"})
                c0_text = rec["pseudo_vi_norm"] if rec["mt_status"] == STATUS_OK else None
                d0_text = d["d0_vi_norm"] if d["d0_status"] == STATUS_OK else None
                rec.update(agreement_features(c0_text, d0_text))
            else:
                rec.update(_D0_DISABLED)
                rec.update({"teacher_d0_agreement_chrf": None, "teacher_d0_agreement_char_ratio": None})
            rec.update(text_length_features(rec["asr_text_norm"], rec["pseudo_vi_norm"]))
            assert_no_reference_fields(rec.keys())
            yield {c: rec.get(c) for c in RAW_RECORD_COLUMNS}


def validation_source_rows(audio_frame: Any) -> List[Dict[str, Any]]:
    """Carried-field rows for G_validation records (identity only, no text)."""
    assert_no_reference_fields(audio_frame.columns, "validation source rows")
    return [
        {"segment_uid": str(r["record_uid"]), "segment_pcm16_sha256": str(r["sha256_pcm"]),
         "duration_seconds": float(r["n_samples"]) / SAMPLE_RATE}
        for _, r in audio_frame.iterrows()
    ]


# --------------------------------------------------------------------------- #
# G_validation audio (calibration only)                                        #
# --------------------------------------------------------------------------- #
VALIDATION_AUDIO_CACHE_VERSION = "rq2_g_validation_audio_cache_v1"
_VALIDATION_SOURCE_BINDING_KEYS = (
    "cache_version", "split", "record_uid", "record_id", "dataset_id", "parquet_revision",
    "parquet_file", "shard_key", "shard_row_index", "manifest_sha256", "audio_pcm_pipeline_version",
    "sample_rate",
)


def _validation_source_binding(
    row: Mapping[str, Any],
    *,
    dataset_id: str,
    parquet_revision: str,
    manifest_sha256: str,
    shard_key: str,
    target_sr: int,
) -> Dict[str, Any]:
    """Source identity for one cached WAV. No absolute path and no reference text."""
    binding = {
        "cache_version": VALIDATION_AUDIO_CACHE_VERSION,
        "split": "validation",
        "record_uid": str(row["record_uid"]),
        "record_id": str(row["record_id"]),
        "dataset_id": str(dataset_id),
        "parquet_revision": str(parquet_revision),
        "parquet_file": str(row["parquet_file"]),
        "shard_key": str(shard_key),
        "shard_row_index": int(row["shard_row_index"]),
        "manifest_sha256": str(manifest_sha256),
        "audio_pcm_pipeline_version": _audio_pcm_pipeline_version(),
        "sample_rate": int(target_sr),
    }
    assert_no_reference_fields(binding.keys(), "validation audio cache binding")
    binding["source_binding_sha256"] = sha256_json({k: binding[k] for k in _VALIDATION_SOURCE_BINDING_KEYS})
    return binding


def _audio_pcm_pipeline_version() -> str:
    from src.asr_full_pcm import AUDIO_PCM_PIPELINE_VERSION

    return AUDIO_PCM_PIPELINE_VERSION


def _cache_paths(audio_root: Path, uid: str) -> tuple:
    from src.data_utils import safe_cache_filename

    wav = audio_root / safe_cache_filename(uid)
    return wav, wav.with_suffix(".json")


def _quarantine_cache_item(audio_root: Path, uid: str, *paths: Path) -> None:
    """Move one record's bad cache files aside. Other records stay in place."""
    qdir = audio_root / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    for path in paths:
        if path.is_file() and not path.name.endswith(".tmp"):
            shutil.move(str(path), str(qdir / f"{path.name}.{tag}"))


def _verified_cached_wav(wav: Path, sidecar_path: Path, expected: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Reuse one WAV only when the file, PCM hash, and source binding all match."""
    if wav.name.endswith(".tmp") or sidecar_path.name.endswith(".tmp"):
        return None
    if not wav.is_file() or not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(sidecar, dict):
        return None
    assert_no_reference_fields(sidecar.keys(), "validation audio sidecar")
    if str(sidecar.get("source_binding_sha256") or "") != expected["source_binding_sha256"]:
        return None
    for key in _VALIDATION_SOURCE_BINDING_KEYS:
        if sidecar.get(key) != expected[key]:
            return None
    if any(isinstance(v, str) and (v.startswith("/") or ":\\" in v) for v in sidecar.values()):
        return None
    try:
        from src.asr_full_pcm import read_pcm16_payload

        info = read_pcm16_payload(wav)
    except Exception:  # noqa: BLE001
        return None
    if int(info["sample_rate"]) != int(expected["sample_rate"]) or int(info["channels"]) != 1:
        return None
    n_samples = int(info["n_samples"])
    if n_samples != int(sidecar.get("n_samples") or -1) or info["sha256_pcm"] != sidecar.get("pcm16_sha256"):
        return None
    if sha256_file(wav) != sidecar.get("wav_sha256"):
        return None
    duration = n_samples / float(info["sample_rate"])
    if abs(float(sidecar.get("duration_seconds") or -1.0) - duration) > 1e-9:
        return None
    return sidecar


def validation_audio_identity_sha256(
    audio_frame: Any,
    *,
    manifest_sha256: str,
    dataset_id: str,
    parquet_revision: str,
) -> str:
    """Order-sensitive identity. Paths are not part of the hash."""
    records = [
        [str(r["record_uid"]), str(r["sha256_pcm"]), int(r["n_samples"]), str(r["shard_key"]), int(r["shard_row_index"])]
        for _, r in audio_frame.iterrows()
    ]
    return sha256_json({
        "cache_version": VALIDATION_AUDIO_CACHE_VERSION,
        "audio_pcm_pipeline_version": _audio_pcm_pipeline_version(),
        "split": "validation",
        "dataset_id": str(dataset_id),
        "parquet_revision": str(parquet_revision),
        "manifest_sha256": str(manifest_sha256),
        "records": records,
    })


def materialize_validation_audio(
    frame: Any,
    *,
    dataset_id: str,
    parquet_revision: str,
    manifest_sha256: str,
    audio_dir: Union[str, Path],
    parquet_cache_dir: Union[str, Path],
    target_sr: int = SAMPLE_RATE,
    reader_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Materialise G_validation audio into a verified, resumable PCM16 cache.

    A later call reuses a WAV only after re-reading it and checking its sidecar
    against this manifest, dataset revision, and PCM hash. One bad record is
    quarantined and rebuilt; the rest of the cache stays. The returned identity
    hash does not contain a machine path or a Vietnamese reference.
    """
    import pandas as pd

    from src.asr_full_pcm import canonical_pcm16_from_bytes, cleanup_shard_download
    from src.asr_full_shards import build_shard_plans, make_hf_parquet_stream_reader
    from src.rq2_pseudo_contract import assert_not_g_test_path

    if not manifest_sha256 or len(str(manifest_sha256)) != 64:
        raise RuntimeError("materialize_validation_audio requires the rq1_validation.csv sha256")
    if set(frame["split"].astype(str)) != {"validation"}:
        raise RuntimeError("materialize_validation_audio only accepts G_validation rows")
    audio_root = assert_not_g_test_path(Path(audio_dir))
    audio_root.mkdir(parents=True, exist_ok=True)
    id_frame = frame[["record_uid", "record_id", "parquet_file", "shard_row_index", "split"]].copy()
    if id_frame["record_uid"].duplicated().any():
        raise RuntimeError("duplicate record_uid in G_validation audio materialisation")
    plans = build_shard_plans({"validation": id_frame}, dataset_id=dataset_id, parquet_revision=parquet_revision)
    plan_for_uid: Dict[str, Any] = {}
    for plan in plans:
        for _, row in plan.per_split["validation"].iterrows():
            plan_for_uid[str(row["record_uid"])] = plan

    reused: Dict[str, Dict[str, Any]] = {}
    pending_rows = []
    n_invalidated = 0
    for _, row in id_frame.iterrows():
        uid = str(row["record_uid"])
        plan = plan_for_uid[uid]
        expected = _validation_source_binding(
            row, dataset_id=dataset_id, parquet_revision=parquet_revision,
            manifest_sha256=manifest_sha256, shard_key=plan.shard_key, target_sr=int(target_sr),
        )
        wav, side = _cache_paths(audio_root, uid)
        cached = _verified_cached_wav(wav, side, expected)
        if cached is not None:
            reused[uid] = cached
            continue
        if wav.exists() or side.exists():
            _quarantine_cache_item(audio_root, uid, wav, side)
            n_invalidated += 1
        pending_rows.append(row)

    built: Dict[str, Dict[str, Any]] = {}
    if pending_rows:
        pending = pd.DataFrame(pending_rows)
        pending_plans = build_shard_plans({"validation": pending}, dataset_id=dataset_id, parquet_revision=parquet_revision)
        pq_root = Path(parquet_cache_dir)
        pq_root.mkdir(parents=True, exist_ok=True)
        downloaded: Dict[str, str] = {}
        reader = (reader_factory or make_hf_parquet_stream_reader)(cache_dir=pq_root, downloaded=downloaded)
        try:
            for plan in pending_plans:
                wanted = {int(r["shard_row_index"]): r for _, r in plan.per_split["validation"].iterrows()}
                seen = set()
                for idx, payload in reader(plan.ref, plan.needed_indices):
                    idx = int(idx)
                    if idx not in wanted:
                        continue
                    if idx in seen:
                        raise RuntimeError(f"duplicate parquet row {idx} in {plan.shard_key}")
                    seen.add(idx)
                    row = wanted[idx]
                    uid = str(row["record_uid"])
                    payload_id = payload.get("id") if isinstance(payload, dict) else None
                    if payload_id is not None and str(payload_id) != str(row["record_id"]):
                        raise RuntimeError(f"G_validation record_id mismatch uid={uid}")
                    raw = payload.get("audio") if isinstance(payload, dict) else None
                    if isinstance(raw, dict):
                        raw = raw.get("bytes")
                    if not raw:
                        raise RuntimeError(f"G_validation null audio uid={uid}")
                    decoded = canonical_pcm16_from_bytes(bytes(raw), target_sr=int(target_sr))
                    binding = _validation_source_binding(
                        row, dataset_id=dataset_id, parquet_revision=parquet_revision,
                        manifest_sha256=manifest_sha256, shard_key=plan.shard_key, target_sr=int(target_sr),
                    )
                    wav, side = _cache_paths(audio_root, uid)
                    duration = int(decoded["n_samples"]) / float(target_sr)
                    sidecar = {
                        **binding,
                        "pcm16_sha256": decoded["sha256_pcm"],
                        "n_samples": int(decoded["n_samples"]),
                        "duration_seconds": f"{duration:.9f}",
                        "cache_filename": wav.name,
                    }
                    staging = wav.with_name(wav.name + ".staging")
                    from src.asr_full_pcm import write_wav_from_pcm16

                    write_wav_from_pcm16(staging, decoded["pcm"], int(target_sr))
                    sidecar["wav_sha256"] = sha256_file(staging)
                    os.replace(staging, wav)
                    write_json(side, sidecar)
                    if _verified_cached_wav(wav, side, binding) is None:
                        raise RuntimeError(f"validation cache write did not verify for {uid}")
                    built[uid] = sidecar
                missing = sorted(set(wanted) - seen)
                if missing:
                    raise RuntimeError(f"G_validation shard missing requested rows {plan.shard_key}: {missing[:5]}")
                local = downloaded.pop(plan.ref.shard_key, None)
                if local:
                    cleanup_shard_download(local)
        finally:
            for p in list(downloaded.values()):
                try:
                    cleanup_shard_download(p)
                except Exception:  # noqa: BLE001
                    pass
            downloaded.clear()

    rows = []
    index_records = []
    total_seconds = 0.0
    for _, row in id_frame.iterrows():
        uid = str(row["record_uid"])
        sidecar = reused.get(uid) or built.get(uid)
        if sidecar is None:
            raise RuntimeError(f"G_validation audio missing after materialisation: {uid}")
        wav, _side = _cache_paths(audio_root, uid)
        rows.append({
            "record_uid": uid,
            "audio_path": str(wav),
            "sha256_pcm": sidecar["pcm16_sha256"],
            "n_samples": int(sidecar["n_samples"]),
            "sample_rate": int(sidecar["sample_rate"]),
            "shard_key": sidecar["shard_key"],
            "shard_row_index": int(sidecar["shard_row_index"]),
        })
        index_records.append({k: sidecar[k] for k in (
            "record_uid", "split", "record_id", "parquet_file", "shard_key", "shard_row_index",
            "dataset_id", "parquet_revision", "manifest_sha256", "pcm16_sha256", "wav_sha256",
            "sample_rate", "n_samples", "duration_seconds", "audio_pcm_pipeline_version",
            "cache_version", "cache_filename", "source_binding_sha256",
        )})
        total_seconds += float(sidecar["duration_seconds"])
    out = pd.DataFrame(rows)
    expected = id_frame["record_uid"].astype(str).tolist()
    if out["record_uid"].astype(str).tolist() != expected:
        raise RuntimeError("G_validation audio materialisation UID accounting failed")
    for rec in index_records:
        assert_no_reference_fields(rec.keys(), "validation audio cache index")
    index_path = audio_root / "validation_audio_cache_index.jsonl"
    atomic_write_text(
        index_path,
        "".join(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n" for rec in index_records),
    )
    identity = validation_audio_identity_sha256(
        out, manifest_sha256=manifest_sha256, dataset_id=dataset_id, parquet_revision=parquet_revision,
    )
    state = {
        "cache_version": VALIDATION_AUDIO_CACHE_VERSION,
        "n_validation_total": len(rows),
        "n_reused": len(reused),
        "n_rebuilt": len(built),
        "n_invalidated": int(n_invalidated),
        "total_audio_seconds": round(total_seconds, 6),
        "ordered_uid_hash": sha256_json(expected),
        "cache_identity_sha256": sha256_json(index_records),
        "validation_audio_identity_sha256": identity,
        "index_sha256": sha256_file(index_path),
    }
    write_json(audio_root / "validation_audio_cache_state.json", state)
    return {"audio_frame": out, **state}
