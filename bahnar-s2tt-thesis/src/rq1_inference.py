"""Model restore/inference helpers for Notebook 06. No function runs on import."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import pandas as pd

from src.data_utils import normalize_bahnar_ctc_v1
from src.mt_normalize import normalize_mt_text_v1


def run_asr_predictions(
    *,
    model: Any,
    processor: Any,
    frame: pd.DataFrame,
    audio_paths: Mapping[str, Union[str, Path]],
    batch_size: int = 1,
) -> pd.DataFrame:
    import soundfile as sf
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, i):
            row = frame.iloc[int(i)]
            uid = str(row["record_uid"])
            p = Path(audio_paths[uid])
            wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if int(sr) != 16000:
                raise RuntimeError(f"ASR audio sample rate {sr} != 16000 for {uid}")
            if getattr(wav, "ndim", 1) > 1:
                wav = np.asarray(wav).mean(axis=-1)
            return uid, np.asarray(wav, dtype=np.float32)

    def collate(items):
        uids = [x[0] for x in items]
        wavs = [x[1] for x in items]
        batch = processor(wavs, sampling_rate=16000, padding=True, return_tensors="pt")
        return uids, batch

    device = next(model.parameters()).device
    model.eval()
    rows = []
    with torch.no_grad():
        for uids, b in DataLoader(DS(), batch_size=int(batch_size), shuffle=False, collate_fn=collate):
            iv = b.input_values.to(device)
            am = getattr(b, "attention_mask", None)
            kwargs = {"input_values": iv}
            if am is not None:
                kwargs["attention_mask"] = am.to(device)
            logits = model(**kwargs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            texts = processor.batch_decode(pred_ids)
            rows.extend(
                {"record_uid": str(uid), "asr_pred_bahnar": normalize_bahnar_ctc_v1(text)}
                for uid, text in zip(uids, texts)
            )
    return pd.DataFrame(rows)


def audit_mt_source_truncation(
    texts: Sequence[str],
    *,
    tokenizer: Any,
    max_source_length: int,
) -> Dict[str, Any]:
    """Count ASR transcripts truncated by the locked MT max_source_length."""
    n = len(texts)
    truncated = 0
    for text in texts:
        ids = tokenizer(
            str(text),
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        if len(ids) > int(max_source_length):
            truncated += 1
    rate = float(truncated) / float(n) if n else 0.0
    return {
        "n": int(n),
        "truncated_count": int(truncated),
        "truncated_rate": rate,
        "max_source_length": int(max_source_length),
    }


from typing import Sequence  # noqa: E402


def run_mt_from_asr(
    *,
    model: Any,
    tokenizer: Any,
    asr_predictions: pd.DataFrame,
    max_source_length: int,
    generation_max_length: int,
    num_beams: int,
    batch_size: int = 8,
) -> pd.DataFrame:
    import torch

    device = next(model.parameters()).device
    model.eval()
    rows = []
    data = asr_predictions.reset_index(drop=True)
    src_all = [normalize_bahnar_ctc_v1(x) for x in data["asr_pred_bahnar"].astype(str)]
    truncation_audit = audit_mt_source_truncation(
        src_all, tokenizer=tokenizer, max_source_length=int(max_source_length)
    )
    with torch.no_grad():
        for start in range(0, len(data), int(batch_size)):
            part = data.iloc[start : start + int(batch_size)]
            src = [normalize_bahnar_ctc_v1(x) for x in part["asr_pred_bahnar"].astype(str)]
            enc = tokenizer(
                src,
                padding=True,
                truncation=True,
                max_length=int(max_source_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outs = model.generate(
                **enc,
                max_length=int(generation_max_length),
                num_beams=int(num_beams),
            )
            pred = tokenizer.batch_decode(outs, skip_special_tokens=True)
            for uid, asr_text, vi in zip(part["record_uid"].astype(str), src, pred):
                rows.append(
                    {
                        "record_uid": uid,
                        "asr_pred_bahnar": asr_text,
                        "c0_pred_vi": normalize_mt_text_v1(vi),
                    }
                )
    out = pd.DataFrame(rows)
    out.attrs["mt_truncation_audit"] = truncation_audit
    return out


def run_direct_predictions(
    *,
    model: Any,
    feature_extractor: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    audio_paths: Mapping[str, Union[str, Path]],
    target_lang: str,
    generation_max_length: int,
    num_beams: int,
    batch_size: int = 1,
) -> pd.DataFrame:
    import soundfile as sf
    import torch

    device = next(model.parameters()).device
    model.eval()
    rows = []
    forced_bos = int(tokenizer.lang_code_to_id[target_lang])
    with torch.no_grad():
        for start in range(0, len(frame), int(batch_size)):
            part = frame.iloc[start : start + int(batch_size)]
            wavs = []
            uids = []
            for _, r in part.iterrows():
                uid = str(r["record_uid"])
                p = Path(audio_paths[uid])
                wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
                if int(sr) != 16000:
                    raise RuntimeError(f"Direct audio sample rate {sr} != 16000 for {uid}")
                if getattr(wav, "ndim", 1) > 1:
                    wav = np.asarray(wav).mean(axis=-1)
                wavs.append(np.asarray(wav, dtype=np.float32))
                uids.append(uid)
            feat = feature_extractor(
                wavs,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            kwargs = {"input_values": feat["input_values"].to(device)}
            if "attention_mask" in feat:
                kwargs["attention_mask"] = feat["attention_mask"].to(device)
            outs = model.generate(
                **kwargs,
                max_length=int(generation_max_length),
                num_beams=int(num_beams),
                forced_bos_token_id=forced_bos,
            )
            pred = tokenizer.batch_decode(outs, skip_special_tokens=True)
            rows.extend(
                {"record_uid": uid, "d0_pred_vi": normalize_mt_text_v1(text)}
                for uid, text in zip(uids, pred)
            )
    return pd.DataFrame(rows)
