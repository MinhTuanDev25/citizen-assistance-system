"""Lazy local-WAV dataset and collator for Direct SpeechEncoderDecoder training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd

from src.asr_full_train import resolve_eligible_audio_path
from src.audio_utils import waveform_to_mono_float32
from src.mt_normalize import normalize_mt_text_v1


class DirectSpeechTranslationDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_extractor: Any,
        tokenizer: Any,
        audio_cache_roots: Sequence[Union[str, Path]],
        sample_rate: int = 16000,
        max_target_length: int = 256,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_cache_roots = [Path(p) for p in audio_cache_roots]
        self.sample_rate = int(sample_rate)
        self.max_target_length = int(max_target_length)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import soundfile as sf

        row = self.frame.iloc[int(idx)]
        wav_path = resolve_eligible_audio_path(row, self.audio_cache_roots)
        if wav_path is None:
            raise RuntimeError(f"Local audio missing for record_uid={row['record_uid']}")
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if int(sr) != self.sample_rate:
            raise RuntimeError(f"Unexpected sample rate {sr} for {wav_path}; expected {self.sample_rate}")
        wav = waveform_to_mono_float32(wav)
        feat = self.feature_extractor(wav, sampling_rate=self.sample_rate, return_attention_mask=True)
        labels = self.tokenizer(
            text_target=normalize_mt_text_v1(row["text_vi_norm"]),
            max_length=self.max_target_length,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]
        return {
            "input_values": feat["input_values"][0],
            "attention_mask": feat.get("attention_mask", [[1] * len(feat["input_values"][0])])[0],
            "labels": labels,
            "record_uid": str(row["record_uid"]),
        }


class DirectDataCollator:
    def __init__(self, feature_extractor: Any, tokenizer: Any):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, features):
        audio = [
            {"input_values": f["input_values"], "attention_mask": f.get("attention_mask")}
            for f in features
        ]
        # Some feature extractors dislike an explicit None attention mask.
        audio = [{k: v for k, v in x.items() if v is not None} for x in audio]
        batch = self.feature_extractor.pad(audio, padding=True, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch
