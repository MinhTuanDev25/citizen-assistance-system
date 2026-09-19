"""Seq2Seq dataset helpers for Notebook 04 MT."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.mt_normalize import normalize_mt_text_v1
from src.mt_runtime_paths import SOURCE_FIELD, TARGET_FIELD


class MtTextDataset:
    """Lightweight map-style dataset; tokenizes on __getitem__ (memory-safe)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        *,
        max_source_length: int,
        max_target_length: int,
        source_col: str = f"{SOURCE_FIELD}_norm",
        target_col: str = f"{TARGET_FIELD}_norm",
        fallback_source_col: str = SOURCE_FIELD,
        fallback_target_col: str = TARGET_FIELD,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)
        self.source_col = source_col if source_col in self.df.columns else fallback_source_col
        self.target_col = target_col if target_col in self.df.columns else fallback_target_col

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        src = normalize_mt_text_v1(row[self.source_col])
        tgt = normalize_mt_text_v1(row[self.target_col])
        # Official HF translation pattern: separate source / text_target encodes.
        # Do not pass max_target_length into a single __call__ (not a public kwarg).
        source_enc = self.tokenizer(
            src,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )
        target_enc = self.tokenizer(
            text_target=tgt,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": list(source_enc["input_ids"]),
            "attention_mask": list(source_enc["attention_mask"]),
            "labels": list(target_enc["input_ids"]),
            "record_uid": str(row["record_uid"]),
        }


def make_seq2seq_collator(tokenizer: Any, model: Optional[Any] = None) -> Any:
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)


def batch_without_record_uid(features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in f.items() if k != "record_uid"} for f in features]
