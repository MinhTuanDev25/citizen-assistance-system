"""Versioned text normalization for Notebook 04 MT (not CTC/ASR)."""
from __future__ import annotations

import re
import unicodedata

from src.mt_runtime_paths import MT_NORMALIZATION_VERSION

_WS = re.compile(r"\s+")


def normalize_mt_text_v1(value: object) -> str:
    """NFC + collapse whitespace + strip. Does not casefold (keep VI casing)."""
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = unicodedata.normalize("NFC", str(value))
    return _WS.sub(" ", text).strip()


def mt_normalization_version() -> str:
    return MT_NORMALIZATION_VERSION
