"""Text normalization for leakage keys / evaluation (keep originals for training)."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_key_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()
