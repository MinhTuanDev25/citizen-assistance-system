"""Keyword check for the real PaddleOCR smoke fixture.

The fixture is an image of unaccented text. Matching is case-insensitive and
ignores punctuation. It does not strip Vietnamese diacritics, so a Latin model
that drops or changes marks still fails a diacritic fixture.
"""

from __future__ import annotations

import re
import unicodedata

EXPECTED_LINES = ("GIAY KHAI SINH", "GIA LAI", "SO 123")
REQUIRED_TOKENS = ("giay", "khai", "sinh", "gia", "lai", "so", "123")


def normalize_ocr_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").lower()
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def missing_smoke_tokens(value: str) -> list[str]:
    text = normalize_ocr_text(value)
    missing: list[str] = []
    for token in REQUIRED_TOKENS:
        if token == "123":
            if token not in text:
                missing.append(token)
            continue
        if re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", text) is None:
            missing.append(token)
    return missing
