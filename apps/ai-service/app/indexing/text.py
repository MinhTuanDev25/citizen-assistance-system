"""Normalize extracted text without changing letters or diacritics."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_SPACE_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACE_RUN.sub(" ", line).strip() for line in text.split("\n")]
    collapsed = "\n".join(lines).strip()
    return _BLANK_RUN.sub("\n\n", collapsed)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
