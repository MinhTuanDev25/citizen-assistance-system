"""Shared data helpers for Bahnar S2TT thesis notebooks (esp. Notebook 02)."""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import math
import random
import re
import socket
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from src.audio_utils import check_waveform, waveform_to_mono_float32

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"

NORMALIZATION_VERSION = "bahnar_ctc_normalization_v1"
AUDIO_PROCESSING_VERSION = "notebook02_audio_v3"

USER_AGENT = "bahnar-s2tt-thesis-notebook02/1.0"
HF_VIEWER_HOSTS = frozenset({"datasets-server.huggingface.co"})
HF_HUB_HOSTS = frozenset({"huggingface.co", "hf.co"})

REQUIRED_MANIFEST_COLUMNS = [
    "record_uid",
    "record_id",
    "source_split",
    "audio_path",
    "duration_seconds",
    "source_label",
    "recording_group_id",
    "group_id",
    "text_bahnar",
    "text_vi",
    "pair_key",
    "split",
]

FROZEN_TEST_ALLOWED_COLUMNS = frozenset(
    {
        "record_uid",
        "record_id",
        "source_split",
        "group_id",
        "recording_group_id",
        "pair_key",
        "split",
    }
)
FROZEN_TEST_FORBIDDEN_COLUMNS = frozenset(
    {
        "text_bahnar",
        "text_vi",
        "text_en",
        "audio_path",
        "duration_seconds",
        "qa_ok",
        "qa_hard_ok",
        "qa_reason",
        "qa_duration_sec",
        "qa_quality_warnings",
    }
)

# Scripts that should be flagged as unexpected in Bahnar text
# These keywords are checked against Unicode character names
UNEXPECTED_SCRIPT_KEYWORDS = frozenset({
    "KHMER", "THAI", "HEBREW", "CYRILLIC", "ARABIC", "CJK",
    "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "KANNADA",
    "MALAYALAM", "SINHALA", "MYANMAR", "GEORGIAN", "ETHIOPIC",
    "CHEROKEE", "HANGUL", "HIRAGANA", "KATAKANA", "BOPOMOFO",
    "TIBETAN", "LAO", "GREEK", "ARMENIAN",
})

_KEEP_PUNCT = set("'-")

# HTTP retry settings
NO_RETRY_STATUSES = frozenset({400, 401, 403, 404, 405, 410, 422})
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_REDIRECTS = 5


# ---------------------------------------------------------------------------
# Run ID / provenance
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run ID for provenance tracking."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Project paths / SHA
# ---------------------------------------------------------------------------


def resolve_project_root(start: Path | None = None) -> Path:
    import sys

    if "google.colab" in sys.modules:
        for cand in (
            Path("/content/drive/MyDrive/bahnar-s2tt-thesis"),
            Path("/content/bahnar-s2tt-thesis"),
        ):
            if _is_project_root(cand):
                return cand.resolve()
    cwd = (start or Path.cwd()).resolve()
    for cand in [cwd, cwd.parent, *cwd.parents]:
        if _is_project_root(cand):
            return cand.resolve()
    raise FileNotFoundError(
        "Cannot locate bahnar-s2tt-thesis root "
        "(needs requirements.txt, src/, data/manifests/)."
    )


def _is_project_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "requirements.txt").is_file()
        and (path / "src").is_dir()
        and (path / "data" / "manifests").is_dir()
    )


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or resolve_project_root()
    return {
        "root": root,
        "manifests": root / "data" / "manifests",
        "audit": root / "data" / "audit",
        "cache_audio": root / "data" / "cache" / "notebook02_audio",
        "artifacts_tokenizer": root / "artifacts" / "tokenizers" / "bahnar_char_ctc",
        "checkpoints": root / "checkpoints",
        "predictions": root / "predictions",
        "metrics": root / "metrics",
        "results": root / "results",
        "configs": root / "configs",
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_sha256(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", text))


def redact_url(url: str) -> str:
    """Strip query/fragment (signed params) for safe error messages."""
    parts = urlsplit(str(url))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


# ---------------------------------------------------------------------------
# Null / bool parsing
# ---------------------------------------------------------------------------


def is_missing_scalar(value: Any) -> bool:
    """True for None / NaN / pd.NA. String 'NA' is NOT missing."""
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, str):
        return False
    try:
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
    except Exception:
        pass
    try:
        if not isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series)):
            if pd.isna(value):
                return True
    except (ValueError, TypeError):
        pass
    return False


def parse_bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ValueError(f"Unrecognized boolean value: {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(parse_bool)


# ---------------------------------------------------------------------------
# Manifest loading (CSV-first; SHA locked on CSV)
# ---------------------------------------------------------------------------

def load_manifest(split: str, root: Path | None = None) -> pd.DataFrame:
    """
    Prefer CSV — Notebook 01 locks SHA256 on CSV files.
    Parquet is used only when CSV is missing AND split_summary lists a
    verified SHA256 for that parquet file that matches on disk.
    """
    paths = project_paths(root)
    csv_path = paths["manifests"] / f"rq1_{split}.csv"
    parquet_path = paths["manifests"] / f"rq1_{split}.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    summary_path = paths["manifests"] / "split_summary.json"
    if parquet_path.exists() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sha_map = summary.get("manifest_sha256") or {}
        expected = sha_map.get(parquet_path.name)
        if is_valid_sha256(expected) and sha256_file(parquet_path) == expected:
            return pd.read_parquet(parquet_path)
        raise FileNotFoundError(
            f"CSV missing for split={split!r} and Parquet is not SHA-verified "
            f"(need valid manifest_sha256[{parquet_path.name!r}] matching file)."
        )
    raise FileNotFoundError(
        f"Missing verified manifest for split={split!r}. Expected {csv_path}."
    )


def load_frozen_test_restricted(root: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load frozen test with ONLY allowlisted columns.
    Never materializes text_bahnar, text_vi, audio_path, etc. into DataFrame.
    """
    paths = project_paths(root)
    path = paths["manifests"] / "rq1_test.csv"
    
    header = pd.read_csv(path, nrows=0).columns.tolist()
    allowed = [c for c in header if c in FROZEN_TEST_ALLOWED_COLUMNS]
    forbidden_in_file = [c for c in header if c in FROZEN_TEST_FORBIDDEN_COLUMNS]
    
    df = pd.read_csv(path, usecols=allowed)
    forbidden_loaded = [c for c in df.columns if c in FROZEN_TEST_FORBIDDEN_COLUMNS]
    
    report = {
        "file_header_columns": header,
        "allowed_columns": sorted(FROZEN_TEST_ALLOWED_COLUMNS),
        "requested_columns": allowed,
        "loaded_columns": list(df.columns),
        "forbidden_columns_in_file": forbidden_in_file,
        "forbidden_columns_loaded": forbidden_loaded,
        "passed": len(forbidden_loaded) == 0,
    }
    return df, report


load_frozen_test_seal_columns = load_frozen_test_restricted


# ---------------------------------------------------------------------------
# Text normalization / vocab / OOV / script audit
# ---------------------------------------------------------------------------


def normalize_bahnar_ctc_v1(text: Any) -> str:
    if is_missing_scalar(text):
        return ""
    if isinstance(text, str) and text == "":
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
        .replace("`", "'")
    )
    s = s.casefold()
    out: list[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace():
            out.append(" ")
        elif cat.startswith("L") or cat.startswith("M") or ch in _KEEP_PUNCT:
            out.append(ch)
        elif cat.startswith("N"):
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def build_char_ctc_vocab(texts: Iterable[str]) -> dict[str, int]:
    charset: set[str] = set()
    for raw in texts:
        norm = normalize_bahnar_ctc_v1(raw)
        for ch in norm:
            if ch == " ":
                continue
            charset.add(ch)
    specials = ["[PAD]", "[UNK]", "|"]
    ordered = specials + sorted(charset)
    return {tok: i for i, tok in enumerate(ordered)}


def encode_text_with_vocab(text: str, vocab: dict[str, int]) -> list[int]:
    norm = normalize_bahnar_ctc_v1(text)
    unk = vocab["[UNK]"]
    ids: list[int] = []
    for ch in norm:
        if ch == " ":
            ids.append(vocab["|"])
        else:
            ids.append(vocab.get(ch, unk))
    return ids


def find_oov_characters(texts: Iterable[str], vocab: dict[str, int]) -> pd.DataFrame:
    known = set(vocab) - {"[PAD]", "[UNK]", "|"}
    counts: dict[str, int] = {}
    for raw in texts:
        norm = normalize_bahnar_ctc_v1(raw)
        for ch in norm:
            if ch == " ":
                continue
            if ch not in known:
                counts[ch] = counts.get(ch, 0) + 1
    rows = []
    for ch, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        rows.append(
            {
                "character": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "unicode_name": name,
                "count": n,
            }
        )
    return pd.DataFrame(rows, columns=["character", "codepoint", "unicode_name", "count"])


def find_oov_rows(
    df: pd.DataFrame,
    vocab: dict[str, int],
    *,
    text_col: str = "text_bahnar",
    uid_col: str = "record_uid",
) -> pd.DataFrame:
    known = set(vocab) - {"[PAD]", "[UNK]", "|"}
    rows = []
    for _, row in df.iterrows():
        raw = row.get(text_col)
        norm = normalize_bahnar_ctc_v1(raw)
        oovs = sorted({ch for ch in norm if ch != " " and ch not in known})
        if not oovs:
            continue
        names = []
        cps = []
        for ch in oovs:
            cps.append(f"U+{ord(ch):04X}")
            try:
                names.append(unicodedata.name(ch))
            except ValueError:
                names.append("")
        rows.append(
            {
                "record_uid": row.get(uid_col),
                "text": raw,
                "text_norm": norm,
                "oov_characters": "".join(oovs),
                "oov_codepoints": "|".join(cps),
                "oov_unicode_names": "|".join(names),
            }
        )
    return pd.DataFrame(rows)


def character_frequency(texts: Iterable[str]) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for raw in texts:
        norm = normalize_bahnar_ctc_v1(raw)
        for ch in norm:
            key = "|" if ch == " " else ch
            counts[key] = counts.get(key, 0) + 1
    rows = [
        {
            "character": ch,
            "codepoint": f"U+{ord(ch):04X}" if ch != "|" else "WORD_DELIM",
            "count": n,
        }
        for ch, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    return pd.DataFrame(rows, columns=["character", "codepoint", "count"])


def classify_character(ch: str) -> str:
    """
    Classify a character into categories for contamination audit.
    
    IMPORTANT: Check unexpected scripts BEFORE classifying as combining_mark.
    This ensures Khmer/Thai vowel signs are flagged as unexpected_script.
    
    Returns one of: latin_letter, combining_mark, digit, allowed_punctuation,
    unexpected_script, control_character, rare_character
    """
    if len(ch) != 1:
        return "invalid"
    
    cat = unicodedata.category(ch)
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = ""
    
    name_upper = name.upper()
    
    # Control characters - check first
    if cat.startswith("C"):
        return "control_character"
    
    # Digits
    if cat.startswith("N"):
        return "digit"
    
    # Allowed punctuation
    if ch in _KEEP_PUNCT:
        return "allowed_punctuation"
    
    # CHECK UNEXPECTED SCRIPTS BEFORE COMBINING MARKS
    # This is critical: Khmer/Thai vowel signs have category M* but should be flagged
    for script in UNEXPECTED_SCRIPT_KEYWORDS:
        if script in name_upper:
            return "unexpected_script"
    
    # Now safe to classify combining marks as valid (they passed the script check)
    if cat.startswith("M"):
        # At this point, the mark is not from an unexpected script
        # Check if it's a Latin combining mark or general diacritic
        if "LATIN" in name_upper or "COMBINING" in name_upper:
            return "combining_mark"
        # Generic combining marks without script indication are OK
        return "combining_mark"
    
    # Letters - check if Latin
    if cat.startswith("L"):
        # Latin letters (including extended blocks) have "LATIN" in name
        if "LATIN" in name_upper:
            return "latin_letter"
        
        # IPA extensions and phonetic extensions are also acceptable for Bahnar
        if any(x in name_upper for x in ("MODIFIER", "PHONETIC", "IPA")):
            return "latin_letter"
        
        # Check codepoint ranges for Latin blocks
        cp = ord(ch)
        
        # Basic Latin, Latin-1 Supplement, Latin Extended-A/B
        if cp <= 0x024F:
            return "latin_letter"
        
        # IPA Extensions (0x0250–0x02AF)
        if 0x0250 <= cp <= 0x02AF:
            return "latin_letter"
        
        # Latin Extended Additional (Vietnamese diacritics)
        if 0x1E00 <= cp <= 0x1EFF:
            return "latin_letter"
        
        # Latin Extended-C, D, E
        if 0x2C60 <= cp <= 0x2C7F:
            return "latin_letter"
        if 0xA720 <= cp <= 0xA7FF:
            return "latin_letter"
        if 0xAB30 <= cp <= 0xAB6F:
            return "latin_letter"
        
        # Phonetic Extensions
        if 0x1D00 <= cp <= 0x1D7F:
            return "latin_letter"
        if 0x1D80 <= cp <= 0x1DBF:
            return "latin_letter"
        
        # If we get here, it's a letter but we can't confirm it's Latin
        return "rare_character"
    
    return "rare_character"


def audit_train_script_contamination(
    vocab: dict[str, int],
    *,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit train vocabulary for unexpected scripts / controls / rare chars.
    
    Returns (char_df, row_df) where:
    - char_df: character-level report with occurrence_count and affected_row_count
    - row_df: row-level report (requires train_df)
    
    Character report columns:
    - character, codepoint, unicode_name, category, classification, flags,
      vocab_id, occurrence_count, affected_row_count
    
    Row report columns:
    - record_uid, record_id, source_label, group_id, recording_group_id,
      text_bahnar, text_bahnar_norm, flagged_characters, flagged_codepoints,
      flagged_unicode_names, flags
    """
    # First pass: identify flagged characters and collect stats
    flagged_chars: dict[str, list[str]] = {}  # char -> list of flags
    char_info: dict[str, dict[str, Any]] = {}  # char -> info dict
    
    for ch, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if ch in {"[PAD]", "[UNK]", "|"}:
            continue
        
        cat = unicodedata.category(ch)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        
        classification = classify_character(ch)
        flags = []
        
        if classification == "control_character":
            flags.append("control_character")
        elif classification == "unexpected_script":
            flags.append("unexpected_script")
        elif classification == "rare_character":
            flags.append("rare_character")
        
        if flags:
            flagged_chars[ch] = flags
            char_info[ch] = {
                "character": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "unicode_name": name,
                "category": cat,
                "vocab_id": idx,
                "classification": classification,
                "flags": "|".join(flags),
                "occurrence_count": 0,
                "affected_row_count": 0,
            }
    
    # Second pass: count occurrences in train data
    row_rows = []
    if train_df is not None:
        text_col = "text_bahnar" if "text_bahnar" in train_df.columns else None
        if text_col and flagged_chars:
            for _, row in train_df.iterrows():
                raw = row.get(text_col, "")
                if is_missing_scalar(raw):
                    continue
                norm = normalize_bahnar_ctc_v1(raw)
                
                found_chars = []
                found_flags = []
                for ch in set(norm):
                    if ch in flagged_chars:
                        found_chars.append(ch)
                        found_flags.extend(flagged_chars[ch])
                        # Count occurrence of this char in this text
                        char_info[ch]["occurrence_count"] += norm.count(ch)
                
                if found_chars:
                    # Update affected_row_count for each char
                    for ch in found_chars:
                        char_info[ch]["affected_row_count"] += 1
                    
                    row_rows.append({
                        "record_uid": row.get("record_uid"),
                        "record_id": row.get("record_id"),
                        "source_label": row.get("source_label"),
                        "group_id": row.get("group_id"),
                        "recording_group_id": row.get("recording_group_id"),
                        "text_bahnar": raw,
                        "text_bahnar_norm": norm,
                        "flagged_characters": "".join(sorted(set(found_chars))),
                        "flagged_codepoints": "|".join(f"U+{ord(c):04X}" for c in sorted(set(found_chars))),
                        "flagged_unicode_names": "|".join(
                            unicodedata.name(c, "") for c in sorted(set(found_chars))
                        ),
                        "flags": "|".join(sorted(set(found_flags))),
                    })
    
    # Build character DataFrame
    char_rows = sorted(char_info.values(), key=lambda x: x["vocab_id"])
    char_df = pd.DataFrame(char_rows) if char_rows else pd.DataFrame(columns=[
        "character", "codepoint", "unicode_name", "category", "vocab_id",
        "classification", "flags", "occurrence_count", "affected_row_count"
    ])
    
    row_df = pd.DataFrame(row_rows) if row_rows else pd.DataFrame(columns=[
        "record_uid", "record_id", "source_label", "group_id", "recording_group_id",
        "text_bahnar", "text_bahnar_norm", "flagged_characters", "flagged_codepoints",
        "flagged_unicode_names", "flags"
    ])
    
    return char_df, row_df


def compute_contamination_summary(
    char_df: pd.DataFrame,
    row_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute contamination summary statistics.
    
    ONLY counts rows/groups with unexpected_script flag, not other flags.
    """
    if char_df.empty:
        return {
            "unexpected_script_characters": 0,
            "unexpected_script_char_list": [],
            "control_characters": 0,
            "rare_characters": 0,
            "affected_rows": 0,
            "affected_groups": 0,
        }
    
    # Count by flag type
    unexpected_chars = char_df.loc[
        char_df["flags"].str.contains("unexpected_script", na=False)
    ] if "flags" in char_df.columns else pd.DataFrame()
    
    control_chars = char_df.loc[
        char_df["flags"].str.contains("control_character", na=False)
    ] if "flags" in char_df.columns else pd.DataFrame()
    
    rare_chars = char_df.loc[
        char_df["flags"].str.contains("rare_character", na=False)
    ] if "flags" in char_df.columns else pd.DataFrame()
    
    # Count affected rows/groups ONLY for unexpected_script
    affected_rows = 0
    affected_groups = 0
    if not row_df.empty and "flags" in row_df.columns:
        unexpected_rows = row_df.loc[
            row_df["flags"].str.contains("unexpected_script", na=False)
        ]
        affected_rows = len(unexpected_rows)
        if "group_id" in unexpected_rows.columns:
            affected_groups = unexpected_rows["group_id"].nunique()
    
    return {
        "unexpected_script_characters": len(unexpected_chars),
        "unexpected_script_char_list": list(unexpected_chars["character"]) if not unexpected_chars.empty else [],
        "control_characters": len(control_chars),
        "rare_characters": len(rare_chars),
        "affected_rows": affected_rows,
        "affected_groups": affected_groups,
    }


# Fixed schema for contamination reports - MUST be consistent even when empty
CONTAMINATION_CHAR_SCHEMA = [
    "character", "codepoint", "unicode_name", "category",
    "classification", "flags", "occurrence_count", "affected_row_count"
]

CONTAMINATION_ROW_SCHEMA = [
    "record_uid", "record_id", "source_label", "group_id", "recording_group_id",
    "split", "text_bahnar", "text_bahnar_norm", "flagged_characters",
    "flagged_codepoints", "flagged_unicode_names", "flags",
    "contamination_ratio", "flag_reason"
]


def _empty_contamination_char_df() -> pd.DataFrame:
    """Return empty character contamination report with fixed schema."""
    return pd.DataFrame(columns=CONTAMINATION_CHAR_SCHEMA)


def _empty_contamination_row_df() -> pd.DataFrame:
    """Return empty row contamination report with fixed schema."""
    return pd.DataFrame(columns=CONTAMINATION_ROW_SCHEMA)


def audit_dataframe_contamination(
    df: pd.DataFrame,
    *,
    text_col: str = "text_bahnar",
    uid_col: str = "record_uid",
    split_name: str = "unknown",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit any DataFrame for script contamination.
    
    This is independent of vocabulary - it directly examines the normalized text.
    Use for validation/train contamination audits.
    
    Returns (char_df, row_df) with FIXED SCHEMA even when empty.
    Character report columns: {CONTAMINATION_CHAR_SCHEMA}
    Row report columns: {CONTAMINATION_ROW_SCHEMA}
    """
    if df is None or len(df) == 0:
        return _empty_contamination_char_df(), _empty_contamination_row_df()
    
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col!r} not found in DataFrame")
    
    # Collect all characters and their classifications
    char_stats: dict[str, dict[str, Any]] = {}  # char -> {flags, occurrence, affected_rows}
    row_records: list[dict[str, Any]] = []
    
    for _, row in df.iterrows():
        raw = row.get(text_col, "")
        if is_missing_scalar(raw):
            continue
        norm = normalize_bahnar_ctc_v1(raw)
        if not norm:
            continue
        
        # Find flagged characters in this row
        found_chars: dict[str, list[str]] = {}  # char -> flags
        for ch in set(norm):
            if ch == " ":
                continue
            classification = classify_character(ch)
            if classification in ("unexpected_script", "control_character", "rare_character"):
                found_chars[ch] = [classification]
        
        # Update character stats
        for ch, flags in found_chars.items():
            if ch not in char_stats:
                cat = unicodedata.category(ch)
                try:
                    name = unicodedata.name(ch)
                except ValueError:
                    name = ""
                char_stats[ch] = {
                    "character": ch,
                    "codepoint": f"U+{ord(ch):04X}",
                    "unicode_name": name,
                    "category": cat,
                    "classification": classify_character(ch),
                    "flags": "|".join(flags),
                    "occurrence_count": 0,
                    "affected_row_count": 0,
                }
            char_stats[ch]["occurrence_count"] += norm.count(ch)
            char_stats[ch]["affected_row_count"] += 1
        
        # Build row record if contaminated
        if found_chars:
            sorted_chars = sorted(found_chars.keys())
            all_flags = sorted(set(f for flags in found_chars.values() for f in flags))
            total_chars = len([c for c in norm if c != " "])
            contam_chars = sum(norm.count(c) for c in found_chars)
            contam_ratio = contam_chars / max(total_chars, 1)
            
            row_records.append({
                "record_uid": row.get(uid_col),
                "record_id": row.get("record_id"),
                "source_label": row.get("source_label"),
                "group_id": row.get("group_id"),
                "recording_group_id": row.get("recording_group_id"),
                "split": split_name,
                "text_bahnar": raw,
                "text_bahnar_norm": norm,
                "flagged_characters": "".join(sorted_chars),
                "flagged_codepoints": "|".join(f"U+{ord(c):04X}" for c in sorted_chars),
                "flagged_unicode_names": "|".join(
                    unicodedata.name(c, "") for c in sorted_chars
                ),
                "flags": "|".join(all_flags),
                "contamination_ratio": round(contam_ratio, 6),
                "flag_reason": f"contains_{all_flags[0]}" if all_flags else "",
            })
    
    # Always return DataFrame with fixed schema, even if empty
    if char_stats:
        char_df = pd.DataFrame(sorted(char_stats.values(), key=lambda x: -x["occurrence_count"]))
        # Ensure column order
        char_df = char_df[CONTAMINATION_CHAR_SCHEMA]
    else:
        char_df = _empty_contamination_char_df()
    
    if row_records:
        row_df = pd.DataFrame(row_records)
        # Ensure column order
        row_df = row_df[CONTAMINATION_ROW_SCHEMA]
    else:
        row_df = _empty_contamination_row_df()
    
    return char_df, row_df


def build_exclusion_list(
    contamination_rows_df: pd.DataFrame,
    *,
    uid_col: str = "record_uid",
    filter_flags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build an exclusion list from contamination row report.
    
    Args:
        contamination_rows_df: Row-level contamination report (from audit_dataframe_contamination)
        uid_col: Column containing record UIDs
        filter_flags: Only include rows with these flags (default: ["unexpected_script"])
    
    Returns:
        DataFrame with columns: record_uid, flags, exclusion_reason
    """
    if contamination_rows_df is None or len(contamination_rows_df) == 0:
        return pd.DataFrame(columns=["record_uid", "flags", "exclusion_reason"])
    
    if filter_flags is None:
        filter_flags = ["unexpected_script"]
    
    mask = pd.Series([False] * len(contamination_rows_df))
    for flag in filter_flags:
        mask |= contamination_rows_df["flags"].str.contains(flag, na=False)
    
    filtered = contamination_rows_df.loc[mask].copy()
    
    exclusions = []
    for _, row in filtered.iterrows():
        exclusions.append({
            "record_uid": row.get(uid_col),
            "flags": row.get("flags", ""),
            "exclusion_reason": f"contamination:{row.get('flags', '')}",
        })
    
    return pd.DataFrame(exclusions).drop_duplicates(subset=["record_uid"])


def build_clean_split(
    original_df: pd.DataFrame,
    exclusion_df: pd.DataFrame,
    *,
    uid_col: str = "record_uid",
) -> pd.DataFrame:
    """
    Build a clean split by anti-joining with exclusion list.
    
    Args:
        original_df: Original manifest DataFrame
        exclusion_df: Exclusion list (must have record_uid column)
        uid_col: Column to use for anti-join
    
    Returns:
        Clean DataFrame with contaminated rows removed
    """
    if exclusion_df is None or len(exclusion_df) == 0:
        return original_df.copy()
    
    excluded_uids = set(exclusion_df[uid_col].astype(str))
    mask = ~original_df[uid_col].astype(str).isin(excluded_uids)
    return original_df.loc[mask].copy().reset_index(drop=True)


def verify_clean_split_no_contamination(
    clean_df: pd.DataFrame,
    *,
    text_col: str = "text_bahnar",
    split_name: str = "clean",
) -> dict[str, Any]:
    """
    Verify a clean split has no unexpected_script contamination.
    
    Returns verification result dict.
    """
    char_df, row_df = audit_dataframe_contamination(
        clean_df, text_col=text_col, split_name=split_name
    )
    
    unexpected_chars = 0
    unexpected_rows = 0
    
    if not char_df.empty and "flags" in char_df.columns:
        unexpected_chars = len(char_df.loc[
            char_df["flags"].str.contains("unexpected_script", na=False)
        ])
    
    if not row_df.empty and "flags" in row_df.columns:
        unexpected_rows = len(row_df.loc[
            row_df["flags"].str.contains("unexpected_script", na=False)
        ])
    
    return {
        "passed": unexpected_chars == 0 and unexpected_rows == 0,
        "unexpected_script_characters": unexpected_chars,
        "unexpected_script_rows": unexpected_rows,
        "total_rows": len(clean_df),
        "split_name": split_name,
    }


def compute_split_hash(df: pd.DataFrame, uid_col: str = "record_uid") -> str:
    """Compute SHA-256 of sorted record_uid list. DEPRECATED: use compute_uid_set_hash."""
    uids = sorted(df[uid_col].astype(str).unique())
    content = "\n".join(uids)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_ordered_uid_hash(df: pd.DataFrame, uid_col: str = "record_uid") -> str:
    """
    Compute SHA-256 of record_uid list preserving DataFrame order.
    
    Requirements:
    - UIDs must not contain null/NaN values
    - UIDs must not have duplicates
    - Empty list returns valid SHA-256 of empty string (not "empty")
    
    Raises:
        ValueError: If UIDs contain null or duplicates
    """
    if df is None or len(df) == 0:
        return hashlib.sha256(b"").hexdigest()
    
    if uid_col not in df.columns:
        raise ValueError(f"Column {uid_col!r} not found in DataFrame")
    
    uids = df[uid_col]
    
    # Check for nulls
    if uids.isna().any():
        raise ValueError(f"Column {uid_col!r} contains null values")
    
    uid_list = uids.astype(str).tolist()
    
    # Check for duplicates
    if len(uid_list) != len(set(uid_list)):
        raise ValueError(f"Column {uid_col!r} contains duplicate values")
    
    # Hash in exact order
    content = "\n".join(uid_list)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_uid_set_hash(df: pd.DataFrame, uid_col: str = "record_uid") -> str:
    """
    Compute SHA-256 of sorted unique record_uid set.
    
    Requirements:
    - UIDs must not contain null/NaN values
    - UIDs must not have duplicates  
    - Empty set returns valid SHA-256 of empty string (not "empty")
    
    Note: Order-independent - [a,b,c] and [c,b,a] produce same hash.
    
    Raises:
        ValueError: If UIDs contain null or duplicates
    """
    if df is None or len(df) == 0:
        return hashlib.sha256(b"").hexdigest()
    
    if uid_col not in df.columns:
        raise ValueError(f"Column {uid_col!r} not found in DataFrame")
    
    uids = df[uid_col]
    
    # Check for nulls
    if uids.isna().any():
        raise ValueError(f"Column {uid_col!r} contains null values")
    
    uid_list = uids.astype(str).tolist()
    
    # Check for duplicates
    if len(uid_list) != len(set(uid_list)):
        raise ValueError(f"Column {uid_col!r} contains duplicate values")
    
    # Hash sorted for order-independence
    content = "\n".join(sorted(uid_list))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_cache_stats_per_split(
    qa_df: pd.DataFrame,
    split_col: str = "final_split",
    reused_col: str = "reused_cache",
) -> dict[str, dict[str, int]]:
    """
    Compute cache fresh/reuse stats independently for each split.
    
    Args:
        qa_df: QA results DataFrame with reused_cache column
        split_col: Column containing split name
        reused_col: Column containing boolean indicating cache reuse
    
    Returns:
        Dict mapping split name to {fresh_count, reuse_count, total}
    """
    result = {}
    
    for split_name in qa_df[split_col].unique():
        split_df = qa_df.loc[qa_df[split_col] == split_name].copy()
        
        # Safely convert reused_cache to boolean
        def safe_bool(val):
            if val is None:
                return False
            if isinstance(val, bool):
                return val
            if isinstance(val, np.bool_):
                return bool(val)
            if isinstance(val, str):
                return val.lower() in ("true", "1")
            if isinstance(val, (int, np.integer)):
                return bool(val)
            return False
        
        reused_series = split_df[reused_col].map(safe_bool)
        reuse_count = int(reused_series.sum())
        fresh_count = len(split_df) - reuse_count
        
        result[str(split_name)] = {
            "fresh_count": fresh_count,
            "reuse_count": reuse_count,
            "total": len(split_df),
        }
    
    return result


def verify_cache_stats_consistency(
    stats: dict[str, dict[str, int]],
    expected_totals: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Verify cache stats are consistent.
    
    Checks:
    - fresh + reuse == total for each split
    - global fresh == sum of split fresh
    - global reuse == sum of split reuse
    - totals match expected if provided
    """
    errors = []
    
    global_fresh = 0
    global_reuse = 0
    global_total = 0
    
    for split_name, split_stats in stats.items():
        fresh = split_stats["fresh_count"]
        reuse = split_stats["reuse_count"]
        total = split_stats["total"]
        
        # Check fresh + reuse == total
        if fresh + reuse != total:
            errors.append(f"{split_name}: fresh({fresh}) + reuse({reuse}) != total({total})")
        
        global_fresh += fresh
        global_reuse += reuse
        global_total += total
        
        # Check expected totals if provided
        if expected_totals and split_name in expected_totals:
            expected = expected_totals[split_name]
            if total != expected:
                errors.append(f"{split_name}: total({total}) != expected({expected})")
    
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "global_fresh": global_fresh,
        "global_reuse": global_reuse,
        "global_total": global_total,
        "per_split": stats,
    }


def texts_match_after_light_norm(a: Any, b: Any) -> bool:
    def _light(x: Any) -> str:
        if is_missing_scalar(x):
            return ""
        if isinstance(x, str) and x == "":
            return ""
        s = unicodedata.normalize("NFC", str(x))
        return re.sub(r"\s+", " ", s).strip()

    return _light(a) == _light(b)


# ---------------------------------------------------------------------------
# HTTP headers / URL validation / retries
# ---------------------------------------------------------------------------


def viewer_headers(token: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def asset_headers() -> dict[str, str]:
    """No Authorization — for signed audio / CDN hosts."""
    return {"User-Agent": USER_AGENT}


def _resolve_all_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve hostname to all IPs (IPv4 and IPv6)."""
    ips = []
    try:
        # Get all addresses (IPv4 and IPv6)
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
                ips.append(ip)
            except ValueError:
                pass
    except socket.gaierror:
        pass
    return ips


def _is_private_ip(
    hostname: str, 
    *, 
    dns_resolver: Callable[[str], list] | None = None
) -> bool:
    """
    Check if hostname resolves to a private/localhost/reserved IP.
    
    Checks ALL resolved IPs (IPv4 and IPv6).
    DNS resolution failure -> fail closed (return True to reject).
    
    dns_resolver is injectable for testing.
    """
    # Check common localhost names directly
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    
    # Resolve and check all IPs
    resolver = dns_resolver or _resolve_all_ips
    ips = resolver(hostname)
    
    # DNS resolution failure -> fail closed
    if not ips:
        return True
    
    # Check all resolved IPs
    for ip in ips:
        if (ip.is_private or ip.is_loopback or ip.is_reserved 
            or ip.is_link_local or ip.is_multicast or ip.is_unspecified):
            return True
    
    return False


# Global DNS resolver override for testing
_DNS_RESOLVER_OVERRIDE: Callable[[str], list] | None = None


def set_dns_resolver_for_testing(resolver: Callable[[str], list] | None) -> None:
    """Set a custom DNS resolver for testing. Pass None to reset."""
    global _DNS_RESOLVER_OVERRIDE
    _DNS_RESOLVER_OVERRIDE = resolver


def validate_https_url(
    url: str,
    *,
    allow_hosts: set[str] | frozenset[str] | None = None,
    reject_private: bool = True,
    dns_resolver: Callable[[str], list] | None = None,
) -> str:
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    parts = urlparse(url.strip())
    if parts.scheme != "https":
        raise ValueError(f"Only HTTPS URLs allowed, got scheme={parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URL must not contain username/password")
    if not parts.hostname:
        raise ValueError("URL missing hostname")
    host = parts.hostname.lower()
    if allow_hosts is not None and host not in allow_hosts:
        raise ValueError(f"Host {host!r} not in allowlist")
    if reject_private:
        resolver = dns_resolver or _DNS_RESOLVER_OVERRIDE
        if _is_private_ip(host, dns_resolver=resolver):
            raise ValueError(f"Private/localhost/reserved IP not allowed: {host!r}")
    return url.strip()


def request_with_retries(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
    max_attempts: int = 6,
    retry_statuses: tuple[int, ...] = RETRY_STATUSES,
    no_retry_statuses: frozenset[int] = NO_RETRY_STATUSES,
    max_sleep: float = 30.0,
    session: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    rng: random.Random | None = None,
) -> Any:
    """
    Shared HTTP helper with Retry-After / exponential backoff + jitter.
    
    Does NOT retry 400/401/403/404 - these indicate permanent errors.
    Only retries network errors and retry_statuses (429, 500, 502, 503, 504).
    """
    import requests

    sleep_fn = sleep_fn or time.sleep
    rng = rng or random.Random(0)
    sess = session or requests.Session()
    last_exc: Exception | None = None
    safe = redact_url(url)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = sess.request(
                method.upper(),
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            
            # No retry for permanent client errors
            if resp.status_code in no_retry_statuses:
                resp.raise_for_status()
                return resp
            
            if resp.status_code in retry_statuses:
                last_exc = RuntimeError(f"retryable_status:{resp.status_code}")
                setattr(last_exc, "response", resp)
                if attempt < max_attempts:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = min(float(retry_after), max_sleep)
                        except ValueError:
                            delay = min((2 ** (attempt - 1)) + rng.random(), max_sleep)
                    else:
                        delay = min((2 ** (attempt - 1)) + rng.random(), max_sleep)
                    sleep_fn(delay)
                    continue
                break
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            # Check if it's a non-retryable HTTP error
            if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
                if exc.response.status_code in no_retry_statuses:
                    raise
            if attempt >= max_attempts:
                break
            delay = min((2 ** (attempt - 1)) + rng.random(), max_sleep)
            sleep_fn(delay)

    status = getattr(getattr(last_exc, "response", None), "status_code", None)
    raise RuntimeError(
        f"HTTP request failed after {max_attempts} attempts "
        f"url={safe} status={status} error={type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _single_request_no_redirect(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    session: Any,
) -> Any:
    """Single HTTP request with allow_redirects=False. Internal helper."""
    return session.request(
        method.upper(),
        url,
        headers=headers,
        timeout=timeout,
        allow_redirects=False,
    )


DEFAULT_MAX_RESPONSE_BYTES = 50 * 1024 * 1024  # 50 MiB


class ResponseTooLargeError(RuntimeError):
    """Raised when response exceeds max_response_bytes."""
    pass


def _read_response_with_limit(
    resp: Any,
    max_bytes: int,
    *,
    chunk_size: int = 64 * 1024,
) -> bytes:
    """
    Read response body with size limit using streaming.
    
    Checks Content-Length header first if available.
    Then reads in chunks, stopping if limit exceeded.
    Always closes response.
    """
    try:
        # Check Content-Length first
        content_length = resp.headers.get("Content-Length")
        if content_length is not None:
            try:
                length = int(content_length)
                if length > max_bytes:
                    raise ResponseTooLargeError(
                        f"Content-Length {length} exceeds limit {max_bytes}"
                    )
            except ValueError:
                pass  # Invalid Content-Length, proceed with streaming
        
        # Stream and accumulate with size check
        chunks = []
        total_bytes = 0
        
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ResponseTooLargeError(
                        f"Response size {total_bytes} exceeds limit {max_bytes}"
                    )
                chunks.append(chunk)
        
        return b"".join(chunks)
    finally:
        resp.close()


def download_audio_bytes_with_redirects(
    url: str,
    *,
    timeout: float = 120.0,
    session: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    max_redirects: int = MAX_REDIRECTS,
    reject_private: bool = True,
    dns_resolver: Callable[[str], list] | None = None,
    max_attempts: int = 6,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> bytes:
    """
    Download audio with manual redirect following AND retry logic.
    
    - Disables automatic redirects
    - Validates each redirect URL (HTTPS, no private IP)
    - Maximum 5 redirects
    - Does NOT forward Authorization to redirected URLs
    - Retries on 429, 500, 502, 503, 504 and connection errors
    - Does NOT retry on 400, 401, 403, 404, 406, 409, 418, etc.
    - Checks Content-Length before reading
    - Uses streaming to enforce max_response_bytes
    - Always closes response on success/error/redirect/retry
    """
    import requests
    
    validate_https_url(url, allow_hosts=None, reject_private=reject_private, dns_resolver=dns_resolver)
    headers = asset_headers()
    assert "Authorization" not in headers, "ASSET_HEADERS must not have Authorization"
    
    sess = session or requests.Session()
    sleep_fn = sleep_fn or time.sleep
    rng = random.Random(0)
    
    current_url = url
    redirect_count = 0
    
    while redirect_count <= max_redirects:
        # Use retry logic for each hop
        last_exc: Exception | None = None
        
        for attempt in range(1, max_attempts + 1):
            resp = None
            try:
                resp = sess.request(
                    "GET",
                    current_url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,  # Enable streaming for size checking
                )
                
                # Check for redirect (don't retry redirects, just follow them)
                if resp.status_code in (301, 302, 303, 307, 308):
                    resp.close()  # Close redirect response
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        raise RuntimeError(f"Too many redirects (>{max_redirects})")
                    
                    location = resp.headers.get("Location")
                    if not location:
                        raise RuntimeError("Redirect without Location header")
                    
                    # Resolve relative URLs
                    new_url = urljoin(current_url, location)
                    
                    # Validate redirect URL
                    try:
                        validate_https_url(new_url, allow_hosts=None, reject_private=reject_private, dns_resolver=dns_resolver)
                    except ValueError as exc:
                        raise RuntimeError(f"Invalid redirect target: {exc}") from exc
                    
                    current_url = new_url
                    break  # Break retry loop, continue redirect loop
                
                # Permanent client errors - don't retry (includes 4xx except 429)
                if resp.status_code in NO_RETRY_STATUSES:
                    resp.close()
                    resp.raise_for_status()
                    # Should not reach here after raise_for_status
                    return b""
                
                # Any other 4xx error (not in NO_RETRY_STATUSES and not 429) - don't retry
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    resp.close()
                    raise RuntimeError(
                        f"Download failed (permanent): HTTP {resp.status_code}"
                    )
                
                # Retryable status codes
                if resp.status_code in RETRY_STATUSES:
                    retry_after = resp.headers.get("Retry-After")
                    resp.close()  # Close before retry
                    last_exc = RuntimeError(f"retryable_status:{resp.status_code}")
                    if attempt < max_attempts:
                        if retry_after is not None:
                            try:
                                delay = min(float(retry_after), 30.0)
                            except ValueError:
                                delay = min((2 ** (attempt - 1)) + rng.random(), 30.0)
                        else:
                            delay = min((2 ** (attempt - 1)) + rng.random(), 30.0)
                        sleep_fn(delay)
                        continue
                    raise RuntimeError(
                        f"HTTP request failed after {max_attempts} attempts, status={resp.status_code}"
                    )
                
                # Success - read with size limit (also closes response)
                resp.raise_for_status()
                return _read_response_with_limit(resp, max_response_bytes)
                
            except ResponseTooLargeError:
                raise
            except RuntimeError:
                raise
            except Exception as exc:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                last_exc = exc
                # Check for non-retryable HTTP error
                if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
                    status = exc.response.status_code
                    if status in NO_RETRY_STATUSES or (400 <= status < 500 and status != 429):
                        raise RuntimeError(f"Download failed (permanent): {type(exc).__name__}: {exc}") from exc
                
                if attempt >= max_attempts:
                    raise RuntimeError(f"Download failed after {max_attempts} attempts: {type(exc).__name__}: {exc}") from exc
                
                delay = min((2 ** (attempt - 1)) + rng.random(), 30.0)
                sleep_fn(delay)
        else:
            # Retry loop exhausted without success
            if last_exc:
                raise RuntimeError(f"Download failed after {max_attempts} attempts") from last_exc
    
    raise RuntimeError(f"Too many redirects (>{max_redirects})")


def download_audio_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    session: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    reject_private: bool = True,
    dns_resolver: Callable[[str], list] | None = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> bytes:
    """Download audio bytes with validation and manual redirect handling."""
    return download_audio_bytes_with_redirects(
        url, 
        timeout=timeout, 
        session=session,
        sleep_fn=sleep_fn,
        reject_private=reject_private,
        dns_resolver=dns_resolver,
        max_response_bytes=max_response_bytes,
    )


def dataset_viewer_get(
    endpoint: str,
    *,
    params: dict[str, Any],
    token: str | None = None,
    timeout: int = 120,
    session: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    pace_seconds: float = 0.15,
) -> dict[str, Any]:
    url = f"https://datasets-server.huggingface.co/{endpoint}"
    validate_https_url(url, allow_hosts=HF_VIEWER_HOSTS)
    if pace_seconds > 0:
        (sleep_fn or time.sleep)(pace_seconds)
    resp = request_with_retries(
        "GET",
        url,
        headers=viewer_headers(token),
        params=params,
        timeout=timeout,
        session=session,
        sleep_fn=sleep_fn,
    )
    return resp.json()


def fetch_hub_dataset_sha(dataset_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(dataset_id, files_metadata=False)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"Hub dataset_info({dataset_id}) returned no SHA")
    return str(sha)


# ---------------------------------------------------------------------------
# Viewer offsets / sampling / audio extract
# ---------------------------------------------------------------------------


def uniform_window_offsets(
    n_rows: int,
    window_length: int,
    max_windows: int,
    seed: int = 42,
) -> list[int]:
    """
    Deterministic head-to-tail offsets:
    first offset is always 0; last is always n_rows - window_length (when >0).
    """
    if n_rows <= 0:
        return []
    window_length = max(1, int(window_length))
    max_windows = max(1, int(max_windows))
    max_start = max(0, n_rows - window_length)
    if max_start == 0:
        return [0]
    if max_windows == 1:
        return [0]

    rng = np.random.default_rng(seed)
    anchors = np.linspace(0, max_start, num=max_windows)
    offsets: list[int] = []
    for i, anchor in enumerate(anchors):
        if i == 0:
            off = 0
        elif i == max_windows - 1:
            off = max_start
        else:
            span = max(1, int(round(max_start / max(max_windows - 1, 1))))
            jitter = int(rng.integers(0, max(1, span // 4 + 1)))
            off = int(min(max_start, max(0, round(float(anchor)) + jitter)))
        offsets.append(off)

    uniq = sorted(set(offsets) | {0, max_start})
    if len(uniq) <= max_windows:
        assert uniq[0] == 0 and uniq[-1] == max_start
        return uniq
    middles = uniq[1:-1]
    need = max_windows - 2
    if need <= 0:
        return [0, max_start]
    pick_idx = np.linspace(0, len(middles) - 1, num=need)
    picked = [middles[int(round(i))] for i in pick_idx]
    out = sorted(set([0, max_start] + picked))
    return out


def extract_audio_src(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        text = obj.strip()
        if text.startswith("https://"):
            return text
        return None
    if isinstance(obj, dict):
        for key in ("src", "url", "path"):
            if key in obj:
                found = extract_audio_src(obj[key])
                if found:
                    return found
        for value in obj.values():
            found = extract_audio_src(value)
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = extract_audio_src(item)
            if found:
                return found
    return None


def duration_bin(seconds: float | None) -> str:
    if is_missing_scalar(seconds):
        return "unknown"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(s):
        return "unknown"
    if s < 3:
        return "0-3s"
    if s < 8:
        return "3-8s"
    if s < 15:
        return "8-15s"
    if s < 30:
        return "15-30s"
    return "30s+"


def select_representative_samples(
    candidates: pd.DataFrame,
    *,
    n: int,
    seed: int = 42,
    max_per_group: int = 5,
    max_per_source_fraction: float = 0.35,
    group_col: str = "group_id",
) -> pd.DataFrame:
    """
    Deterministic round-robin across (source_label, duration_bin),
    preferring diverse groups. No duplicate record_uid.
    
    IMPORTANT: If n <= unique candidates, must return exactly n rows.
    Relaxed pass re-examines ALL unselected records.
    """
    if candidates is None or len(candidates) == 0 or n <= 0:
        return candidates.iloc[0:0].copy() if candidates is not None else pd.DataFrame()

    df = candidates.copy().reset_index(drop=True)
    if "duration_bin" not in df.columns:
        df["duration_bin"] = df.get("duration_seconds", pd.Series([None] * len(df))).map(
            duration_bin
        )
    if "record_uid" not in df.columns:
        raise ValueError("select_representative_samples requires record_uid")

    unique_candidates = df["record_uid"].nunique()
    expected_n = min(n, unique_candidates)
    
    rng = np.random.default_rng(seed)
    
    all_indices = list(range(len(df)))
    rng.shuffle(all_indices)
    
    buckets: dict[tuple[str, str], list[int]] = {}
    for i in all_indices:
        row = df.iloc[i]
        key = (str(row.get("source_label", "")), str(row.get("duration_bin", "unknown")))
        buckets.setdefault(key, []).append(i)
    
    for key, idxs in buckets.items():
        idxs.sort(key=lambda i: (str(df.at[i, group_col]) if group_col in df.columns else "", i))
        shuffled = []
        for j in range(0, len(idxs), 3):
            chunk = idxs[j:j+3]
            rng.shuffle(chunk)
            shuffled.extend(chunk)
        buckets[key] = shuffled
    
    bucket_keys = sorted(buckets.keys())
    rng.shuffle(bucket_keys)

    selected: list[int] = []
    seen_uid: set[str] = set()
    per_group: dict[str, int] = {}
    per_source: dict[str, int] = {}
    per_bin: dict[str, int] = {}
    max_per_source = max(1, int(math.ceil(n * max_per_source_fraction)))

    def _can_take(i: int, relax: bool) -> bool:
        uid = str(df.at[i, "record_uid"])
        if uid in seen_uid:
            return False
        if not relax:
            gid = str(df.at[i, group_col]) if group_col in df.columns else ""
            src = str(df.at[i, "source_label"])
            b = str(df.at[i, "duration_bin"])
            if per_group.get(gid, 0) >= max_per_group:
                return False
            if per_source.get(src, 0) >= max_per_source:
                return False
            if per_bin.get(b, 0) >= max(2, n // 2) and len(selected) < n:
                return False
        return True

    def _take(i: int) -> None:
        uid = str(df.at[i, "record_uid"])
        gid = str(df.at[i, group_col]) if group_col in df.columns else ""
        src = str(df.at[i, "source_label"])
        b = str(df.at[i, "duration_bin"])
        selected.append(i)
        seen_uid.add(uid)
        per_group[gid] = per_group.get(gid, 0) + 1
        per_source[src] = per_source.get(src, 0) + 1
        per_bin[b] = per_bin.get(b, 0) + 1

    pointers = {k: 0 for k in bucket_keys}
    progress = True
    while len(selected) < n and progress:
        progress = False
        for key in bucket_keys:
            if len(selected) >= n:
                break
            idxs = buckets[key]
            ptr = pointers[key]
            while ptr < len(idxs):
                i = idxs[ptr]
                ptr += 1
                pointers[key] = ptr
                if _can_take(i, relax=False):
                    _take(i)
                    progress = True
                    break

    if len(selected) < expected_n:
        unselected = [i for i in all_indices if str(df.at[i, "record_uid"]) not in seen_uid]
        for i in unselected:
            if len(selected) >= n:
                break
            if _can_take(i, relax=True):
                _take(i)

    out = df.iloc[selected[:n]].copy().reset_index(drop=True)
    
    if len(out) != expected_n:
        raise RuntimeError(
            f"Representative sampling returned {len(out)} rows, expected {expected_n} "
            f"(n={n}, unique_candidates={unique_candidates})"
        )
    
    assert out["record_uid"].nunique() == len(out), "Duplicate record_uid in selection"
    return out


def sample_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["source_label", "duration_bin", "n_rows", "n_groups"])
    gcol = "group_id" if "group_id" in df.columns else "recording_group_id"
    rows = (
        df.groupby(["source_label", "duration_bin"], dropna=False)
        .agg(n_rows=("record_uid", "size"), n_groups=(gcol, "nunique"))
        .reset_index()
        .sort_values(["source_label", "duration_bin"])
    )
    return rows


# ---------------------------------------------------------------------------
# Audio mono/resample/cache provenance
# ---------------------------------------------------------------------------


def to_mono_float32(wav: np.ndarray) -> np.ndarray:
    """Canonical: normalize integer PCM first, then mix to mono (via audio_utils)."""
    return waveform_to_mono_float32(wav)


def resample_audio(
    wav: np.ndarray,
    orig_sr: int,
    target_sr: int = 16_000,
) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    mono = to_mono_float32(wav)
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("bad_sampling_rate")
    if orig_sr == target_sr:
        return mono
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(mono, up, down).astype(np.float32)


def write_pcm16_wav(path: Path | str, wav: np.ndarray, sampling_rate: int) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = to_mono_float32(wav)
    clipped = np.clip(mono, -1.0, 1.0)
    sf.write(str(path), clipped, int(sampling_rate), subtype="PCM_16")


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    return np.asarray(data), int(sr)


def safe_cache_filename(record_uid: str, suffix: str = ".wav") -> str:
    digest = hashlib.sha256(str(record_uid).encode("utf-8")).hexdigest()[:32]
    suf = suffix if suffix.startswith(".") else f".{suffix}"
    return f"uid_{digest}{suf}"


def cache_sidecar_path(cache_wav: Path) -> Path:
    return Path(cache_wav).with_suffix(".json")


def build_cache_provenance(
    *,
    record_uid: str,
    dataset_revision: str,
    target_sr: int,
    source_qa: dict[str, Any],
    cache_sha256: str,
) -> dict[str, Any]:
    return {
        "record_uid": record_uid,
        "dataset_revision": dataset_revision,
        "target_sampling_rate": int(target_sr),
        "audio_processing_version": AUDIO_PROCESSING_VERSION,
        "cache_sha256": cache_sha256,
        "source_qa": {
            "ok": source_qa.get("ok"),
            "hard_ok": source_qa.get("hard_ok"),
            "reason": source_qa.get("reason"),
            "quality_warnings": source_qa.get("quality_warnings"),
            "duration_sec": source_qa.get("duration_sec"),
            "rms": source_qa.get("rms"),
            "clip_ratio": source_qa.get("clip_ratio"),
            "sampling_rate": source_qa.get("sampling_rate"),
        },
    }


def cache_provenance_matches(
    prov: dict[str, Any] | None,
    *,
    record_uid: str,
    dataset_revision: str,
    target_sr: int,
    cache_sha256: str,
) -> bool:
    """
    Check if provenance matches expected values.
    Returns False (not crash) for corrupt/missing/invalid provenance.
    """
    if prov is None:
        return False
    if not isinstance(prov, dict):
        return False
    
    try:
        return (
            str(prov.get("record_uid", "")) == str(record_uid)
            and str(prov.get("dataset_revision", "")) == str(dataset_revision)
            and int(prov.get("target_sampling_rate", -1)) == int(target_sr)
            and str(prov.get("audio_processing_version", "")) == AUDIO_PROCESSING_VERSION
            and str(prov.get("cache_sha256", "")) == str(cache_sha256)
            and is_valid_sha256(cache_sha256)
        )
    except (TypeError, ValueError, KeyError):
        return False


def load_cache_sidecar(cache_wav: Path) -> dict[str, Any] | None:
    """Load sidecar, return None on any error (don't crash)."""
    side = cache_sidecar_path(cache_wav)
    if not side.is_file():
        return None
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_cache_sidecar(cache_wav: Path, provenance: dict[str, Any]) -> Path:
    side = cache_sidecar_path(cache_wav)
    side.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return side


def validate_cache_wav(
    cache_path: Path,
    *,
    target_sr: int = 16_000,
    min_duration: float = 0.05,
    max_duration: float = 120.0,
    expected_duration: float | None = None,
    recompute_sha: bool = True,
) -> dict[str, Any]:
    """Strict cache QA: exists, 16 kHz, mono, finite, SHA, hard_ok."""
    out: dict[str, Any] = {
        "cache_exists": False,
        "cache_sr": None,
        "cache_channels": None,
        "cache_duration_sec": None,
        "cache_hard_ok": False,
        "cache_ok": False,
        "cache_reason": "missing",
        "cache_sha256": None,
        "cache_sha256_recomputed": None,
        "cache_sha_match": False,
        "cache_waveform_finite": False,
        "cache_rms": None,
        "cache_clip_ratio": None,
        "cache_quality_warnings": [],
    }
    path = Path(cache_path)
    if not path.is_file():
        return out
    out["cache_exists"] = True
    try:
        wav, sr = read_wav(path)
        arr = np.asarray(wav)
        qa = check_waveform(
            arr,
            sr,
            min_duration=min_duration,
            max_duration=max_duration,
            expected_duration=expected_duration,
        )
        channels = 1 if arr.ndim == 1 else int(arr.shape[-1] if arr.ndim == 2 else -1)
        finite = bool(np.isfinite(arr).all()) if arr.size else False
        digest = sha256_file(path) if recompute_sha else None
        out.update(
            {
                "cache_sr": int(sr),
                "cache_channels": channels,
                "cache_duration_sec": qa.get("duration_sec"),
                "cache_ok": bool(qa.get("ok")),
                "cache_hard_ok": bool(
                    qa.get("hard_ok")
                    and int(sr) == int(target_sr)
                    and channels == 1
                    and finite
                ),
                "cache_reason": qa.get("reason") if qa.get("hard_ok") else qa.get("reason"),
                "cache_sha256": digest,
                "cache_sha256_recomputed": digest,
                "cache_sha_match": is_valid_sha256(digest) if digest else False,
                "cache_waveform_finite": finite,
                "cache_rms": qa.get("rms"),
                "cache_clip_ratio": qa.get("clip_ratio"),
                "cache_quality_warnings": qa.get("quality_warnings") or [],
            }
        )
        if not finite:
            out["cache_hard_ok"] = False
            out["cache_reason"] = "non_finite_waveform"
        elif int(sr) != int(target_sr):
            out["cache_hard_ok"] = False
            out["cache_reason"] = f"bad_sr:{sr}"
        elif channels != 1:
            out["cache_hard_ok"] = False
            out["cache_reason"] = f"bad_channels:{channels}"
        elif not is_valid_sha256(digest):
            out["cache_hard_ok"] = False
            out["cache_reason"] = "invalid_sha256"
    except Exception as exc:
        out["cache_hard_ok"] = False
        out["cache_reason"] = f"cache_read_error:{type(exc).__name__}"
    return out


# ---------------------------------------------------------------------------
# Audio candidate processing (testable helper)
# ---------------------------------------------------------------------------


def init_audio_result_schema(
    *,
    final_split: str,
    record_uid: str,
    record_id: Any,
    source_label: Any,
    group_id: Any,
    duration_seconds_meta: Any,
    text_identity_ok: bool,
    cache_path: str,
    cache_name: str,
) -> dict[str, Any]:
    """Initialize result with ALL fields to ensure consistent schema."""
    return {
        "final_split": final_split,
        "record_uid": str(record_uid),
        "record_id": record_id,
        "source_label": source_label,
        "group_id": group_id,
        "duration_seconds_meta": duration_seconds_meta,
        "text_identity_ok": bool(text_identity_ok),
        "cache_path": cache_path,
        "cache_name": cache_name,
        "reused_cache": False,
        "source_qa_available": False,
        # Source QA fields
        "source_ok": None, "source_hard_ok": None, "source_reason": None,
        "source_quality_warnings": None, "source_duration_sec": None,
        "source_rms": None, "source_clip_ratio": None, "source_sampling_rate": None,
        # Cache QA fields
        "cache_exists": False, "cache_sr": None, "cache_channels": None,
        "cache_duration_sec": None, "cache_hard_ok": False, "cache_ok": False,
        "cache_reason": None, "cache_rms": None, "cache_clip_ratio": None,
        "cache_quality_warnings": None, "cache_sha256": None,
        # Combined QA
        "qa_ok": False, "qa_hard_ok": False, "qa_reason": "not_processed",
    }


def source_qa_to_result_fields(qa: dict[str, Any]) -> dict[str, Any]:
    """Convert source QA dict to result fields."""
    return {
        "source_ok": qa.get("ok"),
        "source_hard_ok": qa.get("hard_ok"),
        "source_reason": qa.get("reason"),
        "source_quality_warnings": "|".join(qa.get("quality_warnings") or []),
        "source_duration_sec": qa.get("duration_sec"),
        "source_rms": qa.get("rms"),
        "source_clip_ratio": qa.get("clip_ratio"),
        "source_sampling_rate": qa.get("sampling_rate"),
    }


def process_audio_candidate(
    *,
    record_uid: str,
    record_id: Any,
    final_split: str,
    source_label: Any,
    group_id: Any,
    duration_seconds_meta: float | None,
    text_identity_ok: bool,
    audio_src: str | None,
    row_idx: int,
    cache_dir: Path,
    project_root: Path,
    dataset_revision: str,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    # Dependency injection for testing
    download_fn: Callable[[str], bytes] | None = None,
    refresh_url_fn: Callable[[int], str | None] | None = None,
    decode_fn: Callable[[bytes], tuple[np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    """
    Process a single audio candidate.
    
    This is the testable helper extracted from the notebook.
    Supports dependency injection for download_fn, refresh_url_fn, decode_fn.
    """
    import soundfile as sf
    
    # Default implementations
    if download_fn is None:
        download_fn = lambda url: download_audio_bytes(url)
    if decode_fn is None:
        decode_fn = lambda b: sf.read(io.BytesIO(b), always_2d=False)
    
    cache_path = cache_dir / safe_cache_filename(record_uid)
    expected_dur = float(duration_seconds_meta) if pd.notna(duration_seconds_meta) else None
    
    result = init_audio_result_schema(
        final_split=final_split,
        record_uid=record_uid,
        record_id=record_id,
        source_label=source_label,
        group_id=group_id,
        duration_seconds_meta=duration_seconds_meta,
        text_identity_ok=text_identity_ok,
        cache_path=str(cache_path.relative_to(project_root)),
        cache_name=cache_path.name,
    )
    
    # Check existing cache with provenance
    if cache_path.is_file():
        cache_qa = validate_cache_wav(
            cache_path, 
            target_sr=target_sr,
            min_duration=min_duration, 
            max_duration=max_duration,
            expected_duration=expected_dur, 
            recompute_sha=True
        )
        prov = load_cache_sidecar(cache_path)
        sha = cache_qa.get("cache_sha256")
        
        if (cache_qa.get("cache_hard_ok") and prov is not None 
            and cache_provenance_matches(
                prov, 
                record_uid=record_uid, 
                dataset_revision=dataset_revision,
                target_sr=target_sr, 
                cache_sha256=sha
            )
            and prov.get("source_qa")):
            # Valid cached result - rehydrate source QA from provenance
            src_qa = dict(prov["source_qa"])
            result.update(source_qa_to_result_fields(src_qa))
            result["source_qa_available"] = True
            result.update({
                "cache_exists": True, 
                "cache_sr": cache_qa["cache_sr"],
                "cache_channels": cache_qa["cache_channels"],
                "cache_duration_sec": cache_qa["cache_duration_sec"],
                "cache_hard_ok": cache_qa["cache_hard_ok"], 
                "cache_ok": cache_qa["cache_ok"],
                "cache_reason": cache_qa["cache_reason"], 
                "cache_rms": cache_qa.get("cache_rms"),
                "cache_clip_ratio": cache_qa.get("cache_clip_ratio"),
                "cache_quality_warnings": "|".join(cache_qa.get("cache_quality_warnings") or []),
                "cache_sha256": sha, 
                "reused_cache": True,
            })
            result["qa_hard_ok"] = bool(result.get("source_hard_ok") and result.get("cache_hard_ok"))
            result["qa_ok"] = bool(result.get("source_ok") and result.get("cache_ok"))
            result["qa_reason"] = "ok" if result["qa_hard_ok"] else f"source:{result.get('source_reason')}|cache:{result.get('cache_reason')}"
            return result
    
    # Need to download source audio
    src = audio_src
    if not src and refresh_url_fn is not None:
        src = refresh_url_fn(row_idx)
    if not src:
        result.update({
            "qa_reason": "missing_audio_src", 
            "source_reason": "missing_audio_src", 
            "cache_reason": "skipped_no_source"
        })
        return result
    
    # Download and decode
    try:
        validate_https_url(src, reject_private=True)
        audio_bytes = download_fn(src)
        array, sr = decode_fn(audio_bytes)
    except Exception as exc:
        # Try refresh URL once
        if refresh_url_fn is not None:
            try:
                src2 = refresh_url_fn(row_idx)
                if src2:
                    validate_https_url(src2, reject_private=True)
                    audio_bytes = download_fn(src2)
                    array, sr = decode_fn(audio_bytes)
                else:
                    raise exc
            except Exception as exc2:
                result.update({
                    "qa_reason": f"download_error:{type(exc2).__name__}", 
                    "source_reason": f"download_error:{type(exc2).__name__}", 
                    "cache_reason": "skipped_download_failed"
                })
                return result
        else:
            result.update({
                "qa_reason": f"download_error:{type(exc).__name__}", 
                "source_reason": f"download_error:{type(exc).__name__}", 
                "cache_reason": "skipped_download_failed"
            })
            return result
    
    # Source QA
    source_qa = check_waveform(
        array, 
        sr, 
        min_duration=min_duration, 
        max_duration=max_duration, 
        expected_duration=expected_dur
    )
    result.update(source_qa_to_result_fields(source_qa))
    result["source_qa_available"] = True
    
    if not source_qa["hard_ok"]:
        result.update({
            "qa_ok": False, 
            "qa_hard_ok": False, 
            "qa_reason": f"source_fail:{source_qa['reason']}", 
            "cache_reason": "skipped_source_fail"
        })
        return result
    
    # Create cache
    try:
        mono = to_mono_float32(array)
        resampled = resample_audio(mono, int(sr), target_sr)
        write_pcm16_wav(cache_path, resampled, target_sr)
        cache_qa = validate_cache_wav(
            cache_path, 
            target_sr=target_sr,
            min_duration=min_duration, 
            max_duration=max_duration,
            expected_duration=expected_dur, 
            recompute_sha=True
        )
        sha = cache_qa["cache_sha256"]
        prov = build_cache_provenance(
            record_uid=record_uid, 
            dataset_revision=dataset_revision,
            target_sr=target_sr, 
            source_qa=source_qa, 
            cache_sha256=sha
        )
        write_cache_sidecar(cache_path, prov)
        
        result.update({
            "cache_exists": True, 
            "cache_sr": cache_qa["cache_sr"],
            "cache_channels": cache_qa["cache_channels"],
            "cache_duration_sec": cache_qa["cache_duration_sec"],
            "cache_hard_ok": cache_qa["cache_hard_ok"], 
            "cache_ok": cache_qa["cache_ok"],
            "cache_reason": cache_qa["cache_reason"], 
            "cache_rms": cache_qa.get("cache_rms"),
            "cache_clip_ratio": cache_qa.get("cache_clip_ratio"),
            "cache_quality_warnings": "|".join(cache_qa.get("cache_quality_warnings") or []),
            "cache_sha256": sha, 
            "reused_cache": False,
        })
        
        if cache_qa["cache_hard_ok"]:
            result.update({"qa_ok": True, "qa_hard_ok": True, "qa_reason": "ok"})
        else:
            result.update({
                "qa_ok": False, 
                "qa_hard_ok": False, 
                "qa_reason": f"cache_fail:{cache_qa['cache_reason']}"
            })
    except Exception as exc:
        result.update({
            "cache_hard_ok": False, 
            "cache_reason": f"cache_error:{type(exc).__name__}",
            "qa_hard_ok": False, 
            "qa_reason": f"cache_error:{type(exc).__name__}"
        })
    
    return result


# ---------------------------------------------------------------------------
# Run metadata / report verification
# ---------------------------------------------------------------------------


def run_metadata(
    *,
    run_id: str,
    dataset_id: str,
    dataset_revision: str,
    seed: int,
    processing_version: str,
    input_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_name": dataset_id,
        "dataset_revision": dataset_revision,
        "seed": int(seed),
        "processing_version": processing_version,
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "input_sha256": input_sha256 or {},
    }


def verify_report_metadata(
    report_path: Path,
    *,
    expected_run_id: str,
    expected_revision: str,
    expected_processing_version: str | None = None,
    verify_sha: bool = True,
    verify_schema: bool = False,
) -> bool:
    """
    Verify that a CSV report belongs to the current run (not stale).
    
    Checks:
    - run_id matches
    - dataset_revision matches
    - processing_version matches (if provided)
    - file_sha256 matches (if verify_sha=True and sidecar has it)
    - columns match sidecar (if verify_schema=True)
    - row_count matches sidecar (if verify_schema=True)
    
    Returns False if CSV was modified after sidecar creation.
    """
    meta_path = report_path.with_suffix(report_path.suffix + ".meta.json")
    if not meta_path.is_file():
        return False
    if not report_path.is_file():
        return False
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        
        # Basic checks
        if meta.get("run_id") != expected_run_id:
            return False
        if meta.get("dataset_revision") != expected_revision:
            return False
        if expected_processing_version is not None:
            if meta.get("processing_version") != expected_processing_version:
                return False
        
        # SHA verification - detect tampered files
        if verify_sha and "file_sha256" in meta:
            actual_sha = sha256_file(report_path)
            if actual_sha != meta["file_sha256"]:
                return False
        
        # Schema verification
        if verify_schema:
            df = pd.read_csv(report_path, nrows=0)  # Just headers
            actual_columns = list(df.columns)
            expected_columns = meta.get("columns", [])
            
            if actual_columns != expected_columns:
                return False
            
            # Row count (requires reading full file)
            if "row_count" in meta:
                df_full = pd.read_csv(report_path)
                if len(df_full) != meta["row_count"]:
                    return False
        
        return True
    except (OSError, json.JSONDecodeError, KeyError, TypeError, pd.errors.EmptyDataError):
        return False


def verify_report_metadata_detailed(
    report_path: Path,
    *,
    expected_run_id: str,
    expected_revision: str,
    expected_processing_version: str | None = None,
) -> dict[str, Any]:
    """
    Detailed verification of CSV report returning all check results.
    
    Unlike verify_report_metadata which returns bool, this returns a dict
    with individual check results for debugging.
    """
    result = {
        "passed": False,
        "errors": [],
        "meta_exists": False,
        "csv_exists": False,
        "run_id_match": False,
        "revision_match": False,
        "processing_version_match": False,
        "sha_match": False,
        "columns_match": False,
        "row_count_match": False,
    }
    
    meta_path = report_path.with_suffix(report_path.suffix + ".meta.json")
    
    if not report_path.is_file():
        result["errors"].append("CSV file not found")
        return result
    result["csv_exists"] = True
    
    if not meta_path.is_file():
        result["errors"].append("Sidecar metadata not found")
        return result
    result["meta_exists"] = True
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        result["errors"].append(f"Failed to read metadata: {e}")
        return result
    
    # Check run_id
    if meta.get("run_id") == expected_run_id:
        result["run_id_match"] = True
    else:
        result["errors"].append(f"run_id mismatch: {meta.get('run_id')} != {expected_run_id}")
    
    # Check revision
    if meta.get("dataset_revision") == expected_revision:
        result["revision_match"] = True
    else:
        result["errors"].append(f"revision mismatch: {meta.get('dataset_revision')} != {expected_revision}")
    
    # Check processing_version
    if expected_processing_version is not None:
        if meta.get("processing_version") == expected_processing_version:
            result["processing_version_match"] = True
        else:
            result["errors"].append(f"processing_version mismatch")
    else:
        result["processing_version_match"] = True
    
    # Check SHA
    if "file_sha256" in meta:
        actual_sha = sha256_file(report_path)
        if actual_sha == meta["file_sha256"]:
            result["sha_match"] = True
        else:
            result["errors"].append("SHA256 mismatch - file was modified")
    else:
        result["sha_match"] = True  # No SHA in metadata, skip check
    
    # Check columns
    if "columns" in meta:
        try:
            df = pd.read_csv(report_path, nrows=0)
            if list(df.columns) == meta["columns"]:
                result["columns_match"] = True
            else:
                result["errors"].append("Column order mismatch")
        except Exception as e:
            result["errors"].append(f"Failed to read CSV: {e}")
    else:
        result["columns_match"] = True
    
    # Check row count
    if "row_count" in meta:
        try:
            df = pd.read_csv(report_path)
            if len(df) == meta["row_count"]:
                result["row_count_match"] = True
            else:
                result["errors"].append(f"Row count mismatch: {len(df)} != {meta['row_count']}")
        except Exception as e:
            result["errors"].append(f"Failed to count rows: {e}")
    else:
        result["row_count_match"] = True
    
    result["passed"] = (
        result["run_id_match"] and result["revision_match"] and
        result["processing_version_match"] and result["sha_match"] and
        result["columns_match"] and result["row_count_match"]
    )
    
    return result


def verify_json_metadata(
    path: Path,
    *,
    expected_run_id: str,
    expected_revision: str,
    expected_processing_version: str,
) -> bool:
    """
    Verify that a JSON report belongs to the current run.
    
    Returns False if:
    - File doesn't exist
    - JSON invalid
    - run_id mismatch
    - dataset_revision mismatch
    - processing_version mismatch
    """
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            data.get("run_id") == expected_run_id
            and data.get("dataset_revision") == expected_revision
            and data.get("processing_version") == expected_processing_version
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


def write_dataframe_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Write CSV after stripping signed URLs with complete sidecar metadata.
    
    Sidecar metadata includes:
    - run_id, dataset_revision, processing_version (from metadata)
    - row_count: actual row count of saved DataFrame
    - columns: column names in order
    - dtypes: column dtypes as strings
    - file_sha256: SHA-256 of the written CSV file
    """
    path = Path(path)
    cleaned = strip_signed_url_columns(df.copy())
    assert_no_signed_urls_persisted(cleaned)
    cleaned.to_csv(path, index=False)
    
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        safe_meta = json.loads(json.dumps(metadata, ensure_ascii=False, default=str))
        for banned in ("HF_TOKEN", "token", "authorization", "Authorization"):
            safe_meta.pop(banned, None)
        
        # Add schema and integrity fields
        safe_meta["row_count"] = len(cleaned)
        safe_meta["columns"] = list(cleaned.columns)
        safe_meta["dtypes"] = {col: str(cleaned[col].dtype) for col in cleaned.columns}
        safe_meta["file_sha256"] = sha256_file(path)
        
        meta_path.write_text(json.dumps(safe_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def strip_signed_url_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    drop = [
        c
        for c in df.columns
        if c.startswith("_")
        or c.lower() in {"audio_src", "signed_url", "src", "temporary_url"}
        or "signed" in c.lower()
    ]
    return df.drop(columns=drop, errors="ignore")


def assert_no_signed_urls_persisted(df: pd.DataFrame) -> None:
    banned = {"_audio_src", "audio_src", "signed_url", "temporary_url"}
    present = banned.intersection(set(map(str, df.columns)))
    if present:
        raise RuntimeError(f"Refusing to persist signed URL columns: {sorted(present)}")
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna().astype(str).head(20)
        if sample.str.contains(r"X-Amz-Signature|X-Amz-Credential|Signature=", regex=True).any():
            raise RuntimeError(f"Column {col!r} appears to contain signed URLs")


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------


def check_manifest_contract(
    *,
    root: Path,
    dataset_id: str,
    expected_revision: str,
    expected_counts: dict[str, int],
    hub_sha: str | None,
    split_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Check manifest integrity. 
    IMPORTANT: Test split is loaded via restricted loader (no text/audio columns).
    """
    paths = project_paths(root)
    manifests = paths["manifests"]
    audit = paths["audit"]
    summary_path = manifests / "split_summary.json"
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("split_summary_exists", summary_path.is_file(), str(summary_path))
    if not summary_path.is_file():
        return pd.DataFrame(checks)

    summary = split_summary or json.loads(summary_path.read_text(encoding="utf-8"))
    add("dataset_id_match", summary.get("dataset_id") == dataset_id, str(summary.get("dataset_id")))
    add(
        "dataset_revision_match",
        summary.get("dataset_commit_sha") == expected_revision
        and summary.get("expected_dataset_revision") == expected_revision,
        str(summary.get("dataset_commit_sha")),
    )
    if hub_sha is not None:
        add("hub_sha_matches_locked", hub_sha == expected_revision, str(hub_sha))

    sha_map = summary.get("manifest_sha256") or {}
    frames: dict[str, pd.DataFrame] = {}
    
    for split, count in expected_counts.items():
        fname = f"rq1_{split}.csv"
        path = manifests / fname
        add(f"{fname}_exists", path.is_file(), str(path))
        if not path.is_file():
            continue
        expected_sha = sha_map.get(fname)
        add(f"{fname}_sha_valid_format", is_valid_sha256(expected_sha), str(expected_sha))
        actual_sha = sha256_file(path)
        add(f"{fname}_sha_match", actual_sha == expected_sha, actual_sha)
        
        if split == "test":
            df, seal_report = load_frozen_test_restricted(root)
            add("test_restricted_load", seal_report["passed"], 
                f"loaded_cols={seal_report['loaded_columns']}")
        else:
            df = pd.read_csv(path)
        
        frames[split] = df
        add(f"{split}_row_count", len(df) == int(count), f"actual={len(df)} expected={count}")
        
        if split == "test":
            required_for_test = [c for c in ["record_uid", "split"] if c in FROZEN_TEST_ALLOWED_COLUMNS]
            missing = [c for c in required_for_test if c not in df.columns]
        else:
            missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
        add(f"{split}_required_columns", not missing, ",".join(missing))
        
        if "record_uid" in df.columns:
            add(
                f"{split}_record_uid_unique",
                df["record_uid"].nunique(dropna=False) == len(df),
                "",
            )

    for name in ("final_integrity_checks.csv", "export_preflight_checks.csv"):
        path = audit / name
        add(f"{name}_exists", path.is_file(), str(path))
        if path.is_file():
            chk = pd.read_csv(path)
            if "passed" in chk.columns:
                try:
                    ok = bool(parse_bool_series(chk["passed"]).all())
                except ValueError as exc:
                    ok = False
                    add(f"{name}_all_pass", False, str(exc))
                    continue
            else:
                ok = False
            add(f"{name}_all_pass", ok, "")

    if "train" in frames and "source_split" in frames["train"].columns:
        add(
            "train_source_split_is_train",
            set(frames["train"]["source_split"].astype(str).unique()) == {"train"},
            "",
        )
    if "validation" in frames and "source_split" in frames["validation"].columns:
        add(
            "validation_source_split_is_train",
            set(frames["validation"]["source_split"].astype(str).unique()) == {"train"},
            "",
        )
    if "test" in frames:
        add("test_rows_215", len(frames["test"]) == 215, str(len(frames["test"])))
        if "source_split" in frames["test"].columns:
            src_counts = frames["test"]["source_split"].astype(str).value_counts().to_dict()
            add(
                "test_from_public_val_test_audio",
                src_counts.get("validation", 0) == 205 and src_counts.get("test", 0) == 10,
                str(src_counts),
            )

    def _overlap(a: pd.DataFrame, b: pd.DataFrame, col: str) -> set:
        if col not in a.columns or col not in b.columns:
            return set()
        return set(a[col].astype(str)) & set(b[col].astype(str))

    if set(frames) >= {"train", "validation", "test"}:
        for col in ("group_id", "recording_group_id", "record_id"):
            if col in frames["train"].columns and col in frames["validation"].columns:
                add(
                    f"no_train_val_{col}_overlap",
                    not bool(_overlap(frames["train"], frames["validation"], col)),
                    "",
                )
            if col in frames["train"].columns and col in frames["test"].columns:
                add(
                    f"no_train_test_{col}_overlap",
                    not bool(_overlap(frames["train"], frames["test"], col)),
                    "",
                )
            if col in frames["validation"].columns and col in frames["test"].columns:
                add(
                    f"no_val_test_{col}_overlap",
                    not bool(_overlap(frames["validation"], frames["test"], col)),
                    "",
                )
        
        if "pair_key" in frames["train"].columns and "pair_key" in frames["test"].columns:
            add(
                "no_train_test_pair_key_overlap",
                not bool(_overlap(frames["train"], frames["test"], "pair_key")),
                "",
            )
        if "pair_key" in frames["validation"].columns and "pair_key" in frames["test"].columns:
            add(
                "no_val_test_pair_key_overlap",
                not bool(_overlap(frames["validation"], frames["test"], "pair_key")),
                "",
            )
        if "pair_key" in frames["train"].columns and "pair_key" in frames["validation"].columns:
            tv_pair = _overlap(frames["train"], frames["validation"], "pair_key")
            add(
                "train_val_pair_key_overlap_reported",
                True,
                f"overlap_n={len(tv_pair)} (informational; not a Notebook 01 hard fail)",
            )

    return pd.DataFrame(checks)


def export_clean_split_contract(
    *,
    output_path: Path,
    run_id: str,
    dataset_id: str,
    dataset_revision: str,
    normalization_version: str,
    base_manifest_sha256: dict[str, str],
    train_original_count: int,
    train_exclusion_count: int,
    train_clean_count: int,
    train_ordered_uid_sha256: str,
    train_uid_set_sha256: str,
    train_exclusion_csv_sha256: str,
    validation_original_count: int,
    validation_exclusion_count: int,
    validation_clean_count: int,
    validation_ordered_uid_sha256: str,
    validation_uid_set_sha256: str,
    validation_exclusion_csv_sha256: str,
) -> dict[str, Any]:
    """
    Export clean split contract to JSON.
    
    This contract captures the exact state of clean splits for reproducibility.
    """
    contract = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "normalization_version": normalization_version,
        "policy": "exclude_unexpected_script",
        "base_manifest_sha256": base_manifest_sha256,
        "train": {
            "original_count": train_original_count,
            "excluded_count": train_exclusion_count,
            "clean_count": train_clean_count,
            "ordered_uid_sha256": train_ordered_uid_sha256,
            "uid_set_sha256": train_uid_set_sha256,
            "exclusion_csv_sha256": train_exclusion_csv_sha256,
        },
        "validation": {
            "original_count": validation_original_count,
            "excluded_count": validation_exclusion_count,
            "clean_count": validation_clean_count,
            "ordered_uid_sha256": validation_ordered_uid_sha256,
            "uid_set_sha256": validation_uid_set_sha256,
            "exclusion_csv_sha256": validation_exclusion_csv_sha256,
        },
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return contract


def verify_clean_split_contract(
    contract_path: Path,
    *,
    expected_run_id: str,
    expected_revision: str,
    train_clean_df: pd.DataFrame,
    validation_clean_df: pd.DataFrame,
    train_exclusion_path: Path,
    validation_exclusion_path: Path,
) -> dict[str, Any]:
    """
    Verify clean split contract by recomputing hashes and comparing.
    
    Returns verification result with all checks.
    """
    result = {
        "passed": False,
        "errors": [],
        "contract_exists": False,
        "run_id_match": False,
        "revision_match": False,
        "train_count_match": False,
        "validation_count_match": False,
        "train_ordered_hash_match": False,
        "train_set_hash_match": False,
        "validation_ordered_hash_match": False,
        "validation_set_hash_match": False,
        "train_exclusion_sha_match": False,
        "validation_exclusion_sha_match": False,
    }
    
    if not contract_path.is_file():
        result["errors"].append("Contract file not found")
        return result
    result["contract_exists"] = True
    
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        result["errors"].append(f"Failed to read contract: {e}")
        return result
    
    # Check run_id
    if contract.get("run_id") == expected_run_id:
        result["run_id_match"] = True
    else:
        result["errors"].append(f"run_id mismatch")
    
    # Check revision
    if contract.get("dataset_revision") == expected_revision:
        result["revision_match"] = True
    else:
        result["errors"].append(f"revision mismatch")
    
    train_contract = contract.get("train", {})
    val_contract = contract.get("validation", {})
    
    # Check train counts
    if len(train_clean_df) == train_contract.get("clean_count"):
        result["train_count_match"] = True
    else:
        result["errors"].append(f"train count: {len(train_clean_df)} != {train_contract.get('clean_count')}")
    
    # Check validation counts
    if len(validation_clean_df) == val_contract.get("clean_count"):
        result["validation_count_match"] = True
    else:
        result["errors"].append(f"validation count: {len(validation_clean_df)} != {val_contract.get('clean_count')}")
    
    # Check train hashes
    try:
        train_ordered = compute_ordered_uid_hash(train_clean_df)
        train_set = compute_uid_set_hash(train_clean_df)
        
        if train_ordered == train_contract.get("ordered_uid_sha256"):
            result["train_ordered_hash_match"] = True
        else:
            result["errors"].append("train ordered hash mismatch")
        
        if train_set == train_contract.get("uid_set_sha256"):
            result["train_set_hash_match"] = True
        else:
            result["errors"].append("train set hash mismatch")
    except Exception as e:
        result["errors"].append(f"train hash error: {e}")
    
    # Check validation hashes
    try:
        val_ordered = compute_ordered_uid_hash(validation_clean_df)
        val_set = compute_uid_set_hash(validation_clean_df)
        
        if val_ordered == val_contract.get("ordered_uid_sha256"):
            result["validation_ordered_hash_match"] = True
        else:
            result["errors"].append("validation ordered hash mismatch")
        
        if val_set == val_contract.get("uid_set_sha256"):
            result["validation_set_hash_match"] = True
        else:
            result["errors"].append("validation set hash mismatch")
    except Exception as e:
        result["errors"].append(f"validation hash error: {e}")
    
    # Check exclusion file SHAs
    if train_exclusion_path.is_file():
        train_exc_sha = sha256_file(train_exclusion_path)
        if train_exc_sha == train_contract.get("exclusion_csv_sha256"):
            result["train_exclusion_sha_match"] = True
        else:
            result["errors"].append("train exclusion SHA mismatch")
    else:
        result["errors"].append("train exclusion file not found")
    
    if validation_exclusion_path.is_file():
        val_exc_sha = sha256_file(validation_exclusion_path)
        if val_exc_sha == val_contract.get("exclusion_csv_sha256"):
            result["validation_exclusion_sha_match"] = True
        else:
            result["errors"].append("validation exclusion SHA mismatch")
    else:
        result["errors"].append("validation exclusion file not found")
    
    result["passed"] = all([
        result["run_id_match"],
        result["revision_match"],
        result["train_count_match"],
        result["validation_count_match"],
        result["train_ordered_hash_match"],
        result["train_set_hash_match"],
        result["validation_ordered_hash_match"],
        result["validation_set_hash_match"],
        result["train_exclusion_sha_match"],
        result["validation_exclusion_sha_match"],
    ])
    
    return result


def save_vocab_artifacts(
    vocab: dict[str, int],
    out_dir: Path,
    *,
    normalization_version: str = NORMALIZATION_VERSION,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    norm_path = out_dir / "normalization_config.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    norm_cfg = {
        "version": normalization_version,
        "description": (
            "NFC, casefold, curly apostrophe→', keep letters/marks/hyphen/apostrophe, "
            "other punct→space, word delimiter=|"
        ),
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "word_delimiter_token": "|",
        "bos_token": None,
        "eos_token": None,
        "pad_token_id": 0,
        "unk_token_id": vocab.get("[UNK]", 1),
        "vocab_size": len(vocab),
        "ctc_blank_token": "[PAD]",
        "ctc_blank_token_id": 0,
    }
    norm_path.write_text(json.dumps(norm_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"vocab": vocab_path, "normalization_config": norm_path}


def save_tokenizer_clean(
    vocab: dict[str, int],
    out_dir: Path,
    *,
    target_sampling_rate: int = 16_000,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save tokenizer artifacts cleanly with backup/recovery and provenance.
    
    1. Create temp dir
    2. Save all artifacts there
    3. Reload and verify EXACT vocabulary mapping
    4. Verify no unexpected scripts in vocabulary
    5. Save provenance.json if provided
    6. If target exists, rename to backup
    7. Rename temp to target
    8. Remove backup only after success
    
    Returns verification report.
    """
    import shutil
    import tempfile
    
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    
    out_dir = Path(out_dir)
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.mkdtemp(dir=parent, prefix=".tokenizer_tmp_"))
    backup_dir = None
    
    try:
        # Save vocab and normalization config
        vocab_path = temp_dir / "vocab.json"
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        
        norm_cfg = {
            "version": NORMALIZATION_VERSION,
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "word_delimiter_token": "|",
            "bos_token": None,
            "eos_token": None,
            "pad_token_id": 0,
            "unk_token_id": vocab.get("[UNK]", 1),
            "vocab_size": len(vocab),
            "ctc_blank_token": "[PAD]",
            "ctc_blank_token_id": 0,
        }
        (temp_dir / "normalization_config.json").write_text(
            json.dumps(norm_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        
        # Create and save tokenizer (no BOS/EOS)
        tokenizer = Wav2Vec2CTCTokenizer(
            str(vocab_path),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            bos_token=None,
            eos_token=None,
        )
        tokenizer.save_pretrained(str(temp_dir))
        
        # Create and save feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=target_sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        feature_extractor.save_pretrained(str(temp_dir))
        
        # Create and save processor
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(str(temp_dir))
        
        # Reload and verify
        reloaded = Wav2Vec2CTCTokenizer.from_pretrained(str(temp_dir))
        reloaded_vocab = reloaded.get_vocab()
        
        verification = {
            "passed": True,
            "errors": [],
            "vocab_size_expected": len(vocab),
            "vocab_size_reloaded": len(reloaded),
            "vocab_exact_match": False,
            "vocab_no_unexpected_scripts": True,
            "unexpected_script_chars": [],
            "pad_token_id": reloaded.pad_token_id,
            "unk_token_id": reloaded.unk_token_id,
            "bos_token": reloaded.bos_token,
            "eos_token": reloaded.eos_token,
            "has_stale_bos_eos": False,
        }
        
        # Check EXACT vocabulary mapping (not just size)
        if reloaded_vocab == vocab:
            verification["vocab_exact_match"] = True
        else:
            verification["passed"] = False
            # Find differences
            missing_in_reloaded = set(vocab.keys()) - set(reloaded_vocab.keys())
            extra_in_reloaded = set(reloaded_vocab.keys()) - set(vocab.keys())
            id_mismatches = [
                k for k in vocab if k in reloaded_vocab and vocab[k] != reloaded_vocab[k]
            ]
            if missing_in_reloaded:
                verification["errors"].append(f"tokens missing in reloaded: {missing_in_reloaded}")
            if extra_in_reloaded:
                verification["errors"].append(f"unexpected tokens in reloaded: {extra_in_reloaded}")
            if id_mismatches:
                verification["errors"].append(f"token ID mismatches: {id_mismatches}")
        
        # Check no unexpected scripts in reloaded vocabulary
        for token in reloaded_vocab:
            if token in {"[PAD]", "[UNK]", "|"}:
                continue
            for ch in token:
                classification = classify_character(ch)
                if classification == "unexpected_script":
                    verification["vocab_no_unexpected_scripts"] = False
                    verification["unexpected_script_chars"].append(ch)
                    verification["passed"] = False
        if verification["unexpected_script_chars"]:
            verification["errors"].append(
                f"unexpected scripts in vocab: {verification['unexpected_script_chars']}"
            )
        
        # Check pad/unk
        if reloaded.pad_token_id != vocab["[PAD]"]:
            verification["passed"] = False
            verification["errors"].append("pad_token_id mismatch")
        if reloaded.pad_token_id != 0:
            verification["passed"] = False
            verification["errors"].append("pad_token_id should be 0")
        
        # Check no stale BOS/EOS
        if reloaded.bos_token is not None:
            verification["passed"] = False
            verification["errors"].append(f"bos_token should be None, got {reloaded.bos_token}")
        if reloaded.eos_token is not None:
            verification["passed"] = False
            verification["errors"].append(f"eos_token should be None, got {reloaded.eos_token}")
        if "<s>" in reloaded_vocab:
            verification["passed"] = False
            verification["has_stale_bos_eos"] = True
            verification["errors"].append("<s> found in vocab")
        if "</s>" in reloaded_vocab:
            verification["passed"] = False
            verification["has_stale_bos_eos"] = True
            verification["errors"].append("</s> found in vocab")
        
        # Check added_tokens.json if it exists
        added_tokens_path = temp_dir / "added_tokens.json"
        if added_tokens_path.is_file():
            added = json.loads(added_tokens_path.read_text(encoding="utf-8"))
            if "<s>" in added or "</s>" in added:
                verification["passed"] = False
                verification["has_stale_bos_eos"] = True
                verification["errors"].append("added_tokens.json contains <s> or </s>")
        
        if not verification["passed"]:
            raise RuntimeError(f"Tokenizer verification failed: {verification['errors']}")
        
        # Save provenance if provided
        if provenance is not None:
            # Compute vocab SHA from the saved file
            vocab_sha = sha256_file(vocab_path)
            provenance_data = {
                **provenance,
                "vocab_sha256": vocab_sha,
            }
            (temp_dir / "provenance.json").write_text(
                json.dumps(provenance_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        
        # Backup existing target if it exists
        if out_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(dir=parent, prefix=".tokenizer_backup_"))
            shutil.move(str(out_dir), str(backup_dir / out_dir.name))
        
        # Move temp to target
        temp_dir.rename(out_dir)
        
        # Success - remove backup
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        return verification
        
    except Exception as exc:
        # Restore from backup if available
        if backup_dir is not None and backup_dir.exists():
            backup_content = backup_dir / out_dir.name
            if backup_content.exists():
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                shutil.move(str(backup_content), str(out_dir))
            shutil.rmtree(backup_dir)
        
        # Cleanup temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def verify_tokenizer_provenance(
    tokenizer_dir: Path,
    *,
    expected_run_id: str,
    expected_revision: str,
    expected_train_clean_ordered_uid_sha256: str,
    expected_train_clean_uid_set_sha256: str,
) -> dict[str, Any]:
    """
    Verify tokenizer provenance against expected values.
    
    Checks:
    - provenance.json exists
    - run_id matches
    - dataset_revision matches
    - train_clean hashes match (ordered and set)
    - vocab.json SHA matches provenance
    
    Returns verification result dict.
    """
    result = {
        "passed": False,
        "errors": [],
        "provenance_exists": False,
        "run_id_match": False,
        "revision_match": False,
        "train_ordered_hash_match": False,
        "train_set_hash_match": False,
        "vocab_sha_match": False,
    }
    
    provenance_path = tokenizer_dir / "provenance.json"
    vocab_path = tokenizer_dir / "vocab.json"
    
    if not provenance_path.is_file():
        result["errors"].append("provenance.json not found")
        return result
    result["provenance_exists"] = True
    
    if not vocab_path.is_file():
        result["errors"].append("vocab.json not found")
        return result
    
    try:
        prov = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        result["errors"].append(f"Failed to read provenance: {e}")
        return result
    
    # Check run_id
    if prov.get("run_id") == expected_run_id:
        result["run_id_match"] = True
    else:
        result["errors"].append(f"run_id mismatch: {prov.get('run_id')} != {expected_run_id}")
    
    # Check revision
    if prov.get("dataset_revision") == expected_revision:
        result["revision_match"] = True
    else:
        result["errors"].append("dataset_revision mismatch")
    
    # Check train clean hashes
    if prov.get("train_clean_ordered_uid_sha256") == expected_train_clean_ordered_uid_sha256:
        result["train_ordered_hash_match"] = True
    else:
        result["errors"].append("train_clean_ordered_uid_sha256 mismatch")
    
    if prov.get("train_clean_uid_set_sha256") == expected_train_clean_uid_set_sha256:
        result["train_set_hash_match"] = True
    else:
        result["errors"].append("train_clean_uid_set_sha256 mismatch")
    
    # Check vocab SHA
    if "vocab_sha256" in prov:
        actual_vocab_sha = sha256_file(vocab_path)
        if actual_vocab_sha == prov["vocab_sha256"]:
            result["vocab_sha_match"] = True
        else:
            result["errors"].append("vocab_sha256 mismatch - vocab was modified")
    else:
        result["errors"].append("vocab_sha256 not in provenance")
    
    result["passed"] = all([
        result["run_id_match"],
        result["revision_match"],
        result["train_ordered_hash_match"],
        result["train_set_hash_match"],
        result["vocab_sha_match"],
    ])
    
    return result
