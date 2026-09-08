"""Group / split helpers for RQ1 manifests."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Named segment markers — always safe to strip once.
_NAMED_SEGMENT_SUFFIX = re.compile(
    r"(?i)(?:[_\-](?:segment|seg|chunk|clip|utterance|utt)[_\-]?\d+)$"
)
_START_END_SUFFIX = re.compile(
    r"(?i)(?:[_\-](?:start|end)[_\-]?\d+(?:[_\-]\d+)?)$"
)
# Zero-padded 4-digit utterance index (_0001) — not calendar years.
_ZERO_PADDED_FOUR = re.compile(r"(?i)(?:[_\-]0\d{3})$")
# Short utterance indices (_1, _10, _001). Do NOT use bare \d{1,4}:
# that would turn show_2025 → show.
_SHORT_INDEX_SUFFIX = re.compile(r"(?i)(?:[_\-]\d{1,3})$")
# Ambiguous trailing 4-digit (possible year / show code) — never auto-stripped.
_FOUR_DIGIT_SUFFIX = re.compile(r"(?i)(?:[_\-]\d{4})$")
_YEARISH_FOUR = re.compile(r"(?i)(?:[_\-](?:19|20)\d{2})$")


def clean_group_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = unicodedata.normalize("NFC", str(value)).strip().casefold()
    value = re.sub(r"\s+", "_", value)
    return value[:240]


def strip_segment_suffix(text: str) -> tuple[str, bool, str | None]:
    """
    Strip one trailing segment index if evidence supports it.

    Returns (reduced, changed, ambiguity_note).
    ambiguity_note is set when a 4-digit suffix was left intact (e.g. show_2025).
    """
    if not text:
        return "", False, None

    for pattern in (_NAMED_SEGMENT_SUFFIX, _START_END_SUFFIX, _ZERO_PADDED_FOUR, _SHORT_INDEX_SUFFIX):
        nxt = pattern.sub("", text)
        if nxt != text and nxt:
            return nxt, True, None

    if _FOUR_DIGIT_SUFFIX.search(text):
        note = "yearish_four_digit_suffix" if _YEARISH_FOUR.search(text) else "four_digit_suffix_not_stripped"
        return text, False, note
    return text, False, None


def recording_group_from_record_id(value) -> tuple[str, bool, str | None]:
    original = clean_group_value(value)
    if not original:
        return "", False, None
    reduced, changed, ambiguity = strip_segment_suffix(original)
    if changed and reduced:
        return reduced, True, ambiguity
    return "", False, ambiguity


def recording_group_from_path(value) -> tuple[str, bool, str | None]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return "", False, None
    parsed = urlparse(str(value))
    path = unquote(parsed.path or str(value)).replace("\\", "/")
    stem = Path(path).stem
    return recording_group_from_record_id(stem)


def derive_group(
    row: pd.Series,
    explicit_group_columns: list[str] | None = None,
) -> pd.Series:
    """
    Priority:
    1) Explicit video/session columns (if present)
    2) recording_group_id from record_id / audio_path (e.g. 1CO_01_001 → 1CO_01)
    3) source_label (renamed speaker_id: program/channel, NOT a person)
    4) row-unique fallback
    """
    explicit_group_columns = explicit_group_columns or []
    for column in explicit_group_columns:
        if column in row.index:
            value = clean_group_value(row.get(column))
            if value:
                return pd.Series(
                    {
                        "recording_group_id": value,
                        "group_id": f"{column}:{value}",
                        "group_source": column,
                        "group_confidence": "high",
                        "group_suffix_note": None,
                    }
                )

    for getter, source in (
        (lambda: recording_group_from_record_id(row.get("record_id")), "record_id_prefix"),
        (lambda: recording_group_from_path(row.get("audio_path")), "audio_path_prefix"),
    ):
        group, ok, note = getter()
        if ok and group:
            return pd.Series(
                {
                    "recording_group_id": group,
                    "group_id": f"recording:{group}",
                    "group_source": source,
                    "group_confidence": "medium",
                    "group_suffix_note": note,
                }
            )
        if note:
            # Keep note even when falling through (ambiguous id like show_2025).
            pass

    # Preserve ambiguity note from record_id attempt when falling back.
    _, _, id_note = recording_group_from_record_id(row.get("record_id"))

    source_label = clean_group_value(row.get("source_label", row.get("speaker_id")))
    if source_label:
        return pd.Series(
            {
                "recording_group_id": f"source:{source_label}",
                "group_id": f"source_label:{source_label}",
                "group_source": "source_label",
                "group_confidence": "low",
                "group_suffix_note": id_note,
            }
        )

    uid = str(row.get("record_uid", ""))
    return pd.Series(
        {
            "recording_group_id": f"row:{uid}",
            "group_id": f"row:{uid}",
            "group_source": "row_unique_fallback",
            "group_confidence": "unresolved",
            "group_suffix_note": id_note,
        }
    )


def namespace_cross_source_collisions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    If the same recording_group_id (prefix-derived) appears under multiple
    source_labels, namespace group_id with source_label to avoid false merges.

    Returns (updated_df, collisions_report).
    """
    out = df.copy()
    required = {"recording_group_id", "group_id", "group_source", "source_label"}
    if out.empty or not required.issubset(out.columns):
        return out, pd.DataFrame(
            columns=[
                "recording_group_id",
                "n_source_labels",
                "source_labels",
                "rows",
            ]
        )

    prefix_mask = out["group_source"].isin(["record_id_prefix", "audio_path_prefix"])
    prefix = out.loc[prefix_mask].copy()
    if prefix.empty:
        return out, pd.DataFrame(
            columns=["recording_group_id", "n_source_labels", "source_labels", "rows"]
        )

    collision_rows = []
    for rgid, g in prefix.groupby("recording_group_id", dropna=False):
        sources = sorted({clean_group_value(s) for s in g["source_label"].tolist() if clean_group_value(s)})
        if len(sources) <= 1:
            continue
        collision_rows.append(
            {
                "recording_group_id": rgid,
                "n_source_labels": len(sources),
                "source_labels": "|".join(sources),
                "rows": int(len(g)),
            }
        )
        collide_ids = set(g.index.tolist())
        for idx in collide_ids:
            src = clean_group_value(out.at[idx, "source_label"]) or "unknown"
            base = out.at[idx, "recording_group_id"]
            out.at[idx, "group_id"] = f"recording:{src}:{base}"

    collisions = pd.DataFrame(collision_rows)
    if not collisions.empty:
        collisions = collisions.sort_values(
            ["n_source_labels", "rows"], ascending=False
        ).reset_index(drop=True)
    return out, collisions


def collect_ambiguous_group_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    """Export ids where a 4-digit suffix was intentionally not stripped."""
    if df.empty or "group_suffix_note" not in df.columns:
        return pd.DataFrame(
            columns=[
                "record_id",
                "audio_path",
                "source_label",
                "group_id",
                "group_suffix_note",
            ]
        )
    cols = [
        c
        for c in (
            "record_id",
            "audio_path",
            "source_label",
            "recording_group_id",
            "group_id",
            "group_source",
            "group_suffix_note",
        )
        if c in df.columns
    ]
    amb = df.loc[df["group_suffix_note"].notna() & (df["group_suffix_note"].astype(str) != ""), cols].copy()
    return amb.drop_duplicates().reset_index(drop=True)


def stratified_group_review_sample(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    max_per_source: int = 20,
    max_per_group: int = 2,
) -> pd.DataFrame:
    """
    Deterministic stratified review sample:
    - cover every source_label
    - <= max_per_source rows per source
    - prefer many group_ids; <= max_per_group per group while alternatives remain
    """
    if df is None or len(df) == 0:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()

    source_col = "source_label" if "source_label" in df.columns else "speaker_id"
    if source_col not in df.columns or "group_id" not in df.columns:
        raise ValueError("stratified_group_review_sample requires source_label and group_id")

    rng = np.random.default_rng(seed)
    picked: list = []

    # sort=True for stable source iteration
    for _, src_df in df.groupby(source_col, dropna=False, sort=True):
        groups = sorted(src_df["group_id"].astype(str).unique().tolist())
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in order]

        per_group = {g: 0 for g in groups}
        selected_idx: list = []
        remaining = {g: src_df.loc[src_df["group_id"].astype(str) == g].index.to_list() for g in groups}
        for g in groups:
            idxs = remaining[g]
            perm = rng.permutation(len(idxs))
            remaining[g] = [idxs[i] for i in perm]

        changed = True
        while len(selected_idx) < max_per_source and changed:
            changed = False
            for g in groups:
                if len(selected_idx) >= max_per_source:
                    break
                if per_group[g] >= max_per_group:
                    continue
                if not remaining[g]:
                    continue
                selected_idx.append(remaining[g].pop(0))
                per_group[g] += 1
                changed = True

        # If still short (few groups), allow filling beyond max_per_group.
        if len(selected_idx) < max_per_source:
            leftovers = []
            for g in groups:
                leftovers.extend(remaining[g])
            need = max_per_source - len(selected_idx)
            selected_idx.extend(leftovers[:need])

        picked.extend(selected_idx)

    return df.loc[picked].copy()


def group_train_validation_indices(groups, test_size: float = 0.1, seed: int = 42):
    """Backward-compatible single split (prefer select_best_group_split)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(range(len(groups)), groups=groups))
    return train_idx, val_idx


def select_best_group_split(
    df: pd.DataFrame,
    *,
    group_col: str = "group_id",
    source_col: str = "source_label",
    test_size: float = 0.1,
    min_val_fraction: float = 0.08,
    max_val_fraction: float = 0.12,
    seed: int = 42,
    n_candidates: int = 200,
) -> dict:
    """
    Try >= n_candidates GroupShuffleSplit seeds; pick best by:
    1) |val_frac - test_size|
    2) source_label distribution L1 vs full pool
    3) penalty if a multi-group source is absent from validation

    Raises if best validation fraction is outside [min_val_fraction, max_val_fraction].
    Group integrity (no shared group_id) is mandatory for every candidate.
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot split empty dataframe")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if source_col not in df.columns:
        raise ValueError(f"Missing source column: {source_col}")
    if df[group_col].nunique(dropna=False) < 2:
        raise ValueError("At least two distinct group_id values are required")

    n = len(df)
    groups = df[group_col].astype(str).to_numpy()
    sources = df[source_col].fillna("__NA__").astype(str)
    full_dist = sources.value_counts(normalize=True)

    # Sources with enough groups to expect validation coverage
    source_group_counts = (
        df.assign(_src=sources, _g=df[group_col].astype(str))
        .groupby("_src")["_g"]
        .nunique()
    )
    multi_group_sources = set(source_group_counts[source_group_counts >= 2].index.tolist())

    best = None
    best_score = float("inf")

    for i in range(n_candidates):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed + i
        )
        train_idx, val_idx = next(splitter.split(np.arange(n), groups=groups))
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)

        if set(groups[train_idx]) & set(groups[val_idx]):
            continue  # should never happen with GroupShuffleSplit

        val_frac = float(len(val_idx) / n)
        size_err = abs(val_frac - test_size)

        val_sources = sources.iloc[val_idx]
        val_dist = val_sources.value_counts(normalize=True)
        all_src = full_dist.index.union(val_dist.index)
        balance_err = float(
            (
                full_dist.reindex(all_src, fill_value=0.0)
                - val_dist.reindex(all_src, fill_value=0.0)
            )
            .abs()
            .sum()
            / 2.0
        )

        val_source_set = set(val_sources.unique().tolist())
        missing_multi = sorted(multi_group_sources - val_source_set)
        missing_penalty = float(len(missing_multi))

        score = (size_err * 10.0) + balance_err + (0.5 * missing_penalty)
        if not (min_val_fraction <= val_frac <= max_val_fraction):
            score += 100.0

        if score < best_score:
            best_score = score
            best = {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_fraction": val_frac,
                "size_error": size_err,
                "source_balance_l1": balance_err,
                "missing_multi_group_sources": missing_multi,
                "missing_penalty": missing_penalty,
                "score": score,
                "candidate_seed": seed + i,
                "candidate_index": i,
            }

    if best is None:
        raise ValueError("No valid group split candidate found")

    if not (min_val_fraction <= best["val_fraction"] <= max_val_fraction):
        raise ValueError(
            f"Best group split validation fraction {best['val_fraction']:.4f} "
            f"is outside required band [{min_val_fraction}, {max_val_fraction}]. "
            f"score={best['score']:.4f} candidate_seed={best['candidate_seed']}"
        )

    # Soft warnings: tiny sources may appear in only one split
    val_set = set(sources.iloc[best["val_idx"]].unique().tolist())
    train_set = set(sources.iloc[best["train_idx"]].unique().tolist())
    single_group_sources = set(source_group_counts[source_group_counts < 2].index.tolist())
    best["warnings"] = {
        "sources_only_in_train": sorted((train_set - val_set) & single_group_sources),
        "sources_only_in_validation": sorted((val_set - train_set) & single_group_sources),
        "multi_group_sources_missing_from_validation": best["missing_multi_group_sources"],
    }
    best["n_candidates"] = n_candidates
    return best
