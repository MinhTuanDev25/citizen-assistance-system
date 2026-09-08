"""Unit tests for split_utils (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.split_utils import (
    collect_ambiguous_group_suffixes,
    derive_group,
    namespace_cross_source_collisions,
    recording_group_from_record_id,
    select_best_group_split,
    stratified_group_review_sample,
    strip_segment_suffix,
)


def _derive(rid: str, source: str = "demo", path: str | None = None):
    return derive_group(
        pd.Series(
            {
                "record_id": rid,
                "audio_path": path or f"{rid}.flac",
                "source_label": source,
                "record_uid": f"uid-{rid}",
            }
        )
    )


def test_same_recording_segments_share_group():
    a = _derive("1CO_01_001")
    b = _derive("1CO_01_002")
    assert a["recording_group_id"] == b["recording_group_id"] == "1co_01"
    assert a["group_id"] == b["group_id"]


def test_oneway_segment_indices_share_group():
    a = _derive("oneway_251021_1")
    b = _derive("oneway_251021_10")
    assert a["recording_group_id"] == b["recording_group_id"] == "oneway_251021"


def test_show_2025_not_stripped():
    reduced, changed, note = strip_segment_suffix("show_2025")
    assert changed is False
    assert reduced == "show_2025"
    assert note is not None

    group, ok, amb = recording_group_from_record_id("show_2025")
    assert ok is False
    assert group == ""
    assert amb is not None

    g = _derive("show_2025", source="radio")
    # Falls back to source_label; must NOT become recording:show
    assert "show" != g["recording_group_id"]
    assert g["group_source"] == "source_label"
    assert "show_2025" not in str(g["group_id"]) or "source_label" in g["group_id"]


def test_cross_source_collision_namespaces_group_id():
    rows = []
    for rid, src in [("recA_001", "yt_1"), ("recA_002", "yt_2")]:
        g = _derive(rid, source=src)
        rows.append(
            {
                "record_id": rid,
                "source_label": src,
                "recording_group_id": g["recording_group_id"],
                "group_id": g["group_id"],
                "group_source": g["group_source"],
                "group_confidence": g["group_confidence"],
                "group_suffix_note": g["group_suffix_note"],
            }
        )
    df = pd.DataFrame(rows)
    out, collisions = namespace_cross_source_collisions(df)
    assert len(collisions) == 1
    assert out.loc[0, "group_id"] != out.loc[1, "group_id"]
    assert "yt_1" in out.loc[0, "group_id"]
    assert "yt_2" in out.loc[1, "group_id"]


def test_stratified_review_covers_sources_and_is_deterministic():
    rows = []
    for src in ["alpha", "beta", "gamma"]:
        for g in range(5):
            for k in range(4):
                rows.append(
                    {
                        "record_id": f"{src}_{g}_{k}",
                        "source_label": src,
                        "group_id": f"g-{src}-{g}",
                        "recording_group_id": f"{src}-{g}",
                        "audio_path": f"{src}_{g}_{k}.flac",
                        "duration_seconds": 1.0,
                    }
                )
    df = pd.DataFrame(rows)
    a = stratified_group_review_sample(df, seed=42, max_per_source=20, max_per_group=2)
    b = stratified_group_review_sample(df, seed=42, max_per_source=20, max_per_group=2)
    assert a["record_id"].tolist() == b["record_id"].tolist()
    assert set(a["source_label"]) == {"alpha", "beta", "gamma"}
    assert a.groupby("source_label").size().max() <= 20
    # Diversity: more than one group per source when possible
    assert a.groupby("source_label")["group_id"].nunique().min() >= 2


def test_select_best_group_split_deterministic_no_overlap_band():
    rows = []
    for src_i, src in enumerate(["s1", "s2", "s3"]):
        for g in range(10):
            for k in range(10):
                rows.append(
                    {
                        "record_id": f"{src}_{g}_{k}",
                        "source_label": src,
                        "group_id": f"{src}-g{g}",
                        "duration_seconds": 1.0,
                    }
                )
    df = pd.DataFrame(rows)
    a = select_best_group_split(df, seed=42, n_candidates=200, test_size=0.1)
    b = select_best_group_split(df, seed=42, n_candidates=200, test_size=0.1)
    assert np.array_equal(a["train_idx"], b["train_idx"])
    assert np.array_equal(a["val_idx"], b["val_idx"])
    assert set(df.iloc[a["train_idx"]]["group_id"]) & set(df.iloc[a["val_idx"]]["group_id"]) == set()
    assert 0.08 <= a["val_fraction"] <= 0.12
    assert "score" in a


def test_select_best_group_split_raises_outside_band():
    # One huge group + tiny groups makes 8–12% impossible for some layouts;
    # construct extreme imbalance where no candidate can land in band.
    rows = [{"record_id": f"a_{i}", "source_label": "only", "group_id": "huge", "duration_seconds": 1.0} for i in range(90)]
    rows += [{"record_id": f"b_{i}", "source_label": "only", "group_id": f"tiny{i}", "duration_seconds": 1.0} for i in range(10)]
    df = pd.DataFrame(rows)
    # With test_size=0.1, GroupShuffleSplit may or may not hit band depending on groups.
    # Force impossible band:
    with pytest.raises(ValueError, match="outside required band"):
        select_best_group_split(
            df,
            seed=42,
            n_candidates=50,
            test_size=0.1,
            min_val_fraction=0.40,
            max_val_fraction=0.45,
        )


def test_ambiguous_suffix_export():
    g = _derive("show_2025", source="radio")
    df = pd.DataFrame(
        [
            {
                "record_id": "show_2025",
                "audio_path": "show_2025.flac",
                "source_label": "radio",
                "recording_group_id": g["recording_group_id"],
                "group_id": g["group_id"],
                "group_source": g["group_source"],
                "group_suffix_note": g["group_suffix_note"],
            }
        ]
    )
    amb = collect_ambiguous_group_suffixes(df)
    assert len(amb) == 1
