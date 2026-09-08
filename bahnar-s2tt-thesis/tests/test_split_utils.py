"""Unit tests for split_utils (no network)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.split_utils import (
    assert_prefix_recording_group_invariant,
    audit_cross_source_collisions,
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


def test_same_recording_group_keeps_same_group_id_across_sources():
    """Different source_label must NOT split a prefix recording into different group_ids."""
    a = _derive("recA_001", source="yt_1")
    b = _derive("recA_002", source="yt_2")
    assert a["recording_group_id"] == b["recording_group_id"] == "reca"
    assert a["group_id"] == b["group_id"] == "recording:reca"
    assert a["group_source"] == "record_id_prefix"


def test_collision_audit_empty_report_keeps_schema():
    df = pd.DataFrame(
        [
            {
                "recording_group_id": "rec_a",
                "group_id": "recording:rec_a",
                "group_source": "record_id_prefix",
                "source_label": "source_1",
            }
        ]
    )

    result = audit_cross_source_collisions(df)

    assert result.empty
    assert result.columns.tolist() == [
        "recording_group_id",
        "n_source_labels",
        "source_labels",
        "rows",
    ]


def test_collision_audit_does_not_mutate_group_id():
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
    before = copy.deepcopy(df.to_dict(orient="list"))

    collisions = audit_cross_source_collisions(df)
    assert len(collisions) == 1
    assert df.to_dict(orient="list") == before
    assert df.loc[0, "group_id"] == df.loc[1, "group_id"] == "recording:reca"

    out, collisions2 = namespace_cross_source_collisions(df)
    assert len(collisions2) == 1
    assert out.loc[0, "group_id"] == out.loc[1, "group_id"] == "recording:reca"
    assert df.to_dict(orient="list") == before
    assert_prefix_recording_group_invariant(out)


def test_prefix_invariant_raises_on_split_group_ids():
    df = pd.DataFrame(
        [
            {
                "recording_group_id": "reca",
                "group_id": "recording:yt_1:reca",
                "group_source": "record_id_prefix",
                "source_label": "yt_1",
            },
            {
                "recording_group_id": "reca",
                "group_id": "recording:yt_2:reca",
                "group_source": "record_id_prefix",
                "source_label": "yt_2",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="recording_group_id|group_id"):
        assert_prefix_recording_group_invariant(df)


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
                        "recording_group_id": f"{src}-g{g}",
                        "duration_seconds": 1.0,
                    }
                )
    df = pd.DataFrame(rows)
    a = select_best_group_split(df, seed=42, n_candidates=200, test_size=0.1)
    b = select_best_group_split(df, seed=42, n_candidates=200, test_size=0.1)
    assert np.array_equal(a["train_idx"], b["train_idx"])
    assert np.array_equal(a["val_idx"], b["val_idx"])
    assert set(df.iloc[a["train_idx"]]["group_id"]) & set(df.iloc[a["val_idx"]]["group_id"]) == set()
    assert (
        set(df.iloc[a["train_idx"]]["recording_group_id"])
        & set(df.iloc[a["val_idx"]]["recording_group_id"])
        == set()
    )
    assert 0.08 <= a["val_fraction"] <= 0.12
    assert "score" in a


def test_select_best_group_split_no_recording_group_leak_with_shared_prefix():
    """Same recording_group_id under different source_labels must stay in one split."""
    rows = []
    # One shared recording with two source labels (collision case)
    for k, src in enumerate(["yt_1", "yt_2"]):
        rows.append(
            {
                "record_id": f"shared_{k}",
                "source_label": src,
                "recording_group_id": "shared_rec",
                "group_id": "recording:shared_rec",
                "group_source": "record_id_prefix",
                "duration_seconds": 1.0,
            }
        )
    # Filler groups so 8–12% band is achievable
    for g in range(20):
        for k in range(10):
            rows.append(
                {
                    "record_id": f"fill_{g}_{k}",
                    "source_label": "fill",
                    "recording_group_id": f"fill-{g}",
                    "group_id": f"recording:fill-{g}",
                    "group_source": "record_id_prefix",
                    "duration_seconds": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    assert_prefix_recording_group_invariant(df)
    result = select_best_group_split(df, seed=42, n_candidates=200, test_size=0.1)
    train = df.iloc[result["train_idx"]]
    val = df.iloc[result["val_idx"]]
    assert set(train["recording_group_id"]) & set(val["recording_group_id"]) == set()
    # Both shared rows must be in the same split
    shared_splits = set()
    if (train["recording_group_id"] == "shared_rec").any():
        shared_splits.add("train")
    if (val["recording_group_id"] == "shared_rec").any():
        shared_splits.add("val")
    assert len(shared_splits) == 1
    assert (df.loc[df["recording_group_id"] == "shared_rec", "group_id"].nunique() == 1)


def test_select_best_group_split_raises_outside_band():
    rows = [{"record_id": f"a_{i}", "source_label": "only", "group_id": "huge", "duration_seconds": 1.0} for i in range(90)]
    rows += [{"record_id": f"b_{i}", "source_label": "only", "group_id": f"tiny{i}", "duration_seconds": 1.0} for i in range(10)]
    df = pd.DataFrame(rows)
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
