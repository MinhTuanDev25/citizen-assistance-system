"""Offline synthetic tests for the RQ2 U_clean pipeline (NB11).

None of these tests require the optional webrtcvad or fpcalc binaries: the VAD
speech mask, frame energies and the acoustic fingerprints are injected
synthetically. Coverage:

* segmentation contract / UID determinism (energy_fallback in the hash);
* planning: single region, long-region split (no overlap), short/low-speech
  drops, lowest-energy fallback, silence-preferred boundary, short-tail drop;
* fingerprint similarity + offset alignment + inverted-index retrieval;
* build + segment QA (dbfs, SILENT exclusion);
* exact dedup owner rule + canonical_segment_uid;
* scalable perceptual dedup + protected-overlap exclusion (+ match evidence);
* fail-closed validation (retained overlap, temporal overlap, missing pcm);
* input contract validation (missing hash, duplicate id, ineligible);
* protected reference: audio identity only, 3-split gate, durable fail-closed;
* the fail-closed success gate (FAIL without protection evidence; SUCCESS only
  with complete evidence);
* retained-only manifest, atomic WAV IO round-trip, stale-checkpoint fail-closed.
"""
from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.rq2_audio_fingerprint import (
    FingerprintIndex,
    SegmentCandidateIndex,
    MatchEvidence,
    OverlapConfig,
    compare_fingerprints,
    compare_fingerprints_detailed,
    fingerprint_shingles,
    is_perceptual_match,
    overlap_contract_sha256,
    pcm16_sha256,
)
from src.rq2_segmentation import (
    DROP_LOW_SPEECH_FRACTION,
    DROP_TOO_SHORT,
    Segment,
    SegmentPlan,
    SegmentationConfig,
    _choose_cut,
    assert_no_overlap,
    compute_frame_energies,
    plan_segments,
    segment_uid,
    segmentation_contract_sha256,
)
from src.rq2_u_clean import (
    EXCLUDED_EXACT_DUPLICATE,
    EXCLUDED_PERCEPTUAL_DUPLICATE,
    EXCLUDED_PROTECTED_OVERLAP,
    EXCLUDED_SILENT,
    EXCLUDED_TOO_SHORT,
    FAIL_STATUS,
    RETAINED_STATUS,
    SUCCESS_STATUS,
    U_CLEAN_MANIFEST_COLUMNS,
    DurableCanonicalReferenceResolver,
    ManifestProtectedReferenceResolver,
    PipelineCompletion,
    ProtectedReferenceEntry,
    ReferenceColumnMapping,
    SyntheticProtectedReferenceResolver,
    UCleanContext,
    assert_checkpoint_compatible,
    assert_no_forbidden_reference_columns,
    bind_protected_audio_identity,
    build_config,
    build_durable_protected_resolver,
    default_protected_manifest_mapping,
    load_protected_audio_identity_index,
    probe_protected_audio_identity,
    require_protected_source_sha256,
    build_segments_for_source,
    build_summary,
    compatibility_key,
    exact_deduplicate,
    exclusion_rows,
    load_reference_identity_frame,
    append_fingerprint_shard,
    load_resumable_checkpoint,
    mark_source_segmentation_complete,
    save_incremental_checkpoint,
    protect_and_deduplicate,
    read_wav_pcm16,
    resolve_protected_references,
    retained_segments,
    scan_nb11_safety,
    u_clean_rows,
    validate_source_contract,
    validate_u_clean,
    verify_protected_reference_audio,
    write_and_verify_segments,
    write_fingerprint_tables,
    write_u_clean_artifacts,
    write_wav_pcm16_atomic,
    EXCLUDED_SEGMENT_WRITE_FAILED,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _config(**seg_over):
    seg = SegmentationConfig(**seg_over)
    ov = OverlapConfig()
    return build_config(project_root=Path.cwd(), segmentation=seg, overlap=ov)


def _speech_mask(n_frames, speech=True):
    return np.full(int(n_frames), bool(speech), dtype=bool)


def _uniform_energy(n_frames, value=0.5):
    return np.full(int(n_frames), float(value), dtype=np.float64)


def _plan(flags, cfg, energies=None):
    flags = np.asarray(flags, dtype=bool)
    if energies is None:
        energies = _uniform_energy(len(flags))
    n_samples = len(flags) * cfg.frame_len_samples
    return plan_segments(flags, energies, n_samples, cfg)


# --------------------------------------------------------------------------- #
# Contract + UID determinism                                                   #
# --------------------------------------------------------------------------- #
def test_segmentation_contract_sha_deterministic_and_sensitive():
    a = SegmentationConfig()
    b = SegmentationConfig()
    assert segmentation_contract_sha256(a) == segmentation_contract_sha256(b)
    c = SegmentationConfig(min_speech_fraction=0.5)
    assert segmentation_contract_sha256(a) != segmentation_contract_sha256(c)
    # energy_fallback IS part of the contract hash
    e = SegmentationConfig(energy_fallback=False)
    assert segmentation_contract_sha256(a) != segmentation_contract_sha256(e)
    # frozen flag is NOT part of the contract hash
    d = SegmentationConfig(frozen=True)
    assert segmentation_contract_sha256(a) == segmentation_contract_sha256(d)


def test_segment_uid_deterministic_and_sensitive():
    sha = segmentation_contract_sha256(SegmentationConfig())
    u1 = segment_uid("VOV4_X", "a" * 64, 0, 48000, sha)
    assert u1 == segment_uid("VOV4_X", "a" * 64, 0, 48000, sha)
    assert u1.startswith("SEG_")
    assert segment_uid("VOV4_X", "a" * 64, 0, 48001, sha) != u1
    assert segment_uid("VOV4_Y", "a" * 64, 0, 48000, sha) != u1
    other = segmentation_contract_sha256(SegmentationConfig(vad_mode=3))
    assert segment_uid("VOV4_X", "a" * 64, 0, 48000, other) != u1


# --------------------------------------------------------------------------- #
# Segmentation planning                                                        #
# --------------------------------------------------------------------------- #
def test_plan_segments_single_region_no_split():
    cfg = SegmentationConfig()
    n_frames = cfg.seconds_to_frames(10.0)
    plan = _plan(_speech_mask(n_frames), cfg)
    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.start_sample == 0
    assert abs(seg.duration_seconds(cfg.sample_rate) - 10.0) < 0.05


def test_plan_segments_long_region_splits_without_overlap():
    cfg = SegmentationConfig()
    n_frames = cfg.seconds_to_frames(100.0)
    plan = _plan(_speech_mask(n_frames), cfg)
    assert len(plan.segments) >= 3
    assert_no_overlap(plan.segments)
    for seg in plan.segments:
        assert seg.duration_seconds(cfg.sample_rate) <= cfg.max_segment_seconds + 1e-9
    for prev, nxt in zip(plan.segments, plan.segments[1:]):
        assert nxt.start_sample >= prev.end_sample


def test_plan_segments_drops_short_region():
    cfg = SegmentationConfig()
    n_frames = cfg.seconds_to_frames(1.0)  # below min 3s
    plan = _plan(_speech_mask(n_frames), cfg)
    assert plan.segments == []
    assert len(plan.drops) == 1
    assert plan.drops[0].reason == DROP_TOO_SHORT


def test_plan_segments_speech_fraction_filter():
    cfg = SegmentationConfig(merge_silence_gap_seconds=5.0, min_speech_fraction=0.6)
    fps = cfg.seconds_to_frames
    flags = np.concatenate([
        _speech_mask(fps(5.0), True),
        _speech_mask(fps(4.0), False),
        _speech_mask(fps(1.0), True),
    ])
    plan = _plan(flags, cfg)
    assert len(plan.segments) == 1
    assert plan.segments[0].vad_speech_fraction >= 0.6 - 1e-6

    strict = SegmentationConfig(merge_silence_gap_seconds=5.0, min_speech_fraction=0.7)
    plan2 = _plan(flags, strict)
    assert plan2.segments == []
    assert any(d.reason == DROP_LOW_SPEECH_FRACTION for d in plan2.drops)


def test_choose_cut_prefers_lowest_energy_when_no_silence():
    cfg = SegmentationConfig()  # target 25s, search 2s
    n = cfg.seconds_to_frames(60.0)
    flags = _speech_mask(n)  # all speech, no silence boundary
    energies = _uniform_energy(n, 0.9)
    low = cfg.seconds_to_frames(24.0)
    energies[low] = 0.0  # deep minimum inside the search window
    cut = _choose_cut(0, n, flags, energies, cfg)
    assert cut == low
    # disabling the fallback -> hard cut at the target
    cfg_no = replace(cfg, energy_fallback=False)
    assert _choose_cut(0, n, flags, energies, cfg_no) == cfg.seconds_to_frames(25.0)


def test_choose_cut_prefers_silence_over_energy():
    cfg = SegmentationConfig()
    n = cfg.seconds_to_frames(60.0)
    flags = _speech_mask(n)
    silence = cfg.seconds_to_frames(25.0)
    flags[silence] = False
    energies = _uniform_energy(n, 0.9)
    energies[cfg.seconds_to_frames(24.0)] = 0.0  # a lower-energy frame nearby
    assert _choose_cut(0, n, flags, energies, cfg) == silence


def test_plan_segments_short_tail_dropped():
    cfg = SegmentationConfig(target_segment_seconds=37.0, max_segment_seconds=40.0, min_segment_seconds=5.0)
    n = cfg.seconds_to_frames(41.0)
    flags = _speech_mask(n)
    energies = _uniform_energy(n, 0.9)
    energies[cfg.seconds_to_frames(37.0)] = 0.0
    plan = _plan(flags, cfg, energies)
    assert len(plan.segments) == 1
    assert abs(plan.segments[0].duration_seconds(cfg.sample_rate) - 37.0) < 0.1
    assert any(d.reason == DROP_TOO_SHORT for d in plan.drops)


# --------------------------------------------------------------------------- #
# Fingerprint similarity (pure)                                                #
# --------------------------------------------------------------------------- #
def test_fingerprint_identical_and_different():
    cfg = OverlapConfig(min_overlap_items=4)
    rng = np.random.default_rng(0)
    fp = rng.integers(0, 2 ** 32, size=100, dtype=np.uint64).astype("<u4").tolist()
    assert compare_fingerprints(fp, fp, cfg) == 1.0
    other = rng.integers(0, 2 ** 32, size=100, dtype=np.uint64).astype("<u4").tolist()
    assert compare_fingerprints(fp, other, cfg) < 0.75


def test_fingerprint_offset_alignment_partial_overlap():
    cfg = OverlapConfig(min_overlap_items=10)
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype("<u4").tolist()
    trimmed = base[30:]
    score, offset, overlap = compare_fingerprints_detailed(base, trimmed, cfg)
    assert score == 1.0
    assert overlap == len(trimmed)
    assert offset == -30


def test_inverted_index_retrieves_one_bit_perturbation():
    cfg = OverlapConfig(shingle_k=4, min_shared_shingles=1, min_overlap_items=4, similarity_threshold=0.90)
    rng = np.random.default_rng(3)
    original = rng.integers(0, 2 ** 32, size=40, dtype=np.uint64).astype("<u4").tolist()
    flipped = [int(item) ^ 1 for item in original]
    index = FingerprintIndex(cfg)
    index.add("A", original)
    assert "A" in index.candidates(flipped)
    score, _offset, _overlap = compare_fingerprints_detailed(original, flipped, cfg)
    assert score >= 0.90
    cfg = OverlapConfig(shingle_k=4, min_shared_shingles=1, min_overlap_items=4)
    rng = np.random.default_rng(2)
    fp = rng.integers(0, 2 ** 32, size=60, dtype=np.uint64).astype("<u4").tolist()
    other = rng.integers(0, 2 ** 32, size=60, dtype=np.uint64).astype("<u4").tolist()
    index = FingerprintIndex(cfg)
    index.add("A", fp)
    index.add("B", other)
    assert "A" in index.candidates(fp[10:])
    assert fingerprint_shingles(fp, 4)


def test_is_perceptual_match_requires_frozen():
    cfg = OverlapConfig(frozen=False)
    with pytest.raises(RuntimeError):
        is_perceptual_match([1, 2, 3], [1, 2, 3], cfg)
    frozen = OverlapConfig(frozen=True, min_overlap_items=1, similarity_threshold=0.9)
    assert is_perceptual_match([1, 2, 3], [1, 2, 3], frozen) is True


# --------------------------------------------------------------------------- #
# Segment building + QA                                                        #
# --------------------------------------------------------------------------- #
def _source_row(source_id="VOV4_A", **over):
    row = {
        "source_id": source_id,
        "article_url": f"https://vov4.vov.vn/bahnar/x/{source_id}.vov4",
        "article_date": "2026-02-01",
        "source_section": "kotong-ang-lom-topol-thoi-su-xa-hoi",
        "content_class": "news",
        "phase": "A",
        "wav_local_path": f"artifacts/rq2/vov4_full/wav16k/{source_id}.wav",
        "wav_sha256": "0" * 64,
        "pcm16_sha256": "0" * 64,
        "media_url_sha256": "f" * 64,
        "duration_seconds": "600.0",
        "source_level_status": "ELIGIBLE_SOURCE",
        "technical_status": "PASS",
    }
    row.update(over)
    return row


def _sine_pcm(seconds, freq=220.0, sr=16000, amp=0.3):
    t = np.arange(int(seconds * sr)) / float(sr)
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2")


def test_build_segments_for_source_fields():
    cfg = _config()
    pcm = _sine_pcm(10.0)
    frames = len(pcm) // cfg.segmentation.frame_len_samples
    rows = build_segments_for_source(_source_row(), pcm, _speech_mask(frames), cfg)
    retained = retained_segments(rows)
    assert len(retained) == 1
    r = retained[0]
    assert r["segment_uid"].startswith("SEG_")
    assert r["canonical_segment_uid"] == r["segment_uid"]
    assert r["source_group_id"] == r["source_id"]
    assert r["media_url_sha256"] == "f" * 64
    assert r["rms_dbfs"] <= 0.0 and r["peak_dbfs"] <= 0.0
    assert abs(r["duration_seconds"] - (r["end_sample"] - r["start_sample"]) / 16000.0) < 1e-6
    assert set(U_CLEAN_MANIFEST_COLUMNS).issubset(set(r.keys()))


def test_build_segments_excludes_silent():
    cfg = _config()
    pcm = np.zeros(int(10.0 * 16000), dtype="<i2")  # pure silence
    frames = len(pcm) // cfg.segmentation.frame_len_samples
    rows = build_segments_for_source(_source_row(), pcm, _speech_mask(frames), cfg)
    assert rows and all(r["u_clean_status"] == EXCLUDED_SILENT for r in rows if r["start_sample"] == 0 or True)
    assert any(r["u_clean_status"] == EXCLUDED_SILENT for r in rows)


def test_exact_dedup_owner_rule_and_canonical_uid():
    cfg = _config()
    pcm = _sine_pcm(6.0)
    frames = len(pcm) // cfg.segmentation.frame_len_samples
    a = build_segments_for_source(_source_row("VOV4_A"), pcm, _speech_mask(frames), cfg)
    b = build_segments_for_source(_source_row("VOV4_B"), pcm, _speech_mask(frames), cfg)
    segments = a + b
    assert a[0]["segment_pcm16_sha256"] == b[0]["segment_pcm16_sha256"]
    exact_deduplicate(segments)
    retained = retained_segments(segments)
    assert len(retained) == 1
    # owner = smallest (source_id, start_sample, segment_uid) -> VOV4_A
    assert retained[0]["source_id"] == "VOV4_A"
    owner_uid = retained[0]["segment_uid"]
    assert all(r["canonical_segment_uid"] == owner_uid for r in segments)
    loser = [r for r in segments if r["u_clean_status"] == EXCLUDED_EXACT_DUPLICATE]
    assert len(loser) == 1 and loser[0]["exact_duplicate"] is True


# --------------------------------------------------------------------------- #
# Protected reference: audio identity only                                     #
# --------------------------------------------------------------------------- #
def _write_reference_csv(path):
    import pandas as pd

    pd.DataFrame([{
        "record_uid": "R1",
        "audio_path": "audio/R1.flac",
        "duration_seconds": 8.4,
        "group_id": "recording:KT01",
        "text_bahnar": "SECRET BAHNAR",
        "text_vi": "SECRET VI",
        "text_en": "SECRET EN",
    }]).to_csv(path, index=False)


def _mapping():
    return ReferenceColumnMapping(
        reference_uid="record_uid", audio_locator="audio_path",
        duration_seconds="duration_seconds", group_id="group_id",
    )


def test_reference_loader_projects_identity_only(tmp_path):
    csv = tmp_path / "rq1_like.csv"
    _write_reference_csv(csv)
    frame = load_reference_identity_frame(csv, _mapping())
    cols = set(frame.columns)
    assert cols == {"record_uid", "audio_path", "duration_seconds", "group_id"}
    for forbidden in ("text_bahnar", "text_vi", "text_en"):
        assert forbidden not in cols


def test_reference_loader_rejects_forbidden_mapping():
    with pytest.raises(RuntimeError):
        assert_no_forbidden_reference_columns(["record_uid", "text_bahnar"])
    with pytest.raises(RuntimeError):
        assert_no_forbidden_reference_columns(["prediction_bleu"])


def test_manifest_resolver_three_splits(tmp_path):
    paths = {}
    for split in ("g_train", "g_validation", "frozen_test"):
        p = tmp_path / f"{split}.csv"
        _write_reference_csv(p)
        paths[split] = p
    resolver = ManifestProtectedReferenceResolver(paths, _mapping())
    entries = resolver.resolve()
    assert {e.split for e in entries} == {"g_train", "g_validation", "frozen_test"}


def test_durable_resolver_fails_closed_without_chain():
    resolver = DurableCanonicalReferenceResolver(
        durable_root=None, contract_hash=None, reference_index=None, mapping=None,
    )
    with pytest.raises(RuntimeError):
        resolver.resolve()


def test_resolve_references_requires_all_three_splits():
    cfg = _config()
    cfg.protected_reference_resolver = SyntheticProtectedReferenceResolver([
        ProtectedReferenceEntry("R1", "g_train", "a/R1.flac"),
        ProtectedReferenceEntry("R2", "g_validation", "a/R2.flac"),
    ])
    ctx = UCleanContext(config=cfg)
    with pytest.raises(RuntimeError):
        resolve_protected_references(cfg, ctx)


# --------------------------------------------------------------------------- #
# Input contract validation                                                    #
# --------------------------------------------------------------------------- #
def test_validate_source_contract_missing_hash():
    row = _source_row()
    row["pcm16_sha256"] = ""
    with pytest.raises(RuntimeError):
        validate_source_contract([row])


def test_validate_source_contract_duplicate_id():
    with pytest.raises(RuntimeError):
        validate_source_contract([_source_row("VOV4_A"), _source_row("VOV4_A")])


def test_validate_source_contract_rejects_ineligible():
    with pytest.raises(RuntimeError):
        validate_source_contract([_source_row(source_level_status="TOO_SHORT_SOURCE")])


def test_validate_source_contract_passes_good_rows():
    validate_source_contract([_source_row("VOV4_A", wav_sha256="a" * 64, pcm16_sha256="b" * 64)])


# --------------------------------------------------------------------------- #
# Perceptual dedup + protected overlap + fail-closed validation               #
# --------------------------------------------------------------------------- #
def _frozen_overlap_config():
    return OverlapConfig(frozen=True, shingle_k=4, min_shared_shingles=1, min_overlap_items=4, similarity_threshold=0.90)


def _distinct_segments():
    cfg = _config()
    cfg.overlap = _frozen_overlap_config()
    cfg.segmentation = replace(cfg.segmentation, frozen=True)
    segs = []
    for sid, freq in (("VOV4_A", 200.0), ("VOV4_B", 440.0)):
        pcm = _sine_pcm(6.0, freq=freq)
        frames = len(pcm) // cfg.segmentation.frame_len_samples
        segs += build_segments_for_source(_source_row(sid), pcm, _speech_mask(frames), cfg)
    return cfg, segs


def _rand_fp(seed, size=80):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 32, size=size, dtype=np.uint64).astype("<u4").tolist()


def test_perceptual_dedup_records_evidence():
    cfg, segs = _distinct_segments()
    ctx = UCleanContext(config=cfg)
    fp_a = _rand_fp(7)
    uids = [s["segment_uid"] for s in segs]
    seg_fps = {uids[0]: fp_a, uids[1]: fp_a}  # perceptual near-duplicates
    protect_and_deduplicate(segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=[])
    assert len(retained_segments(segs)) == 1
    assert any(s["u_clean_status"] == EXCLUDED_PERCEPTUAL_DUPLICATE for s in segs)
    assert any(e.match_type == "u_u" for e in ctx.match_evidence)


def test_protected_overlap_excludes_not_fails():
    cfg, segs = _distinct_segments()
    ctx = UCleanContext(config=cfg)
    ctx.completion.references_resolved = True
    fp_a, fp_b = _rand_fp(11), _rand_fp(12)
    uids = [s["segment_uid"] for s in segs]
    seg_fps = {uids[0]: fp_a, uids[1]: fp_b}
    ref_fps = [{"uid": "REF_FT_1", "split": "frozen_test", "fingerprint": fp_a}]
    protect_and_deduplicate(segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=ref_fps)
    excluded = [s for s in segs if s["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP]
    assert len(excluded) == 1 and excluded[0]["overlap_frozen_test"] is True
    assert any(e.match_type == "protected" and e.reference_split == "frozen_test" for e in ctx.match_evidence)
    validate_u_clean(segs, ctx)


def test_validate_fails_closed_on_retained_overlap():
    cfg, segs = _distinct_segments()
    ctx = UCleanContext(config=cfg)
    segs[0]["overlap_frozen_test"] = True
    with pytest.raises(RuntimeError):
        validate_u_clean(segs, ctx)


def test_validate_fails_closed_on_temporal_overlap():
    cfg, segs = _distinct_segments()
    for s in segs:
        s["source_id"] = "VOV4_SAME"
    segs[0]["start_sample"], segs[0]["end_sample"] = 0, 16000 * 6
    segs[1]["start_sample"], segs[1]["end_sample"] = 16000 * 3, 16000 * 9
    ctx = UCleanContext(config=cfg)
    with pytest.raises(RuntimeError):
        validate_u_clean(segs, ctx)


# --------------------------------------------------------------------------- #
# Success gate (fail closed)                                                   #
# --------------------------------------------------------------------------- #
def _three_split_entries():
    return [
        ProtectedReferenceEntry("REF_TR", "g_train", "audio/tr.wav", source_sha256="a" * 64),
        ProtectedReferenceEntry("REF_VA", "g_validation", "audio/va.wav", source_sha256="b" * 64),
        ProtectedReferenceEntry("REF_TE", "frozen_test", "audio/te.wav", source_sha256="c" * 64),
    ]


def _full_success_context(tmp_path):
    cfg, segs = _distinct_segments()
    cfg.run_full_pipeline = True
    cfg.project_root = tmp_path
    cfg.out_dir = tmp_path / "u_clean"
    for row in segs:
        if row["u_clean_status"] != RETAINED_STATUS:
            continue
        wav_rel = row["source_wav_local_path"]
        wav_path = tmp_path / wav_rel
        pcm = _sine_pcm(6.0, freq=200.0 if "VOV4_A" in row["source_id"] else 440.0)
        from src.rq2_u_clean import write_wav_pcm16_atomic
        write_wav_pcm16_atomic(wav_path, pcm, 16000)
        row["source_wav_local_path"] = wav_rel
    ctx = UCleanContext(
        config=cfg,
        eligible_sources=[_source_row("VOV4_A"), _source_row("VOV4_B")],
        nb10_status="SUCCESS_RQ2_VOV4_FULL_SOURCE_POOL",
    )
    ctx.nb10_locks = {"summary.json": "a" * 64}
    ctx.protected_entries = _three_split_entries()
    ctx.completion.nb10_locked = True
    fp_a, fp_b = _rand_fp(21), _rand_fp(22)
    fp_tr, fp_va, fp_te = _rand_fp(23), _rand_fp(24), _rand_fp(25)
    uids = [s["segment_uid"] for s in segs]
    seg_fps = {uids[0]: fp_a, uids[1]: fp_b}
    ref_fps = [
        {"uid": "REF_TR", "split": "g_train", "fingerprint": fp_tr, "source_sha256": "a" * 64},
        {"uid": "REF_VA", "split": "g_validation", "fingerprint": fp_va, "source_sha256": "b" * 64},
        {"uid": "REF_TE", "split": "frozen_test", "fingerprint": fp_te, "source_sha256": "c" * 64},
    ]
    from src.rq2_u_clean import write_and_verify_segments
    write_and_verify_segments(segs, ctx)
    protect_and_deduplicate(segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=ref_fps)
    return cfg, segs, ctx, seg_fps, ref_fps


def test_success_gate_fails_without_full_run():
    cfg, segs = _distinct_segments()
    ctx = UCleanContext(config=cfg)
    summary = build_summary(segs, ctx)
    assert summary["status"] == FAIL_STATUS


def test_success_gate_fails_without_protection_evidence():
    # full run + frozen configs but NO protection completed -> must FAIL, not SUCCESS
    cfg, segs = _distinct_segments()
    cfg.run_full_pipeline = True
    ctx = UCleanContext(config=cfg, nb10_status="SUCCESS_RQ2_VOV4_FULL_SOURCE_POOL")
    ctx.completion.references_resolved = True
    ctx.completion.nb10_locked = True
    ctx.completion.segments_written_verified = True
    # note: protect_and_deduplicate NOT called -> protection_complete stays False
    summary = build_summary(segs, ctx)
    assert summary["status"] == FAIL_STATUS
    assert summary["gates"]["protection_complete"] is False


def test_success_gate_fails_without_reference_fingerprints():
    cfg, segs = _distinct_segments()
    cfg.run_full_pipeline = True
    ctx = UCleanContext(
        config=cfg,
        eligible_sources=[_source_row("VOV4_A"), _source_row("VOV4_B")],
        nb10_status="SUCCESS_RQ2_VOV4_FULL_SOURCE_POOL",
    )
    ctx.protected_entries = _three_split_entries()
    ctx.completion.nb10_locked = True
    ctx.completion.segments_written_verified = True
    ctx.completion.artifacts_reverified = True
    fp_a, fp_b = _rand_fp(41), _rand_fp(42)
    uids = [s["segment_uid"] for s in segs]
    protect_and_deduplicate(
        segs, ctx,
        segment_fingerprints_by_uid={uids[0]: fp_a, uids[1]: fp_b},
        reference_fingerprints=[],
    )
    summary = build_summary(segs, ctx)
    assert summary["status"] == FAIL_STATUS
    assert summary["gates"]["protected_overlap_check_complete"] is False
    assert summary["gates"]["reference_fingerprint_coverage_complete"] is False


def test_success_gate_fails_on_empty_pool():
    cfg, _segs = _distinct_segments()
    cfg.run_full_pipeline = True
    ctx = UCleanContext(config=cfg, eligible_sources=[_source_row()])
    ctx.completion.references_resolved = True
    ctx.completion.reference_fingerprint_coverage_complete = True
    ctx.completion.segment_fingerprint_coverage_complete = True
    ctx.completion.u_u_protection_complete = True
    ctx.completion.protected_overlap_check_complete = True
    ctx.completion.nb10_locked = True
    ctx.completion.segments_written_verified = True
    ctx.completion.artifacts_reverified = True
    summary = build_summary([], ctx)
    assert summary["status"] == FAIL_STATUS
    assert summary["n_u_clean"] == 0


def test_success_gate_succeeds_with_complete_evidence(tmp_path):
    _cfg, segs, ctx, _seg_fps, _ref_fps = _full_success_context(tmp_path)
    summary = build_summary(segs, ctx)
    assert summary["gates"]["protection_complete"] is True
    assert summary["gates"]["nonempty_u_clean"] is True
    assert summary["gates"]["artifacts_reverified"] is True
    assert summary["n_u_clean"] > 0
    assert summary["status"] == SUCCESS_STATUS
    for key in (
        "n_source_input", "source_input_hours", "n_segments_generated", "generated_segment_hours",
        "n_segment_qa_pass", "n_segment_qa_rejected", "n_exact_duplicates", "n_perceptual_duplicates_u",
        "n_overlap_g_train", "n_overlap_g_validation", "n_overlap_frozen_test", "n_u_clean", "u_clean_hours",
        "min_segment_seconds", "median_segment_seconds", "p90_segment_seconds", "max_segment_seconds",
        "source_count_u_clean", "per_source_segment_counts", "per_source_hours",
        "protected_reference_fingerprints", "generated_at_utc",
    ):
        assert key in summary


# --------------------------------------------------------------------------- #
# Manifest / exclusions                                                        #
# --------------------------------------------------------------------------- #
def test_manifest_is_retained_only():
    cfg, segs = _distinct_segments()
    ctx = UCleanContext(config=cfg)
    fp_a = _rand_fp(31)
    uids = [s["segment_uid"] for s in segs]
    protect_and_deduplicate(segs, ctx, segment_fingerprints_by_uid={uids[0]: fp_a, uids[1]: fp_a})
    rows = u_clean_rows(segs)
    assert len(rows) == len(retained_segments(segs)) == 1
    assert list(rows[0].keys()) == U_CLEAN_MANIFEST_COLUMNS
    excl = exclusion_rows(segs)
    assert any(r["u_clean_status"] == EXCLUDED_PERCEPTUAL_DUPLICATE for r in excl)


def test_manifest_has_no_transcript_columns():
    for col in U_CLEAN_MANIFEST_COLUMNS:
        assert "text_" not in col and "transcript" not in col and "bleu" not in col


# --------------------------------------------------------------------------- #
# Checkpoint compatibility                                                     #
# --------------------------------------------------------------------------- #
def test_checkpoint_stale_fails_closed():
    cfg, _ = _distinct_segments()
    ctx = UCleanContext(config=cfg)
    ctx.nb10_locks = {"summary.json": "a" * 64}
    key = compatibility_key(ctx)
    assert_checkpoint_compatible(key, dict(key))  # identical -> ok
    stale = dict(key)
    stale["nb10_locks"] = {"summary.json": "b" * 64}
    with pytest.raises(RuntimeError):
        assert_checkpoint_compatible(stale, key)


# --------------------------------------------------------------------------- #
# WAV IO round-trip                                                            #
# --------------------------------------------------------------------------- #
def test_read_wav_pcm16_roundtrip(tmp_path):
    pcm = _sine_pcm(1.0)
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())
    info = read_wav_pcm16(path)
    assert info["sample_rate"] == 16000 and info["channels"] == 1
    assert info["pcm16_sha256"] == pcm16_sha256(pcm)


def test_atomic_write_wav_roundtrip(tmp_path):
    pcm = _sine_pcm(1.0)
    path = tmp_path / "seg" / "s.wav"
    wav_sha = write_wav_pcm16_atomic(path, pcm, 16000)
    info = read_wav_pcm16(path)
    assert info["pcm16_sha256"] == pcm16_sha256(pcm)
    assert info["wav_sha256"] == wav_sha


def test_read_wav_pcm16_rejects_non_16k(tmp_path):
    pcm = _sine_pcm(1.0, sr=8000)
    path = tmp_path / "b.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(pcm.tobytes())
    with pytest.raises(RuntimeError):
        read_wav_pcm16(path)


def test_durable_resolver_verifies_canonical_hash(tmp_path):
    import json

    from src.rq1_contract import build_final_contract, sha256_file, sha256_json
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths

    project = tmp_path / "proj"
    durable = tmp_path / "durable"
    names = {
        "g_train": "train_identity.csv",
        "g_validation": "validation_identity.csv",
        "frozen_test": "frozen_identity.csv",
    }
    shas = {}
    index = {}
    for split, name in names.items():
        path = durable / "protected_manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "record_uid,audio_path,duration_seconds,group_id\n"
            + split + "_1,audio/" + split + ".wav,1.0,g\n",
            encoding="utf-8",
        )
        shas[split] = sha256_file(path)
        index[split] = path
    identity = durable / "protected_manifests" / "audio_identity.csv"
    identity_rows = []
    for split, name in names.items():
        identity_rows.append(
            "%s,%s,audio/%s.wav,%s" % (split + "_1", split, split, hashlib.sha256(split.encode()).hexdigest())
        )
    identity.write_text(
        "reference_uid,split,audio_locator,source_sha256\n" + "\n".join(identity_rows) + "\n",
        encoding="utf-8",
    )
    identity_sha = sha256_file(identity)
    test_body = {
        "contract_version": "rq1_final_contract_v1",
        "test_count": 1,
        "manifest_sha256": shas["frozen_test"],
        "train_manifest_sha256": shas["g_train"],
        "validation_manifest_sha256": shas["g_validation"],
    }
    test_body["rq1_test_contract_hash"] = sha256_json(dict(test_body))
    final = build_final_contract(
        test_contract={"rq1_test_contract_hash": test_body["rq1_test_contract_hash"]},
        asr_handoff_hash="a" * 64,
        mt_handoff_hash="b" * 64,
        direct_handoff_hash="c" * 64,
        source_fingerprint_sha256="d" * 64,
        runtime_versions={"python": "3.9"},
        seed=1,
        bootstrap_samples=10,
    )
    runtime = resolve_rq1_runtime_paths(project_root=project, durable_root=durable)
    state = runtime.state_dir(final["rq1_final_contract_hash"])
    state.mkdir(parents=True)
    (state / "rq1_final_contract.json").write_text(json.dumps(final), encoding="utf-8")
    (state / "rq1_test_contract.json").write_text(json.dumps(test_body), encoding="utf-8")
    missing_index = DurableCanonicalReferenceResolver(
        durable_root=durable,
        project_root=project,
        contract_hash=final["rq1_final_contract_hash"],
        reference_index=None,
        mapping=_mapping(),
    )
    with pytest.raises(RuntimeError):
        missing_index.resolve()
    resolver = DurableCanonicalReferenceResolver(
        durable_root=durable,
        project_root=project,
        contract_hash=final["rq1_final_contract_hash"],
        reference_index=index,
        mapping=_mapping(),
        audio_identity_index=identity,
        audio_identity_sha256=identity_sha,
    )
    entries = resolver.resolve()
    assert {entry.split for entry in entries} == set(names)
    assert all(len(entry.source_sha256) == 64 for entry in entries)
    provenance = resolver.provenance()
    assert "durable_root" not in json.dumps(provenance)
    assert provenance["rq1_final_contract_hash"] == final["rq1_final_contract_hash"]
    assert provenance["splits"]["g_train"]["manifest_root"] == "durable"
    assert provenance["splits"]["g_train"]["manifest_relative_path"] == "protected_manifests/train_identity.csv"
    assert provenance["splits"]["g_train"]["manifest_sha256"] == shas["g_train"]
    assert provenance["splits"]["frozen_test"]["n_rows"] == 1
    (runtime.durable_state_root / "LATEST_UNLOCKED.json").write_text(
        json.dumps({
            "state_dir": str(tmp_path / "not-the-real-state"),
            "final_contract_hash": final["rq1_final_contract_hash"],
        }),
        encoding="utf-8",
    )
    built = build_durable_protected_resolver(
        project_root=project,
        reference_index=index,
        mapping=_mapping(),
        durable_root=durable,
        audio_identity_index=identity,
        audio_identity_sha256=identity_sha,
    )
    assert {entry.split for entry in built.resolve()} == set(names)
    assert built.provenance()["rq1_final_contract_hash"] == final["rq1_final_contract_hash"]
    tampered = dict(test_body)
    tampered["test_count"] = 9
    (state / "rq1_test_contract.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError):
        resolver.resolve()


def test_parquet_manifest_projects_identity_columns_only(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame([{
        "record_uid": "R1",
        "audio_path": "audio/r1.wav",
        "duration_seconds": 1.0,
        "group_id": "g",
        "text_bahnar": "do-not-load",
    }])
    path = tmp_path / "refs.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_reference_identity_frame(path, _mapping())
    assert "text_bahnar" not in set(loaded.columns)
    assert loaded.iloc[0]["record_uid"] == "R1"


def test_segment_write_failure_excludes_row_only(tmp_path, monkeypatch):
    cfg, segs = _distinct_segments()
    cfg.project_root = tmp_path
    cfg.out_dir = tmp_path / "out"
    ctx = UCleanContext(config=cfg)
    calls = {"n": 0}
    real = write_wav_pcm16_atomic

    def _flaky(path, pcm, sample_rate=16000):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk")
        return real(path, pcm, sample_rate)

    for row in segs:
        wav_rel = row["source_wav_local_path"]
        freq = 200.0 if row["source_id"] == "VOV4_A" else 440.0
        write_wav_pcm16_atomic(tmp_path / wav_rel, _sine_pcm(6.0, freq=freq), 16000)
    monkeypatch.setattr("src.rq2_u_clean.write_wav_pcm16_atomic", _flaky)
    write_and_verify_segments(segs, ctx)
    assert any(row["u_clean_status"] == EXCLUDED_SEGMENT_WRITE_FAILED for row in segs)
    assert ctx.completion.segments_written_verified is True
    assert retained_segments(segs)


def test_fingerprint_checkpoint_roundtrip_and_corrupt_cache(tmp_path):
    pytest.importorskip("pyarrow")
    cfg, segs, ctx, seg_fps, ref_fps = _full_success_context(tmp_path)
    ctx.reference_summary = {"reference_set_sha256": "e" * 64}
    written = write_u_clean_artifacts(
        segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=ref_fps,
    )
    assert written["written"] is True
    assert (tmp_path / "u_clean" / "fingerprints" / "u_segments.parquet").is_file()
    assert (tmp_path / "u_clean" / "fingerprints" / "protected_reference.parquet").is_file()
    loaded = load_resumable_checkpoint(ctx)
    assert set(loaded["segment_fingerprints_by_uid"]) == set(seg_fps)
    shard = next((tmp_path / "u_clean" / "checkpoint" / "segment_fingerprints").glob("part-*.parquet"))
    digest = shard.read_bytes()
    shard.write_bytes(bytes([digest[0] ^ 0xFF]) + digest[1:])
    reloaded = load_resumable_checkpoint(ctx)
    assert reloaded["segment_fingerprints_by_uid"] == {}
    assert reloaded["discarded_segment_fingerprints"] >= 1


def test_fingerprint_shards_append_without_rewriting(tmp_path):
    pytest.importorskip("pyarrow")
    cfg, _segs = _distinct_segments()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg)
    first = {"uid": "REF_A", "split": "g_train", "source_sha256": "a" * 64, "fingerprint": _rand_fp(4)}
    second = {"uid": "REF_B", "split": "g_validation", "source_sha256": "b" * 64, "fingerprint": _rand_fp(5)}
    append_fingerprint_shard(ctx, "protected_fingerprints", [first])
    part1 = tmp_path / "out" / "checkpoint" / "protected_fingerprints" / "part-000001.parquet"
    digest = part1.read_bytes()
    append_fingerprint_shard(ctx, "protected_fingerprints", [first, second])
    assert part1.read_bytes() == digest
    assert (tmp_path / "out" / "checkpoint" / "protected_fingerprints" / "part-000002.parquet").is_file()
    part1.write_bytes(bytes([digest[0] ^ 0xFF]) + digest[1:])
    loaded = load_resumable_checkpoint(ctx)
    assert {item["uid"] for item in loaded["reference_fingerprints"]} == {"REF_B"}
    assert "part-000001.parquet" in loaded["discarded_reference_shards"]


def test_protected_parquet_compacts_verified_shards(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    _cfg, segs, ctx, seg_fps, ref_fps = _full_success_context(tmp_path)
    append_fingerprint_shard(ctx, "protected_fingerprints", ref_fps[:1])
    append_fingerprint_shard(ctx, "protected_fingerprints", ref_fps[1:])
    written = write_u_clean_artifacts(
        segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=None,
    )
    assert written["written"] is True
    frame = pd.read_parquet(tmp_path / "u_clean" / "fingerprints" / "protected_reference.parquet")
    assert set(frame["reference_uid"]) == {row["uid"] for row in ref_fps}
    assert set(frame["split"]) == {"g_train", "g_validation", "frozen_test"}


def test_zero_segment_source_stays_complete(tmp_path):
    cfg, segs = _distinct_segments()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg)
    mark_source_segmentation_complete(ctx, "EMPTY", [])
    mark_source_segmentation_complete(ctx, segs[0]["source_id"], [segs[0]])
    shard = next((tmp_path / "out" / "checkpoint" / "segments").glob("part-*.jsonl"))
    digest = shard.read_bytes()
    mark_source_segmentation_complete(ctx, "EMPTY", [])
    assert shard.read_bytes() == digest
    loaded = load_resumable_checkpoint(ctx)
    assert "EMPTY" in loaded["completed_source_ids"]
    assert loaded["source_status"]["EMPTY"]["n_segments"] == 0
    assert loaded["source_status"]["EMPTY"]["segmentation_status"] == "COMPLETE"


def test_stale_checkpoint_fails_closed(tmp_path):
    cfg, _segs = _distinct_segments()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg)
    mark_source_segmentation_complete(ctx, "EMPTY", [])
    ctx.nb10_locks = {"summary.json": "b" * 64}
    with pytest.raises(RuntimeError):
        load_resumable_checkpoint(ctx)


def test_u_index_matches_protected_index_candidates():
    cfg = OverlapConfig(shingle_k=4, min_shared_shingles=1, min_overlap_items=4, similarity_threshold=0.90)
    u_index = SegmentCandidateIndex(cfg)
    p_index = FingerprintIndex(cfg)
    u_fps = [_rand_fp(seed) for seed in range(6)]
    u_fps.append([7, 7, 7, 9, 9, 1, 2, 3, 4])
    p_fp = list(u_fps[1])
    p_fp[0] = (int(p_fp[0]) ^ 1) & 0xFFFFFFFF
    for index, fp in enumerate(u_fps):
        u_index.add("U%d" % index, fp)
    p_index.add("P", p_fp)
    expected = {"U%d" % index for index, fp in enumerate(u_fps) if "P" in p_index.candidates(fp)}
    assert set(u_index.candidates_for_reference(p_fp)) == expected


def test_failed_fingerprint_publish_does_not_leave_success(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    _cfg, segs, ctx, seg_fps, ref_fps = _full_success_context(tmp_path)
    summary_path = tmp_path / "u_clean" / "summary.json"
    summary_path.write_text('{"status": "OLD"}', encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr("pandas.DataFrame.to_parquet", _boom)
    with pytest.raises(ImportError):
        write_u_clean_artifacts(
            segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=ref_fps,
        )
    assert summary_path.read_text(encoding="utf-8") == '{"status": "OLD"}'
    assert not (tmp_path / "u_clean" / ".staging").exists()


def test_corrupt_cached_segment_is_recomputed(tmp_path):
    cfg, segs = _distinct_segments()
    cfg.project_root = tmp_path
    cfg.out_dir = tmp_path / "out"
    ctx = UCleanContext(config=cfg)
    for row in retained_segments(segs):
        freq = 200.0 if row["source_id"] == "VOV4_A" else 440.0
        write_wav_pcm16_atomic(tmp_path / row["source_wav_local_path"], _sine_pcm(6.0, freq=freq), 16000)
    row = [item for item in retained_segments(segs) if item["source_id"] == "VOV4_A"][0]
    row["segment_local_path"] = "missing/cached.wav"
    write_and_verify_segments(segs, ctx)
    assert row["u_clean_status"] == RETAINED_STATUS
    assert row["segment_local_path"] != "missing/cached.wav"
    assert (tmp_path / row["segment_local_path"]).is_file()


def test_missing_source_provenance_fails_closed():
    row = _source_row()
    row["article_url"] = ""
    with pytest.raises(RuntimeError):
        validate_source_contract([row])


def test_static_safety_scan_passes_current_sources():
    root = Path(__file__).resolve().parents[1]
    scan_nb11_safety([
        root / "src" / "rq2_segmentation.py",
        root / "src" / "rq2_audio_fingerprint.py",
        root / "src" / "rq2_u_clean.py",
        root / "notebooks" / "11_RQ2_UReal_Segmentation_Dedup_Freeze_UClean.ipynb",
    ])


def test_full_run_mapping_cannot_emit_empty_source_sha(tmp_path):
    mapping = default_protected_manifest_mapping()
    assert mapping.source_sha256 is None
    manifest = tmp_path / "refs.csv"
    manifest.write_text(
        "record_uid,audio_path,duration_seconds,group_id,text_bahnar\n"
        "R1,audio/r1.flac,1.0,g,do-not-load\n",
        encoding="utf-8",
    )
    entries = ManifestProtectedReferenceResolver(
        {"g_train": manifest, "g_validation": manifest, "frozen_test": manifest},
        mapping,
    ).resolve()
    assert entries[0].source_sha256 == ""
    with pytest.raises(RuntimeError):
        require_protected_source_sha256(entries)
    identity = tmp_path / "audio_identity.csv"
    lines = ["reference_uid,split,audio_locator,source_sha256"]
    for split in ("g_train", "g_validation", "frozen_test"):
        lines.append("R1,%s,audio/r1.flac,%s" % (split, hashlib.sha256(split.encode()).hexdigest()))
    identity.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from src.rq1_contract import sha256_file
    pin = sha256_file(identity)
    bound = bind_protected_audio_identity(entries, load_protected_audio_identity_index(identity, pin))
    require_protected_source_sha256(bound)
    assert all(entry.source_sha256 for entry in bound)
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "11_RQ2_UReal_Segmentation_Dedup_Freeze_UClean.ipynb"
    text = notebook.read_text(encoding="utf-8")
    assert "default_protected_manifest_mapping" in text
    assert "audio_identity_sha256" in text


def test_corrupt_segment_shard_reopens_source(tmp_path):
    cfg, segs = _distinct_segments()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg)
    mark_source_segmentation_complete(ctx, segs[0]["source_id"], [segs[0]])
    mark_source_segmentation_complete(ctx, "EMPTY", [])
    shard = next((tmp_path / "out" / "checkpoint" / "segments").glob("part-*.jsonl"))
    shard.write_bytes(b"not-jsonl")
    loaded = load_resumable_checkpoint(ctx)
    assert segs[0]["source_id"] not in loaded["completed_source_ids"]
    assert loaded["source_status"].get(segs[0]["source_id"], {}).get("segmentation_status") != "COMPLETE"
    assert "EMPTY" in loaded["completed_source_ids"]
    assert loaded["source_status"]["EMPTY"]["segmentation_status"] == "COMPLETE"
    assert loaded["source_status"]["EMPTY"]["n_segments"] == 0


def test_shard_ids_do_not_reuse_holes(tmp_path):
    pytest.importorskip("pyarrow")
    cfg, _segs = _distinct_segments()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg)
    directory = tmp_path / "out" / "checkpoint" / "protected_fingerprints"
    directory.mkdir(parents=True)
    existing = directory / "part-000002.parquet"
    existing.write_bytes(b"keep-me")
    append_fingerprint_shard(ctx, "protected_fingerprints", [{
        "uid": "REF_A", "split": "g_train", "source_sha256": "a" * 64, "fingerprint": _rand_fp(4),
    }])
    assert existing.read_bytes() == b"keep-me"
    assert (directory / "part-000003.parquet").is_file()
    assert not (directory / "part-000001.parquet").exists()


def test_mismatched_segment_audio_hash_drops_only_that_fingerprint(tmp_path):
    pytest.importorskip("pyarrow")
    cfg, segs = _distinct_segments()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg)
    row = dict(segs[0])
    row["segment_pcm16_sha256"] = "a" * 64
    other = dict(segs[1])
    other["segment_pcm16_sha256"] = "b" * 64
    mark_source_segmentation_complete(ctx, row["source_id"], [row])
    mark_source_segmentation_complete(ctx, other["source_id"], [other])
    append_fingerprint_shard(ctx, "segment_fingerprints", [
        {"segment_uid": row["segment_uid"], "fingerprint": _rand_fp(3), "audio_sha256": "c" * 64},
        {"segment_uid": other["segment_uid"], "fingerprint": _rand_fp(4), "audio_sha256": "b" * 64},
    ])
    loaded = load_resumable_checkpoint(ctx)
    assert row["segment_uid"] not in loaded["segment_fingerprints_by_uid"]
    assert other["segment_uid"] in loaded["segment_fingerprints_by_uid"]


def test_real_rq1_manifest_probe_is_identity_only():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "manifests" / "rq1_train.csv").is_file():
        pytest.skip("canonical RQ1 manifests are not in this checkout")
    paths = {
        "g_train": root / "data" / "manifests" / "rq1_train.csv",
        "g_validation": root / "data" / "manifests" / "rq1_validation.csv",
        "frozen_test": root / "data" / "manifests" / "rq1_test.csv",
    }
    entries = []
    for split, path in paths.items():
        header = path.open(encoding="utf-8").readline()
        assert "sha256" not in header.lower()
        import pandas as pd
        frame = pd.read_csv(path, usecols=["record_uid", "audio_path", "duration_seconds", "group_id"], nrows=1)
        locator = str(frame.iloc[0]["audio_path"])
        assert not Path(locator).is_absolute()
        assert "text_" not in locator
        entries.append(ProtectedReferenceEntry(
            str(frame.iloc[0]["record_uid"]),
            split,
            locator,
            group_id=str(frame.iloc[0]["group_id"]),
            duration_seconds=float(frame.iloc[0]["duration_seconds"] or 0),
        ))

    class _Root:
        def __init__(self, root_path):
            self.root_path = root_path

        def resolve_audio_path(self, entry):
            from src.rq2_u_clean import _durable_audio_path
            return _durable_audio_path(self.root_path, entry.audio_locator)

    report = probe_protected_audio_identity(entries, _Root(root))
    assert [item["split"] for item in report] == ["g_train", "g_validation", "frozen_test"]
    for item in report:
        assert set(item) == {
            "split", "reference_uid", "locator_resolvable", "expected_sha_available", "hash_matches",
        }
        assert item["expected_sha_available"] is False
        assert item["locator_resolvable"] is False
        assert item["hash_matches"] is False


def test_generation_publish_points_at_complete_snapshot(tmp_path):
    pytest.importorskip("pyarrow")
    _cfg, segs, ctx, seg_fps, ref_fps = _full_success_context(tmp_path)
    written = write_u_clean_artifacts(
        segs, ctx, segment_fingerprints_by_uid=seg_fps, reference_fingerprints=ref_fps,
    )
    assert written["written"] is True
    current = (tmp_path / "u_clean" / "CURRENT").read_text(encoding="utf-8").strip()
    complete = json.loads((tmp_path / "u_clean" / "generations" / current / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["files"][-1] == "summary.json"
    assert (tmp_path / "u_clean" / "summary.json").is_file()


# --------------------------------------------------------------------------- #
# Derived protected identity + on-demand Parquet materialization              #
# --------------------------------------------------------------------------- #
_REV = "3d88d3951b1a6e3388559b341cd7bd274879d696"
_OTHER_REV = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_DATASET_REV = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_PCM = "ab" * 32
_PIPELINE = "full_pcm16_le_v1"
_GOLDEN_UID = "0c29a2618a487f074ff215521cb95aaaff71435e"
_GOLDEN_PCM = "d73e4bf03f1b6603fee759ce2e0c699b1a5503ac7fcc6bee37bb5aa40e4ec9ba"


def _identity_manifest(path, rows):
    lines = ["record_uid,record_id,audio_path,parquet_file,shard_row_index,duration_seconds,group_id,text_vi"]
    for uid, record_id, audio, parquet_file, index in rows:
        lines.append(
            "%s,%s,%s,%s,%s,1.0,g,SECRET_TRANSCRIPT" % (uid, record_id, audio, parquet_file, index)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audio_index(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record) for record in records)
    (directory / "default_train_0000.jsonl").write_text(payload + "\n", encoding="utf-8")


def _sidecar(uid, **overrides):
    row = {
        "record_uid": uid,
        "sha256_pcm": _PCM,
        "sha256_source": "cd" * 32,
        "n_samples": 1600,
        "sample_rate": 16000,
        "audio_pcm_pipeline_version": _PIPELINE,
        "parquet_revision": _REV,
        "dataset_revision": _DATASET_REV,
        "split": "train",
        "shard_key": "default/train/0000.parquet",
        "shard_row_index": 1,
    }
    row.update(overrides)
    return row


def _build_identity(tmp_path, *, train_rows, index_records, frozen_rows, integrity_rows, counts):
    from src.rq2_u_clean import build_derived_protected_identity_rows

    root = tmp_path / "proj"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    _identity_manifest(manifests / "train.csv", train_rows)
    _identity_manifest(
        manifests / "validation.csv",
        [("VAL1", "val-1", "val.flac", "default/train/0000.parquet", 2)],
    )
    _identity_manifest(manifests / "test.csv", frozen_rows)
    index_dir = tmp_path / "audio_index"
    _write_audio_index(index_dir, index_records)
    integrity = tmp_path / "rq1_audio_integrity.csv"
    header = "record_uid,audio_path,sha256_pcm,sha256_source,n_samples,sample_rate,shard_key,shard_row_index"
    body = [
        "%s,audio/%s.wav,%s,%s,%s,%s,%s,%s" % (
            item["record_uid"], item["record_uid"], item["sha256_pcm"], item.get("sha256_source", ""),
            item["n_samples"], item["sample_rate"], item["shard_key"], item["shard_row_index"],
        )
        for item in integrity_rows
    ]
    before = {
        "index": (index_dir / "default_train_0000.jsonl").read_bytes(),
        "integrity": None,
    }
    integrity.write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")
    before["integrity"] = integrity.read_bytes()
    sentinel = tmp_path / "rq1_final_contract.json"
    sentinel.write_text('{"do_not_touch": true}\n', encoding="utf-8")
    sentinel_bytes = sentinel.read_bytes()
    rows = build_derived_protected_identity_rows(
        reference_index={
            "g_train": manifests / "train.csv",
            "g_validation": manifests / "validation.csv",
            "frozen_test": manifests / "test.csv",
        },
        project_root=root,
        dataset_id="org/bahnar",
        dataset_revision=_DATASET_REV,
        parquet_revision=_REV,
        audio_index_dir=index_dir,
        frozen_integrity_csv=integrity,
        expected_counts=counts,
    )
    assert sentinel.read_bytes() == sentinel_bytes
    assert (index_dir / "default_train_0000.jsonl").read_bytes() == before["index"]
    assert integrity.read_bytes() == before["integrity"]
    return rows


def test_derived_index_covers_manifest_universe_without_transcripts(tmp_path):
    from src.rq2_u_clean import write_derived_protected_identity_index, load_derived_protected_identity_index

    rows = _build_identity(
        tmp_path,
        train_rows=[
            ("TR1", "tr-1", "tr.flac", "default/train/0000.parquet", 1),
            ("EXCLUDED1", "ex-1", "ex.flac", "default/train/0000.parquet", 9),
        ],
        index_records=[
            _sidecar("TR1", shard_row_index=1),
            _sidecar("VAL1", split="validation", shard_row_index=2),
        ],
        frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
        integrity_rows=[{
            "record_uid": "TE1",
            "sha256_pcm": _PCM,
            "sha256_source": "ef" * 32,
            "n_samples": 134232,
            "sample_rate": 16000,
            "shard_key": "default/test/0000.parquet",
            "shard_row_index": 8,
        }],
        counts={"g_train": 2, "g_validation": 1, "frozen_test": 1},
    )
    assert [row["reference_uid"] for row in rows] == ["TE1", "EXCLUDED1", "TR1", "VAL1"]
    excluded = next(row for row in rows if row["reference_uid"] == "EXCLUDED1")
    assert excluded["sha256_pcm"] == ""
    assert excluded["parquet_file"] == "default/train/0000.parquet"
    assert excluded["shard_row_index"] == 9
    assert all("text_vi" not in row for row in rows)
    dest = tmp_path / "protected_audio_identity.csv"
    digest = write_derived_protected_identity_index(rows, dest)
    text = dest.read_text(encoding="utf-8")
    assert "SECRET_TRANSCRIPT" not in text
    assert "text_vi" not in text.splitlines()[0]
    loaded = load_derived_protected_identity_index(dest, digest)
    assert set(loaded) == {("frozen_test", "TE1"), ("g_train", "EXCLUDED1"), ("g_train", "TR1"), ("g_validation", "VAL1")}


def test_derived_index_rejects_duplicate_uid(tmp_path):
    with pytest.raises(RuntimeError, match="duplicate"):
        _build_identity(
            tmp_path,
            train_rows=[
                ("TR1", "tr-1", "tr.flac", "default/train/0000.parquet", 1),
                ("TR1", "tr-1b", "tr2.flac", "default/train/0000.parquet", 2),
            ],
            index_records=[_sidecar("TR1")],
            frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
            integrity_rows=[{
                "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
                "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
            }],
            counts={"g_train": 2, "g_validation": 1, "frozen_test": 1},
        )


def test_derived_index_rejects_missing_shard_row(tmp_path):
    with pytest.raises(RuntimeError, match="missing shard row"):
        _build_identity(
            tmp_path,
            train_rows=[("TR1", "tr-1", "tr.flac", "", -1)],
            index_records=[],
            frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
            integrity_rows=[{
                "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
                "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
            }],
            counts={"g_train": 1, "g_validation": 1, "frozen_test": 1},
        )


def test_derived_index_rejects_wrong_parquet_revision(tmp_path):
    with pytest.raises(RuntimeError, match="wrong parquet revision"):
        _build_identity(
            tmp_path,
            train_rows=[("TR1", "tr-1", "tr.flac", "default/train/0000.parquet", 1)],
            index_records=[_sidecar("TR1", parquet_revision=_OTHER_REV)],
            frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
            integrity_rows=[{
                "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
                "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
            }],
            counts={"g_train": 1, "g_validation": 1, "frozen_test": 1},
        )


def test_derived_index_rejects_forbidden_transcript_column(tmp_path):
    from src.rq2_u_clean import load_derived_protected_identity_index

    path = tmp_path / "bad.csv"
    path.write_text(
        "reference_uid,split,manifest_audio_locator,parquet_file,shard_key,shard_row_index,"
        "dataset_revision,parquet_revision,pcm_pipeline_version,sha256_pcm,sha256_source,"
        "n_samples,sample_rate,record_id,text_vi\n"
        "TR1,g_train,tr.flac,default/train/0000.parquet,default/train/0000.parquet,1,"
        + _DATASET_REV + "," + _REV + "," + _PIPELINE + "," + _PCM + "," + ("cd" * 32) + ",1600,16000,tr-1,SECRET\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="forbidden"):
        load_derived_protected_identity_index(path, digest)


def test_derived_index_rejects_unknown_column(tmp_path):
    from src.rq2_u_clean import DERIVED_PROTECTED_IDENTITY_COLUMNS, load_derived_protected_identity_index

    header = ",".join(DERIVED_PROTECTED_IDENTITY_COLUMNS) + ",extra_note"
    row = ",".join([
        "TR1", "g_train", "tr.flac", "default/train/0000.parquet", "default/train/0000.parquet", "0",
        _DATASET_REV, _REV, _PIPELINE, _PCM, "cd" * 32, "1600", "16000", "tr-1", "note",
    ])
    path = tmp_path / "extra.csv"
    path.write_text(header + "\n" + row + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="schema mismatch"):
        load_derived_protected_identity_index(path, digest)


def test_audio_index_row_zero_matches_manifest(tmp_path):
    rows = _build_identity(
        tmp_path,
        train_rows=[("TR0", "tr-0", "tr.flac", "default/train/0000.parquet", 0)],
        index_records=[_sidecar("TR0", shard_row_index=0)],
        frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
        integrity_rows=[{
            "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
            "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
        }],
        counts={"g_train": 1, "g_validation": 1, "frozen_test": 1},
    )
    train = next(row for row in rows if row["reference_uid"] == "TR0")
    assert train["shard_row_index"] == 0
    assert train["sha256_pcm"] == _PCM


def test_audio_index_row_zero_rejects_manifest_one(tmp_path):
    with pytest.raises(RuntimeError, match="shard_row_index mismatch"):
        _build_identity(
            tmp_path,
            train_rows=[("TR0", "tr-0", "tr.flac", "default/train/0000.parquet", 1)],
            index_records=[_sidecar("TR0", shard_row_index=0)],
            frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
            integrity_rows=[{
                "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
                "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
            }],
            counts={"g_train": 1, "g_validation": 1, "frozen_test": 1},
        )


def test_audio_index_rejects_off_by_one_row(tmp_path):
    with pytest.raises(RuntimeError, match="shard_row_index mismatch"):
        _build_identity(
            tmp_path,
            train_rows=[("TR1", "tr-1", "tr.flac", "default/train/0000.parquet", 11)],
            index_records=[_sidecar("TR1", shard_row_index=10)],
            frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
            integrity_rows=[{
                "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
                "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
            }],
            counts={"g_train": 1, "g_validation": 1, "frozen_test": 1},
        )


def test_audio_index_missing_shard_row_index_fails_closed(tmp_path):
    sidecar = _sidecar("TR1", shard_row_index=1)
    sidecar.pop("shard_row_index")
    with pytest.raises(RuntimeError, match="shard_row_index missing"):
        _build_identity(
            tmp_path,
            train_rows=[("TR1", "tr-1", "tr.flac", "default/train/0000.parquet", 1)],
            index_records=[sidecar],
            frozen_rows=[("TE1", "te-1", "te.flac", "default/test/0000.parquet", 8)],
            integrity_rows=[{
                "record_uid": "TE1", "sha256_pcm": _PCM, "n_samples": 1, "sample_rate": 16000,
                "shard_key": "default/test/0000.parquet", "shard_row_index": 8,
            }],
            counts={"g_train": 1, "g_validation": 1, "frozen_test": 1},
        )


def test_bind_fails_when_identity_is_missing():
    entries = [ProtectedReferenceEntry("TR1", "g_train", "tr.flac")]
    with pytest.raises(RuntimeError, match="missing"):
        bind_protected_audio_identity(entries, {})


def _wav_payload(n_samples=1600):
    import io

    pcm = np.zeros(n_samples, dtype="<i2")
    pcm[::2] = 1000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _decoded_wav(n_samples=1600):
    pytest.importorskip("soundfile")
    from src.asr_full_pcm import canonical_pcm16_from_bytes

    raw = _wav_payload(n_samples)
    return raw, canonical_pcm16_from_bytes(raw, target_sr=16000)


def _reconstruct_entry(uid, index, decoded, **overrides):
    values = dict(
        reference_uid=uid,
        split="g_train",
        audio_locator=uid + ".flac",
        manifest_audio_locator=uid + ".flac",
        parquet_file="default/train/0000.parquet",
        shard_key="default/train/0000.parquet",
        shard_row_index=index,
        dataset_revision=_DATASET_REV,
        parquet_revision=_REV,
        pcm_pipeline_version=_PIPELINE,
        sha256_pcm=decoded["sha256_pcm"],
        sha256_source=decoded["sha256_source"],
        n_samples=int(decoded["n_samples"]),
        sample_rate=int(decoded["sample_rate"]),
        record_id="id-%s" % index,
    )
    values.update(overrides)
    return ProtectedReferenceEntry(**values)


def _materializer(tmp_path, payloads, calls):
    from src.rq2_u_clean import ProtectedParquetMaterializer

    shard = tmp_path / "downloaded.parquet"
    shard.write_bytes(b"shard-bytes")

    def factory(downloaded):
        downloaded["default/train/0000.parquet"] = str(shard)

        def reader(ref, indices):
            calls["n"] += 1
            calls["indices"].append(list(indices))
            for index in indices:
                yield index, payloads[index]
        return reader

    return ProtectedParquetMaterializer(
        dataset_id="org/bahnar",
        cache_dir=tmp_path / "cache",
        reader_factory=factory,
    ), shard


def test_materialize_verifies_pcm_and_deletes_temp_wav(tmp_path):
    raw, decoded = _decoded_wav()
    calls = {"n": 0, "indices": []}
    materializer, shard = _materializer(tmp_path, {
        1: {"id": "id-1", "audio": raw},
        8: {"id": "id-8", "audio": raw},
    }, calls)
    first = _reconstruct_entry("A", 1, decoded)
    second = _reconstruct_entry("B", 8, decoded)
    paths = []
    for item in materializer.iter_materialized_audio([second, first]):
        paths.append(item.path)
        assert item.path.is_file()
        assert item.sha256_pcm == decoded["sha256_pcm"]
        assert item.n_samples == decoded["n_samples"]
        assert item.sample_rate == 16000
    assert calls["n"] == 1
    assert calls["indices"] == [[1, 8]]
    assert all(not path.exists() for path in paths)
    assert not shard.exists()


def test_materialize_rejects_wrong_pcm_and_source_hash(tmp_path):
    raw, decoded = _decoded_wav()
    calls = {"n": 0, "indices": []}
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, calls)
    wrong_pcm = _reconstruct_entry("A", 1, decoded, sha256_pcm="ff" * 32)
    with pytest.raises(RuntimeError, match="sha256_pcm mismatch"):
        list(materializer.iter_materialized_audio([wrong_pcm]))
    calls = {"n": 0, "indices": []}
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, calls)
    wrong_source = _reconstruct_entry("A", 1, decoded, sha256_source="11" * 32)
    with pytest.raises(RuntimeError, match="sha256_source mismatch"):
        list(materializer.iter_materialized_audio([wrong_source]))


def test_materialize_rejects_missing_shard_row(tmp_path):
    raw, decoded = _decoded_wav()
    calls = {"n": 0, "indices": []}
    materializer, _shard = _materializer(tmp_path, {}, calls)

    def factory(downloaded):
        def reader(ref, indices):
            return iter(())
        return reader

    materializer._reader_factory = factory
    entry = _reconstruct_entry("A", 1, decoded)
    with pytest.raises(RuntimeError, match="missing shard row"):
        list(materializer.iter_materialized_audio([entry]))


def test_durable_resolver_refuses_permanent_audio_path():
    resolver = DurableCanonicalReferenceResolver(
        durable_root=None, contract_hash=None, reference_index=None, mapping=None,
    )
    entry = ProtectedReferenceEntry("TR1", "g_train", "KT-XH02_029.flac")
    with pytest.raises(RuntimeError, match="materialize_audio"):
        resolver.resolve_audio_path(entry)


def _fingerprint_context(tmp_path, entries):
    cfg = _config()
    cfg.out_dir = tmp_path / "out"
    cfg.out_dir.mkdir()
    ctx = UCleanContext(config=cfg, protected_entries=list(entries))
    ctx.reference_provenance = {
        "rq1_final_contract_hash": "a" * 64,
        "rq1_test_contract_hash": "b" * 64,
        "protected_audio_identity_sha256": "c" * 64,
        "parquet_revision": _REV,
        "pcm_pipeline_version": _PIPELINE,
    }
    ctx.reference_summary = {"reference_set_sha256": "d" * 64}
    return ctx


def test_protected_fingerprint_resume_reuses_valid_shards(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("soundfile")
    from src.rq2_u_clean import generate_protected_fingerprints

    raw, decoded = _decoded_wav(81600)
    calls = {"n": 0, "indices": []}
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, calls)
    entry = _reconstruct_entry("A", 1, decoded)
    ctx = _fingerprint_context(tmp_path, [entry])
    fingerprints = {"n": 0}

    def fingerprint_fn(path):
        fingerprints["n"] += 1
        assert Path(path).is_file()
        return _rand_fp(3)

    assert generate_protected_fingerprints(ctx, materializer, fingerprint_fn=fingerprint_fn) == 1
    assert fingerprints["n"] == 1
    assert calls["n"] == 1
    assert generate_protected_fingerprints(ctx, materializer, fingerprint_fn=fingerprint_fn) == 0
    assert fingerprints["n"] == 1
    assert calls["n"] == 1


def test_changed_identity_index_sha_invalidates_resume(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("soundfile")
    from src.rq2_u_clean import generate_protected_fingerprints

    raw, decoded = _decoded_wav(81600)
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, {"n": 0, "indices": []})
    ctx = _fingerprint_context(tmp_path, [_reconstruct_entry("A", 1, decoded)])
    generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(4))
    ctx.reference_provenance["protected_audio_identity_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="stale"):
        generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(4))


def test_changed_overlap_contract_invalidates_resume(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("soundfile")
    from src.rq2_u_clean import generate_protected_fingerprints

    raw, decoded = _decoded_wav(81600)
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, {"n": 0, "indices": []})
    ctx = _fingerprint_context(tmp_path, [_reconstruct_entry("A", 1, decoded)])
    generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(5))
    ctx.config.overlap = OverlapConfig(similarity_threshold=0.50)
    with pytest.raises(RuntimeError, match="stale"):
        generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(5))


def _eligibility_entry(uid, split, **overrides):
    values = dict(
        reference_uid=uid,
        split=split,
        audio_locator=uid + ".flac",
        sha256_pcm="ab" * 32,
        n_samples=0,
        sample_rate=0,
    )
    values.update(overrides)
    return ProtectedReferenceEntry(**values)


class _WavResolver:
    def __init__(self, root, samples_by_uid, calls, corrupt=()):
        self.root = Path(root)
        self.samples_by_uid = dict(samples_by_uid)
        self.calls = calls
        self.corrupt = set(corrupt)

    def iter_materialized_audio(self, entries):
        from src.rq2_u_clean import MaterializedProtectedAudio

        self.calls["n"] += 1
        self.calls["uids"].extend(entry.reference_uid for entry in entries)
        for entry in entries:
            path = self.root / (entry.reference_uid + ".wav")
            n_samples = int(self.samples_by_uid.get(entry.reference_uid) or 0)
            if entry.reference_uid in self.corrupt:
                path.write_bytes(b"not-a-wav")
            else:
                pcm = np.zeros(n_samples, dtype="<i2")
                write_wav_pcm16_atomic(path, pcm, 16000)
            yield MaterializedProtectedAudio(
                entry=entry,
                path=path,
                sha256_pcm=entry.sha256_pcm or ("ab" * 32),
                sha256_source=entry.sha256_source or ("cd" * 32),
                n_samples=n_samples,
                sample_rate=16000,
            )


def _short_rows(ctx):
    path = ctx.config.out_dir / "checkpoint" / "protected_short_eligibility.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_short_protected_reference_is_audited_without_fpcalc(tmp_path):
    from src.rq2_u_clean import (
        SHORT_NOT_PERCEPTUALLY_ELIGIBLE,
        generate_protected_fingerprints,
        protected_fingerprint_coverage_complete,
        protected_reference_accounting,
    )

    calls = {"n": 0}
    samples = {"TR": 81599, "VA": 32000, "TE": 24000}
    entries = [
        _eligibility_entry("TR", "g_train"),
        _eligibility_entry("VA", "g_validation"),
        _eligibility_entry("TE", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    resolver = _WavResolver(tmp_path, samples, {"n": 0, "uids": []})

    def fingerprint_fn(path):
        calls["n"] += 1
        return _rand_fp(1)

    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=fingerprint_fn) == 0
    assert calls["n"] == 0
    rows = _short_rows(ctx)
    assert [row["reference_uid"] for row in rows] == ["TE", "TR", "VA"]
    assert {row["status"] for row in rows} == {SHORT_NOT_PERCEPTUALLY_ELIGIBLE}
    assert {row["reason"] for row in rows} == {SHORT_NOT_PERCEPTUALLY_ELIGIBLE}
    by_uid = {row["reference_uid"]: row for row in rows}
    assert by_uid["TR"]["actual_n_samples"] == 81599
    assert by_uid["TR"]["actual_sample_rate"] == 16000
    assert by_uid["TR"]["actual_duration_seconds"] < 5.1
    assert by_uid["TR"]["protected_perceptual_min_duration_seconds"] == 5.1
    assert by_uid["TR"]["min_overlap_items"] == 20
    stats = protected_reference_accounting(ctx)
    assert stats["n_protected_total"] == 3
    assert stats["n_protected_fingerprinted"] == 0
    assert stats["n_protected_short_not_perceptually_eligible"] == 3
    assert stats["n_protected_unaccounted"] == 0
    assert protected_fingerprint_coverage_complete(ctx) is True


def test_duration_at_boundary_persists_fingerprint(tmp_path):
    from src.rq2_u_clean import generate_protected_fingerprints, iter_protected_fingerprint_shards

    calls = {"n": 0}
    entries = [
        _eligibility_entry("TR", "g_train"),
        _eligibility_entry("VA", "g_validation"),
        _eligibility_entry("TE", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    resolver = _WavResolver(tmp_path, {"TR": 81600, "VA": 16000, "TE": 16000}, {"n": 0, "uids": []})

    def fingerprint_fn(path):
        calls["n"] += 1
        return _rand_fp(9)

    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=fingerprint_fn) == 1
    assert calls["n"] == 1
    yielded = [row for batch in iter_protected_fingerprint_shards(ctx) for row in batch]
    assert [row["uid"] for row in yielded] == ["TR"]
    assert len(yielded[0]["fingerprint"]) >= 20


def test_eligible_duration_fails_closed_on_empty_fingerprint(tmp_path):
    from src.rq2_u_clean import generate_protected_fingerprints

    entry = _eligibility_entry("TR", "g_train", n_samples=81600, sample_rate=16000)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []})
    with pytest.raises(RuntimeError, match="TR"):
        generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: [])
    (tmp_path / "err").mkdir()
    ctx = _fingerprint_context(tmp_path / "err", [entry])
    resolver = _WavResolver(tmp_path / "err", {"TR": 81600}, {"n": 0, "uids": []})

    def boom(path):
        raise RuntimeError("Empty fingerprint")

    with pytest.raises(RuntimeError, match="Empty fingerprint") as caught:
        generate_protected_fingerprints(ctx, resolver, fingerprint_fn=boom)
    message = str(caught.value)
    assert "TR" in message and "g_train" in message and "5.1" in message


def test_eligible_duration_fails_closed_when_fingerprint_is_too_short(tmp_path):
    from src.rq2_u_clean import generate_protected_fingerprints

    entry = _eligibility_entry("TR", "g_train", n_samples=81600, sample_rate=16000)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []})
    with pytest.raises(RuntimeError, match="items=19") as caught:
        generate_protected_fingerprints(
            ctx, resolver, fingerprint_fn=lambda path: _rand_fp(2, size=19),
        )
    message = str(caught.value)
    assert "TR" in message and "g_train" in message and "required=20" in message


def test_mixed_protected_coverage_requires_every_uid(tmp_path):
    from src.rq2_u_clean import (
        generate_protected_fingerprints,
        protected_fingerprint_coverage_complete,
        protected_reference_accounting,
    )

    entries = [
        _eligibility_entry("TR", "g_train"),
        _eligibility_entry("VA", "g_validation"),
        _eligibility_entry("TE", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    resolver = _WavResolver(tmp_path, {"TR": 81600, "VA": 24000, "TE": 32000}, {"n": 0, "uids": []})
    generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(3))
    stats = protected_reference_accounting(ctx)
    assert stats["n_protected_fingerprinted"] == 1
    assert stats["n_protected_short_not_perceptually_eligible"] == 2
    assert stats["n_protected_perceptual_eligible"] == 1
    assert stats["n_protected_unaccounted"] == 0
    assert protected_fingerprint_coverage_complete(ctx) is True
    ctx.protected_entries = entries + [_eligibility_entry("MISSING", "g_train")]
    stats = protected_reference_accounting(ctx)
    assert stats["n_protected_unaccounted"] == 1
    assert protected_fingerprint_coverage_complete(ctx) is False


def test_zero_metadata_uses_materialized_wav_dimensions(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("soundfile")
    from src.rq2_u_clean import generate_protected_fingerprints

    raw_short, decoded_short = _decoded_wav(1600)
    raw_long, decoded_long = _decoded_wav(81600)
    calls = {"n": 0, "indices": []}
    materializer, _shard = _materializer(tmp_path, {
        1: {"id": "id-1", "audio": raw_short},
        2: {"id": "id-2", "audio": raw_long},
    }, calls)
    short = _reconstruct_entry("S", 1, decoded_short, n_samples=0, sample_rate=0, split="g_train")
    long = _reconstruct_entry("L", 2, decoded_long, n_samples=0, sample_rate=0, split="g_validation")
    ctx = _fingerprint_context(tmp_path, [short, long])
    fingerprints = {"n": 0}

    def fingerprint_fn(path):
        fingerprints["n"] += 1
        return _rand_fp(6)

    assert generate_protected_fingerprints(ctx, materializer, fingerprint_fn=fingerprint_fn) == 1
    assert fingerprints["n"] == 1
    rows = _short_rows(ctx)
    assert len(rows) == 1
    assert rows[0]["reference_uid"] == "S"
    assert rows[0]["actual_n_samples"] == int(decoded_short["n_samples"])
    assert rows[0]["actual_sample_rate"] == 16000
    assert rows[0]["actual_n_samples"] > 0


def test_nonzero_metadata_mismatch_fails_closed(tmp_path):
    pytest.importorskip("soundfile")
    from src.rq2_u_clean import generate_protected_fingerprints

    calls = {"n": 0}
    entry = _eligibility_entry("TR", "g_train", n_samples=100, sample_rate=16000)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []})
    with pytest.raises(RuntimeError, match="n_samples mismatch"):
        generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(1))
    assert calls["n"] == 0

    raw, decoded = _decoded_wav(1600)
    (tmp_path / "mat").mkdir()
    materializer, _shard = _materializer(tmp_path / "mat", {1: {"id": "id-1", "audio": raw}}, {"n": 0, "indices": []})
    mismatched = _reconstruct_entry("A", 1, decoded, n_samples=1)
    with pytest.raises(RuntimeError, match="n_samples mismatch"):
        list(materializer.iter_materialized_audio([mismatched]))


def test_invalid_reconstructed_wav_fails_closed(tmp_path):
    from src.rq2_u_clean import generate_protected_fingerprints

    calls = {"n": 0}
    entry = _eligibility_entry("TR", "g_train")
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []}, corrupt={"TR"})
    with pytest.raises(RuntimeError, match="invalid reconstructed"):
        generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(1))
    assert calls["n"] == 0


def test_protected_fingerprint_and_short_accounting_resume(tmp_path):
    from src.rq2_u_clean import generate_protected_fingerprints, protected_reference_accounting

    calls = {"n": 0}
    samples = {"TR": 81600, "VA": 24000, "TE": 32000}
    entries = [
        _eligibility_entry("TR", "g_train"),
        _eligibility_entry("VA", "g_validation"),
        _eligibility_entry("TE", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    resolver = _WavResolver(tmp_path, samples, {"n": 0, "uids": []})

    def fingerprint_fn(path):
        calls["n"] += 1
        return _rand_fp(8)

    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=fingerprint_fn) == 1
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    short_path = ctx.config.out_dir / "checkpoint" / "protected_short_eligibility.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["segment_shards"] = [{"name": "keep.parquet", "sha256": "cd" * 32, "uids": ["seg-1"]}]
    state["segment_fingerprint_shards"] = [{"name": "keep-fp.parquet", "sha256": "ef" * 32, "uids": ["seg-1"]}]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    short_bytes = short_path.read_bytes()
    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=fingerprint_fn) == 0
    assert calls["n"] == 1
    assert resolver.calls["n"] == 1
    assert short_path.read_bytes() == short_bytes
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed["segment_shards"] == state["segment_shards"]
    assert resumed["segment_fingerprint_shards"] == state["segment_fingerprint_shards"]

    original = short_path.read_bytes()
    original_state = state_path.read_bytes()
    duplicated = original.decode("utf-8") + original.decode("utf-8").splitlines()[0] + "\n"
    short_path.write_text(duplicated, encoding="utf-8")
    tampered = json.loads(original_state.decode("utf-8"))
    tampered["protected_short_eligibility"]["sha256"] = hashlib.sha256(short_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate protected short-eligibility"):
        protected_reference_accounting(ctx)

    short_path.write_bytes(original)
    state_path.write_bytes(original_state)
    both = original.decode("utf-8") + json.dumps({
        "reference_uid": "TR",
        "split": "g_train",
        "actual_n_samples": 1000,
        "actual_sample_rate": 16000,
        "actual_duration_seconds": 0.0625,
        "status": "SHORT_NOT_PERCEPTUALLY_ELIGIBLE",
        "reason": "SHORT_NOT_PERCEPTUALLY_ELIGIBLE",
        "protected_perceptual_min_duration_seconds": 5.1,
        "min_overlap_items": 20,
        "fpcalc_version": "1.5.1",
        "sha256_pcm": "ab" * 32,
        "policy": "protected_perceptual_min_duration_v1",
    }, sort_keys=True) + "\n"
    short_path.write_text(both, encoding="utf-8")
    tampered = json.loads(original_state.decode("utf-8"))
    tampered["protected_short_eligibility"]["sha256"] = hashlib.sha256(short_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate protected accounting"):
        protected_reference_accounting(ctx)


def test_short_rows_never_reach_protected_matcher(tmp_path):
    from src.rq2_u_clean import (
        generate_protected_fingerprints,
        iter_protected_fingerprint_shards,
        match_protected_batch,
    )

    entries = [
        _eligibility_entry("TR", "g_train"),
        _eligibility_entry("VA", "g_validation"),
        _eligibility_entry("TE", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    resolver = _WavResolver(tmp_path, {"TR": 81600, "VA": 24000, "TE": 16000}, {"n": 0, "uids": []})
    generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(7))
    index = FingerprintIndex(ctx.config.overlap)
    seen = []
    for batch in iter_protected_fingerprint_shards(ctx):
        assert all(row["uid"] != "VA" and row["uid"] != "TE" for row in batch)
        match_protected_batch([], SegmentCandidateIndex(ctx.config.overlap), batch, ctx.config)
        for row in batch:
            seen.append(row["uid"])
            index.add(row["uid"], row["fingerprint"])
    assert seen == ["TR"]
    with pytest.raises(KeyError):
        index.fingerprint_of("VA")


def _open_segment(uid, pcm_sha, duration=4.0):
    return {
        "segment_uid": uid,
        "source_id": "VOV4",
        "start_sample": 0,
        "u_clean_status": RETAINED_STATUS,
        "segment_pcm16_sha256": pcm_sha,
        "duration_seconds": duration,
        "overlap_g_train": False,
        "overlap_g_validation": False,
        "overlap_frozen_test": False,
    }


def test_exact_protected_pcm_excludes_short_reference(tmp_path):
    from src.rq2_u_clean import (
        EXCLUDED_PROTECTED_OVERLAP,
        apply_protected_exact_identity,
        generate_protected_fingerprints,
    )

    pcm = "ab" * 32
    calls = {"n": 0}
    entry = _eligibility_entry("TR", "g_train", sha256_pcm=pcm)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 64000}, {"n": 0, "uids": []})
    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: calls.__setitem__("n", 1) or _rand_fp(1)) == 0
    assert calls["n"] == 0
    segment = _open_segment("SEG", pcm, duration=4.0)
    evidence = apply_protected_exact_identity([segment], ctx)
    assert segment["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    assert segment["overlap_g_train"] is True
    assert evidence[0].match_type == "protected_exact"
    assert evidence[0].reference_uid == "TR"
    assert evidence[0].reference_split == "g_train"


def test_exact_protected_pcm_keeps_different_short_reference(tmp_path):
    from src.rq2_u_clean import apply_protected_exact_identity, generate_protected_fingerprints

    entry = _eligibility_entry("TR", "g_train", sha256_pcm="ab" * 32)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 64000}, {"n": 0, "uids": []})
    generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(1))
    segment = _open_segment("SEG", "cd" * 32, duration=4.0)
    evidence = apply_protected_exact_identity([segment], ctx)
    assert segment["u_clean_status"] == RETAINED_STATUS
    assert evidence == []


def test_exact_protected_pcm_records_reference_split(tmp_path):
    from src.rq2_u_clean import apply_protected_exact_identity, generate_protected_fingerprints

    pcm = "ef" * 32
    entry = _eligibility_entry("TE", "frozen_test", sha256_pcm=pcm)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TE": 48000}, {"n": 0, "uids": []})
    generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(1))
    segment = _open_segment("SEG", pcm, duration=3.0)
    evidence = apply_protected_exact_identity([segment], ctx)
    assert segment["overlap_frozen_test"] is True
    assert segment["overlap_g_train"] is False
    assert evidence[0].reference_split == "frozen_test"
    assert evidence[0].reference_uid == "TE"
    assert evidence[0].match_type == "protected_exact"


def test_exact_protected_pcm_uses_materialized_hash_when_metadata_is_zero(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("soundfile")
    from src.rq2_u_clean import (
        EXCLUDED_PROTECTED_OVERLAP,
        apply_protected_exact_identity,
        generate_protected_fingerprints,
    )

    raw, decoded = _decoded_wav(64000)
    calls = {"n": 0, "indices": []}
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, calls)
    entry = _reconstruct_entry("S", 1, decoded, n_samples=0, sample_rate=0, split="g_train")
    assert entry.n_samples == 0 and entry.sample_rate == 0
    ctx = _fingerprint_context(tmp_path, [entry])
    assert generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(1)) == 0
    row = _short_rows(ctx)[0]
    assert row["actual_n_samples"] == int(decoded["n_samples"])
    assert row["actual_sample_rate"] == 16000
    assert row["sha256_pcm"] == decoded["sha256_pcm"]
    segment = _open_segment("SEG", decoded["sha256_pcm"], duration=4.0)
    apply_protected_exact_identity([segment], ctx)
    assert segment["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP


def test_exact_protected_check_reruns_when_ledger_is_missing(tmp_path):
    from src.rq2_u_clean import (
        EXCLUDED_PROTECTED_OVERLAP,
        apply_protected_exact_identity,
        generate_protected_fingerprints,
        protected_exact_identity_complete,
    )

    pcm = "ab" * 32
    entry = _eligibility_entry("TR", "g_train", sha256_pcm=pcm)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 64000}, {"n": 0, "uids": []})
    generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(1))
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["segment_shards"] = [{"name": "keep.parquet", "sha256": "cd" * 32, "uids": ["seg-1"]}]
    state["segment_fingerprint_shards"] = [{"name": "keep-fp.parquet", "sha256": "ef" * 32, "uids": ["seg-1"]}]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    segment = _open_segment("SEG", pcm, duration=4.0)
    apply_protected_exact_identity([segment], ctx)
    assert segment["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    segment["u_clean_status"] = RETAINED_STATUS
    segment["overlap_g_train"] = False
    exact_path = ctx.config.out_dir / "checkpoint" / "protected_exact_identity.jsonl"
    exact_path.unlink()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["protected_exact_identity"] = {}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert protected_exact_identity_complete(ctx) is False
    apply_protected_exact_identity([segment], ctx)
    assert segment["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed["segment_shards"][0]["name"] == "keep.parquet"
    assert resumed["segment_fingerprint_shards"][0]["name"] == "keep-fp.parquet"


def test_exact_protected_pcm_also_matches_long_reference(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import (
        EXCLUDED_PROTECTED_OVERLAP,
        apply_protected_exact_identity,
        generate_protected_fingerprints,
    )

    pcm = "12" * 32
    entry = _eligibility_entry("TR", "g_validation", sha256_pcm=pcm)
    ctx = _fingerprint_context(tmp_path, [entry])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []})
    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(4)) == 1
    segment = _open_segment("SEG", pcm, duration=6.0)
    evidence = apply_protected_exact_identity([segment], ctx)
    assert segment["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    assert segment["overlap_g_validation"] is True
    assert evidence[0].match_type == "protected_exact"


def test_swapped_protected_split_cannot_complete_coverage(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import (
        append_fingerprint_shard,
        generate_protected_fingerprints,
        protected_fingerprint_coverage_complete,
    )

    entries = [
        _eligibility_entry("TR", "g_train"),
        _eligibility_entry("VA", "g_validation"),
        _eligibility_entry("TE", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    append_fingerprint_shard(ctx, "protected_fingerprints", [{
        "uid": "TR",
        "split": "frozen_test",
        "source_sha256": "ab" * 32,
        "fingerprint": _rand_fp(1),
    }])
    completed = None
    with pytest.raises(RuntimeError, match="split mismatch"):
        completed = protected_fingerprint_coverage_complete(ctx)
    assert completed is not True

    (tmp_path / "shorts").mkdir()
    short_ctx = _fingerprint_context(tmp_path / "shorts", entries)
    resolver = _WavResolver(tmp_path / "shorts", {"TR": 24000, "VA": 24000, "TE": 24000}, {"n": 0, "uids": []})
    generate_protected_fingerprints(short_ctx, resolver, fingerprint_fn=lambda path: _rand_fp(1))
    path = short_ctx.config.out_dir / "checkpoint" / "protected_short_eligibility.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in lines:
        if row["reference_uid"] == "TR":
            row["split"] = "frozen_test"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in lines)
    path.write_text(payload, encoding="utf-8")
    state_path = short_ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["protected_short_eligibility"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    completed = None
    with pytest.raises(RuntimeError, match="split mismatch"):
        completed = protected_fingerprint_coverage_complete(short_ctx)
    assert completed is not True


def _plant_protected_fingerprint(ctx, uid, pcm, n_items, split="g_train"):
    from src.rq2_u_clean import append_fingerprint_shard

    append_fingerprint_shard(ctx, "protected_fingerprints", [{
        "uid": uid,
        "split": split,
        "source_sha256": pcm,
        "fingerprint": _rand_fp(1, size=n_items),
    }])


def _plant_segment_checkpoint(ctx):
    from src.rq2_u_clean import append_fingerprint_shard, mark_source_segmentation_complete

    mark_source_segmentation_complete(ctx, "VOV4", [{
        "source_id": "VOV4",
        "segment_uid": "S1",
        "segment_pcm16_sha256": "cd" * 32,
    }])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "S1",
        "fingerprint": _rand_fp(3, size=19),
        "audio_sha256": "cd" * 32,
    }])


def test_old_short_protected_fingerprint_is_discarded_on_resume(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import iter_protected_fingerprint_shards, load_resumable_checkpoint, protected_reference_uids

    pcm = "ab" * 32
    ctx = _fingerprint_context(tmp_path, [_eligibility_entry("TR", "g_train", sha256_pcm=pcm)])
    _plant_protected_fingerprint(ctx, "TR", pcm, 19)
    with pytest.raises(RuntimeError, match="items=19"):
        list(iter_protected_fingerprint_shards(ctx))
    loaded = load_resumable_checkpoint(ctx)
    assert loaded["discarded_reference_fingerprints"] == 1
    assert "TR" not in protected_reference_uids(ctx)
    state = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert state["protected_fingerprint_shards"] == []
    assert list(iter_protected_fingerprint_shards(ctx)) == []


def test_discarded_short_fingerprint_is_rematerialized_as_ineligible(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import (
        SHORT_NOT_PERCEPTUALLY_ELIGIBLE,
        generate_protected_fingerprints,
        load_resumable_checkpoint,
    )

    pcm = "ab" * 32
    calls = {"n": 0}
    ctx = _fingerprint_context(tmp_path, [_eligibility_entry("TR", "g_train", sha256_pcm=pcm)])
    resolver = _WavResolver(tmp_path, {"TR": 64000}, {"n": 0, "uids": []})
    _plant_protected_fingerprint(ctx, "TR", pcm, 19)
    load_resumable_checkpoint(ctx)
    assert generate_protected_fingerprints(
        ctx, resolver, fingerprint_fn=lambda path: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(1),
    ) == 0
    assert resolver.calls["n"] == 1
    assert calls["n"] == 0
    rows = _short_rows(ctx)
    assert rows[0]["reference_uid"] == "TR"
    assert rows[0]["status"] == SHORT_NOT_PERCEPTUALLY_ELIGIBLE
    assert rows[0]["actual_n_samples"] == 64000


def test_discarded_short_fingerprint_is_recomputed_when_long_enough(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import generate_protected_fingerprints, iter_protected_fingerprint_shards, load_resumable_checkpoint

    pcm = "ab" * 32
    calls = {"n": 0}
    ctx = _fingerprint_context(tmp_path, [_eligibility_entry("TR", "g_train", sha256_pcm=pcm)])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []})
    _plant_protected_fingerprint(ctx, "TR", pcm, 19)
    load_resumable_checkpoint(ctx)

    def fingerprint_fn(path):
        calls["n"] += 1
        return _rand_fp(4)

    assert generate_protected_fingerprints(ctx, resolver, fingerprint_fn=fingerprint_fn) == 1
    assert resolver.calls["n"] == 1
    assert calls["n"] == 1
    yielded = [row for batch in iter_protected_fingerprint_shards(ctx) for row in batch]
    assert [row["uid"] for row in yielded] == ["TR"]
    assert len(yielded[0]["fingerprint"]) >= 20


def test_valid_protected_fingerprint_is_reused_after_resume(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import generate_protected_fingerprints, load_resumable_checkpoint, protected_reference_uids

    pcm = "ab" * 32
    calls = {"n": 0}
    ctx = _fingerprint_context(tmp_path, [_eligibility_entry("TR", "g_train", sha256_pcm=pcm)])
    resolver = _WavResolver(tmp_path, {"TR": 81600}, {"n": 0, "uids": []})
    _plant_protected_fingerprint(ctx, "TR", pcm, 80)
    loaded = load_resumable_checkpoint(ctx)
    assert loaded["discarded_reference_fingerprints"] == 0
    assert "TR" in protected_reference_uids(ctx)
    assert generate_protected_fingerprints(
        ctx, resolver, fingerprint_fn=lambda path: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(1),
    ) == 0
    assert resolver.calls["n"] == 0
    assert calls["n"] == 0


def test_protected_fingerprint_migration_leaves_segment_shards(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import load_resumable_checkpoint

    pcm = "ab" * 32
    ctx = _fingerprint_context(tmp_path, [_eligibility_entry("TR", "g_train", sha256_pcm=pcm)])
    _plant_segment_checkpoint(ctx)
    _plant_protected_fingerprint(ctx, "TR", pcm, 19)
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    before = json.loads(state_path.read_text(encoding="utf-8"))
    loaded = load_resumable_checkpoint(ctx)
    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded["discarded_reference_fingerprints"] == 1
    assert after["segment_shards"] == before["segment_shards"]
    assert after["segment_fingerprint_shards"] == before["segment_fingerprint_shards"]
    assert "S1" in loaded["segment_fingerprints_by_uid"]
    assert len(loaded["segment_fingerprints_by_uid"]["S1"]) == 19
    assert after["protected_fingerprint_shards"] == []


def _u_span_row(uid, n_samples, pcm):
    return {
        "segment_uid": uid,
        "source_id": "VOV4",
        "start_sample": 0,
        "end_sample": int(n_samples),
        "duration_seconds": round(int(n_samples) / 16000.0, 6),
        "u_clean_status": RETAINED_STATUS,
        "segment_pcm16_sha256": pcm,
        "exclusion_reason": "",
        "canonical_segment_uid": uid,
    }


def test_u_segment_below_5_1_skips_fpcalc_and_u_clean(tmp_path):
    from src.rq2_u_clean import (
        EXCLUDED_PERCEPTUAL_INELIGIBLE,
        apply_u_perceptual_eligibility,
        fingerprint_retained_u_segments,
    )

    cfg = _config()
    row = _u_span_row("S", 64000, "cd" * 32)
    calls = {"n": 0}
    ctx = _fingerprint_context(tmp_path, [])
    ctx.config.overlap = cfg.overlap
    ctx.config.segmentation = cfg.segmentation
    assert apply_u_perceptual_eligibility([row], ctx.config) == 1
    assert row["u_clean_status"] == EXCLUDED_PERCEPTUAL_INELIGIBLE
    assert row["exclusion_reason"] == EXCLUDED_PERCEPTUAL_INELIGIBLE
    assert row["u_perceptual_min_duration_seconds"] == 5.1
    assert row["perceptual_eligibility_min_overlap_items"] == 20
    assert row["duration_seconds"] == 4.0
    assert fingerprint_retained_u_segments(
        [row], ctx, lambda item: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(1),
    ) == {}
    assert calls["n"] == 0
    assert retained_segments([row]) == []


def test_u_segment_at_5_1_requires_20_item_fingerprint(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import apply_u_perceptual_eligibility, fingerprint_retained_u_segments

    row = _u_span_row("S", 81600, "cd" * 32)
    calls = {"n": 0}
    ctx = _fingerprint_context(tmp_path, [])
    assert apply_u_perceptual_eligibility([row], ctx.config) == 0

    def fingerprint_fn(item):
        calls["n"] += 1
        return _rand_fp(2, size=20)

    fps = fingerprint_retained_u_segments([row], ctx, fingerprint_fn)
    assert calls["n"] == 1
    assert row["u_clean_status"] == RETAINED_STATUS
    assert len(fps["S"]) >= 20


def test_u_segment_at_least_5_1_rejects_19_item_fingerprint(tmp_path):
    from src.rq2_u_clean import fingerprint_retained_u_segments

    row = _u_span_row("S", 81600, "cd" * 32)
    ctx = _fingerprint_context(tmp_path, [])
    with pytest.raises(RuntimeError, match="S"):
        fingerprint_retained_u_segments([row], ctx, lambda item: _rand_fp(3, size=19))


def test_cached_short_u_fingerprint_excludes_without_recompute(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import (
        EXCLUDED_PERCEPTUAL_INELIGIBLE,
        append_fingerprint_shard,
        fingerprint_retained_u_segments,
        load_resumable_checkpoint,
        migrate_cached_u_fingerprints,
    )

    pcm = "cd" * 32
    long_pcm = "ef" * 32
    ctx = _fingerprint_context(tmp_path, [])
    short = _u_span_row("SHORT", 64000, pcm)
    long = _u_span_row("LONG", 81600, long_pcm)
    from src.rq2_u_clean import mark_source_segmentation_complete

    mark_source_segmentation_complete(ctx, "VOV4", [short, long])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "SHORT",
        "fingerprint": _rand_fp(5, size=19),
        "audio_sha256": pcm,
    }])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "LONG",
        "fingerprint": _rand_fp(6, size=80),
        "audio_sha256": long_pcm,
    }])
    _plant_protected_fingerprint(ctx, "TR", "ab" * 32, 80)
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    segment_file = next((ctx.config.out_dir / "checkpoint" / "segments").iterdir())
    segment_bytes = segment_file.read_bytes()
    before = json.loads(state_path.read_text(encoding="utf-8"))
    loaded = load_resumable_checkpoint(ctx)
    calls = {"n": 0}
    cached = migrate_cached_u_fingerprints(loaded["segments"], ctx, loaded["segment_fingerprints_by_uid"])
    after = json.loads(state_path.read_text(encoding="utf-8"))
    by_uid = {row["segment_uid"]: row for row in loaded["segments"]}
    assert by_uid["SHORT"]["u_clean_status"] == EXCLUDED_PERCEPTUAL_INELIGIBLE
    assert "SHORT" not in cached
    assert len(cached["LONG"]) >= 20
    assert fingerprint_retained_u_segments(
        loaded["segments"], ctx, lambda item: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(7), cached,
    )
    assert calls["n"] == 0
    assert after["segment_shards"] == before["segment_shards"]
    assert after["protected_fingerprint_shards"] == before["protected_fingerprint_shards"]
    assert segment_file.read_bytes() == segment_bytes
    assert [list(shard["uids"]) for shard in after["segment_fingerprint_shards"]] == [["LONG"]]


def test_cached_long_u_fingerprint_below_20_is_recomputed(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import (
        append_fingerprint_shard,
        fingerprint_retained_u_segments,
        load_resumable_checkpoint,
        mark_source_segmentation_complete,
        migrate_cached_u_fingerprints,
    )

    pcm = "cd" * 32
    ctx = _fingerprint_context(tmp_path, [])
    row = _u_span_row("LONG", 81600, pcm)
    mark_source_segmentation_complete(ctx, "VOV4", [row])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "LONG",
        "fingerprint": _rand_fp(8, size=19),
        "audio_sha256": pcm,
    }])
    loaded = load_resumable_checkpoint(ctx)
    calls = {"n": 0}
    cached = migrate_cached_u_fingerprints(loaded["segments"], ctx, loaded["segment_fingerprints_by_uid"])
    assert cached == {}
    assert loaded["segments"][0]["u_clean_status"] == RETAINED_STATUS

    def fingerprint_fn(item):
        calls["n"] += 1
        return _rand_fp(9, size=24)

    first = fingerprint_retained_u_segments(loaded["segments"], ctx, fingerprint_fn, cached)
    assert calls["n"] == 1
    assert len(first["LONG"]) >= 20
    fingerprint_retained_u_segments(loaded["segments"], ctx, fingerprint_fn, first)
    assert calls["n"] == 1
    resumed = load_resumable_checkpoint(ctx)
    reused = migrate_cached_u_fingerprints(resumed["segments"], ctx, resumed["segment_fingerprints_by_uid"])
    fingerprint_retained_u_segments(resumed["segments"], ctx, fingerprint_fn, reused)
    assert calls["n"] == 1
    assert len(reused["LONG"]) >= 20


def test_cached_valid_u_fingerprint_is_reused(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import (
        append_fingerprint_shard,
        fingerprint_retained_u_segments,
        load_resumable_checkpoint,
        mark_source_segmentation_complete,
        migrate_cached_u_fingerprints,
    )

    pcm = "cd" * 32
    ctx = _fingerprint_context(tmp_path, [])
    row = _u_span_row("LONG", 96000, pcm)
    mark_source_segmentation_complete(ctx, "VOV4", [row])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "LONG",
        "fingerprint": _rand_fp(10, size=80),
        "audio_sha256": pcm,
    }])
    before = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    loaded = load_resumable_checkpoint(ctx)
    calls = {"n": 0}
    cached = migrate_cached_u_fingerprints(loaded["segments"], ctx, loaded["segment_fingerprints_by_uid"])
    after = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert len(cached["LONG"]) >= 20
    fingerprint_retained_u_segments(
        loaded["segments"], ctx, lambda item: calls.__setitem__("n", calls["n"] + 1) or _rand_fp(11), cached,
    )
    assert calls["n"] == 0
    assert after["segment_fingerprint_shards"] == before["segment_fingerprint_shards"]
    assert after["segment_shards"] == before["segment_shards"]
    assert after["protected_fingerprint_shards"] == before["protected_fingerprint_shards"]


def test_coverage_rejects_u_fingerprint_below_min_overlap_items():
    from src.rq2_u_clean import EXCLUDED_PERCEPTUAL_INELIGIBLE, _fingerprint_coverage_complete, apply_u_perceptual_eligibility

    cfg = _config()
    eligible = _u_span_row("LONG", 81600, "cd" * 32)
    assert _fingerprint_coverage_complete([eligible], {"LONG": _rand_fp(12, size=19)}, cfg) is False
    assert _fingerprint_coverage_complete([eligible], {"LONG": _rand_fp(12, size=20)}, cfg) is True
    short = _u_span_row("SHORT", 64000, "ef" * 32)
    apply_u_perceptual_eligibility([short], cfg)
    assert short["u_clean_status"] == EXCLUDED_PERCEPTUAL_INELIGIBLE
    assert _fingerprint_coverage_complete([short], {}, cfg) is True


def test_matcher_indexes_keep_only_long_enough_u_fingerprints(monkeypatch):
    from src.rq2_u_clean import build_u_candidate_index, perceptual_deduplicate

    cfg = _config()
    cfg.overlap = OverlapConfig(frozen=True, min_overlap_items=20, similarity_threshold=0.775236)
    short = _u_span_row("SHORT", 81600, "cd" * 32)
    long = _u_span_row("LONG", 81600, "ef" * 32)
    long["start_sample"] = 81600
    long["end_sample"] = 163200
    fps = {"SHORT": _rand_fp(13, size=19), "LONG": _rand_fp(14, size=80)}
    added = []
    real_add = FingerprintIndex.add

    def spy_add(self, item_id, fingerprint, meta=None):
        added.append((item_id, len(list(fingerprint))))
        return real_add(self, item_id, fingerprint, meta)

    monkeypatch.setattr(FingerprintIndex, "add", spy_add)
    ctx = UCleanContext(config=cfg)
    perceptual_deduplicate([short, long], fps, cfg)
    assert added == [("LONG", 80)]
    index, _elapsed = build_u_candidate_index([short, long], fps, cfg)
    assert index.fingerprint_of("LONG")
    with pytest.raises(KeyError):
        index.fingerprint_of("SHORT")


def test_golden_rq1_frozen_test_reconstruction():
    import os

    if os.environ.get("BAHNAR_NB11_INTEGRATION") != "1":
        pytest.skip("set BAHNAR_NB11_INTEGRATION=1 to reconstruct the golden frozen-test row")
    pytest.importorskip("soundfile")
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths
    from src.rq2_u_clean import ProtectedParquetMaterializer

    project = Path(os.environ.get("BAHNAR_PROJECT_ROOT", "/workspace/citizen-assistance-system/bahnar-s2tt-thesis"))
    runtime = resolve_rq1_runtime_paths(project_root=project)
    entry = ProtectedReferenceEntry(
        reference_uid=_GOLDEN_UID,
        split="frozen_test",
        audio_locator="golden.flac",
        manifest_audio_locator="golden.flac",
        parquet_file="default/test/0000.parquet",
        shard_key="default/test/0000.parquet",
        shard_row_index=8,
        dataset_revision=os.environ["BAHNAR_DATASET_REVISION"],
        parquet_revision=os.environ["BAHNAR_PARQUET_REVISION"],
        pcm_pipeline_version=_PIPELINE,
        sha256_pcm=_GOLDEN_PCM,
        n_samples=134232,
        sample_rate=16000,
        record_id=os.environ.get("BAHNAR_GOLDEN_RECORD_ID", ""),
    )
    materializer = ProtectedParquetMaterializer(
        dataset_id=os.environ["BAHNAR_DATASET_ID"],
        cache_dir=runtime.hf_parquet_cache_dir,
    )
    with materializer.materialize_audio(entry) as item:
        assert item.sha256_pcm == _GOLDEN_PCM
        assert item.n_samples == 134232
        assert item.sample_rate == 16000
        assert item.path.is_file()
        wav_path = item.path
    assert not wav_path.exists()


# --------------------------------------------------------------------------- #
# Resumable protected matching                                                 #
# --------------------------------------------------------------------------- #
_MATCH_FP = [0xAAAAAAAA] * 8
_OTHER_FP = [0xFFFFFFFF] * 8
_NOISE_FP = [0x55555555] * 8


def _match_overlap(**overrides):
    values = dict(
        frozen=True,
        shingle_k=4,
        min_shared_shingles=1,
        min_overlap_items=4,
        similarity_threshold=0.90,
    )
    values.update(overrides)
    return OverlapConfig(**values)


def _match_segment(uid, pcm):
    row = _u_span_row(uid, 160000, pcm)
    row["overlap_g_train"] = False
    row["overlap_g_validation"] = False
    row["overlap_frozen_test"] = False
    row["exclusion_reason"] = ""
    return row


def _evidence_tuples(evidence):
    return [
        (
            item.candidate_uid,
            item.reference_uid,
            item.reference_split,
            item.matched_duration_seconds,
            item.alignment_offset,
            item.similarity,
            item.match_type,
        )
        for item in evidence
    ]


def _segment_views(segments):
    return [
        (
            row["segment_uid"],
            row.get("u_clean_status"),
            row.get("exclusion_reason"),
            row.get("overlap_g_train"),
            row.get("overlap_g_validation"),
            row.get("overlap_frozen_test"),
        )
        for row in segments
    ]


def _plant_protected_match_case(tmp_path):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import append_fingerprint_shard, build_u_candidate_index, mark_source_segmentation_complete

    pcm = "cd" * 32
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    ctx = _fingerprint_context(root, [
        _eligibility_entry("R0A", "g_train", sha256_pcm=pcm),
        _eligibility_entry("R0B", "g_train", sha256_pcm=pcm),
        _eligibility_entry("R1", "g_validation", sha256_pcm=pcm),
        _eligibility_entry("R2", "frozen_test", sha256_pcm=pcm),
    ])
    ctx.config.overlap = _match_overlap()
    segment = _match_segment("SEG", pcm)
    other = _match_segment("OTHER", "ef" * 32)
    mark_source_segmentation_complete(ctx, segment["source_id"], [segment])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "SEG",
        "fingerprint": list(_MATCH_FP),
        "audio_sha256": pcm,
    }])
    append_fingerprint_shard(ctx, "protected_fingerprints", [
        {"uid": "R0A", "split": "g_train", "source_sha256": pcm, "fingerprint": list(_MATCH_FP)},
        {"uid": "R0B", "split": "g_train", "source_sha256": pcm, "fingerprint": list(_MATCH_FP)},
    ])
    append_fingerprint_shard(ctx, "protected_fingerprints", [
        {"uid": "R1", "split": "g_validation", "source_sha256": pcm, "fingerprint": list(_MATCH_FP)},
    ])
    append_fingerprint_shard(ctx, "protected_fingerprints", [
        {"uid": "R2", "split": "frozen_test", "source_sha256": pcm, "fingerprint": list(_MATCH_FP)},
    ])
    fingerprints = {"SEG": list(_MATCH_FP), "OTHER": list(_OTHER_FP)}
    segments = [segment, other]
    index, _elapsed = build_u_candidate_index(segments, fingerprints, ctx.config)
    return ctx, segments, index, fingerprints


def _legacy_match_protected_batch(segments, index, reference_batch, config):
    """Historical single-process matcher, kept as an equivalence oracle."""
    from src.rq2_audio_fingerprint import is_valid_fingerprint, matched_duration_seconds
    from src.rq2_u_clean import _SPLIT_TO_FLAG, _exclude

    ov = config.overlap
    by_uid = {str(row["segment_uid"]): row for row in segments}
    evidence = []
    for ref in reference_batch:
        ref_fp = ref.get("fingerprint")
        if not is_valid_fingerprint(ref_fp):
            continue
        split = str(ref.get("split") or "")
        for uid in index.candidates_for_reference(ref_fp):
            row = by_uid.get(uid)
            status = row.get("u_clean_status") if row is not None else ""
            if row is None or status not in (RETAINED_STATUS, EXCLUDED_PROTECTED_OVERLAP):
                continue
            score, offset, overlap = compare_fingerprints_detailed(index.fingerprint_of(uid), ref_fp, ov)
            if score < ov.similarity_threshold:
                continue
            flag = _SPLIT_TO_FLAG.get(split)
            if flag:
                row[flag] = True
            evidence.append(MatchEvidence(
                candidate_uid=uid,
                reference_uid=str(ref.get("uid") or ""),
                reference_split=split,
                matched_duration_seconds=matched_duration_seconds(overlap, ov),
                alignment_offset=offset,
                similarity=round(score, 6),
                match_type="protected",
            ))
            _exclude(row, EXCLUDED_PROTECTED_OVERLAP)
    return evidence


def _checkpoint_lists(ctx):
    path = ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    return {key: state.get(key) for key in (
        "sources",
        "segment_shards",
        "segment_fingerprint_shards",
        "protected_fingerprint_shards",
        "protected_short_eligibility",
        "protected_exact_identity",
    )}


def _match_file_bytes(ctx):
    directory = ctx.config.out_dir / "checkpoint" / "protected_matches"
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.glob("match-*.json"))
    }


def test_pure_matcher_matches_legacy_single_process():
    import copy
    from src.rq2_u_clean import (
        _exclude,
        apply_protected_match_evidence,
        build_u_candidate_index,
        compute_protected_matches,
        match_protected_batch,
    )

    cfg = _config()
    cfg.overlap = _match_overlap()
    pcm = "cd" * 32
    batch = [
        {"uid": "R0A", "split": "g_train", "fingerprint": list(_MATCH_FP)},
        {"uid": "R0B", "split": "g_train", "fingerprint": list(_MATCH_FP)},
        {"uid": "R1", "split": "g_validation", "fingerprint": list(_MATCH_FP)},
        {"uid": "R2", "split": "frozen_test", "fingerprint": list(_MATCH_FP)},
        {"uid": "NOISE", "split": "g_train", "fingerprint": list(_NOISE_FP)},
    ]
    legacy_rows = [_match_segment("SEG", pcm), _match_segment("OTHER", "ef" * 32)]
    new_rows = copy.deepcopy(legacy_rows)
    pure_rows = copy.deepcopy(legacy_rows)
    fingerprints = {"SEG": list(_MATCH_FP), "OTHER": list(_OTHER_FP)}
    legacy_index, _ = build_u_candidate_index(legacy_rows, fingerprints, cfg)
    new_index, _ = build_u_candidate_index(new_rows, fingerprints, cfg)
    pure_index, _ = build_u_candidate_index(pure_rows, fingerprints, cfg)
    legacy = _legacy_match_protected_batch(legacy_rows, legacy_index, batch, cfg)
    current = match_protected_batch(new_rows, new_index, batch, cfg)
    assert _evidence_tuples(current) == _evidence_tuples(legacy)
    assert _segment_views(new_rows) == _segment_views(legacy_rows)

    def forbid_exclude(*_args, **_kwargs):
        raise AssertionError("pure matching must not exclude")

    original = _exclude
    try:
        import src.rq2_u_clean as clean
        clean._exclude = forbid_exclude
        computed = compute_protected_matches(
            pure_index,
            batch,
            cfg.overlap,
            {row["segment_uid"] for row in pure_rows},
        )
    finally:
        import src.rq2_u_clean as clean
        clean._exclude = original
    assert _segment_views(pure_rows) == [
        ("SEG", RETAINED_STATUS, "", False, False, False),
        ("OTHER", RETAINED_STATUS, "", False, False, False),
    ]
    assert _evidence_tuples(computed["evidence"]) == _evidence_tuples(legacy)
    apply_protected_match_evidence(computed["evidence"], {row["segment_uid"]: row for row in pure_rows})
    assert _segment_views(pure_rows) == _segment_views(legacy_rows)


def test_parallel_protected_matches_match_single_worker(tmp_path):
    import copy
    from src.rq2_u_clean import match_protected_references_resumable

    ctx_one, rows_one, index_one, _fps = _plant_protected_match_case(tmp_path / "one")
    ctx_many, rows_many, index_many, _fps = _plant_protected_match_case(tmp_path / "many")
    pristine = copy.deepcopy(rows_one)
    one = match_protected_references_resumable(rows_one, index_one, ctx_one, workers=1)
    many = match_protected_references_resumable(rows_many, index_many, ctx_many, workers=2)
    assert one["benchmark"]["workers"] == 1
    assert many["benchmark"]["workers"] == 2
    assert one["benchmark"]["n_match_shards_computed"] == one["benchmark"]["n_match_shards_total"] == 3
    assert many["benchmark"]["n_matches"] == one["benchmark"]["n_matches"] == 4
    assert _evidence_tuples(many["evidence"]) == _evidence_tuples(one["evidence"])
    assert _segment_views(rows_many) == _segment_views(rows_one)
    assert _segment_views(rows_one) != _segment_views(pristine)
    assert list(_match_file_bytes(ctx_many)) == list(_match_file_bytes(ctx_one))
    assert [json.loads(blob)["result_sha256"] for blob in _match_file_bytes(ctx_many).values()] == [
        json.loads(blob)["result_sha256"] for blob in _match_file_bytes(ctx_one).values()
    ]


def test_multiple_protected_references_and_split_flags_accumulate(tmp_path):
    from src.rq2_u_clean import match_protected_references_resumable

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    report = match_protected_references_resumable(rows, index, ctx, workers=2)
    matched = [item for item in report["evidence"] if item.candidate_uid == "SEG"]
    assert [item.reference_uid for item in matched] == ["R0A", "R0B", "R1", "R2"]
    assert [item.reference_split for item in matched] == ["g_train", "g_train", "g_validation", "frozen_test"]
    seg = rows[0]
    assert seg["segment_uid"] == "SEG"
    assert seg["overlap_g_train"] is True
    assert seg["overlap_g_validation"] is True
    assert seg["overlap_frozen_test"] is True
    assert seg["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    assert seg["exclusion_reason"] == EXCLUDED_PROTECTED_OVERLAP
    assert rows[1]["u_clean_status"] == RETAINED_STATUS
    assert rows[1]["overlap_g_train"] is False
    progress = json.loads((ctx.config.out_dir / "checkpoint" / "protected_matches" / "progress.json").read_text(encoding="utf-8"))
    assert progress["n_match_shards_completed"] == 3
    assert progress["n_references_processed"] == 4


def test_protected_match_resume_recomputes_only_missing_shards(tmp_path, monkeypatch):
    import copy
    import src.rq2_u_clean as clean

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path / "resume")
    uninterrupted_ctx, uninterrupted_rows, uninterrupted_index, _fps = _plant_protected_match_case(tmp_path / "full")
    pristine = copy.deepcopy(rows)
    calls = {"n": 0}
    real_commit = clean._commit_protected_match_shard

    def stop_after_first(context, document, batch_length):
        loaded = real_commit(context, document, batch_length)
        calls["n"] += 1
        if calls["n"] >= 1:
            raise RuntimeError("simulated interrupt")
        return loaded

    monkeypatch.setattr(clean, "_commit_protected_match_shard", stop_after_first)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        clean.match_protected_references_resumable(rows, index, ctx, workers=1)
    assert calls["n"] == 1
    assert _segment_views(rows) == _segment_views(pristine)
    assert len(_match_file_bytes(ctx)) == 1
    monkeypatch.setattr(clean, "_commit_protected_match_shard", real_commit)
    resumed = clean.match_protected_references_resumable(rows, index, ctx, workers=1)
    full = clean.match_protected_references_resumable(
        uninterrupted_rows, uninterrupted_index, uninterrupted_ctx, workers=1,
    )
    assert resumed["benchmark"]["n_match_shards_reused"] == 1
    assert resumed["benchmark"]["n_match_shards_computed"] == 2
    assert resumed["benchmark"]["n_matches"] == full["benchmark"]["n_matches"] == 4
    assert _evidence_tuples(resumed["evidence"]) == _evidence_tuples(full["evidence"])
    assert _segment_views(rows) == _segment_views(uninterrupted_rows)
    assert [json.loads(blob)["result_sha256"] for blob in _match_file_bytes(ctx).values()] == [
        json.loads(blob)["result_sha256"] for blob in _match_file_bytes(uninterrupted_ctx).values()
    ]


def test_corrupt_match_shard_is_recomputed(tmp_path):
    import copy
    from src.rq2_u_clean import match_protected_references_resumable

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    first = match_protected_references_resumable(rows, index, ctx, workers=1)
    path = ctx.config.out_dir / "checkpoint" / "protected_matches" / "match-000001.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["result_sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    fresh = copy.deepcopy(pristine)
    second = match_protected_references_resumable(fresh, index, ctx, workers=1)
    assert second["benchmark"]["n_match_shards_reused"] == 2
    assert second["benchmark"]["n_match_shards_computed"] == 1
    assert _evidence_tuples(second["evidence"]) == _evidence_tuples(first["evidence"])
    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert repaired["result_sha256"] != "0" * 64
    assert repaired["status"] == "COMPLETE"


def test_changed_overlap_contract_recomputes_match_shards(tmp_path):
    import copy
    from src.rq2_u_clean import build_u_candidate_index, compatibility_key, match_protected_references_resumable

    ctx, rows, index, fingerprints = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    match_protected_references_resumable(rows, index, ctx, workers=1)
    ctx.config.overlap = _match_overlap(similarity_threshold=0.99)
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["compatibility_key"] = compatibility_key(ctx)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    fresh = copy.deepcopy(pristine)
    rebuilt, _elapsed = build_u_candidate_index(fresh, fingerprints, ctx.config)
    report = match_protected_references_resumable(fresh, rebuilt, ctx, workers=1)
    assert report["benchmark"]["n_match_shards_reused"] == 0
    assert report["benchmark"]["n_match_shards_computed"] == 3
    stored = json.loads((ctx.config.out_dir / "checkpoint" / "protected_matches" / "match-000000.json").read_text(encoding="utf-8"))
    assert stored["overlap_contract_sha256"] == overlap_contract_sha256(ctx.config.overlap)


def test_changed_u_fingerprint_identity_recomputes_match_shards(tmp_path):
    import copy
    from src.rq2_u_clean import build_u_candidate_index, match_protected_references_resumable

    ctx, rows, index, fingerprints = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    match_protected_references_resumable(rows, index, ctx, workers=1)
    changed = dict(fingerprints)
    changed["SEG"] = [0xAAAAAAAA] * 7 + [0xAAAAAAAB]
    fresh = copy.deepcopy(pristine)
    rebuilt, _elapsed = build_u_candidate_index(fresh, changed, ctx.config)
    report = match_protected_references_resumable(fresh, rebuilt, ctx, workers=1)
    assert report["benchmark"]["n_match_shards_reused"] == 0
    assert report["benchmark"]["n_match_shards_computed"] == 3


def test_changed_protected_shard_recomputes_only_that_match(tmp_path):
    import os
    import copy
    import pandas as pd
    from src.rq2_audio_fingerprint import fingerprint_sha256
    from src.rq2_u_clean import match_protected_references_resumable

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    match_protected_references_resumable(rows, index, ctx, workers=1)
    before = _match_file_bytes(ctx)
    shard = ctx.config.out_dir / "checkpoint" / "protected_fingerprints" / "part-000002.parquet"
    frame = pd.read_parquet(shard)
    frame.at[0, "fingerprint"] = list(_NOISE_FP)
    frame.at[0, "fingerprint_sha256"] = fingerprint_sha256(_NOISE_FP)
    tmp = shard.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, shard)
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["protected_fingerprint_shards"][1]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    fresh = copy.deepcopy(pristine)
    report = match_protected_references_resumable(fresh, index, ctx, workers=1)
    after = _match_file_bytes(ctx)
    assert report["benchmark"]["n_match_shards_reused"] == 2
    assert report["benchmark"]["n_match_shards_computed"] == 1
    assert after["match-000000.json"] == before["match-000000.json"]
    assert after["match-000002.json"] == before["match-000002.json"]
    assert after["match-000001.json"] != before["match-000001.json"]
    assert [item.reference_uid for item in report["evidence"]] == ["R0A", "R0B", "R2"]


def test_match_completion_order_does_not_change_output(tmp_path, monkeypatch):
    from concurrent.futures import ALL_COMPLETED, wait as real_wait
    from src.rq2_u_clean import match_protected_references_resumable

    def reversed_completed(inflight):
        done, _pending = real_wait(set(inflight), return_when=ALL_COMPLETED)
        return list(reversed(list(done)))

    monkeypatch.setattr("src.rq2_u_clean._completed_match_futures", reversed_completed)
    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path / "ordered")
    other_ctx, other_rows, other_index, _fps = _plant_protected_match_case(tmp_path / "reversed")
    forward = match_protected_references_resumable(rows, index, ctx, workers=1)
    backward = match_protected_references_resumable(other_rows, other_index, other_ctx, workers=2)
    assert _evidence_tuples(backward["evidence"]) == _evidence_tuples(forward["evidence"])
    assert _segment_views(other_rows) == _segment_views(rows)
    assert [json.loads(blob)["result_sha256"] for blob in _match_file_bytes(other_ctx).values()] == [
        json.loads(blob)["result_sha256"] for blob in _match_file_bytes(ctx).values()
    ]


def test_parallel_executor_uses_requested_workers(tmp_path, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from src.rq2_u_clean import match_protected_references_resumable

    seen = {}

    class SpyExecutor(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            seen["max_workers"] = kwargs.get("max_workers")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("src.rq2_u_clean.ProcessPoolExecutor", SpyExecutor)
    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    report = match_protected_references_resumable(rows, index, ctx, workers=2)
    assert seen["max_workers"] == 2
    assert report["benchmark"]["workers"] == 2
    assert report["benchmark"]["n_match_shards_computed"] == 3
    assert report["benchmark"]["n_references_processed"] == 4
    assert report["benchmark"]["n_candidate_comparisons"] >= report["benchmark"]["n_matches"]
    assert report["benchmark"]["elapsed_seconds"] >= 0
    assert report["benchmark"]["refs_per_second"] > 0


def test_existing_checkpoint_without_match_state_stays_reusable(tmp_path):
    from src.rq2_u_clean import load_resumable_checkpoint, match_protected_references_resumable

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    loaded = load_resumable_checkpoint(ctx)
    assert loaded["source_status"]["VOV4"]["segmentation_status"] == "COMPLETE"
    assert "SEG" in loaded["segment_fingerprints_by_uid"]
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("protected_match_shards", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    before = _checkpoint_lists(ctx)
    report = match_protected_references_resumable(rows, index, ctx, workers=1)
    after = _checkpoint_lists(ctx)
    assert after == before
    assert report["benchmark"]["n_matches"] == 4
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(written["protected_match_shards"]) == 3
    assert all(row["status"] == "COMPLETE" for row in written["protected_match_shards"])


def test_match_worker_count_is_runtime_only(monkeypatch):
    from src.rq2_u_clean import protected_match_worker_count

    monkeypatch.delenv("BAHNAR_NB11_MATCH_WORKERS", raising=False)
    monkeypatch.setattr("src.rq2_u_clean.os.cpu_count", lambda: 128)
    assert protected_match_worker_count() == 32
    monkeypatch.setenv("BAHNAR_NB11_MATCH_WORKERS", "32")
    assert protected_match_worker_count() == 32
    assert protected_match_worker_count(1) == 1
    with pytest.raises(RuntimeError):
        protected_match_worker_count(0)
    same = overlap_contract_sha256(_match_overlap())
    monkeypatch.setenv("BAHNAR_NB11_MATCH_WORKERS", "8")
    assert overlap_contract_sha256(_match_overlap()) == same


def _plant_many_match_shards(tmp_path, n_shards):
    pytest.importorskip("pyarrow")
    from src.rq2_u_clean import append_fingerprint_shard, build_u_candidate_index, mark_source_segmentation_complete

    pcm = "cd" * 32
    splits = ("g_train", "g_validation", "frozen_test")
    entries = [
        _eligibility_entry("R%d" % index, splits[index % 3], sha256_pcm=pcm)
        for index in range(n_shards)
    ]
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    ctx = _fingerprint_context(root, entries)
    ctx.config.overlap = _match_overlap()
    segment = _match_segment("SEG", pcm)
    mark_source_segmentation_complete(ctx, segment["source_id"], [segment])
    append_fingerprint_shard(ctx, "segment_fingerprints", [{
        "segment_uid": "SEG",
        "fingerprint": list(_MATCH_FP),
        "audio_sha256": pcm,
    }])
    for entry in entries:
        append_fingerprint_shard(ctx, "protected_fingerprints", [{
            "uid": entry.reference_uid,
            "split": entry.split,
            "source_sha256": pcm,
            "fingerprint": list(_MATCH_FP),
        }])
    fingerprints = {"SEG": list(_MATCH_FP)}
    segments = [segment]
    index, _elapsed = build_u_candidate_index(segments, fingerprints, ctx.config)
    return ctx, segments, index


def test_u_candidate_identity_computed_once(tmp_path, monkeypatch):
    import copy
    import src.rq2_u_clean as clean

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    calls = {"n": 0}
    real_identity = clean._u_candidate_identity

    def spy(candidate_index):
        calls["n"] += 1
        return real_identity(candidate_index)

    monkeypatch.setattr(clean, "_u_candidate_identity", spy)
    clean.match_protected_references_resumable(rows, index, ctx, workers=1)
    assert calls["n"] == 1
    clean.match_protected_references_resumable(copy.deepcopy(pristine), index, ctx, workers=1)
    assert calls["n"] == 2


def test_state_json_not_rewritten_per_match_shard(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    writes = {"n": 0}
    real_write = clean._write_checkpoint_state

    def spy(context, state):
        writes["n"] += 1
        return real_write(context, state)

    monkeypatch.setattr(clean, "_write_checkpoint_state", spy)
    clean.match_protected_references_resumable(rows, index, ctx, workers=1)
    assert writes["n"] == 1
    state = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert len(state["protected_match_shards"]) == 3


def test_committed_match_file_resumes_without_state_record(tmp_path, monkeypatch):
    import copy
    import src.rq2_u_clean as clean

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path / "resume")
    full_ctx, full_rows, full_index, _fps = _plant_protected_match_case(tmp_path / "full")
    pristine = copy.deepcopy(rows)
    writes = {"n": 0}
    real_write = clean._write_checkpoint_state
    real_commit = clean._commit_protected_match_shard

    def spy_write(context, state):
        writes["n"] += 1
        return real_write(context, state)

    def stop_after_first(context, document, batch_length):
        loaded = real_commit(context, document, batch_length)
        raise KeyboardInterrupt()

    monkeypatch.setattr(clean, "_write_checkpoint_state", spy_write)
    monkeypatch.setattr(clean, "_commit_protected_match_shard", stop_after_first)
    with pytest.raises(KeyboardInterrupt):
        clean.match_protected_references_resumable(rows, index, ctx, workers=2)
    assert writes["n"] == 0
    assert _segment_views(rows) == _segment_views(pristine)
    assert len(_match_file_bytes(ctx)) == 1
    state = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert state.get("protected_match_shards") == []
    monkeypatch.setattr(clean, "_commit_protected_match_shard", real_commit)
    monkeypatch.setattr(clean, "_write_checkpoint_state", real_write)
    resumed = clean.match_protected_references_resumable(rows, index, ctx, workers=1)
    full = clean.match_protected_references_resumable(full_rows, full_index, full_ctx, workers=1)
    assert resumed["benchmark"]["n_match_shards_reused"] == 1
    assert resumed["benchmark"]["n_match_shards_computed"] == 2
    assert _evidence_tuples(resumed["evidence"]) == _evidence_tuples(full["evidence"])
    assert _segment_views(rows) == _segment_views(full_rows)


def test_logical_batch_change_invalidates_only_that_match(tmp_path):
    import copy
    from src.rq2_u_clean import match_protected_references_resumable

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    match_protected_references_resumable(rows, index, ctx, workers=1)
    before = _match_file_bytes(ctx)
    unchanged = match_protected_references_resumable(copy.deepcopy(pristine), index, ctx, workers=1)
    assert unchanged["benchmark"]["n_match_shards_reused"] == 3
    assert unchanged["benchmark"]["n_match_shards_computed"] == 0
    assert _match_file_bytes(ctx) == before
    shard_path = ctx.config.out_dir / "checkpoint" / "protected_fingerprints" / "part-000001.parquet"
    parquet_bytes = shard_path.read_bytes()
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    dropped = state["protected_fingerprint_shards"][0]["uids"].pop()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert shard_path.read_bytes() == parquet_bytes
    report = match_protected_references_resumable(copy.deepcopy(pristine), index, ctx, workers=1)
    after = _match_file_bytes(ctx)
    assert shard_path.read_bytes() == parquet_bytes
    assert report["benchmark"]["n_match_shards_reused"] == 2
    assert report["benchmark"]["n_match_shards_computed"] == 1
    assert after["match-000000.json"] != before["match-000000.json"]
    assert after["match-000001.json"] == before["match-000001.json"]
    assert after["match-000002.json"] == before["match-000002.json"]
    assert dropped not in [item.reference_uid for item in report["evidence"]]


def test_match_workers_execute_and_stay_bounded(tmp_path, monkeypatch):
    import os
    import src.rq2_u_clean as clean

    monkeypatch.setenv("BAHNAR_NB11_MATCH_READY_BARRIER", "1")
    ctx, rows, index = _plant_many_match_shards(tmp_path / "parallel", 8)
    other_ctx, other_rows, other_index = _plant_many_match_shards(tmp_path / "serial", 8)
    parallel = clean.match_protected_references_resumable(rows, index, ctx, workers=2)
    worker_pids = list(clean._LAST_MATCH_WORKER_PIDS)
    max_inflight = clean._LAST_MATCH_MAX_INFLIGHT
    serial = clean.match_protected_references_resumable(other_rows, other_index, other_ctx, workers=1)
    assert len(worker_pids) == 8
    assert len(set(worker_pids)) > 1
    assert os.getpid() not in set(worker_pids)
    assert max_inflight == clean._protected_match_max_inflight(2) == 4
    assert _evidence_tuples(parallel["evidence"]) == _evidence_tuples(serial["evidence"])
    assert _segment_views(rows) == _segment_views(other_rows)
    for blob in _match_file_bytes(ctx).values():
        document = json.loads(blob)
        assert "worker_pid" not in document
        assert "worker_pid" not in document["compatibility"]
    assert "worker_pid" not in parallel["benchmark"]


def test_notebook_uses_resumable_protected_matcher():
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "11_RQ2_UReal_Segmentation_Dedup_Freeze_UClean.ipynb"
    text = notebook.read_text(encoding="utf-8")
    assert "match_protected_references_resumable" in text
    assert "for batch in uc.iter_protected_fingerprint_shards" not in text
    assert "RUN_FULL_PIPELINE = True" in text
    assert "SEGMENTATION_CONFIG_FROZEN = True" in text
    assert "OVERLAP_CONFIG_FROZEN = True" in text
    assert "load_resumable_checkpoint(context, load_protected_fingerprints=False)" in text


def test_executor_starts_before_protected_iterator_is_consumed(tmp_path, monkeypatch):
    import os
    from concurrent.futures import ProcessPoolExecutor
    import src.rq2_u_clean as clean

    monkeypatch.setenv("BAHNAR_NB11_MATCH_READY_BARRIER", "1")
    n_shards = 8
    ctx, rows, index = _plant_many_match_shards(tmp_path, n_shards)
    consumed = {"n": 0}
    observed = {}
    real_units = clean._iter_protected_match_units
    real_progress = clean._write_match_progress

    def counting_units(context, base):
        for unit in real_units(context, base):
            consumed["n"] += 1
            yield unit

    class SpyExecutor(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            observed["consumed_at_pool"] = consumed["n"]
            observed["max_workers"] = kwargs.get("max_workers")
            super().__init__(*args, **kwargs)

        def submit(self, fn, *args, **kwargs):
            if "consumed_at_first_submit" not in observed:
                observed["consumed_at_first_submit"] = consumed["n"]
            return super().submit(fn, *args, **kwargs)

    def spy_progress(context, progress):
        if "first_progress" not in observed:
            observed["first_progress"] = dict(progress)
            observed["consumed_at_first_progress"] = consumed["n"]
        return real_progress(context, progress)

    monkeypatch.setattr(clean, "_iter_protected_match_units", counting_units)
    monkeypatch.setattr(clean, "ProcessPoolExecutor", SpyExecutor)
    monkeypatch.setattr(clean, "_write_match_progress", spy_progress)
    report = clean.match_protected_references_resumable(rows, index, ctx, workers=2)
    progress = json.loads(
        (ctx.config.out_dir / "checkpoint" / "protected_matches" / "progress.json").read_text(encoding="utf-8")
    )
    assert observed["consumed_at_pool"] == 0
    assert observed["consumed_at_first_submit"] < n_shards
    assert observed["consumed_at_first_submit"] <= clean._protected_match_max_inflight(2)
    assert n_shards - observed["consumed_at_first_submit"] > 0
    assert observed["consumed_at_first_progress"] == 0
    assert observed["first_progress"]["status"] == "running"
    assert observed["first_progress"]["workers"] == 2
    assert observed["first_progress"]["total_shards"] == n_shards
    assert observed["first_progress"]["completed_shards"] == 0
    assert clean._LAST_MATCH_PULLED_AT_POOL_START == 0
    assert clean._LAST_MATCH_PULLED_AT_FIRST_SUBMIT < n_shards
    assert clean._LAST_MATCH_MAX_INFLIGHT <= clean._protected_match_max_inflight(2)
    assert clean._LAST_MATCH_MAX_INFLIGHT == 4
    assert len(set(clean._LAST_MATCH_WORKER_PIDS)) > 1
    assert os.getpid() not in set(clean._LAST_MATCH_WORKER_PIDS)
    assert consumed["n"] == n_shards
    assert report["benchmark"]["n_match_shards_total"] == n_shards
    assert progress["status"] == "complete"
    for key in (
        "status", "workers", "total_shards", "completed_shards", "reused_shards",
        "submitted_shards", "references_processed", "matches", "elapsed_seconds",
    ):
        assert key in progress


def test_streaming_matches_materialized_reference(tmp_path):
    import src.rq2_u_clean as clean

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path / "stream")
    other_ctx, other_rows, other_index, _fps = _plant_protected_match_case(tmp_path / "materialized")
    jobs = []
    for job in clean._iter_protected_fingerprint_jobs(other_ctx):
        job["input_batch_sha256"] = clean._protected_input_batch_sha256(job["batch"])
        jobs.append(job)
    assert len(jobs) == 3
    eligible = clean._eligible_protected_match_uids({row["segment_uid"]: row for row in other_rows})
    evidence = []
    for job in sorted(jobs, key=lambda item: int(item["ordinal"])):
        payload = clean._compute_protected_match_job(job, other_index, other_ctx.config.overlap, eligible)
        evidence.extend(payload["evidence"])
    clean.apply_protected_match_evidence(evidence, {row["segment_uid"]: row for row in other_rows})
    streamed = clean.match_protected_references_resumable(rows, index, ctx, workers=2)
    assert _evidence_tuples(streamed["evidence"]) == _evidence_tuples(evidence)
    assert _segment_views(rows) == _segment_views(other_rows)


def test_resume_skips_array_materialization_for_completed_shards(tmp_path, monkeypatch):
    import copy
    import src.rq2_u_clean as clean

    ctx, rows, index, _fps = _plant_protected_match_case(tmp_path)
    pristine = copy.deepcopy(rows)
    clean.match_protected_references_resumable(rows, index, ctx, workers=1)
    clean._PROTECTED_SHARD_CACHE.clear()
    arrays = {"n": 0}
    computes = {"n": 0}
    real_arrays = clean._load_protected_match_arrays
    real_compute = clean.compute_protected_matches

    def spy_arrays(context, shard, path):
        arrays["n"] += 1
        return real_arrays(context, shard, path)

    def spy_compute(index_arg, batch, overlap, eligible):
        computes["n"] += 1
        return real_compute(index_arg, batch, overlap, eligible)

    monkeypatch.setattr(clean, "_load_protected_match_arrays", spy_arrays)
    monkeypatch.setattr(clean, "compute_protected_matches", spy_compute)
    missing = ctx.config.out_dir / "checkpoint" / "protected_matches" / "match-000001.json"
    missing.unlink()
    report = clean.match_protected_references_resumable(copy.deepcopy(pristine), index, ctx, workers=1)
    assert arrays["n"] == 1
    assert computes["n"] == 1
    assert report["benchmark"]["n_match_shards_reused"] == 2
    assert report["benchmark"]["n_match_shards_computed"] == 1
    arrays["n"] = 0
    computes["n"] = 0
    again = clean.match_protected_references_resumable(copy.deepcopy(pristine), index, ctx, workers=1)
    assert arrays["n"] == 0
    assert computes["n"] == 0
    assert again["benchmark"]["n_match_shards_reused"] == 3
    assert again["benchmark"]["n_match_shards_computed"] == 0


def test_deferred_resume_does_not_read_protected_parquet(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    ctx, _rows, _index, _fps = _plant_protected_match_case(tmp_path)
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    before = json.loads(state_path.read_text(encoding="utf-8"))["protected_fingerprint_shards"]
    reads = {"n": 0}
    real_read = clean._read_expected_file_bytes

    def spy_read(path, expected_sha):
        reads["n"] += 1
        return real_read(path, expected_sha)

    monkeypatch.setattr(clean, "_read_expected_file_bytes", spy_read)
    clean._PROTECTED_SHARD_CACHE.clear()
    loaded = clean.load_resumable_checkpoint(ctx, load_protected_fingerprints=False)
    assert loaded["discarded_reference_shards"] == []
    assert reads["n"] == 0
    after = json.loads(state_path.read_text(encoding="utf-8"))["protected_fingerprint_shards"]
    assert after == before
    clean._indexed_protected_pcm(ctx)
    verified_reads = reads["n"]
    assert verified_reads == 3
    clean.protected_fingerprint_coverage_complete(ctx)
    assert reads["n"] == verified_reads


def _protected_hash_set(indexed):
    hashes = dict(indexed["fingerprinted"])
    hashes.update(indexed["shorts"])
    return hashes


def test_pinned_exact_pcm_hashes_match_verified_scan(tmp_path):
    import src.rq2_u_clean as clean

    ctx, rows, _index, _fps = _plant_protected_match_case(tmp_path / "long")
    pinned = clean._pinned_protected_pcm(ctx)
    verified = clean._indexed_protected_pcm(ctx)
    assert pinned["unaccounted"] == []
    assert verified["unaccounted"] == []
    assert _protected_hash_set(pinned) == _protected_hash_set(verified)
    other_ctx, other_rows, _index, _fps = _plant_protected_match_case(tmp_path / "apply")
    old_evidence = clean._commit_exact_protected_identity(
        other_rows, other_ctx, _protected_hash_set(clean._indexed_protected_pcm(other_ctx)),
    )
    new_evidence = clean.apply_protected_exact_identity(rows, ctx)
    assert _evidence_tuples(new_evidence) == _evidence_tuples(old_evidence)
    assert _segment_views(rows) == _segment_views(other_rows)

    pcm = "ab" * 32
    other = "cd" * 32
    (tmp_path / "mixed").mkdir()
    mixed = _fingerprint_context(tmp_path / "mixed", [
        _eligibility_entry("TR", "g_train", sha256_pcm=pcm),
        _eligibility_entry("VA", "g_validation", sha256_pcm=other),
    ])
    resolver = _WavResolver(tmp_path / "mixed", {"TR": 81600, "VA": 24000}, {"n": 0, "uids": []})
    clean.generate_protected_fingerprints(mixed, resolver, fingerprint_fn=lambda path: _rand_fp(4))
    mixed_pinned = clean._pinned_protected_pcm(mixed)
    mixed_verified = clean._indexed_protected_pcm(mixed)
    assert mixed_pinned["unaccounted"] == []
    assert _protected_hash_set(mixed_pinned) == _protected_hash_set(mixed_verified)
    assert set(mixed_pinned["shorts"]) == {("g_validation", "VA")}
    assert set(mixed_pinned["fingerprinted"]) == {("g_train", "TR")}


def test_notebook_path_reaches_matcher_before_all_protected_parquets_are_read(tmp_path, monkeypatch):
    import os
    from concurrent.futures import ProcessPoolExecutor
    import src.rq2_u_clean as clean

    monkeypatch.setenv("BAHNAR_NB11_MATCH_READY_BARRIER", "1")
    n_shards = 8
    ctx, _rows, _index = _plant_many_match_shards(tmp_path / "notebook", n_shards)
    ref_ctx, _ref_rows, _ref_index = _plant_many_match_shards(tmp_path / "reference", n_shards)
    state_path = ctx.config.out_dir / "checkpoint" / "state.json"
    shard_list = json.loads(state_path.read_text(encoding="utf-8"))["protected_fingerprint_shards"]
    reference_total = sum(len(shard.get("uids") or []) for shard in shard_list)
    assert reference_total == n_shards
    reads = {"n": 0}
    observed = {}
    real_read = clean._read_expected_file_bytes
    real_progress = clean._write_match_progress

    def spy_read(path, expected_sha):
        reads["n"] += 1
        return real_read(path, expected_sha)

    class SpyExecutor(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            observed["reads_at_pool"] = reads["n"]
            super().__init__(*args, **kwargs)

        def submit(self, fn, *args, **kwargs):
            if "reads_at_first_submit" not in observed:
                observed["reads_at_first_submit"] = reads["n"]
            return super().submit(fn, *args, **kwargs)

    def spy_progress(context, progress):
        if "first_progress" not in observed:
            observed["first_progress"] = dict(progress)
            observed["reads_at_first_progress"] = reads["n"]
            observed["progress_exists"] = (
                context.config.out_dir / "checkpoint" / "protected_matches" / "progress.json"
            ).is_file()
        return real_progress(context, progress)

    monkeypatch.setattr(clean, "_read_expected_file_bytes", spy_read)
    monkeypatch.setattr(clean, "ProcessPoolExecutor", SpyExecutor)
    monkeypatch.setattr(clean, "_write_match_progress", spy_progress)

    class UnusedResolver:
        def iter_materialized_audio(self, entries):
            raise AssertionError("resume reused protected fingerprints without materializing audio")

    loaded = clean.load_resumable_checkpoint(ctx, load_protected_fingerprints=False)
    assert reads["n"] == 0
    assert json.loads(state_path.read_text(encoding="utf-8"))["protected_fingerprint_shards"] == shard_list
    assert clean.generate_protected_fingerprints(ctx, UnusedResolver()) == 0
    assert reads["n"] == 0
    segments = loaded["segments"]
    fingerprints = loaded["segment_fingerprints_by_uid"]
    for row in segments:
        row["segment_pcm16_sha256"] = "ef" * 32
    exact = clean.apply_protected_exact_identity(segments, ctx)
    assert exact == []
    assert reads["n"] == 0
    clean.perceptual_deduplicate(segments, fingerprints, ctx.config)
    assert reads["n"] == 0
    index, _elapsed = clean.build_u_candidate_index(segments, fingerprints, ctx.config)
    assert reads["n"] == 0
    report = clean.match_protected_references_resumable(segments, index, ctx, workers=2)
    progress = json.loads(
        (ctx.config.out_dir / "checkpoint" / "protected_matches" / "progress.json").read_text(encoding="utf-8")
    )
    assert observed["reads_at_pool"] < n_shards
    assert observed["reads_at_first_progress"] < n_shards
    assert observed["reads_at_first_submit"] < n_shards
    assert observed["reads_at_first_submit"] <= clean._protected_match_max_inflight(2)
    assert observed["first_progress"]["status"] == "running"
    assert observed["first_progress"]["total_shards"] == n_shards
    assert observed["first_progress"]["completed_shards"] == 0
    assert observed["first_progress"]["n_references_total"] == reference_total
    assert progress["n_references_total"] == reference_total
    assert progress["total_shards"] == n_shards
    assert progress["status"] == "complete"
    assert len(set(clean._LAST_MATCH_WORKER_PIDS)) > 1
    assert os.getpid() not in set(clean._LAST_MATCH_WORKER_PIDS)
    after_match = reads["n"]
    assert clean.protected_fingerprint_coverage_complete(ctx) is True
    assert clean.protected_exact_identity_complete(ctx) is True
    assert reads["n"] == after_match
    assert json.loads(state_path.read_text(encoding="utf-8"))["protected_fingerprint_shards"] == shard_list

    ref_verified = clean._indexed_protected_pcm(ref_ctx)
    ref_pinned = clean._pinned_protected_pcm(ref_ctx)
    assert ref_verified["unaccounted"] == []
    assert _protected_hash_set(ref_pinned) == _protected_hash_set(ref_verified)
    ref_loaded = clean.load_resumable_checkpoint(ref_ctx, load_protected_fingerprints=False)
    ref_segments = ref_loaded["segments"]
    ref_fingerprints = ref_loaded["segment_fingerprints_by_uid"]
    for row in ref_segments:
        row["segment_pcm16_sha256"] = "ef" * 32
    ref_exact = clean._commit_exact_protected_identity(
        ref_segments, ref_ctx, _protected_hash_set(ref_verified),
    )
    clean.perceptual_deduplicate(ref_segments, ref_fingerprints, ref_ctx.config)
    ref_index, _elapsed = clean.build_u_candidate_index(ref_segments, ref_fingerprints, ref_ctx.config)
    ref_report = clean.match_protected_references_resumable(ref_segments, ref_index, ref_ctx, workers=1)
    assert exact == ref_exact
    assert _evidence_tuples(report["evidence"]) == _evidence_tuples(ref_report["evidence"])
    assert _segment_views(segments) == _segment_views(ref_segments)
    assert report["benchmark"]["n_matches"] > 0


class _CountingExpected(dict):
    """Counts full scans. Direct membership does not increment."""

    def __init__(self):
        super().__init__()
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


def test_valid_protected_identity_lookup_does_not_scan_expected():
    from src.rq2_u_clean import _bind_protected_identity

    splits = ("g_train", "g_validation", "frozen_test")
    expected = _CountingExpected()
    count = 6000
    for index in range(count):
        expected[(splits[index % 3], "U%d" % index)] = index
    for index in range(count):
        split = splits[index % 3]
        uid = "U%d" % index
        assert _bind_protected_identity(expected, "fingerprint", split, uid) == (split, uid)
        assert _bind_protected_identity(expected, "short-eligibility", split, uid) == (split, uid)
    assert expected.scans == 0
    with pytest.raises(RuntimeError, match="split mismatch"):
        _bind_protected_identity(expected, "fingerprint", "frozen_test", "U0")
    assert expected.scans == 1


def test_pinned_uid_owner_index_is_built_once(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    splits = ("g_train", "g_validation", "frozen_test")
    count = 6000
    pcm = "ab" * 32
    entries = [
        _eligibility_entry("U%d" % index, splits[index % 3], sha256_pcm=pcm)
        for index in range(count)
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    state = clean._empty_checkpoint_state(ctx)
    state["protected_fingerprint_shards"] = [{
        "name": "part-000001.parquet",
        "sha256": "cd" * 32,
        "uids": ["U%d" % index for index in range(count)],
    }]
    clean._write_checkpoint_state(ctx, state)
    calls = {"n": 0}
    real_index = clean._protected_uid_owner_index

    def spy(expected):
        calls["n"] += 1
        return real_index(expected)

    monkeypatch.setattr(clean, "_protected_uid_owner_index", spy)
    pinned = clean._pinned_protected_pcm(ctx)
    assert calls["n"] == 1
    assert len(pinned["fingerprinted"]) == count
    assert pinned["unaccounted"] == []
    clean._pinned_protected_pcm(ctx)
    assert calls["n"] == 2


def test_ambiguous_and_unknown_protected_uids_fail_closed(tmp_path):
    import src.rq2_u_clean as clean

    pcm = "ab" * 32
    ctx = _fingerprint_context(tmp_path, [
        _eligibility_entry("U", "g_train", sha256_pcm=pcm),
        _eligibility_entry("U", "g_validation", sha256_pcm=pcm),
    ])
    state = clean._empty_checkpoint_state(ctx)
    state["protected_fingerprint_shards"] = [{
        "name": "part-000001.parquet",
        "sha256": "cd" * 32,
        "uids": ["U", "MISSING"],
    }]
    clean._write_checkpoint_state(ctx, state)
    with pytest.raises(RuntimeError, match="ambiguous protected fingerprint uid U"):
        clean._pinned_protected_pcm(ctx)
    state["protected_fingerprint_shards"][0]["uids"] = ["MISSING"]
    clean._write_checkpoint_state(ctx, state)
    with pytest.raises(RuntimeError, match="unknown protected fingerprint uid MISSING"):
        clean._pinned_protected_pcm(ctx)


def test_optimized_identity_matches_verified_exact_results(tmp_path):
    import copy
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pcm = {
        "TR": "ab" * 32,
        "VA": "cd" * 32,
        "TE": "ef" * 32,
    }
    ctx = _fingerprint_context(tmp_path, [
        _eligibility_entry("TR", "g_train", sha256_pcm=pcm["TR"]),
        _eligibility_entry("VA", "g_validation", sha256_pcm=pcm["VA"]),
        _eligibility_entry("TE", "frozen_test", sha256_pcm=pcm["TE"]),
    ])
    resolver = _WavResolver(tmp_path, {"TR": 81600, "VA": 24000, "TE": 81600}, {"n": 0, "uids": []})
    assert clean.generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(4)) == 2
    pinned = clean._pinned_protected_pcm(ctx)
    verified = clean._indexed_protected_pcm(ctx)
    assert pinned["fingerprinted"] == verified["fingerprinted"]
    assert pinned["shorts"] == verified["shorts"]
    assert pinned["unaccounted"] == verified["unaccounted"] == []
    assert set(pinned["fingerprinted"]) == {("g_train", "TR"), ("frozen_test", "TE")}
    assert set(pinned["shorts"]) == {("g_validation", "VA")}
    rows = [
        _open_segment("SEG-TR", pcm["TR"], duration=6.0),
        _open_segment("SEG-VA", pcm["VA"], duration=1.5),
        _open_segment("OTHER", "12" * 32, duration=6.0),
    ]
    left = copy.deepcopy(rows)
    right = copy.deepcopy(rows)
    new_evidence = clean.apply_protected_exact_identity(left, ctx)
    old_evidence = clean._commit_exact_protected_identity(right, ctx, _protected_hash_set(verified))
    assert _evidence_tuples(new_evidence) == _evidence_tuples(old_evidence)
    assert _segment_views(left) == _segment_views(right)
    assert left[0]["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    assert left[1]["u_clean_status"] == EXCLUDED_PROTECTED_OVERLAP
    assert left[2]["u_clean_status"] == RETAINED_STATUS
    ctx.protected_entries = list(ctx.protected_entries) + [
        _eligibility_entry("MISSING", "frozen_test", sha256_pcm="34" * 32),
    ]
    assert clean._pinned_protected_pcm(ctx)["unaccounted"] == clean._indexed_protected_pcm(ctx)["unaccounted"]


def test_corrupt_listed_protected_shard_fails_immediately(tmp_path, monkeypatch):
    import os
    import pandas as pd
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")

    def plant(name):
        ctx, rows, index = _plant_many_match_shards(tmp_path / name, 4)
        state_path = ctx.config.out_dir / "checkpoint" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        shard = state["protected_fingerprint_shards"][0]
        path = ctx.config.out_dir / "checkpoint" / "protected_fingerprints" / shard["name"]
        return ctx, rows, index, state, state_path, shard, path

    def run(ctx, rows, index, state, state_path):
        state_path.write_text(json.dumps(state), encoding="utf-8")
        clean._PROTECTED_SHARD_CACHE.pop(id(ctx), None)
        read_names = []
        real_read = clean._read_expected_file_bytes

        def spy(path, expected_sha):
            read_names.append(path.name)
            return real_read(path, expected_sha)

        monkeypatch.setattr(clean, "_read_expected_file_bytes", spy)
        with pytest.raises(RuntimeError, match="protected fingerprint shard failed verification") as caught:
            clean.match_protected_references_resumable(rows, index, ctx, workers=2)
        monkeypatch.setattr(clean, "_read_expected_file_bytes", real_read)
        return str(caught.value), read_names

    ctx, rows, index, state, state_path, shard, path = plant("missing")
    path.unlink()
    message, read_names = run(ctx, rows, index, state, state_path)
    assert "ordinal=0" in message
    assert "shard=%s" % shard["name"] in message
    assert "reason=missing file" in message
    assert read_names == []
    assert list((ctx.config.out_dir / "checkpoint" / "protected_matches").glob("match-*.json")) == []

    ctx, rows, index, state, state_path, shard, path = plant("sha")
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])
    message, read_names = run(ctx, rows, index, state, state_path)
    assert "ordinal=0" in message and "reason=file sha256 mismatch" in message
    assert read_names == [shard["name"]]
    assert list((ctx.config.out_dir / "checkpoint" / "protected_matches").glob("match-*.json")) == []

    ctx, rows, index, state, state_path, shard, path = plant("fingerprint")
    frame = pd.read_parquet(path)
    frame.at[0, "fingerprint_sha256"] = "0" * 64
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    state["protected_fingerprint_shards"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    message, read_names = run(ctx, rows, index, state, state_path)
    assert "ordinal=0" in message and "reason=invalid fingerprint_sha256" in message
    assert read_names == [shard["name"]]
    assert list((ctx.config.out_dir / "checkpoint" / "protected_matches").glob("match-*.json")) == []

    ctx, rows, index, state, state_path, shard, path = plant("batch")
    frame = pd.read_parquet(path)
    frame.at[0, "source_audio_sha256"] = "11" * 32
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    state["protected_fingerprint_shards"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    message, read_names = run(ctx, rows, index, state, state_path)
    assert "ordinal=0" in message and "reason=invalid logical batch" in message
    assert read_names == [shard["name"]]
    assert list((ctx.config.out_dir / "checkpoint" / "protected_matches").glob("match-*.json")) == []
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert written["protected_fingerprint_shards"] == state["protected_fingerprint_shards"]


def test_streaming_does_not_change_scientific_contract_fields():
    from src.rq2_pseudo_contract import EXPECTED_OVERLAP_CONTRACT_SHA256, EXPECTED_SEGMENTATION_CONTRACT_SHA256
    from src.rq2_u_clean import U_CLEAN_SCHEMA_VERSION, compatibility_key

    assert EXPECTED_SEGMENTATION_CONTRACT_SHA256 == "0ce196cc20aca45bc707c4347f1eaa602cd04eacb0ced64c4665f2870a8cefab"
    assert EXPECTED_OVERLAP_CONTRACT_SHA256 == "d5830cfa285b6cf5cf82f817262c612e254c2cd03ac13b33078c06f31c6afb2b"
    assert U_CLEAN_SCHEMA_VERSION == "rq2-uclean-1.1"
    fields = set(compatibility_key.__code__.co_consts)
    assert "workers" not in fields
    assert "protected_match_shards" not in fields
    assert "progress" not in fields


def _blank_entry(uid, split):
    return _eligibility_entry(uid, split, sha256_pcm="", source_sha256="")


def _plant_fingerprint_rows(ctx, rows):
    append_fingerprint_shard(ctx, "protected_fingerprints", rows)


def _reload_fingerprint_context(ctx):
    reloaded = UCleanContext(config=ctx.config, protected_entries=list(ctx.protected_entries))
    reloaded.reference_provenance = dict(ctx.reference_provenance)
    reloaded.reference_summary = dict(ctx.reference_summary)
    return reloaded


def _spy_protected_reads(monkeypatch):
    import src.rq2_u_clean as clean

    reads = []
    real_read = clean._read_expected_file_bytes

    def spy(path, expected_sha):
        reads.append(Path(path).name)
        return real_read(path, expected_sha)

    monkeypatch.setattr(clean, "_read_expected_file_bytes", spy)
    return reads


def test_blank_pinned_sha_recovered_from_fingerprint_shard(tmp_path):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pcm = "ab" * 32
    ctx = _fingerprint_context(tmp_path, [_blank_entry("LEGACY", "g_train")])
    _plant_fingerprint_rows(ctx, [{
        "uid": "LEGACY",
        "split": "g_train",
        "source_sha256": pcm,
        "fingerprint": _rand_fp(1),
    }])

    class UnusedResolver:
        def iter_materialized_audio(self, entries):
            raise AssertionError("blank pinned identity must reuse the fingerprint shard")

    assert clean.generate_protected_fingerprints(ctx, UnusedResolver()) == 0
    pinned = clean._pinned_protected_pcm(ctx)
    assert pinned["fingerprinted"][("g_train", "LEGACY")] == pcm
    assert pinned["unaccounted"] == []
    assert pinned["shorts"] == {}


def test_fallback_reads_only_unresolved_shards(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pinned_pcm = "11" * 32
    fallback_pcm = "22" * 32
    entries = [
        _eligibility_entry("P0", "g_train", sha256_pcm=pinned_pcm),
        _eligibility_entry("P1", "g_validation", sha256_pcm=pinned_pcm),
        _eligibility_entry("P2", "frozen_test", sha256_pcm=pinned_pcm),
        _blank_entry("LEG", "g_train"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    for entry in entries[:3]:
        _plant_fingerprint_rows(ctx, [{
            "uid": entry.reference_uid,
            "split": entry.split,
            "source_sha256": pinned_pcm,
            "fingerprint": _rand_fp(1),
        }])
    _plant_fingerprint_rows(ctx, [{
        "uid": "LEG",
        "split": "g_train",
        "source_sha256": fallback_pcm,
        "fingerprint": _rand_fp(2),
    }])
    state = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    pinned_names = [shard["name"] for shard in state["protected_fingerprint_shards"][:3]]
    fallback_name = state["protected_fingerprint_shards"][3]["name"]
    clean._PROTECTED_SHARD_CACHE.clear()
    reads = _spy_protected_reads(monkeypatch)
    pinned = clean._pinned_protected_pcm(ctx)
    assert pinned["fingerprinted"][("g_train", "LEG")] == fallback_pcm
    assert reads == [fallback_name]
    assert len(reads) <= 1
    for name in pinned_names:
        assert name not in reads


def test_unresolved_uids_in_one_shard_are_read_once(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pcm = {"A": "ab" * 32, "B": "cd" * 32, "C": "ef" * 32}
    entries = [
        _blank_entry("A", "g_train"),
        _blank_entry("B", "g_validation"),
        _blank_entry("C", "frozen_test"),
    ]
    ctx = _fingerprint_context(tmp_path, entries)
    _plant_fingerprint_rows(ctx, [
        {"uid": uid, "split": entry.split, "source_sha256": pcm[uid], "fingerprint": _rand_fp(index + 1)}
        for index, (uid, entry) in enumerate(zip(pcm, entries))
    ])
    writes = {"n": 0}
    real_write = clean._write_checkpoint_state

    def spy_write(context, state):
        writes["n"] += 1
        return real_write(context, state)

    monkeypatch.setattr(clean, "_write_checkpoint_state", spy_write)
    clean._PROTECTED_SHARD_CACHE.clear()
    reads = _spy_protected_reads(monkeypatch)
    pinned = clean._pinned_protected_pcm(ctx)
    assert writes["n"] == 1
    assert len(reads) == 1
    assert pinned["fingerprinted"] == {
        ("g_train", "A"): pcm["A"],
        ("g_validation", "B"): pcm["B"],
        ("frozen_test", "C"): pcm["C"],
    }
    shard = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    shard_name = shard["protected_fingerprint_shards"][0]["name"]
    path = ctx.config.out_dir / "checkpoint" / "protected_fingerprints" / shard_name
    before = len(reads)
    clean._match_shard_rows(ctx, shard["protected_fingerprint_shards"][0], path, keep_arrays=False)
    assert len(reads) == before


def _mutate_protected_parquet(path, state, mutate):
    import os
    import pandas as pd

    frame = pd.read_parquet(path)
    mutate(frame)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    state["protected_fingerprint_shards"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrupt_fallback_shard_fails_closed(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")

    def plant(name):
        root = tmp_path / name
        root.mkdir()
        ctx = _fingerprint_context(root, [
            _blank_entry("LEG", "g_train"),
            _blank_entry("NEXT", "g_validation"),
        ])
        _plant_fingerprint_rows(ctx, [{
            "uid": "LEG",
            "split": "g_train",
            "source_sha256": "ab" * 32,
            "fingerprint": _rand_fp(1),
        }])
        _plant_fingerprint_rows(ctx, [{
            "uid": "NEXT",
            "split": "g_validation",
            "source_sha256": "cd" * 32,
            "fingerprint": _rand_fp(2),
        }])
        state_path = ctx.config.out_dir / "checkpoint" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        shard = state["protected_fingerprint_shards"][0]
        path = ctx.config.out_dir / "checkpoint" / "protected_fingerprints" / shard["name"]
        return ctx, state, state_path, shard, path

    def fail(ctx, state, state_path, match):
        state_path.write_text(json.dumps(state), encoding="utf-8")
        clean._PROTECTED_SHARD_CACHE.pop(id(ctx), None)
        verified = []
        real_rows = clean._match_shard_rows

        def spy(context, shard, path, *, keep_arrays):
            verified.append(str(shard.get("name")))
            return real_rows(context, shard, path, keep_arrays=keep_arrays)

        monkeypatch.setattr(clean, "_match_shard_rows", spy)
        with pytest.raises(RuntimeError, match=match):
            clean._pinned_protected_pcm(ctx)
        monkeypatch.setattr(clean, "_match_shard_rows", real_rows)
        written = json.loads(state_path.read_text(encoding="utf-8"))
        assert written["protected_fingerprint_shards"] == state["protected_fingerprint_shards"]
        assert not (ctx.config.out_dir / "checkpoint" / "protected_materialized_pcm.json").is_file()
        assert verified == [state["protected_fingerprint_shards"][0]["name"]]

    ctx, state, state_path, shard, path = plant("missing")
    path.unlink()
    fail(ctx, state, state_path, "ordinal=0 shard=%s reason=missing file" % shard["name"])

    ctx, state, state_path, shard, path = plant("sha")
    state["protected_fingerprint_shards"][0]["sha256"] = "ab" * 32
    fail(ctx, state, state_path, "ordinal=0 shard=%s reason=file sha256 mismatch" % shard["name"])

    ctx, state, state_path, shard, path = plant("fingerprint")
    _mutate_protected_parquet(path, state, lambda frame: frame.__setitem__("fingerprint_sha256", "0" * 64))
    fail(ctx, state, state_path, "ordinal=0 shard=%s reason=invalid fingerprint_sha256" % shard["name"])

    ctx, state, state_path, shard, path = plant("source")
    _mutate_protected_parquet(path, state, lambda frame: frame.__setitem__("source_audio_sha256", ""))
    fail(ctx, state, state_path, "invalid source_audio_sha256")

    ctx, state, state_path, shard, path = plant("split")
    _mutate_protected_parquet(path, state, lambda frame: frame.__setitem__("split", "frozen_test"))
    fail(ctx, state, state_path, "split mismatch for LEG")

    ctx, state, state_path, shard, path = plant("missing-uid")
    _mutate_protected_parquet(path, state, lambda frame: frame.__setitem__("reference_uid", "OTHER"))
    fail(ctx, state, state_path, "LEG missing from shard")

    ctx, state, state_path, shard, path = plant("duplicate")
    state["protected_fingerprint_shards"].append({
        "name": "part-000099.parquet",
        "sha256": "ef" * 32,
        "uids": ["LEG"],
    })
    state_path.write_text(json.dumps(state), encoding="utf-8")
    clean._PROTECTED_SHARD_CACHE.pop(id(ctx), None)
    verified = []
    real_rows = clean._match_shard_rows

    def spy(context, shard, path, *, keep_arrays):
        verified.append(str(shard.get("name")))
        return real_rows(context, shard, path, keep_arrays=keep_arrays)

    monkeypatch.setattr(clean, "_match_shard_rows", spy)
    with pytest.raises(RuntimeError, match="duplicate protected fingerprint identity g_train:LEG"):
        clean._pinned_protected_pcm(ctx)
    assert verified == []


def test_recovered_pcm_persists_across_restart(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pcm = "ab" * 32
    ctx = _fingerprint_context(tmp_path, [_blank_entry("LEGACY", "g_train")])
    _plant_fingerprint_rows(ctx, [{
        "uid": "LEGACY",
        "split": "g_train",
        "source_sha256": pcm,
        "fingerprint": _rand_fp(3),
    }])
    clean._PROTECTED_SHARD_CACHE.clear()
    reads = _spy_protected_reads(monkeypatch)
    first = clean._pinned_protected_pcm(ctx)
    assert reads == [
        json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))[
            "protected_fingerprint_shards"
        ][0]["name"]
    ]
    ledger_path = ctx.config.out_dir / "checkpoint" / "protected_materialized_pcm.json"
    document = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "rq2-protected-materialized-pcm-1"
    assert document["protected_audio_identity_sha256"] == ctx.reference_provenance["protected_audio_identity_sha256"]
    assert document["compatibility_key"] == clean.compatibility_key(ctx)
    assert document["records"] == [{"sha256_pcm": pcm, "split": "g_train", "uid": "LEGACY"}]
    state = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert state["protected_materialized_pcm"]["sha256"] == hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert state["protected_materialized_pcm"]["n_records"] == 1

    clean._PROTECTED_SHARD_CACHE.clear()
    reads.clear()
    reloaded = _reload_fingerprint_context(ctx)
    second = clean._pinned_protected_pcm(reloaded)
    assert reads == []
    assert second["fingerprinted"] == first["fingerprinted"]
    assert second["fingerprinted"][("g_train", "LEGACY")] == pcm


def test_pinned_fallback_matches_verified_pcm_index(tmp_path):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pcm = {"TR": "ab" * 32, "TE": "cd" * 32, "VA": "ef" * 32}
    ctx = _fingerprint_context(tmp_path, [
        _eligibility_entry("TR", "g_train", sha256_pcm=pcm["TR"]),
        _blank_entry("TE", "frozen_test"),
        _eligibility_entry("VA", "g_validation", sha256_pcm=pcm["VA"]),
    ])
    _plant_fingerprint_rows(ctx, [{
        "uid": "TR",
        "split": "g_train",
        "source_sha256": pcm["TR"],
        "fingerprint": _rand_fp(4),
    }])
    _plant_fingerprint_rows(ctx, [{
        "uid": "TE",
        "split": "frozen_test",
        "source_sha256": pcm["TE"],
        "fingerprint": _rand_fp(5),
    }])
    resolver = _WavResolver(tmp_path, {"VA": 24000}, {"n": 0, "uids": []})
    assert clean.generate_protected_fingerprints(ctx, resolver, fingerprint_fn=lambda path: _rand_fp(4)) == 0
    pinned = clean._pinned_protected_pcm(ctx)
    verified = clean._indexed_protected_pcm(ctx)
    assert pinned["fingerprinted"] == verified["fingerprinted"]
    assert pinned["shorts"] == verified["shorts"]
    assert pinned["unaccounted"] == verified["unaccounted"] == []
    assert set(pinned["expected"]) == set(verified["expected"])
    assert set(pinned["fingerprinted"]) == {("g_train", "TR"), ("frozen_test", "TE")}
    assert pinned["fingerprinted"][("frozen_test", "TE")] == pcm["TE"]
    assert set(pinned["shorts"]) == {("g_validation", "VA")}


def test_blank_pinned_recovery_does_not_scan_every_shard(tmp_path, monkeypatch):
    import src.rq2_u_clean as clean

    pytest.importorskip("pyarrow")
    pinned_pcm = "11" * 32
    entries = [
        _eligibility_entry("P%d" % index, ("g_train", "g_validation", "frozen_test")[index % 3], sha256_pcm=pinned_pcm)
        for index in range(5)
    ]
    entries.extend([
        _blank_entry("L0", "g_train"),
        _blank_entry("L1", "g_validation"),
    ])
    ctx = _fingerprint_context(tmp_path, entries)
    for entry in entries[:5]:
        _plant_fingerprint_rows(ctx, [{
            "uid": entry.reference_uid,
            "split": entry.split,
            "source_sha256": pinned_pcm,
            "fingerprint": _rand_fp(1),
        }])
    _plant_fingerprint_rows(ctx, [
        {"uid": "L0", "split": "g_train", "source_sha256": "22" * 32, "fingerprint": _rand_fp(2)},
        {"uid": "L1", "split": "g_validation", "source_sha256": "33" * 32, "fingerprint": _rand_fp(3)},
    ])
    state = json.loads((ctx.config.out_dir / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    shards = state["protected_fingerprint_shards"]
    assert len(shards) == 6
    pinned_names = {shard["name"] for shard in shards[:5]}
    unresolved_name = shards[5]["name"]
    clean._PROTECTED_SHARD_CACHE.clear()
    reads = _spy_protected_reads(monkeypatch)
    pinned = clean._pinned_protected_pcm(ctx)
    unresolved_shards = 1
    assert len(reads) <= unresolved_shards
    assert reads == [unresolved_name]
    assert pinned_names.isdisjoint(reads)
    assert len(pinned["fingerprinted"]) == 7
