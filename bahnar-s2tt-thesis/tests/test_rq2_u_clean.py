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


def _decoded_wav():
    pytest.importorskip("soundfile")
    from src.asr_full_pcm import canonical_pcm16_from_bytes

    raw = _wav_payload()
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

    raw, decoded = _decoded_wav()
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

    raw, decoded = _decoded_wav()
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

    raw, decoded = _decoded_wav()
    materializer, _shard = _materializer(tmp_path, {1: {"id": "id-1", "audio": raw}}, {"n": 0, "indices": []})
    ctx = _fingerprint_context(tmp_path, [_reconstruct_entry("A", 1, decoded)])
    generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(5))
    ctx.config.overlap = OverlapConfig(similarity_threshold=0.50)
    with pytest.raises(RuntimeError, match="stale"):
        generate_protected_fingerprints(ctx, materializer, fingerprint_fn=lambda path: _rand_fp(5))


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
