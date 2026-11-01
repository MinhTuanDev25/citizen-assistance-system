"""Unit tests of deploy/runpod/rootfs/opt/cas/lib/cas_storage.py (the all-in-one pod's
storage keeper): fingerprints, Postgres cross-check, base backup schedule and pruning.
The end-to-end behaviour runs in deploy/runpod/tests/persist_test.sh."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "deploy" / "runpod" / "rootfs" / "opt" / "cas" / "lib" / "cas_storage.py"


@pytest.fixture()
def cs(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cas_storage_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "W", tmp_path / "cas")
    monkeypatch.setattr(mod, "WAL", tmp_path / "cas" / "pg" / "wal")
    monkeypatch.setattr(mod, "BASE", tmp_path / "cas" / "pg" / "base")
    monkeypatch.setattr(mod, "STATE", tmp_path / "cas" / "pg" / "state")
    monkeypatch.setattr(mod, "NEED_BASE", tmp_path / "run" / "need-base-backup")
    for d in (mod.WAL, mod.BASE, mod.STATE, tmp_path / "run"):
        d.mkdir(parents=True)
    return mod


def _points(order):
    pts = {
        "a": {"id": "a", "payload": {"generation_id": "g1", "chunk_id": "c1"}, "vector": [0.1, 0.2]},
        "b": {"id": "b", "payload": {"generation_id": "g1", "chunk_id": "c2"}, "vector": [0.3, 0.4]},
        "c": {"id": "c", "payload": {"generation_id": "g2", "chunk_id": "c3"}, "vector": [0.5, 0.6]},
    }
    return [pts[k] for k in order]


def _fake_scroll(points, page=2):
    def http(method, path, body=None, timeout=60):
        offset = body.get("offset") or 0
        chunk = points[offset: offset + page]
        nxt = offset + page if offset + page < len(points) else None
        return {"result": {"points": chunk, "next_page_offset": nxt}}
    return http


def test_fingerprint_is_order_independent_and_per_generation(cs, monkeypatch):
    monkeypatch.setattr(cs, "http", _fake_scroll(_points("abc")))
    one = cs.fingerprint("x")
    monkeypatch.setattr(cs, "http", _fake_scroll(_points("cba"), page=1))
    two = cs.fingerprint("x")
    assert one == two
    assert one["points"] == 3
    assert one["generations"]["g1"]["count"] == 2 and one["generations"]["g2"]["count"] == 1


@pytest.mark.parametrize("field,change", [("payload", {"generation_id": "g1", "chunk_id": "zz"}), ("vector", [0.1, 0.2000001]), ("id", "a2")])
def test_fingerprint_detects_any_change(cs, monkeypatch, field, change):
    monkeypatch.setattr(cs, "http", _fake_scroll(_points("abc")))
    base = cs.fingerprint("x")
    pts = _points("abc")
    pts[0] = dict(pts[0], **{field: change})
    monkeypatch.setattr(cs, "http", _fake_scroll(pts))
    changed = cs.fingerprint("x")
    assert changed["generations"]["g1"] != base["generations"]["g1"]
    assert changed["generations"]["g2"] == base["generations"]["g2"]


def test_vector_bytes_are_float32_and_named_vectors_sorted(cs):
    assert cs._vector_bytes([1.0, 2.0]) == struct.pack("<2f", 1.0, 2.0)
    assert cs._vector_bytes({"b": [1.0], "a": [2.0]}) == cs._vector_bytes({"a": [2.0], "b": [1.0]})


def test_verify_against_pg(cs):
    fp = {"generations": {"g1": {"count": 2}, "old": {"count": 5}}}
    assert cs.verify_against_pg(fp, {"g1": 2}) == []
    problems = cs.verify_against_pg(fp, {"g1": 3, "g9": 1})
    assert any("g1: 2 points, Postgres vector_count 3" in p for p in problems)
    assert any("g9 has no points" in p for p in problems)


def test_ready_digest_is_order_independent(cs):
    assert cs.ready_digest({"a": 1, "b": 2}) == cs.ready_digest({"b": 2, "a": 1})
    assert cs.ready_digest({"a": 1}) != cs.ready_digest({"a": 2})


def _base(cs, name: str, start_wal: str, age_s: float = 0):
    d = cs.BASE / name
    d.mkdir()
    (d / "backup_label").write_text(f"START WAL LOCATION: 0/2000028 (file {start_wal})\n")
    marker = d / ".cas-complete"
    marker.write_text(json.dumps({"start_wal": start_wal}))
    t = time.time() - age_s
    os.utime(marker, (t, t))
    return d


def test_base_due(cs, monkeypatch):
    assert "no base backup" in cs.base_due()
    _base(cs, "20260101T000000Z", "000000010000000000000002")
    assert cs.base_due() is None
    cs.NEED_BASE.write_text("")
    assert "required at this boot" in cs.base_due()
    cs.NEED_BASE.unlink()
    monkeypatch.setattr(cs, "BASE_WAL_GB", 0.00001)
    (cs.WAL / "000000010000000000000003").write_bytes(b"x" * 20000)
    assert "of WAL since" in cs.base_due()
    monkeypatch.setattr(cs, "BASE_WAL_GB", 1)
    monkeypatch.setattr(cs, "BASE_HOURS", 0.0001)
    time.sleep(0.5)
    assert "h old" in cs.base_due()


def test_partial_or_unmarked_bases_are_not_complete(cs):
    (cs.BASE / ".20260101T000000Z.partial").mkdir()
    (cs.BASE / "20260102T000000Z").mkdir()   # no .cas-complete
    assert cs.complete_bases() == []


def test_prune_keeps_newest_and_cleans_wal_from_oldest_kept(cs, monkeypatch):
    calls = []
    monkeypatch.setattr(cs, "as_postgres", lambda args, timeout=None, check=True: calls.append(args))
    monkeypatch.setattr(cs, "BASE_KEEP", 2)
    for i, wal in enumerate(["000000010000000000000002", "000000010000000000000005", "000000020000000000000007"]):
        _base(cs, f"2026010{i + 1}T000000Z", wal)
    cs.prune_bases()
    assert [p.name for p in cs.complete_bases()] == ["20260102T000000Z", "20260103T000000Z"]
    assert calls and calls[-1][-2:] == [str(cs.WAL), "000000010000000000000005"]


# ---- end of recovery: wal_end_check on synthetic WAL segments --------------------------
SEG, BLK, MAGIC = 16 * 1024 * 1024, 8192, 0xD113
SEG_START = 9 << 24   # segment 9


def _segment(lengths, fill=0xAB):
    """A WAL segment holding records of the given total lengths from the first record
    position, with page headers (continuation flag + remaining length) where records cross
    pages. Returns (bytes, [(start, end)] offsets of each record)."""
    buf = bytearray(SEG)
    struct.pack_into("<HHIQI", buf, 0, MAGIC, 0x0002, 1, SEG_START, 0)
    pos, spans = 40, []
    for n in lengths:
        pos = (pos + 7) & ~7
        if pos % BLK == 0:
            struct.pack_into("<HHIQI", buf, pos, MAGIC, 0, 1, SEG_START + pos, 0)
            pos += 24
        start, remaining = pos, n
        first = True
        while remaining:
            if pos % BLK == 0:
                struct.pack_into("<HHIQI", buf, pos, MAGIC, 0x0001, 1, SEG_START + pos, remaining)
                pos += 24
            take = min(remaining, BLK - pos % BLK)
            buf[pos:pos + take] = bytes([fill]) * take
            if first:
                struct.pack_into("<I", buf, pos, n)
                first = False
            pos += take
            remaining -= take
        spans.append((start, pos))
    return buf, spans


def test_wal_end_only_zeros_after_the_end_is_accepted(cs):
    buf, spans = _segment([100, 300])
    assert cs.wal_end_check(bytes(buf), SEG_START + spans[-1][1]) is None


def test_wal_end_complete_record_after_the_end_is_refused(cs):
    buf, spans = _segment([100, 300, 120])
    problem = cs.wal_end_check(bytes(buf), SEG_START + spans[0][1])
    assert problem and "complete record" in problem


def test_wal_end_incomplete_record_cut_at_a_page_boundary_is_accepted(cs):
    buf, spans = _segment([100, 30000])
    cut = (spans[1][0] // BLK + 2) * BLK          # two pages of the long record written
    buf[cut:] = bytes(SEG - cut)
    assert cs.wal_end_check(bytes(buf), SEG_START + spans[0][1]) is None


def test_wal_end_inconsistent_continuation_page_is_refused(cs):
    buf, spans = _segment([100, 30000])
    page = (spans[1][0] // BLK + 1) * BLK
    struct.pack_into("<I", buf, page + 16, 5)     # xlp_rem_len no longer matches
    problem = cs.wal_end_check(bytes(buf), SEG_START + spans[0][1])
    assert problem and "does not continue" in problem


def test_wal_end_garbage_after_the_end_is_refused(cs):
    buf, spans = _segment([100])
    end = spans[0][1]
    buf[end:end + 64] = b"\xff" * 64
    problem = cs.wal_end_check(bytes(buf), SEG_START + end)
    assert problem and "invalid record" in problem


def test_wal_end_zeroed_record_header_with_data_after_is_refused(cs):
    buf, spans = _segment([100, 200, 200])
    s, _ = spans[1]
    buf[s:s + 24] = bytes(24)                      # damaged header, the next record still there
    problem = cs.wal_end_check(bytes(buf), SEG_START + spans[0][1])
    assert problem and "invalid record (length 0)" in problem


def test_wal_end_at_a_page_boundary(cs):
    buf = bytearray(SEG)
    struct.pack_into("<HHIQI", buf, 0, MAGIC, 0x0002, 1, SEG_START, 0)
    buf[40:BLK] = b"\xab" * (BLK - 40)
    struct.pack_into("<HHIQI", buf, BLK, MAGIC, 0, 1, SEG_START + BLK, 0)   # empty next page
    assert cs.wal_end_check(bytes(buf), SEG_START + BLK) is None
    struct.pack_into("<HHIQI", buf, BLK, MAGIC, 0x0001, 1, SEG_START + BLK, 50)
    assert "continuation data" in cs.wal_end_check(bytes(buf), SEG_START + BLK)


def test_wal_end_truncated_segment_is_refused(cs):
    buf, spans = _segment([100])
    assert "not 16777216" in cs.wal_end_check(bytes(buf[: SEG // 2]), SEG_START + spans[0][1])


def test_lsn_round_trip(cs):
    assert cs.lsn_int("0/9000100") >> 24 == cs.lsn_int("0/9FF0000") >> 24 == 9
    assert cs.lsn_int("0/9FF0000") > cs.lsn_int("0/9000100")
    assert cs.lsn_str(cs.lsn_int("1A/2B3C")) == "1A/2B3C"


# ---- Qdrant: READY generations against their recorded fingerprint ----------------------
def _gen(count=2, v="v"):
    return {"count": count, "ids_sha256": "i", "payload_sha256": "p", "vector_sha256": v}


def _baseline(gens):
    return {gid: {"generation_id": gid, "fingerprint": fp, "snapshot": "s1.snapshot"} for gid, fp in gens.items()}


def test_same_count_but_changed_vectors_is_refused(cs):
    fp = {"generations": {"g1": _gen(v="tampered")}}
    assert cs.verify_against_pg(fp, {"g1": 2}) == []          # the count alone passes
    problems = cs.verify_against_baselines(fp, {"g1": 2}, _baseline({"g1": _gen()}))
    assert problems and "vector_sha256 differ" in problems[0]


def test_unrecorded_ready_generation_is_refused_and_identical_passes(cs):
    fp = {"generations": {"g1": _gen(), "g2": _gen()}}
    recorded = _baseline({"g1": _gen()})
    problems = cs.verify_against_baselines(fp, {"g1": 2, "g2": 2}, recorded)
    assert len(problems) == 1 and "g2 has no recorded fingerprint" in problems[0]
    assert cs.verify_against_baselines(fp, {"g1": 2}, recorded) == []
    assert cs.verify_against_baselines(fp, {"g1": 2, "g2": 2}, recorded, allow_new=True) == []


@pytest.fixture()
def qd(cs, monkeypatch, tmp_path):
    """take_snapshot against a fake Qdrant (live fingerprint and READY set settable)."""
    monkeypatch.setattr(cs, "SNAP_ROOT", tmp_path / "snaps")
    monkeypatch.setattr(cs, "BASELINE_ROOT", tmp_path / "baselines")
    monkeypatch.setattr(cs, "SNAP_KEEP", 1)
    live = {"ready": {"g1": 2}, "fp": {"collection": cs.COLL, "points": 2, "generations": {"g1": _gen()}}}
    monkeypatch.setattr(cs, "pg_ready", lambda conn=None: dict(live["ready"]))
    monkeypatch.setattr(cs, "fingerprint", lambda: json.loads(json.dumps(live["fp"])))
    monkeypatch.setattr(cs, "http", lambda *a, **k: {"result": {"name": "q.snapshot"}})

    class Stream:
        status_code = 200

        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_bytes(self, n):
            return [b"snapshot"]

    import httpx
    monkeypatch.setattr(httpx, "stream", Stream)
    stamps = iter(f"2026{i:04d}T000000000000Z" for i in range(101, 9999))
    monkeypatch.setattr(cs, "utc_stamp", lambda: next(stamps))
    return live


def test_snapshot_refused_when_a_ready_generation_changed_since_its_baseline(cs, qd):
    """S1 records g1; a g1 vector changes (same count); g2 is published: the next snapshot
    is refused, and the boot check still sees the change after S1 was pruned."""
    cs.take_snapshot()
    assert set(cs.baselines()) == {"g1"}
    qd["fp"]["generations"]["g1"] = _gen(v="tampered")
    qd["fp"]["generations"]["g2"] = _gen()
    qd["ready"]["g2"] = 2
    with pytest.raises(cs.QdrantIntegrity, match="snapshot refused: READY generation g1: vector_sha256 differ"):
        cs.take_snapshot()
    assert len(cs.snapshots()) == 1 and set(cs.baselines()) == {"g1"}
    assert cs.verify_against_baselines(qd["fp"], qd["ready"], cs.baselines())  # boot refuses
    # The baseline is never replaced: even a snapshot taken after repairing g1 keeps it.
    qd["fp"]["generations"]["g1"] = _gen()
    cs.take_snapshot()
    snaps = cs.snapshots()
    assert len(snaps) == 1 and snaps[0].name != "20260101T000000000000Z.snapshot"   # S1 pruned
    assert set(cs.baselines()) == {"g1", "g2"} and cs.baselines()["g1"]["snapshot"] == "20260101T000000000000Z.snapshot"
    qd["fp"]["generations"]["g1"] = _gen(v="tampered")
    assert "g1: vector_sha256 differ" in cs.verify_against_baselines(qd["fp"], qd["ready"], cs.baselines())[0]


def test_snapshot_refused_when_a_ready_generation_is_incomplete(cs, qd):
    qd["fp"]["generations"]["g1"] = _gen(count=1)
    with pytest.raises(cs.QdrantIntegrity, match="snapshot refused: READY generation g1: 1 points, Postgres vector_count 2"):
        cs.take_snapshot()
    assert cs.snapshots() == [] and cs.baselines() == {}


# ---- keeper readiness (qdrant_tick) ---------------------------------------------------------
def _tick(cs, keeper, ready, now):
    problems, warnings = [], []
    keeper.qdrant_tick(dict(ready), problems, warnings, now=now)
    return problems, warnings


def test_keeper_stays_not_ready_when_the_ready_set_returns_to_the_snapshotted_one(cs, qd):
    """G1 tampered -> G2 published -> snapshot refused -> G2 retired: still not ready until
    G1 is valid again (the READY digest equal to the last snapshot's clears nothing)."""
    k = cs.Keeper()
    assert _tick(cs, k, qd["ready"], 0) == ([], [])          # S1 + baseline of g1
    qd["fp"]["generations"]["g1"] = _gen(v="tampered")
    qd["fp"]["generations"]["g2"] = _gen()
    qd["ready"]["g2"] = 2
    p, _ = _tick(cs, k, qd["ready"], 10)
    assert p and "snapshot refused: READY generation g1: vector_sha256 differ" in p[0]
    del qd["ready"]["g2"]                                      # g2 retired
    del qd["fp"]["generations"]["g2"]
    p, _ = _tick(cs, k, qd["ready"], 20)                       # before the re-check: kept
    assert p and "g1: vector_sha256 differ" in p[0]
    p, _ = _tick(cs, k, qd["ready"], 20 + cs.QDRANT_RETRY_S)   # re-checked, still changed
    assert p and "g1: vector_sha256 differ" in p[0]
    qd["fp"]["generations"]["g1"] = _gen()                     # repaired (e.g. restored)
    assert _tick(cs, k, qd["ready"], 30 + 2 * cs.QDRANT_RETRY_S) == ([], [])
    assert len(cs.snapshots()) == 1


def test_keeper_not_ready_when_a_ready_generation_has_missing_points(cs, qd):
    """G2 READY in Postgres, a point missing in Qdrant: the snapshot is refused and the
    keeper reports a problem (not only a warning) until a check passes."""
    k = cs.Keeper()
    _tick(cs, k, qd["ready"], 0)
    qd["ready"]["g2"] = 2
    qd["fp"]["generations"]["g2"] = _gen(count=1)
    p, w = _tick(cs, k, qd["ready"], 10)
    assert w == [] and p and "READY generation g2: 1 points, Postgres vector_count 2" in p[0]
    assert _tick(cs, k, qd["ready"], 11)[0] == p               # kept between re-checks
    qd["fp"]["generations"]["g2"] = _gen()                     # points complete
    assert _tick(cs, k, qd["ready"], 10 + cs.QDRANT_RETRY_S) == ([], [])
    assert set(cs.baselines()) == {"g1", "g2"}


def test_keeper_detects_a_change_without_any_ready_set_change(cs, qd):
    k = cs.Keeper()
    _tick(cs, k, qd["ready"], 0)
    qd["fp"]["generations"]["g1"] = _gen(v="tampered")
    assert _tick(cs, k, qd["ready"], cs.QDRANT_VERIFY_S - 1) == ([], [])   # not due yet
    p, _ = _tick(cs, k, qd["ready"], cs.QDRANT_VERIFY_S)
    assert p and "g1: vector_sha256 differ" in p[0] and "snapshot refused" not in p[0]


def test_keeper_snapshot_failing_for_other_reasons_warns_then_becomes_a_problem(cs, qd, monkeypatch):
    k = cs.Keeper()
    _tick(cs, k, qd["ready"], 0)
    qd["ready"]["g2"] = 2
    qd["fp"]["generations"]["g2"] = _gen()

    def down(*a, **k):
        raise cs.Fail("qdrant POST: HTTP 500")
    monkeypatch.setattr(cs, "http", down)
    p, w = _tick(cs, k, qd["ready"], 10)
    assert p == [] and "qdrant snapshot failed (retrying)" in w[0]
    p, w = _tick(cs, k, qd["ready"], 11 + cs.SNAPSHOT_STALE_S)
    assert p and "no snapshot of the current READY set" in p[0]


def test_baseline_write_failure_after_publish_is_retried_and_completed_at_boot(cs, qd, monkeypatch):
    """The snapshot is published, writing the new generation's baseline fails: the keeper
    retries (a new snapshot records it); if the container stops first, bootstrap completes
    the baseline from the published snapshot (only with identical points)."""
    k = cs.Keeper()
    _tick(cs, k, qd["ready"], 0)
    qd["ready"]["g2"] = 2
    qd["fp"]["generations"]["g2"] = _gen()
    real = cs.write_once

    def fail_once(path, data):
        raise OSError("EIO")
    monkeypatch.setattr(cs, "write_once", fail_once)
    p, w = _tick(cs, k, qd["ready"], 10)
    assert p == [] and "EIO" in w[0] and "g2" not in cs.baselines()
    assert "g2" in json.loads(cs.snapshots()[-1].with_suffix(".json").read_text())["pg_ready"]   # published
    # a) the keeper retries at the next tick
    monkeypatch.setattr(cs, "write_once", real)
    k2 = cs.Keeper()
    k2.last_digest = None
    assert _tick(cs, k2, qd["ready"], 20) == ([], []) and "g2" in cs.baselines()
    # b) bootstrap: published snapshot without the baseline -> completed if the points match
    (cs.BASELINE_ROOT / cs.COLL / "g2.json").unlink()
    qd["fp"]["generations"]["g2"] = _gen(v="tampered")
    assert cs.complete_baselines(qd["fp"], qd["ready"]) == [] and "g2" not in cs.baselines()
    qd["fp"]["generations"]["g2"] = _gen()
    assert cs.complete_baselines(qd["fp"], qd["ready"]) == ["g2"]
    assert cs.baselines()["g2"]["snapshot"] == cs.snapshots()[-1].name


def test_write_once_never_replaces(cs, tmp_path, monkeypatch):
    p = tmp_path / "b.json"
    assert cs.write_once(p, b"1") and not cs.write_once(p, b"2") and p.read_bytes() == b"1"

    def no_links(a, b):
        raise PermissionError("no hard links")
    monkeypatch.setattr(cs.os, "link", no_links)
    q = tmp_path / "c.json"
    assert cs.write_once(q, b"1") and not cs.write_once(q, b"2") and q.read_bytes() == b"1"
    assert sorted(x.name for x in tmp_path.iterdir() if x.name.startswith(".")) == []
