"""Durable-storage operations of the all-in-one pod (run with /opt/cas/venv-ai/bin/python).

Layout (deploy/runpod/README.md, "Storage"):
  container disk  /var/lib/cas           pgdata, Qdrant storage, TLS (POSIX, rebuilt per container)
  network volume  /workspace/cas         pg/wal (pg_receivewal --synchronous), pg/base (verified
                                         base backups), pg/state, backups/, qdrant/snapshots, objects/
The volume is the durable copy; nothing on it is a secret or a private key.

Subcommands:
  keeper                 supervisord program (root): base backups, Qdrant snapshots when the
                         READY generation set changes, Qdrant integrity re-checks, storage/receiver
                         guard, status file
  base-backup            take, verify and publish one base backup now (root)
  qdrant-snapshot        take one snapshot + fingerprint now (refused when a READY generation
                         differs from its baseline; records the baseline of new ones)
  qdrant-fingerprint     print the fingerprint of the live collection
  qdrant-restore-verify  bootstrap: restore an empty collection from the newest snapshot, then
                         verify it against the snapshot fingerprint, Postgres and the per-generation
                         baselines (fail-closed)
  objects-verify         bootstrap: every document's object exists with its size and SHA-256
  receiver-wait SECONDS  bootstrap: wait until the WAL receiver streams synchronously
  wal-end-check FILE LSN restore: nothing replayable may follow the end of recovery (LSN) in
                         FILE, the copy of the last WAL segment recovery read
  status                 print the last status written by the keeper
Every failure prints one line "cas-storage: FAIL <reason>" and exits 1.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

W = Path(os.environ.get("CAS_WORKSPACE", "/workspace/cas"))
L = Path(os.environ.get("CAS_LOCAL", "/var/lib/cas"))
WAL = W / "pg" / "wal"
BASE = W / "pg" / "base"
STATE = W / "pg" / "state"
SNAP_ROOT = W / "qdrant" / "snapshots"
BASELINE_ROOT = W / "qdrant" / "baselines"
OBJ_ROOT = Path(os.environ.get("OBJECT_STORAGE_FILE_ROOT", str(W / "objects")))
BUCKET = os.environ.get("OBJECT_STORAGE_BUCKET", "cas-documents")
QDRANT = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
COLL = os.environ.get("QDRANT_COLLECTION", "knowledge_chunks")
RUN = Path("/run/cas")
STATUS = RUN / "storage-status.json"
BLOCKED = RUN / "storage-blocked"
READONLY_MARK = RUN / "keeper-set-read-only"
NEED_BASE = RUN / "need-base-backup"
BASE_OK = RUN / "base-backup.ok"
SLOT = "cas_vol"
DURABLE_LSN = STATE / "flushed_lsn.json"
DURABLE_LSN_S = float(os.environ.get("CAS_DURABLE_LSN_INTERVAL_SECONDS", "1"))
DURABLE_STALE_S = 30
durable_ok_at = 0.0   # last time durable_lsn_loop read and recorded the position
PGBIN = "/usr/lib/postgresql/16/bin"

BASE_KEEP = int(os.environ.get("CAS_BASE_KEEP", "2"))
BASE_HOURS = float(os.environ.get("CAS_BASEBACKUP_HOURS", "24"))
BASE_WAL_GB = float(os.environ.get("CAS_BASEBACKUP_WAL_GB", "1"))
SNAP_KEEP = int(os.environ.get("CAS_SNAPSHOT_KEEP", "3"))
QDRANT_VERIFY_S = float(os.environ.get("CAS_QDRANT_VERIFY_SECONDS", "300"))
QDRANT_RETRY_S = 60
SNAPSHOT_STALE_S = 600
QUOTA_GB = float(os.environ.get("CAS_VOLUME_QUOTA_GB", "200"))
MIN_FREE_GB = float(os.environ.get("CAS_VOLUME_MIN_FREE_GB", "10"))
RESUME_FREE_GB = MIN_FREE_GB + float(os.environ.get("CAS_VOLUME_RESUME_MARGIN_GB", "5"))
WAL_BUDGET_GB = float(os.environ.get("CAS_WAL_BUDGET_GB", "10"))
VOLUME_SCAN_S = float(os.environ.get("CAS_VOLUME_SCAN_SECONDS", "1800"))
GB = 1024 ** 3


class Fail(Exception):
    pass


def log(msg: str) -> None:
    print(f"[cas-storage] {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}", file=sys.stderr, flush=True)


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# ---- small helpers -----------------------------------------------------------------
def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def write_once(path: Path, data: bytes) -> bool:
    """Create path with data, atomically, never replacing an existing file: False if it exists."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        except OSError:
            # A filesystem without hard links: the writers (cas-keeper, backup.sh) run in the
            # one container holding the workspace lease, so checking first is enough.
            if path.exists():
                return False
            os.replace(tmp, path)
        return True
    finally:
        tmp.unlink(missing_ok=True)
        fsync_dir(path.parent)


def as_postgres(args: list[str], timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["gosu", "postgres", *args] if os.geteuid() == 0 else args
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise Fail(f"{os.path.basename(args[0])} failed: {(p.stderr or p.stdout).strip()[-400:]}")
    return p


def su_sql(sql: str, timeout: float = 30) -> str:
    """Superuser SQL over the local socket (peer auth as postgres)."""
    p = as_postgres([f"{PGBIN}/psql", "-h", "/run/postgresql", "-d", "citizen_assistance", "-XAtq", "-v", "ON_ERROR_STOP=1", "-c", sql], timeout)
    return p.stdout.strip()


def db():
    import psycopg

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise Fail("DATABASE_URL is not set")
    return psycopg.connect(url, connect_timeout=5, autocommit=True)


def http(method: str, path: str, body=None, timeout: float = 60):
    import httpx

    r = httpx.request(method, f"{QDRANT}{path}", json=body, timeout=timeout)
    if r.status_code >= 400:
        raise Fail(f"qdrant {method} {path}: HTTP {r.status_code} {r.text[:200]}")
    return r.json() if r.content else {}


def dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except FileNotFoundError:
                pass
    return total


# ---- WAL receiver --------------------------------------------------------------------
def receiver_state(conn=None) -> tuple[str | None, str | None]:
    """(state, sync_state) of the WAL receiver connection, or (None, None)."""
    sql = "select state, sync_state from pg_stat_replication where application_name = %s"
    own = conn is None
    conn = conn or db()
    try:
        row = conn.execute(sql, (SLOT,)).fetchone()
    finally:
        if own:
            conn.close()
    return (row[0], row[1]) if row else (None, None)


def receiver_wait(seconds: float) -> None:
    end = time.monotonic() + seconds
    last = (None, None)
    while time.monotonic() < end:
        try:
            last = receiver_state()
            if last == ("streaming", "sync"):
                log("WAL receiver streaming, synchronous (commits wait for the volume)")
                return
        except Exception as exc:  # noqa: BLE001 - keep waiting until the deadline
            last = (f"error: {type(exc).__name__}", None)
        time.sleep(1)
    raise Fail(f"WAL receiver not streaming synchronously after {seconds:.0f}s (last state {last})")


def lsn_int(lsn: str) -> int:
    hi, lo = lsn.split("/")
    return (int(hi, 16) << 32) | int(lo, 16)


def lsn_str(value: int) -> str:
    return f"{value >> 32:X}/{value & 0xFFFFFFFF:X}"


def durable_lsn_loop() -> None:
    """Record on the volume, about every second, the WAL position the receiver has flushed
    there (pg_stat_replication.flush_lsn of cas_vol). Every commit acknowledged before a
    record has an LSN at or below it, so a restore must recover at least that far
    (cas-pg-restore.sh). Commits acknowledged after the newest record are protected only by
    the WAL bytes themselves."""
    global durable_ok_at
    conn, sysid, last = None, None, None
    while True:
        try:
            if conn is None:
                conn = db()
                sysid = conn.execute("select system_identifier::text from pg_control_system()").fetchone()[0]
            row = conn.execute("select flush_lsn::text, sync_state from pg_stat_replication where application_name = %s", (SLOT,)).fetchone()
            if row and row[0] and row[0] != last:
                write_atomic(DURABLE_LSN, json.dumps({"system_identifier": sysid, "flushed_lsn": row[0], "sync_state": row[1],
                                                      "recorded_at": int(time.time())}).encode())
                last = row[0]
                write_atomic(RUN / "durable-lsn", row[0].encode())
            if row and row[0]:
                durable_ok_at = time.time()
        except Exception as exc:  # noqa: BLE001 - reconnect and keep recording
            log(f"durable LSN not recorded: {type(exc).__name__}: {exc}")
            try:
                conn and conn.close()
            except Exception:  # noqa: BLE001
                pass
            conn = None
            time.sleep(5)
        time.sleep(DURABLE_LSN_S)


WAL_SEG = 16 * 1024 * 1024
WAL_BLK = 8192
XLP_FIRST_IS_CONTRECORD = 0x0001
MAX_RECORD = 1020 * 1024 * 1024


def wal_end_check(data: bytes, end_lsn: int) -> str | None:
    """None if nothing replayable follows end_lsn in this WAL segment, else the reason.

    The receiver's copy of the WAL is cut either at a record end (a flush for a commit)
    or at a page boundary (a background flush in the middle of a long record). So after
    the end of a recovery that consumed everything, the segment holds either only zeros,
    or ONE incomplete record whose continuation pages carry consistent headers and that
    stops at a page boundary. Anything else (a complete record, a second record, garbage)
    means recovery stopped at damaged WAL with data after it: refused."""
    if len(data) != WAL_SEG:
        return f"segment is {len(data)} bytes, not {WAL_SEG}"
    seg_start = end_lsn - end_lsn % WAL_SEG
    magic = struct.unpack_from("<H", data, 0)[0]
    used = len(data.rstrip(b"\0"))
    pos = ((end_lsn % WAL_SEG) + 7) & ~7   # the next record starts MAXALIGNed

    def page_header(off: int) -> tuple[int, int, int, int]:
        m, info, _tli, addr, rem = struct.unpack_from("<HHIQI", data, off)
        return m, info, addr, rem

    def header_size(off: int) -> int:
        return 40 if off == 0 else 24

    if pos % WAL_BLK == 0 and pos < WAL_SEG and used > pos:
        m, info, addr, rem = page_header(pos)
        if m != magic or addr != seg_start + pos:
            return f"invalid page header at {lsn_str(seg_start + pos)} after the end of recovery"
        if info & XLP_FIRST_IS_CONTRECORD:
            return f"continuation data at {lsn_str(seg_start + pos)} where recovery found the end"
        pos += header_size(pos)
    if used <= pos:
        return None   # only zeros after the end: the WAL ended there
    if pos + 4 > WAL_SEG:
        return f"data after the end of recovery at {lsn_str(seg_start + pos)}"
    tot_len = struct.unpack_from("<I", data, pos)[0]
    if tot_len < 24 or tot_len > MAX_RECORD:
        return f"invalid record (length {tot_len}) with data after it at {lsn_str(seg_start + pos)}: corrupt WAL"
    remaining, cur = tot_len, pos
    take = min(remaining, WAL_BLK - cur % WAL_BLK)
    remaining -= take
    cur += take
    while remaining > 0:
        if cur >= WAL_SEG:
            return None   # an incomplete record running into a segment that does not exist
        if used <= cur:
            return None   # an incomplete record cut at a page boundary
        m, info, addr, rem = page_header(cur)
        if m != magic or addr != seg_start + cur:
            return f"invalid page header at {lsn_str(seg_start + cur)} inside the record at {lsn_str(seg_start + pos)}: corrupt WAL"
        if not info & XLP_FIRST_IS_CONTRECORD or rem != remaining:
            return (f"page {lsn_str(seg_start + cur)} does not continue the record at {lsn_str(seg_start + pos)} "
                    f"(remaining {remaining}, page says {rem}): corrupt WAL")
        cur += header_size(cur)
        take = min(remaining, WAL_BLK - cur % WAL_BLK)
        remaining -= take
        cur += take
    return (f"a complete record at {lsn_str(seg_start + pos)} (length {tot_len}) follows the end of recovery: "
            "recovery stopped at damaged WAL")


# ---- base backups ---------------------------------------------------------------------
def complete_bases() -> list[Path]:
    if not BASE.is_dir():
        return []
    return sorted(p for p in BASE.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / ".cas-complete").is_file())


def backup_start_wal(base: Path) -> str:
    for line in (base / "backup_label").read_text().splitlines():
        if line.startswith("START WAL LOCATION:") and "(file " in line:
            return line.split("(file ")[1].rstrip(")").strip()
    raise Fail(f"{base}/backup_label has no START WAL LOCATION")


def base_backup() -> Path:
    if os.geteuid() != 0:
        raise Fail("base-backup runs as root (it drops to postgres)")
    BASE.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    part = BASE / f".{stamp}.partial"
    final = BASE / stamp
    t0 = time.monotonic()
    try:
        as_postgres([f"{PGBIN}/pg_basebackup", "-h", "/run/postgresql", "-U", "postgres", "-D", str(part), "-Fp", "-X", "stream",
                     "-c", "fast", "--manifest-checksums=SHA256", "--no-password", "-l", f"cas-{stamp}"], timeout=3600)
        as_postgres([f"{PGBIN}/pg_verifybackup", str(part)], timeout=3600)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)   # never leave a half backup on the volume
        raise
    meta = {"created_at": stamp, "start_wal": backup_start_wal(part), "bytes": dir_bytes(part),
            "system_identifier": (STATE / "system_identifier").read_text().strip() if (STATE / "system_identifier").is_file() else None,
            "verified": "pg_verifybackup", "seconds": round(time.monotonic() - t0, 1)}
    write_atomic(part / ".cas-complete", json.dumps(meta, indent=1).encode())
    os.replace(part, final)
    fsync_dir(BASE)
    log(f"base backup {final.name} verified ({meta['bytes'] / 1e6:.1f} MB, start WAL {meta['start_wal']}, {meta['seconds']}s)")
    prune_bases()
    return final


def prune_bases() -> None:
    bases = complete_bases()
    for old in bases[:-BASE_KEEP]:
        shutil.rmtree(old, ignore_errors=True)
        log(f"pruned base backup {old.name}")
    cutoff = time.time() - 86400
    for p in BASE.glob(".*.partial"):
        if p.stat().st_mtime < cutoff:
            shutil.rmtree(p, ignore_errors=True)
            log(f"removed abandoned partial base backup {p.name}")
    kept = complete_bases()
    if kept and WAL.is_dir():
        oldest = backup_start_wal(kept[0])
        # Removes only WAL segments older than the oldest kept base backup's start.
        as_postgres([f"{PGBIN}/pg_archivecleanup", str(WAL), oldest], timeout=600)


def base_due() -> str | None:
    if NEED_BASE.exists():
        return "required at this boot (fresh cluster or restore)"
    bases = complete_bases()
    if not bases:
        return "no base backup on the volume"
    meta = json.loads((bases[-1] / ".cas-complete").read_text())
    age_h = (time.time() - (bases[-1] / ".cas-complete").stat().st_mtime) / 3600
    if age_h >= BASE_HOURS:
        return f"last base backup {age_h:.1f} h old"
    start = meta.get("start_wal", "")
    newer = sum(p.stat().st_size for p in WAL.glob("[0-9A-F]" * 24) if p.name > start) if WAL.is_dir() else 0
    if newer >= BASE_WAL_GB * GB:
        return f"{newer / GB:.1f} GB of WAL since the last base backup"
    return None


# ---- Qdrant snapshots and fingerprints -----------------------------------------------------
def _vector_bytes(vec) -> bytes:
    if isinstance(vec, dict):  # named vectors: deterministic order
        return b"".join(name.encode() + b"=" + _vector_bytes(vec[name]) for name in sorted(vec))
    return struct.pack(f"<{len(vec)}f", *vec)


def fingerprint(collection: str = COLL) -> dict:
    """Per generation: point count and SHA-256 of the sorted ids, payloads and float32 vectors."""
    groups: dict[str, list] = {}
    offset = None
    while True:
        body = {"limit": 256, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        res = http("POST", f"/collections/{collection}/points/scroll", body)["result"]
        for point in res["points"]:
            gid = str((point.get("payload") or {}).get("generation_id") or "_none")
            groups.setdefault(gid, []).append(point)
        offset = res.get("next_page_offset")
        if offset is None:
            break
    gens = {}
    for gid, points in sorted(groups.items()):
        points.sort(key=lambda p: str(p["id"]))
        hi, hp, hv = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for p in points:
            pid = str(p["id"]).encode()
            hi.update(pid + b"\n")
            hp.update(pid + b"\0" + json.dumps(p.get("payload"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
            hv.update(pid + b"\0" + _vector_bytes(p.get("vector") or []) + b"\n")
        gens[gid] = {"count": len(points), "ids_sha256": hi.hexdigest(), "payload_sha256": hp.hexdigest(), "vector_sha256": hv.hexdigest()}
    return {"collection": collection, "points": sum(g["count"] for g in gens.values()), "generations": gens}


def pg_ready(conn=None) -> dict[str, int]:
    own = conn is None
    conn = conn or db()
    try:
        rows = conn.execute("select id::text, vector_count from document_index_generations where status = 'READY' order by id").fetchall()
    finally:
        if own:
            conn.close()
    return {r[0]: int(r[1]) for r in rows}


def ready_digest(ready: dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(sorted(ready.items())).encode()).hexdigest()


def snapshots() -> list[Path]:
    d = SNAP_ROOT / COLL
    return sorted(d.glob("*.snapshot")) if d.is_dir() else []


def take_snapshot() -> Path:
    import httpx

    d = SNAP_ROOT / COLL
    d.mkdir(parents=True, exist_ok=True)
    # A snapshot never records a READY generation that is incomplete or differs from its
    # baseline: the changed points would become what the next boot restores and accepts.
    try:
        ready, stable, fp = checked_fingerprint()
    except QdrantIntegrity as exc:
        raise QdrantIntegrity(f"snapshot refused: {exc}") from None
    name = http("POST", f"/collections/{COLL}/snapshots?wait=true", timeout=600)["result"]["name"]
    stamp = utc_stamp()
    final = d / f"{stamp}.snapshot"
    tmp = d / f".{stamp}.snapshot.tmp"
    h = hashlib.sha256()
    try:
        with httpx.stream("GET", f"{QDRANT}/collections/{COLL}/snapshots/{name}", timeout=600) as r:
            if r.status_code != 200:
                raise Fail(f"snapshot download HTTP {r.status_code}")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                f.flush()
                os.fsync(f.fileno())
        # The fingerprint must describe exactly what the snapshot holds: re-read after it.
        if fingerprint() != fp:
            raise Fail("collection changed while the snapshot was taken; retry later")
        meta = {"created_at": stamp, "snapshot": final.name, "snapshot_sha256": h.hexdigest(), "qdrant_snapshot_name": name,
                "pg_ready": ready, "pg_ready_digest": ready_digest(ready), "fingerprint": fp}
        write_atomic(d / f"{stamp}.json", json.dumps(meta, indent=1).encode())
        os.replace(tmp, final)
        fsync_dir(d)
        # Baselines only after the snapshot is published, so each names a snapshot that
        # exists. If this fails, the keeper retries (its READY digest is not updated) and
        # bootstrap completes them from this snapshot (complete_baselines).
        new = record_baselines(fp, stable, final.name)
    finally:
        if tmp.exists():
            tmp.unlink()
        try:
            http("DELETE", f"/collections/{COLL}/snapshots/{name}")
        except Fail:
            pass
    log(f"qdrant snapshot {final.name}: {fp['points']} points, {len(fp['generations'])} generations"
        + (f"; baseline recorded for {len(new)} new READY generation(s)" if new else ""))
    for old in snapshots()[:-SNAP_KEEP]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
        log(f"pruned qdrant snapshot {old.name}")
    return final


def last_snapshot_meta() -> dict | None:
    for snap in reversed(snapshots()):
        meta = snap.with_suffix(".json")
        if meta.is_file():
            return json.loads(meta.read_text())
    return None


def verify_against_pg(fp: dict, ready: dict[str, int]) -> list[str]:
    problems = []
    for gid, want in ready.items():
        got = fp["generations"].get(gid)
        if got is None:
            problems.append(f"READY generation {gid} has no points in Qdrant (expected {want})")
        elif got["count"] != want:
            problems.append(f"READY generation {gid}: {got['count']} points, Postgres vector_count {want}")
    return problems


class QdrantIntegrity(Fail):
    """A READY generation is incomplete (point count differs from Postgres) or differs from
    its baseline: not ready until a later check finds it valid again."""


def checked_fingerprint() -> tuple[dict[str, int], dict[str, int], dict]:
    """(READY set, the generations READY both before and after the fingerprint, fingerprint)
    of the live collection. Raises QdrantIntegrity when one of the stable READY generations
    has a point count other than Postgres vector_count, or points other than its baseline.
    A generation retired meanwhile may be losing its points to the reaper: not checked."""
    ready = pg_ready()
    fp = fingerprint()
    after = pg_ready()
    stable = {g: n for g, n in ready.items() if after.get(g) == n}
    problems = verify_against_pg(fp, stable) + verify_against_baselines(fp, stable, baselines(), allow_new=True)
    if problems:
        raise QdrantIntegrity("; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
    return ready, stable, fp


def baselines() -> dict[str, dict]:
    """Per READY generation ever snapshotted: its fingerprint when the keeper first recorded it
    (qdrant/baselines/<collection>/<generation>.json). A READY generation never changes
    (re-indexing writes a new generation), so its baseline is written once, never replaced,
    and not pruned with the snapshots."""
    d = BASELINE_ROOT / COLL
    out = {}
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            rec = json.loads(f.read_text())
            out[rec["generation_id"]] = rec
    return out


def record_baselines(fp: dict, ready: dict[str, int], snapshot: str) -> list[str]:
    """Write the baseline of each READY generation that has none yet; returns their ids."""
    d = BASELINE_ROOT / COLL
    d.mkdir(parents=True, exist_ok=True)
    new = []
    for gid in sorted(ready):
        got = fp["generations"].get(gid)
        if got is None:
            continue
        rec = {"generation_id": gid, "fingerprint": got, "recorded_at": utc_stamp(), "snapshot": snapshot}
        if write_once(d / f"{gid}.json", json.dumps(rec, indent=1).encode()):
            new.append(gid)
    return new


def complete_baselines(fp: dict, ready: dict[str, int]) -> list[str]:
    """A snapshot published without the baselines of its new READY generations (the write
    failed or the container stopped in between): record them from the newest snapshot that
    holds each one as READY, only if the live points equal that snapshot's fingerprint.
    Returns the generations completed."""
    recorded = baselines()
    done = []
    for gid in sorted(set(ready) - set(recorded)):
        for snap in reversed(snapshots()):
            meta_path = snap.with_suffix(".json")
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            want = meta["fingerprint"]["generations"].get(gid)
            if gid not in meta.get("pg_ready", {}) or want is None:
                continue
            if want == fp["generations"].get(gid) and record_baselines(fp, {gid: ready[gid]}, snap.name):
                log(f"qdrant: baseline of READY generation {gid} completed from {snap.name}")
                done.append(gid)
            break
    return done


def verify_against_baselines(fp: dict, ready: dict[str, int], recorded: dict[str, dict], allow_new: bool = False) -> list[str]:
    """Ids, payloads and vectors of every READY generation equal its baseline. allow_new: a
    READY generation without a baseline is not a problem (the snapshot about to be taken
    records it)."""
    problems = []
    for gid in ready:
        got = fp["generations"].get(gid)
        if got is None:
            continue   # reported by verify_against_pg
        if gid not in recorded:
            if not allow_new:
                problems.append(f"READY generation {gid} has no recorded fingerprint (published after the newest snapshot): "
                                "cannot verify its points")
            continue
        want = recorded[gid]["fingerprint"]
        diff = [k for k in ("count", "ids_sha256", "payload_sha256", "vector_sha256") if got.get(k) != want.get(k)]
        if diff:
            problems.append(f"READY generation {gid}: {', '.join(diff)} differ from the fingerprint recorded in its baseline "
                            f"(first snapshot {recorded[gid].get('snapshot')})")
    return problems


def collection_exists() -> bool:
    import httpx

    r = httpx.get(f"{QDRANT}/collections/{COLL}", timeout=10)
    if r.status_code == 404:
        return False
    if r.status_code != 200:
        raise Fail(f"qdrant GET collection: HTTP {r.status_code}")
    return True


def qdrant_restore_verify() -> None:
    import httpx

    ready = pg_ready()
    if not collection_exists():
        meta = last_snapshot_meta()
        if meta is None:
            if ready:
                raise Fail(f"Qdrant is empty, Postgres has {len(ready)} READY generation(s) and the volume has no snapshot: reindex needed")
            log("qdrant: no collection and no snapshot; bootstrap creates an empty collection")
            return
        snap = SNAP_ROOT / COLL / meta["snapshot"]
        h = hashlib.sha256()
        with open(snap, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != meta["snapshot_sha256"]:
            raise Fail(f"snapshot {snap.name} SHA-256 does not match its metadata (corrupted)")
        with open(snap, "rb") as f:
            r = httpx.post(f"{QDRANT}/collections/{COLL}/snapshots/upload?priority=snapshot&wait=true",
                           files={"snapshot": (snap.name, f, "application/octet-stream")}, timeout=1800)
        if r.status_code != 200:
            raise Fail(f"snapshot upload HTTP {r.status_code} {r.text[:200]}")
        fp = fingerprint()
        if fp != meta["fingerprint"]:
            raise Fail(f"restored collection does not match the fingerprint of {snap.name}")
        log(f"qdrant restored from {snap.name}: fingerprint identical ({fp['points']} points)")
    else:
        fp = fingerprint()
    complete_baselines(fp, ready)
    problems = verify_against_pg(fp, ready) + verify_against_baselines(fp, ready, baselines())
    if problems:
        raise Fail("; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
    extra = sorted(set(fp["generations"]) - set(ready) - {"_none"})
    log(f"qdrant verified: {len(ready)} READY generation(s) match Postgres vector_count and their baselines"
        + (f"; {len(extra)} non-READY generation(s) left for the reaper" if extra else ""))


# ---- objects ------------------------------------------------------------------------------
def objects_verify() -> None:
    sys.path.insert(0, "/opt/cas/ai")
    from app.indexing.errors import IndexFailure
    from app.indexing.file_objects import read_object

    with db() as conn:
        rows = conn.execute("select id::text, storage_uri, checksum, file_size_bytes from documents order by id").fetchall()
    bad = []
    for doc_id, uri, checksum, size in rows:
        prefix = f"s3://{BUCKET}/"
        if not uri.startswith(prefix):
            bad.append(f"{doc_id}: storage_uri not in bucket {BUCKET}")
            continue
        try:
            data = read_object(str(OBJ_ROOT), BUCKET, uri[len(prefix):], 64 * 1024 * 1024)
        except IndexFailure as exc:
            bad.append(f"{doc_id}: {exc.code}")
            continue
        if len(data) != size or hashlib.sha256(data).hexdigest() != checksum:
            bad.append(f"{doc_id}: size or SHA-256 differs from the documents row")
    if bad:
        raise Fail(f"{len(bad)} of {len(rows)} document object(s) missing or corrupt: " + "; ".join(bad[:5]))
    log(f"objects verified: {len(rows)} document object(s) present with matching size and SHA-256")


# ---- keeper -----------------------------------------------------------------------------------
class Keeper:
    def __init__(self) -> None:
        self.volume_used: int | None = None
        self.components_at_scan = 0
        self.volume_scanned_at = 0.0
        self.scan_lock = threading.Lock()
        self.last_digest: str | None = None
        self.base_retry_at = 0.0
        self.qdrant_next = 0.0                       # next integrity check / refused snapshot retry
        self.qdrant_problem: str | None = None       # cleared only by a check that passes
        self.snap_pending_since: float | None = None  # READY set changed, no snapshot yet
        meta = last_snapshot_meta()
        if meta:
            self.last_digest = meta.get("pg_ready_digest")

    def scan_volume(self) -> None:
        if not self.scan_lock.acquire(blocking=False):
            return
        try:
            components = self.components()
            p = subprocess.run(["du", "-sxk", str(W.parent)], capture_output=True, text=True, timeout=1800)
            if p.returncode == 0:
                self.volume_used = int(p.stdout.split()[0]) * 1024
                self.components_at_scan = components
                self.volume_scanned_at = time.time()
            else:
                log(f"volume scan failed: {p.stderr.strip()[-200:]}")
        except Exception as exc:  # noqa: BLE001
            log(f"volume scan failed: {exc}")
        finally:
            self.scan_lock.release()

    def set_read_only(self, on: bool, reason: str) -> None:
        if on and not READONLY_MARK.exists():
            su_sql("ALTER SYSTEM SET default_transaction_read_only = on")
            su_sql("SELECT pg_reload_conf()")
            READONLY_MARK.write_text(reason)
            log(f"STORAGE: database set READ-ONLY: {reason}")
        elif not on and READONLY_MARK.exists():
            su_sql("ALTER SYSTEM RESET default_transaction_read_only")
            su_sql("SELECT pg_reload_conf()")
            READONLY_MARK.unlink()
            log("STORAGE: free space recovered; database writable again")

    @staticmethod
    def sizes() -> dict[str, int]:
        return {name: dir_bytes(path) for name, path in
                (("pg_wal", WAL), ("pg_base", BASE), ("backups", W / "backups"), ("qdrant_snapshots", SNAP_ROOT), ("objects", OBJ_ROOT))}

    def components(self) -> int:
        return sum(self.sizes().values())

    def tick(self) -> dict:
        problems, warnings = [], []
        sizes = self.sizes()
        if time.time() - self.volume_scanned_at > VOLUME_SCAN_S:
            threading.Thread(target=self.scan_volume, daemon=True).start()
        free_gb = None
        if self.volume_used is None:
            warnings.append("volume usage not measured yet (scan running)")
        else:
            # Between two scans, the growth of this system's own data is added (writes
            # to the research data in between are only seen at the next scan).
            grown = max(0, sum(sizes.values()) - self.components_at_scan)
            free_gb = QUOTA_GB - (self.volume_used + grown) / GB
            if free_gb < MIN_FREE_GB:
                problems.append(f"volume free {free_gb:.1f} GB < {MIN_FREE_GB} GB (quota {QUOTA_GB} GB)")
            elif free_gb < RESUME_FREE_GB:
                warnings.append(f"volume free {free_gb:.1f} GB is low")
        if sizes["pg_wal"] > WAL_BUDGET_GB * GB:
            warnings.append(f"WAL on the volume {sizes['pg_wal'] / GB:.1f} GB > budget {WAL_BUDGET_GB} GB")
        try:
            with db() as conn:
                state = receiver_state(conn)
                slot = conn.execute("select wal_status from pg_replication_slots where slot_name = %s", (SLOT,)).fetchone()
                ready = pg_ready(conn)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"database unreachable: {type(exc).__name__}")
            state, slot, ready = (None, None), None, None
        if state != ("streaming", "sync"):
            problems.append(f"WAL receiver not streaming synchronously ({state}): commits wait")
        if time.time() - durable_ok_at > DURABLE_STALE_S:
            problems.append(f"durable LSN not recorded on the volume for {time.time() - durable_ok_at:.0f}s")
        if slot and slot[0] == "lost":
            problems.append("replication slot lost: a new base backup and slot are needed (cas-pg-rebase)")
        blocked = any(p.startswith("volume free") for p in problems)
        try:
            if blocked:
                self.set_read_only(True, problems[0])
            elif free_gb is not None and free_gb >= RESUME_FREE_GB:
                self.set_read_only(False, "")
        except Fail as exc:
            problems.append(f"cannot change read-only mode: {exc}")
        # Base backup when due (not while the volume is too full or the receiver is down).
        if not blocked and state == ("streaming", "sync") and time.time() >= self.base_retry_at:
            why = base_due()
            if why:
                try:
                    log(f"base backup due: {why}")
                    base_backup()
                    NEED_BASE.unlink(missing_ok=True)
                    BASE_OK.write_text(Path("/run/cas/boot_id").read_text() if Path("/run/cas/boot_id").exists() else "")
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"base backup failed: {exc}")
                    self.base_retry_at = time.time() + 60   # back off; the failed attempt left nothing behind
            elif not BASE_OK.exists() and not NEED_BASE.exists():
                BASE_OK.write_text(Path("/run/cas/boot_id").read_text() if Path("/run/cas/boot_id").exists() else "")
        if ready is not None and Path("/run/cas/bootstrap.ok").exists() and not blocked:
            self.qdrant_tick(ready, problems, warnings)
        elif self.qdrant_problem:
            problems.append(self.qdrant_problem)   # never dropped without a passing check
        status = {"ok": not problems, "problems": problems, "warnings": warnings, "updated_at": int(time.time()),
                  "receiver": list(state) if state else None, "volume_free_gb": None if free_gb is None else round(free_gb, 1),
                  "volume_scanned_at": int(self.volume_scanned_at) or None, "sizes_bytes": sizes,
                  "base_backups": [p.name for p in complete_bases()], "snapshots": [p.name for p in snapshots()]}
        write_atomic(STATUS, json.dumps(status, indent=1).encode())
        if problems:
            BLOCKED.write_text("; ".join(problems))
        else:
            BLOCKED.unlink(missing_ok=True)
        for line in problems + warnings:
            log(("PROBLEM " if line in problems else "WARN ") + line)
        return status

    def qdrant_tick(self, ready: dict[str, int], problems: list[str], warnings: list[str], now: float | None = None) -> None:
        """Snapshot when the READY set changed; otherwise re-check the live collection every
        QDRANT_VERIFY_S. An incomplete or changed READY generation is a problem (not ready)
        until a later check passes, re-checked every QDRANT_RETRY_S (a check reads the whole
        collection). A snapshot failing for other reasons is retried every tick, and becomes
        a problem after SNAPSHOT_STALE_S without a snapshot of the current READY set."""
        now = time.time() if now is None else now
        digest = ready_digest(ready)
        changed = digest != self.last_digest
        if not changed:
            self.snap_pending_since = None
        elif self.snap_pending_since is None:
            self.snap_pending_since = now
        if now >= self.qdrant_next or (changed and self.qdrant_problem is None):
            try:
                if changed:
                    take_snapshot()
                    self.last_digest = digest
                    self.snap_pending_since = None
                else:
                    checked_fingerprint()
                self.qdrant_problem = None
                self.qdrant_next = now + QDRANT_VERIFY_S
            except QdrantIntegrity as exc:
                self.qdrant_problem = f"qdrant {exc}"
                self.qdrant_next = now + QDRANT_RETRY_S
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"qdrant {'snapshot' if changed else 'check'} failed (retrying): {exc}")
        if self.qdrant_problem:
            problems.append(self.qdrant_problem)
        if self.snap_pending_since is not None and now - self.snap_pending_since > SNAPSHOT_STALE_S:
            problems.append(f"qdrant: no snapshot of the current READY set for {now - self.snap_pending_since:.0f}s")

    def run(self) -> None:
        global durable_ok_at
        durable_ok_at = time.time()   # grace period for the first record
        threading.Thread(target=durable_lsn_loop, daemon=True, name="durable-lsn").start()
        log(f"keeper started: quota {QUOTA_GB} GB, min free {MIN_FREE_GB} GB, base backup every {BASE_HOURS} h, keep {BASE_KEEP} bases / {SNAP_KEEP} snapshots")
        # The volume scan (du of the whole volume, research data included) can take
        # minutes on the network volume: it runs in a thread, never in the tick.
        while True:
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the keeper must keep running
                log(f"PROBLEM keeper tick failed: {type(exc).__name__}: {exc}")
                BLOCKED.write_text(f"keeper tick failed: {exc}")
            time.sleep(float(os.environ.get("CAS_KEEPER_INTERVAL_SECONDS", "10")))


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        if cmd == "keeper":
            Keeper().run()
        elif cmd == "base-backup":
            print(base_backup())
        elif cmd == "qdrant-snapshot":
            print(take_snapshot())
        elif cmd == "qdrant-fingerprint":
            print(json.dumps(fingerprint(), indent=1))
        elif cmd == "qdrant-restore-verify":
            qdrant_restore_verify()
        elif cmd == "objects-verify":
            objects_verify()
        elif cmd == "receiver-wait":
            receiver_wait(float(argv[2]) if len(argv) > 2 else 120)
        elif cmd == "wal-end-check":
            problem = wal_end_check(Path(argv[2]).read_bytes(), lsn_int(argv[3]))
            if problem:
                raise Fail(problem)
            print("ok: nothing replayable after the end of recovery")
        elif cmd == "status":
            print(STATUS.read_text() if STATUS.exists() else '{"ok": false, "problems": ["keeper has not written a status yet"]}')
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except Fail as exc:
        print(f"cas-storage: FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
