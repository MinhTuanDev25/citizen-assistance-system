#!/usr/bin/env python3
"""Metadata fingerprint of the research data on the volume (read-only).

Lists every entry under <root> except the given excluded subtrees, as
"<relative path>\t<type>\t<size>\t<mtime_ns>" sorted, and prints its SHA-256.
File contents are not read, so the walk is fast on the network volume; any write
changes size or mtime. Used before the image switch (pod_stage.sh apply) and after
it (verify_after_switch.sh) to show the new image did not modify RQ1/RQ2 data.

  rq_fingerprint.py <root> [--exclude <path>]... [--listing-out <file.gz>] [--compare <file.gz>]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import stat
import sys


def walk(root: str, excludes: set[str]) -> list[str]:
    rows: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if full in excludes:
                continue
            keep.append(d)
            filenames.append(d)
        dirnames[:] = keep
        for name in filenames:
            full = os.path.join(dirpath, name)
            if full in excludes:
                continue
            st = os.lstat(full)
            kind = "d" if stat.S_ISDIR(st.st_mode) else "l" if stat.S_ISLNK(st.st_mode) else "f"
            size = 0 if kind == "d" else st.st_size
            rows.append(f"{os.path.relpath(full, root)}\t{kind}\t{size}\t{st.st_mtime_ns}")
    rows.sort()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--listing-out")
    ap.add_argument("--compare")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    rows = walk(root, {os.path.abspath(p) for p in a.exclude})
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    total = sum(int(r.split("\t")[2]) for r in rows)
    print(f"rq_fingerprint sha256={digest} entries={len(rows)} bytes={total} root={root}")
    if a.listing_out:
        with gzip.open(a.listing_out, "wt", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    if a.compare:
        with gzip.open(a.compare, "rt", encoding="utf-8") as f:
            before = [line.rstrip("\n") for line in f if line.strip()]
        if before == rows:
            print("rq_fingerprint compare: IDENTICAL to the saved listing")
            return 0
        old, new = set(before), set(rows)
        print(f"rq_fingerprint compare: DIFFERENT ({len(old - new)} removed/changed, {len(new - old)} added/changed)")
        for r in sorted(old - new)[:20]:
            print(f"  - {r}")
        for r in sorted(new - old)[:20]:
            print(f"  + {r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
