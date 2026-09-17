"""Small helper to apply exact-string patches to notebook code cells.

Usage: python tools/nbpatch.py <notebook> <patchfile.json>

The patch file is a list of {"cell": int, "old": str, "new": str, "count": int?}
entries. Every patch must match exactly ``count`` times (default 1) or the run
aborts without writing, so a silent no-op edit is impossible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def apply(nb_path: Path, patches: list[dict]) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for i, patch in enumerate(patches):
        idx = int(patch["cell"])
        cell = cells[idx]
        if cell["cell_type"] != "code":
            raise SystemExit(f"patch {i}: cell {idx} is not code")
        source = "".join(cell["source"])
        old, new = patch["old"], patch["new"]
        want = int(patch.get("count", 1))
        got = source.count(old)
        if got != want:
            raise SystemExit(
                f"patch {i}: cell {idx} matched {got}x, expected {want}x\n  old={old[:160]!r}"
            )
        source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"applied {len(patches)} patch(es) to {nb_path.name}")


if __name__ == "__main__":
    apply(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
