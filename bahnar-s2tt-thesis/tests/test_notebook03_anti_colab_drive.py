"""
Anti-Colab/Drive guards with precise allowlisting.

Scans Notebook 03 code cells and production AST in NB03 modules. Allowlists only:
  * the COLAB_DRIVE_FORBIDDEN_MARKERS definition node in asr_runtime_paths.py
  * intentional strings inside dedicated anti-residue test fixtures
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.asr_runtime_paths import (
    COLAB_DRIVE_FORBIDDEN_MARKERS,
    assert_no_colab_drive_markers,
    find_colab_drive_markers,
)

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "03_asr_baseline_training.ipynb"
SRC = REPO / "src"

NB03_PRODUCTION_MODULES = (
    "asr_runtime_paths.py",
    "asr_full_data.py",
    "asr_full_pcm.py",
    "asr_full_shards.py",
    "asr_full_train.py",
    "asr_utils.py",
    "seed.py",
)


def _nb_code() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )


def _marker_definition_lineno_range(path: Path) -> set[int]:
    """Line numbers belonging to the COLAB_DRIVE_FORBIDDEN_MARKERS assignment."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: set[int] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "COLAB_DRIVE_FORBIDDEN_MARKERS":
                start = int(getattr(node, "lineno", 0) or 0)
                end = int(getattr(node, "end_lineno", start) or start)
                lines.update(range(start, end + 1))
    return lines


def _production_hits(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.name != "asr_runtime_paths.py":
        return find_colab_drive_markers(text)
    allow = _marker_definition_lineno_range(path)
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if i in allow:
            continue
        found = find_colab_drive_markers(line)
        if found:
            hits.extend(f"L{i}:{m}" for m in found)
    return hits


class TestAntiColabDriveGuards:
    def test_notebook03_code_has_no_colab_drive_markers(self):
        assert_no_colab_drive_markers(_nb_code(), label="notebook03")

    def test_production_modules_allowlist_marker_definition_only(self):
        problems = []
        for name in NB03_PRODUCTION_MODULES:
            hits = _production_hits(SRC / name)
            if hits:
                problems.append(f"{name}: {hits}")
        assert not problems, problems

    def test_asr_runtime_paths_is_not_fully_skipped(self):
        """A Drive string outside the marker list definition must still fail."""
        # Simulate a polluted helper line.
        fake = "def sync():\n    return '/content/drive/MyDrive'\n"
        assert find_colab_drive_markers(fake)

    def test_forbidden_marker_tuple_is_nonempty(self):
        assert "google.colab" in COLAB_DRIVE_FORBIDDEN_MARKERS
        assert "require_drive" not in COLAB_DRIVE_FORBIDDEN_MARKERS  # cleaned alias, not path
