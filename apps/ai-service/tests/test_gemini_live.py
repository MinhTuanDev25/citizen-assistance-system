"""Opt-in live Gemini calls. Skipped unless GEMINI_API_KEY is in the environment.

Never prints the key. Default pytest (no marker) skips this module's tests.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or "").strip(),
    reason="GEMINI_LIVE_CONNECTIVITY=NOT_RUN_NO_KEY",
)


def test_live_gemini_placeholder_skipped_without_key():
    """If a key is present, Step 02 probe script records live results.

    This test exists so `pytest -k gemini_live` documents the skip reason
    when the key is absent. A live PASS is recorded in
    reports/step02_gemini_connectivity/live_connectivity_results.json,
    not by fabricating a response here.
    """
    assert (os.getenv("GEMINI_API_KEY") or "").strip()
