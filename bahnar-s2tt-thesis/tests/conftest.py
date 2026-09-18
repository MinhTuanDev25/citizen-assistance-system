"""Shared pytest fixtures for Notebook 03 / ASR full-pipeline tests."""
from __future__ import annotations

import os

import pytest

from src.asr_full_train import ENV_DURABLE_CHECKPOINT_BUDGET_BYTES


@pytest.fixture(autouse=True)
def _default_durable_checkpoint_budget_env(monkeypatch: pytest.MonkeyPatch):
    """
    Sync stages require an explicit budget. Tests that are not about missing
    budget get a generous default via env so call sites stay focused.
    """
    monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10 ** 12))
    yield
