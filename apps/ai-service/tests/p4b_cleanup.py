"""Cleanup helpers for resources an integration test created."""

from __future__ import annotations

import uuid
import warnings


def generation_delete_body(generation_id: uuid.UUID) -> dict:
    """Delete filter for one generation. Other points in the collection stay."""
    return {"filter": {"must": [{"key": "generation_id", "match": {"value": str(generation_id)}}]}}


def run_cleanups(actions: list) -> list[BaseException]:
    """Run every cleanup. A failure does not skip the remaining actions."""
    errors: list[BaseException] = []
    for action in actions:
        try:
            action()
        except Exception as exc:
            errors.append(exc)
    return errors


class OwnedResources:
    """Register a test-owned id before the write that may persist it.

    The cleanup action must delete only that id and stay safe when the write
    never persisted. A second finish does not run the actions again.
    """

    def __init__(self) -> None:
        self._actions: list = []

    def own(self, action) -> None:
        self._actions.append(action)

    def finish(self, primary: BaseException | None = None) -> list[BaseException]:
        actions = list(reversed(self._actions))
        self._actions.clear()
        errors = run_cleanups(actions)
        if errors and primary is None:
            raise errors[0]
        if errors and primary is not None:
            detail = "; ".join(f"{type(item).__name__}: {item}" for item in errors)
            warnings.warn("CLEANUP_FAILED " + detail, RuntimeWarning, stacklevel=2)
        return errors
