"""Stable indexing error codes. None of these carry document text."""

from __future__ import annotations


class IndexFailure(Exception):
    def __init__(self, code: str) -> None:
        if not code or len(code) > 64:
            code = "worker_failed"
        super().__init__(code)
        self.code = code
