"""Deterministic page-aware chunking. Token counts use one configured tokenizer."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


@dataclass(frozen=True)
class Chunk:
    index: int
    page_start: int
    page_end: int
    text: str
    text_sha256: str
    token_count: int
    source: str


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int
    hard_max_tokens: int
    overlap_tokens: int
    tokenizer_id: str = "unicode-word-v1"

    def hash(self) -> str:
        raw = f"{self.tokenizer_id}|{self.target_tokens}|{self.hard_max_tokens}|{self.overlap_tokens}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if self.target_tokens < 1 or self.hard_max_tokens < self.target_tokens:
            raise ValueError("chunk target exceeds hard maximum")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError("chunk overlap must be smaller than the target")


def chunk_pages(pages: list[tuple[int, str, str]], config: ChunkConfig) -> list[Chunk]:
    """pages are (page_number, normalized_text, source native|ocr)."""
    config.validate()
    units: list[tuple[str, int, str]] = []
    for number, text, source in pages:
        for token in tokenize(text):
            units.append((token, number, source))
    if not units:
        return []
    step = config.target_tokens - config.overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        end = min(start + config.target_tokens, len(units))
        window = units[start:end]
        if len(window) > config.hard_max_tokens:
            window = window[: config.hard_max_tokens]
            end = start + len(window)
        text = _join(window)
        sources = {item[2] for item in window}
        source = next(iter(sources)) if len(sources) == 1 else "mixed"
        from app.indexing.text import sha256_text

        chunks.append(
            Chunk(
                index=len(chunks),
                page_start=window[0][1],
                page_end=window[-1][1],
                text=text,
                text_sha256=sha256_text(text),
                token_count=len(window),
                source=source,
            )
        )
        if end >= len(units):
            break
        start += step
    return chunks


def manifest_hash(chunks: list[Chunk]) -> str:
    lines = [
        f"{item.index}|{item.page_start}|{item.page_end}|{item.text_sha256}|{item.token_count}|{item.source}"
        for item in chunks
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _join(window: list[tuple[str, int, str]]) -> str:
    parts: list[str] = []
    for token, _, _ in window:
        if not parts:
            parts.append(token)
            continue
        if re.fullmatch(r"\w+", token, re.UNICODE) and re.fullmatch(r"\w+", parts[-1], re.UNICODE):
            parts.append(" ")
        parts.append(token)
    return "".join(parts)
