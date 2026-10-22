"""Strict contract for POST /v1/retrieve."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "retrieve.v1"
MAX_RETRIEVE_BODY_BYTES = 64 * 1024
MAX_QUERY_LEN = 2000
MAX_TOP_K = 20
DEFAULT_TOP_K = 8
MAX_ALLOWED_SOURCES = 256
MAX_STATEMENTS = 30
MAX_STATEMENT_LEN = 400


class RetrieveSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    generation_id: str

    @field_validator("document_id", "generation_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        text = str(value or "").strip()
        if len(text) != 36:
            raise ValueError("must be a uuid")
        return text


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["retrieve.v1"]
    request_id: str = ""
    query: str
    xa_id: str
    procedure_id: str = ""
    procedure_version_id: str
    allowed_sources: list[RetrieveSource]
    top_k: int = DEFAULT_TOP_K
    statements: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("query is required")
        if len(text) > MAX_QUERY_LEN:
            raise ValueError("query too long")
        return text

    @field_validator("xa_id", "procedure_version_id")
    @classmethod
    def _required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("required")
        return text

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, value: int) -> int:
        if value < 1 or value > MAX_TOP_K:
            raise ValueError("top_k out of range")
        return value

    @field_validator("allowed_sources")
    @classmethod
    def _sources(cls, value: list[RetrieveSource]) -> list[RetrieveSource]:
        if len(value) > MAX_ALLOWED_SOURCES:
            raise ValueError("too many allowed_sources")
        return value

    @field_validator("statements")
    @classmethod
    def _statements(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value[:MAX_STATEMENTS]:
            text = " ".join(str(item or "").split())
            if text:
                out.append(text[:MAX_STATEMENT_LEN])
        return out

    @model_validator(mode="after")
    def _scope(self) -> "RetrieveRequest":
        return self


class RetrieveHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    generation_id: str
    procedure_version_id: str
    title: str
    document_number: str = ""
    page_start: int
    page_end: int
    excerpt: str
    excerpt_sha256: str = ""
    content_sha256: str = ""
    retrieval_score: float | None = None
    evidence_status: Literal["supporting", "related"]
    availability: Literal["available", "archived", "unavailable"] = "available"


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["retrieve.v1"] = SCHEMA_VERSION
    status: Literal[
        "ok",
        "no_relevant_evidence",
        "all_candidates_rejected",
        "retrieval_unavailable",
        "retrieval_incompatible",
    ]
    hits: list[RetrieveHit] = Field(default_factory=list)
    reason: str = ""
