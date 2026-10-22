"""Strict contract for POST /v1/index.

P4A returns a mock lifecycle result. It never writes knowledge_chunks,
embeddings, or OCR text.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "index.v1"
MAX_INDEX_BODY_BYTES = 8 * 1024
MAX_ERROR_CODE_LEN = 64
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PAGE = re.compile(r"^[1-9][0-9]{0,3}(-[1-9][0-9]{0,3})?$")
# Reserved checksum so tests can force the failure path without a real model.
FORCE_FAIL_CHECKSUM = "f" * 64


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["index.v1"]
    document_id: uuid.UUID
    xa_id: str = Field(min_length=1, max_length=64)
    domain_id: str = Field(min_length=1, max_length=64)
    checksum: str
    procedure_version_id: uuid.UUID
    relationship_type: Literal["SOURCE", "SUPERSEDES"]
    page_range: str | None = None

    @field_validator("checksum")
    @classmethod
    def checksum_sha256(cls, value: str) -> str:
        if _SHA.fullmatch(value) is None:
            raise ValueError("checksum must be sha256 hex")
        return value

    @field_validator("page_range")
    @classmethod
    def page_range_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _PAGE.fullmatch(value) is None:
            raise ValueError("page_range is invalid")
        if "-" in value:
            start_s, end_s = value.split("-", 1)
            if int(start_s) > int(end_s):
                raise ValueError("page_range start must be <= end")
        return value


class IndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["index.v1"]
    document_id: uuid.UUID
    procedure_version_id: uuid.UUID
    outcome: Literal["READY", "FAILED"]
    error_code: str | None = Field(default=None, max_length=MAX_ERROR_CODE_LEN)

    @model_validator(mode="after")
    def outcome_matches_error(self) -> "IndexResponse":
        if self.outcome == "READY" and self.error_code is not None:
            raise ValueError("READY must not include error_code")
        if self.outcome == "FAILED" and not self.error_code:
            raise ValueError("FAILED requires error_code")
        return self


def mock_index(request: IndexRequest) -> IndexResponse:
    if request.checksum == FORCE_FAIL_CHECKSUM:
        return IndexResponse(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            procedure_version_id=request.procedure_version_id,
            outcome="FAILED",
            error_code="mock_failed",
        )
    return IndexResponse(
        schema_version=SCHEMA_VERSION,
        document_id=request.document_id,
        procedure_version_id=request.procedure_version_id,
        outcome="READY",
        error_code=None,
    )
