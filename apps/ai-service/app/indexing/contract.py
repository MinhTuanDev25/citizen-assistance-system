"""Strict POST /v1/index body for the offline pipeline. Mock v1 stays in app.models.index."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.indexing.pipeline import PIPELINE_VERSION

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PAGE = re.compile(r"^[1-9][0-9]{0,3}(-[1-9][0-9]{0,3})?$")
_KEY = re.compile(r"^[a-z0-9_./-]{1,512}$")


class IndexRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["index.v2"]
    xa_id: str = Field(min_length=1, max_length=64)
    document_id: uuid.UUID
    procedure_id: uuid.UUID
    procedure_version_id: uuid.UUID
    job_id: uuid.UUID
    claim_token: uuid.UUID
    generation_id: uuid.UUID
    bucket: str = Field(min_length=1, max_length=128)
    object_key: str
    checksum: str
    page_range: str | None = None
    pipeline_version: Literal["p4b.1"] = PIPELINE_VERSION

    @field_validator("checksum")
    @classmethod
    def checksum_sha(cls, value: str) -> str:
        if _SHA.fullmatch(value) is None:
            raise ValueError("checksum must be sha256 hex")
        return value

    @field_validator("object_key")
    @classmethod
    def object_key_internal(cls, value: str) -> str:
        if _KEY.fullmatch(value) is None or ".." in value or "://" in value:
            raise ValueError("object_key is not an internal key")
        return value

    @field_validator("page_range")
    @classmethod
    def page_range_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _PAGE.fullmatch(value) is None:
            raise ValueError("page_range is invalid")
        if "-" in value and int(value.split("-", 1)[0]) > int(value.split("-", 1)[1]):
            raise ValueError("page_range start must be <= end")
        return value


class IndexResponseV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["index.v2"]
    document_id: uuid.UUID
    procedure_version_id: uuid.UUID
    xa_id: str
    job_id: uuid.UUID
    generation_id: uuid.UUID
    outcome: Literal["READY", "FAILED"]
    error_code: str | None = Field(default=None, max_length=64)
    source_sha256: str | None = None
    content_sha256: str | None = None
    pages_processed: int = 0
    native_pages: int = 0
    ocr_pages: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    manifest_hash: str | None = None
    pipeline_version: str | None = None
    extraction_version: str | None = None
    ocr_version: str | None = None
    embedding_model_id: str | None = None
    embedding_revision: str | None = None
    embedding_checksum: str | None = None
    vector_dimension: int = 0
