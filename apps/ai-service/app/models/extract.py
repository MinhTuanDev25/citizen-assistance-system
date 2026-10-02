"""Strict Pydantic contract for POST /v1/extract.

Every model uses `extra="forbid"` so unknown fields are rejected rather than
silently ignored. The LLM may only ever pick a procedure_code from the
candidates the caller supplied and extract values for slots that are declared
on that candidate — it cannot invent procedures, slots, or free-text replies.
"""

from __future__ import annotations

import math
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "extract.v1"

# Hard structural ceilings — independent of provider — to keep prompts bounded
# and reject pathological payloads before they ever reach a provider adapter.
MAX_MESSAGE_CODEPOINTS = 4000
MAX_CANDIDATES = 20
MAX_SLOTS_PER_CANDIDATE = 30
MAX_INTENT_EXAMPLES = 20
MAX_INTENT_EXAMPLE_LEN = 300
MAX_ENUM_VALUES = 30
MAX_ENUM_VALUE_LEN = 80
MAX_QUESTION_LEN = 500
MAX_SLOT_KEY_LEN = 64
MAX_STRING_VALUE_LEN = 500
MAX_ALTERNATIVES = 5
MAX_EVIDENCE_LEN = 300
MAX_ABSTAIN_LEN = 300
MAX_PROVIDER_LEN = 32
MAX_MODEL_LEN = 80
MAX_ALLOWED_SLOT_KEYS = 30
MAX_SLOT_STATE = 30
MAX_REQUEST_BODY_BYTES = 256 * 1024

SLOT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
MODEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,79}$")
VALID_PROVIDERS = ("mock", "openai", "gemini")

SlotType = Literal["string", "boolean", "number", "enum"]
SlotOperation = Literal["set", "correct"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _is_finite_unit_interval(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("confidence must be a number")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError("confidence must be finite")
    if f < 0.0 or f > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return f


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SlotSpec(StrictModel):
    type: SlotType
    question: str = ""
    enum_values: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_len(cls, v: str) -> str:
        if len(v) > MAX_QUESTION_LEN:
            raise ValueError(f"question exceeds max of {MAX_QUESTION_LEN}")
        return v

    @field_validator("enum_values")
    @classmethod
    def _cap_enum_values(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_ENUM_VALUES:
            raise ValueError(f"enum_values exceeds max of {MAX_ENUM_VALUES}")
        for item in v:
            if len(item) > MAX_ENUM_VALUE_LEN:
                raise ValueError(f"enum value exceeds max of {MAX_ENUM_VALUE_LEN}")
        return v

    @model_validator(mode="after")
    def _enum_requires_values(self) -> "SlotSpec":
        if self.type == "enum" and not self.enum_values:
            raise ValueError("enum slot must declare enum_values")
        return self


class Candidate(StrictModel):
    procedure_code: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    intent_examples: list[str] = Field(default_factory=list)
    slots: dict[str, SlotSpec] = Field(default_factory=dict)

    @field_validator("intent_examples")
    @classmethod
    def _cap_examples(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_INTENT_EXAMPLES:
            raise ValueError(f"intent_examples exceeds max of {MAX_INTENT_EXAMPLES}")
        for item in v:
            if len(item) > MAX_INTENT_EXAMPLE_LEN:
                raise ValueError(f"intent example exceeds max of {MAX_INTENT_EXAMPLE_LEN}")
        return v

    @field_validator("slots")
    @classmethod
    def _cap_slots(cls, v: dict[str, SlotSpec]) -> dict[str, SlotSpec]:
        if len(v) > MAX_SLOTS_PER_CANDIDATE:
            raise ValueError(f"slots exceeds max of {MAX_SLOTS_PER_CANDIDATE}")
        for key in v:
            if not SLOT_KEY_RE.fullmatch(key):
                raise ValueError("invalid slot key")
        return v


class SlotStateEntry(StrictModel):
    value: str | float | bool | None = None
    status: Literal["MISSING", "KNOWN", "CONFIRMED"] = "MISSING"


class PinnedContext(StrictModel):
    procedure_code: str = Field(min_length=1, max_length=200)
    allowed_slot_keys: list[str] = Field(default_factory=list)
    slot_state: dict[str, SlotStateEntry] = Field(default_factory=dict)

    @field_validator("allowed_slot_keys")
    @classmethod
    def _cap_allowed(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_ALLOWED_SLOT_KEYS:
            raise ValueError(f"allowed_slot_keys exceeds max of {MAX_ALLOWED_SLOT_KEYS}")
        for key in v:
            if not SLOT_KEY_RE.fullmatch(key):
                raise ValueError("invalid allowed slot key")
        return v

    @field_validator("slot_state")
    @classmethod
    def _cap_state(cls, v: dict[str, SlotStateEntry]) -> dict[str, SlotStateEntry]:
        if len(v) > MAX_SLOT_STATE:
            raise ValueError(f"slot_state exceeds max of {MAX_SLOT_STATE}")
        for key, entry in v.items():
            if not SLOT_KEY_RE.fullmatch(key):
                raise ValueError("invalid slot_state key")
            if isinstance(entry.value, str) and len(entry.value) > MAX_STRING_VALUE_LEN:
                raise ValueError(f"slot_state value exceeds max of {MAX_STRING_VALUE_LEN}")
        return v


class ExtractRequest(StrictModel):
    schema_version: Literal["extract.v1"] = SCHEMA_VERSION
    request_id: str
    message: str = Field(min_length=1)
    candidates: list[Candidate] = Field(default_factory=list)
    pinned_context: PinnedContext | None = None

    @field_validator("request_id")
    @classmethod
    def _request_id_is_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(str(v))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("request_id must be a UUID") from exc
        return v

    @field_validator("message")
    @classmethod
    def _message_bounds(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be empty")
        if len(v) > MAX_MESSAGE_CODEPOINTS:
            raise ValueError(f"message exceeds max of {MAX_MESSAGE_CODEPOINTS} code points")
        return v

    @field_validator("candidates")
    @classmethod
    def _candidates_bounds(cls, v: list[Candidate]) -> list[Candidate]:
        if len(v) > MAX_CANDIDATES:
            raise ValueError(f"candidates exceeds max of {MAX_CANDIDATES}")
        seen: set[str] = set()
        for c in v:
            if c.procedure_code in seen:
                raise ValueError(f"duplicate candidate procedure_code {c.procedure_code!r}")
            seen.add(c.procedure_code)
        return v

    @model_validator(mode="after")
    def _pinned_must_be_candidate_if_present(self) -> "ExtractRequest":
        if self.pinned_context is not None and self.candidates:
            codes = {c.procedure_code for c in self.candidates}
            if self.pinned_context.procedure_code not in codes:
                raise ValueError("pinned_context.procedure_code must be one of candidates")
        return self


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class Alternative(StrictModel):
    procedure_code: str = Field(min_length=1, max_length=200)
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _confidence_bounds(cls, v: float) -> float:
        return _is_finite_unit_interval(v)


class IntentResult(StrictModel):
    procedure_code: str | None = None
    confidence: float = 0.0
    alternatives: list[Alternative] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _confidence_bounds(cls, v: float) -> float:
        return _is_finite_unit_interval(v)

    @field_validator("alternatives")
    @classmethod
    def _cap_alternatives(cls, v: list[Alternative]) -> list[Alternative]:
        if len(v) > MAX_ALTERNATIVES:
            raise ValueError(f"alternatives exceeds max of {MAX_ALTERNATIVES}")
        return v

    @model_validator(mode="after")
    def _null_code_means_no_confidence_claim(self) -> "IntentResult":
        if self.procedure_code is None and self.confidence not in (0.0,):
            # Model must not claim confidence for a null pick.
            raise ValueError("confidence must be 0 when procedure_code is null")
        return self


class SlotResult(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    value: str | float | bool
    confidence: float
    evidence: str = Field(default="", max_length=MAX_EVIDENCE_LEN)
    operation: SlotOperation = "set"

    @field_validator("confidence")
    @classmethod
    def _confidence_bounds(cls, v: float) -> float:
        return _is_finite_unit_interval(v)

    @field_validator("value")
    @classmethod
    def _string_value_len(cls, v):
        if isinstance(v, str) and len(v) > MAX_STRING_VALUE_LEN:
            raise ValueError(f"string slot value exceeds max of {MAX_STRING_VALUE_LEN}")
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("slot value must be finite")
        return v


class ExtractResponse(StrictModel):
    schema_version: Literal["extract.v1"] = SCHEMA_VERSION
    intent: IntentResult | None = None
    slots_for_procedure_code: str | None = None
    slots: list[SlotResult] = Field(default_factory=list)
    abstain_reason: str | None = None
    provider: str = Field(min_length=1, max_length=MAX_PROVIDER_LEN)
    model: str = Field(min_length=1, max_length=MAX_MODEL_LEN)

    @field_validator("provider")
    @classmethod
    def _provider_allowlist(cls, v: str) -> str:
        if v not in VALID_PROVIDERS:
            raise ValueError("invalid provider")
        return v

    @field_validator("model")
    @classmethod
    def _model_shape(cls, v: str) -> str:
        if not MODEL_RE.fullmatch(v):
            raise ValueError("invalid model")
        return v

    @field_validator("abstain_reason")
    @classmethod
    def _abstain_len(cls, v: str | None) -> str | None:
        if v is not None and len(v) > MAX_ABSTAIN_LEN:
            raise ValueError(f"abstain_reason exceeds max of {MAX_ABSTAIN_LEN}")
        return v

    @field_validator("slots")
    @classmethod
    def _no_duplicate_slot_keys(cls, v: list[SlotResult]) -> list[SlotResult]:
        seen: set[str] = set()
        for s in v:
            if s.key in seen:
                raise ValueError(f"duplicate slot key {s.key!r} in response")
            seen.add(s.key)
        return v

    @model_validator(mode="after")
    def _slots_require_target(self) -> "ExtractResponse":
        if self.slots and not self.slots_for_procedure_code:
            raise ValueError("slots present but slots_for_procedure_code is missing")
        return self
