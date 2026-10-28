"""Strict Pydantic contract for POST /v1/draft_definition.

The FastAPI service never treats a caller-supplied verified flag as proof that
an excerpt is an official government document. Go resolves excerpts from
PostgreSQL; this service only checks identity hashes, quotation membership,
and the Definition-shaped JSON the model emits.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

SCHEMA_VERSION = "draft_definition.v1"
SLOT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

MAX_EXCERPTS = 80
MAX_EXCERPT_CHARS = 4000
MAX_TOTAL_SOURCE_CHARS = 40000
MAX_NAME_LEN = 300
MAX_QUESTION_LEN = 500
MAX_SLOTS = 30
MAX_REQUIRED = 30
MAX_CONDITIONAL = 20
MAX_INTENT_EXAMPLES = 20
MAX_INTENT_EXAMPLE_LEN = 300
MAX_CHECKLIST = 40
MAX_CHECKLIST_ITEM_LEN = 400
MAX_NOTES = 20
MAX_NOTE_LEN = 500
MAX_EVIDENCE = 80
MAX_QUOTE_LEN = 800
MAX_WARNING_LEN = 400
MAX_WARNINGS = 40
MAX_MISSING = 40
MAX_ENUM_VALUES = 30
MAX_ENUM_VALUE_LEN = 80
MAX_REQUEST_BODY_BYTES = 256 * 1024
MAX_RESPONSE_CHARS = 200_000

SlotType = Literal["string", "boolean", "number", "enum"]
EvidenceKind = Literal["slot", "required_slot", "conditional", "checklist", "guidance", "citation"]
Readiness = Literal["needs_review", "invalid", "schema_valid_unapproved"]
CitationSourceType = Literal["ai_draft_unapproved"]
IssueSeverity = Literal["error", "warning"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _finite_confidence(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("confidence must be a number")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError("confidence must be finite")
    if f < 0.0 or f > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return f


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SourceExcerpt(StrictModel):
    excerpt_id: str
    document_id: str
    generation_id: str
    page_start: int
    page_end: int
    text_sha256: str
    content: str

    @field_validator("excerpt_id", "document_id", "generation_id")
    @classmethod
    def _uuid_id(cls, v: str) -> str:
        if UUID_RE.fullmatch(v or "") is None:
            raise ValueError("identifier must be a UUID")
        return v.lower()

    @field_validator("text_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if SHA256_RE.fullmatch(v or "") is None:
            raise ValueError("text_sha256 must be 64 lowercase hex chars")
        return v

    @field_validator("page_start", "page_end")
    @classmethod
    def _page(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError("page must be in [1, 10000]")
        return v

    @field_validator("content")
    @classmethod
    def _content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("excerpt content is required")
        if len(v) > MAX_EXCERPT_CHARS:
            raise ValueError(f"excerpt exceeds {MAX_EXCERPT_CHARS} characters")
        return v

    @model_validator(mode="after")
    def _range_and_hash(self) -> SourceExcerpt:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        digest = content_sha256(self.content)
        if digest != self.text_sha256:
            raise ValueError("text_sha256 does not match excerpt content")
        return self


MAX_BATCHES = 100
MAX_KNOWN_SLOTS = 30


class KnownSlot(StrictModel):
    """A slot an earlier batch already proposed; reusing its key lets Go merge it."""

    key: str
    type: SlotType
    question: str

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        if SLOT_KEY_RE.fullmatch(v or "") is None:
            raise ValueError("known slot key is invalid")
        return v

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        if len(v) > MAX_QUESTION_LEN:
            raise ValueError("known slot question is too long")
        return v


class DraftContext(StrictModel):
    """Set when the excerpts are one batch of a longer procedure section."""

    partial: bool = False
    section_title: str = ""
    batch_index: int = 1
    batch_count: int = 1
    known_slots: list[KnownSlot] = Field(default_factory=list)

    @field_validator("section_title")
    @classmethod
    def _title(cls, v: str) -> str:
        if len(v) > MAX_NAME_LEN:
            raise ValueError("section_title is too long")
        return v.strip()

    @field_validator("known_slots")
    @classmethod
    def _known(cls, v: list[KnownSlot]) -> list[KnownSlot]:
        if len(v) > MAX_KNOWN_SLOTS:
            raise ValueError("too many known_slots")
        return v

    @model_validator(mode="after")
    def _batch(self) -> DraftContext:
        if not (1 <= self.batch_index <= self.batch_count <= MAX_BATCHES):
            raise ValueError("batch_index/batch_count are out of range")
        return self


class DraftRequest(StrictModel):
    schema_version: Literal["draft_definition.v1"] = SCHEMA_VERSION
    request_id: uuid.UUID
    procedure_code: str
    procedure_name: str
    xa_id: str
    document_id: str
    generation_id: str
    source_page_start: int
    source_page_end: int
    excerpts: list[SourceExcerpt]
    context: DraftContext | None = None

    @property
    def partial(self) -> bool:
        return bool(self.context and self.context.partial)

    @field_validator("procedure_code")
    @classmethod
    def _code(cls, v: str) -> str:
        if SLOT_KEY_RE.fullmatch(v or "") is None:
            raise ValueError("procedure_code is invalid")
        return v

    @field_validator("procedure_name", "xa_id")
    @classmethod
    def _name(cls, v: str) -> str:
        text = (v or "").strip()
        if not text or len(text) > MAX_NAME_LEN:
            raise ValueError("name/xa_id is required and bounded")
        return text

    @field_validator("document_id", "generation_id")
    @classmethod
    def _ids(cls, v: str) -> str:
        if UUID_RE.fullmatch(v or "") is None:
            raise ValueError("identifier must be a UUID")
        return v.lower()

    @field_validator("source_page_start", "source_page_end")
    @classmethod
    def _pages(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError("page must be in [1, 10000]")
        return v

    @field_validator("excerpts")
    @classmethod
    def _excerpts(cls, v: list[SourceExcerpt]) -> list[SourceExcerpt]:
        if not v:
            raise ValueError("at least one excerpt is required")
        if len(v) > MAX_EXCERPTS:
            raise ValueError(f"excerpts exceed {MAX_EXCERPTS}")
        total = sum(len(item.content) for item in v)
        if total > MAX_TOTAL_SOURCE_CHARS:
            raise ValueError("combined excerpt text exceeds the drafting budget")
        ids = [item.excerpt_id for item in v]
        if len(set(ids)) != len(ids):
            raise ValueError("excerpt_id values must be unique")
        return v

    @model_validator(mode="after")
    def _scope(self) -> DraftRequest:
        if self.source_page_end < self.source_page_start:
            raise ValueError("source_page_end must be >= source_page_start")
        for item in self.excerpts:
            if item.document_id != self.document_id:
                raise ValueError("excerpt document_id does not match request")
            if item.generation_id != self.generation_id:
                raise ValueError("excerpt generation_id does not match request")
            if item.page_start < self.source_page_start or item.page_end > self.source_page_end:
                raise ValueError("excerpt pages are outside the approved source scope")
        return self


class SlotProposal(StrictModel):
    key: str
    type: SlotType
    question: str
    enum_values: list[str] = Field(default_factory=list)
    # Citizen-facing text for each enum value, same order (e.g. "Trực tiếp").
    enum_labels: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        if SLOT_KEY_RE.fullmatch(v or "") is None:
            raise ValueError("slot key is invalid")
        return v

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_QUESTION_LEN:
            raise ValueError("slot question is required and bounded")
        return v

    @field_validator("enum_values")
    @classmethod
    def _enums(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_ENUM_VALUES:
            raise ValueError("too many enum_values")
        for item in v:
            if not item or len(item) > MAX_ENUM_VALUE_LEN:
                raise ValueError("enum value is empty or too long")
        return v

    @model_validator(mode="after")
    def _enum_required(self) -> SlotProposal:
        if self.type == "enum" and not self.enum_values:
            raise ValueError("enum slots require enum_values")
        if self.type != "enum" and self.enum_values:
            raise ValueError("enum_values are only allowed on enum slots")
        if self.enum_labels:
            if len(self.enum_labels) != len(self.enum_values):
                raise ValueError("enum_labels must match enum_values one to one")
            for label in self.enum_labels:
                if not label.strip() or len(label) > MAX_ENUM_VALUE_LEN:
                    raise ValueError("enum label is empty or too long")
        return self


class WhenClause(StrictModel):
    slot: str
    equals: str | int | float | bool

    @field_validator("slot")
    @classmethod
    def _slot(cls, v: str) -> str:
        if SLOT_KEY_RE.fullmatch(v or "") is None:
            raise ValueError("when.slot is invalid")
        return v

    @field_validator("equals")
    @classmethod
    def _equals(cls, v: str | int | float | bool) -> str | int | float | bool:
        if isinstance(v, str) and len(v) > MAX_ENUM_VALUE_LEN:
            raise ValueError("equals is too long")
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("equals must be finite")
        return v


class ConditionalProposal(StrictModel):
    when: WhenClause
    require: list[str]

    @field_validator("require")
    @classmethod
    def _require(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("conditional require cannot be empty")
        if len(v) > MAX_REQUIRED:
            raise ValueError("too many required keys")
        for key in v:
            if SLOT_KEY_RE.fullmatch(key or "") is None:
                raise ValueError("required slot key is invalid")
        return v


class GuidanceRule(StrictModel):
    """A checklist item or note that applies only to one answer of a question.

    when.slot may name a question proposed by another batch of the same
    section, so it is not checked against this batch's slots; the server checks
    it against the merged Definition.
    """

    when: WhenClause
    item: str

    @field_validator("item")
    @classmethod
    def _item(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_NOTE_LEN:
            raise ValueError("conditional guidance item is empty or too long")
        return v


class GuidanceProposal(StrictModel):
    summary: str
    checklist: list[str]
    where_to_submit: str = ""
    notes: list[str] = Field(default_factory=list)
    fee_hint: str = ""
    processing_time_hint: str = ""
    conditional_checklist: list[GuidanceRule] = Field(default_factory=list)
    conditional_notes: list[GuidanceRule] = Field(default_factory=list)

    @field_validator("conditional_checklist", "conditional_notes")
    @classmethod
    def _rules(cls, v: list[GuidanceRule]) -> list[GuidanceRule]:
        if len(v) > MAX_CHECKLIST:
            raise ValueError("too many conditional guidance items")
        return v

    @field_validator("summary", "where_to_submit", "fee_hint", "processing_time_hint")
    @classmethod
    def _text(cls, v: str) -> str:
        if len(v) > MAX_NOTE_LEN:
            raise ValueError("guidance text is too long")
        return v

    @field_validator("checklist")
    @classmethod
    def _checklist(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_CHECKLIST:
            raise ValueError("too many checklist items")
        for item in v:
            if not item.strip() or len(item) > MAX_CHECKLIST_ITEM_LEN:
                raise ValueError("checklist item is empty or too long")
        return v

    @field_validator("notes")
    @classmethod
    def _notes(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_NOTES:
            raise ValueError("too many notes")
        for item in v:
            if len(item) > MAX_NOTE_LEN:
                raise ValueError("note is too long")
        return v


class CitationProposal(StrictModel):
    doc_id: str
    title: str
    source_type: CitationSourceType = "ai_draft_unapproved"

    @field_validator("doc_id", "title")
    @classmethod
    def _cite(cls, v: str) -> str:
        text = (v or "").strip()
        if not text or len(text) > MAX_NAME_LEN:
            raise ValueError("citation fields are required and bounded")
        return text


def _rule(rule: GuidanceRule) -> dict[str, Any]:
    return {"when": {"slot": rule.when.slot, "equals": rule.when.equals}, "item": rule.item}


class DefinitionProposal(StrictModel):
    procedure_code: str
    name: str
    intent_examples: list[str] = Field(default_factory=list)
    slots: list[SlotProposal]
    required_slots: list[str] = Field(default_factory=list)
    conditional_slots: list[ConditionalProposal] = Field(default_factory=list)
    guidance: GuidanceProposal
    citations: list[CitationProposal] = Field(default_factory=list)

    @field_validator("procedure_code")
    @classmethod
    def _code(cls, v: str) -> str:
        if SLOT_KEY_RE.fullmatch(v or "") is None:
            raise ValueError("procedure_code is invalid")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_NAME_LEN:
            raise ValueError("name is required")
        return v

    @field_validator("intent_examples")
    @classmethod
    def _examples(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_INTENT_EXAMPLES:
            raise ValueError("too many intent_examples")
        for item in v:
            if not item.strip() or len(item) > MAX_INTENT_EXAMPLE_LEN:
                raise ValueError("intent example is empty or too long")
        return v

    @field_validator("slots")
    @classmethod
    def _slots(cls, v: list[SlotProposal], info: ValidationInfo) -> list[SlotProposal]:
        # One batch of a long section may hold no citizen question (e.g. only fees);
        # the merged Definition is still validated by Go.
        partial = bool((info.context or {}).get("partial"))
        if not v and not partial:
            raise ValueError("at least one slot is required")
        if len(v) > MAX_SLOTS:
            raise ValueError("too many slots")
        keys = [item.key for item in v]
        if len(set(keys)) != len(keys):
            raise ValueError("slot keys must be unique")
        return v

    @field_validator("required_slots")
    @classmethod
    def _required(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_REQUIRED:
            raise ValueError("too many required_slots")
        for key in v:
            if SLOT_KEY_RE.fullmatch(key or "") is None:
                raise ValueError("required slot key is invalid")
        return v

    @field_validator("conditional_slots")
    @classmethod
    def _cond(cls, v: list[ConditionalProposal]) -> list[ConditionalProposal]:
        if len(v) > MAX_CONDITIONAL:
            raise ValueError("too many conditional_slots")
        return v

    @field_validator("citations")
    @classmethod
    def _citations(cls, v: list[CitationProposal]) -> list[CitationProposal]:
        if len(v) > 10:
            raise ValueError("too many citations")
        return v

    @model_validator(mode="after")
    def _refs(self) -> DefinitionProposal:
        keys = {item.key for item in self.slots}
        for key in self.required_slots:
            if key not in keys:
                raise ValueError(f"required_slots references unknown slot {key}")
        for rule in self.conditional_slots:
            if rule.when.slot not in keys:
                raise ValueError("conditional when.slot is unknown")
            for key in rule.require:
                if key not in keys:
                    raise ValueError(f"conditional require references unknown slot {key}")
        return self

    def to_go_definition(self) -> dict[str, Any]:
        slots = {
            item.key: {
                "type": item.type,
                "question": item.question,
                **({"enum_values": item.enum_values} if item.enum_values else {}),
                **({"enum_labels": item.enum_labels} if item.enum_labels else {}),
            }
            for item in self.slots
        }
        return {
            "procedure_code": self.procedure_code,
            "name": self.name,
            "intent_examples": list(self.intent_examples),
            "slots": slots,
            "required_slots": list(self.required_slots),
            "conditional_slots": [
                {
                    "when": {"slot": rule.when.slot, "equals": rule.when.equals},
                    "require": list(rule.require),
                }
                for rule in self.conditional_slots
            ],
            "guidance": {
                "summary": self.guidance.summary,
                "checklist": list(self.guidance.checklist),
                "where_to_submit": self.guidance.where_to_submit,
                "notes": list(self.guidance.notes),
                "fee_hint": self.guidance.fee_hint,
                "processing_time_hint": self.guidance.processing_time_hint,
                **(
                    {"conditional_checklist": [_rule(r) for r in self.guidance.conditional_checklist]}
                    if self.guidance.conditional_checklist
                    else {}
                ),
                **(
                    {"conditional_notes": [_rule(r) for r in self.guidance.conditional_notes]}
                    if self.guidance.conditional_notes
                    else {}
                ),
            },
            "citations": [
                {
                    "doc_id": item.doc_id,
                    "title": item.title,
                    "source_type": item.source_type,
                }
                for item in self.citations
            ],
        }


class EvidenceItem(StrictModel):
    proposal_id: str
    kind: EvidenceKind
    excerpt_id: str
    page: int
    quotation: str
    review_warning: str = ""

    @field_validator("proposal_id")
    @classmethod
    def _pid(cls, v: str) -> str:
        if not v.strip() or len(v) > 80:
            raise ValueError("proposal_id is required")
        return v.strip()

    @field_validator("excerpt_id")
    @classmethod
    def _eid(cls, v: str) -> str:
        if UUID_RE.fullmatch(v or "") is None:
            raise ValueError("excerpt_id must be a UUID")
        return v.lower()

    @field_validator("page")
    @classmethod
    def _page(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError("page must be in [1, 10000]")
        return v

    @field_validator("quotation")
    @classmethod
    def _quote(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_QUOTE_LEN:
            raise ValueError("quotation is required and bounded")
        return v

    @field_validator("review_warning")
    @classmethod
    def _warn(cls, v: str) -> str:
        if len(v) > MAX_WARNING_LEN:
            raise ValueError("review_warning is too long")
        return v


class ModelDraft(StrictModel):
    """Provider-facing structured object. Converted to DraftResponse after checks."""

    schema_version: Literal["draft_definition.v1"] = SCHEMA_VERSION
    procedure_code: str
    definition: DefinitionProposal
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    abstain_reason: str | None = None

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v: list[EvidenceItem]) -> list[EvidenceItem]:
        if len(v) > MAX_EVIDENCE:
            raise ValueError("too many evidence items")
        return v

    @field_validator("warnings", "missing_requirements")
    @classmethod
    def _lists(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_WARNINGS:
            raise ValueError("too many warning/missing items")
        for item in v:
            if len(item) > MAX_WARNING_LEN:
                raise ValueError("warning/missing item is too long")
        return v

    @field_validator("abstain_reason")
    @classmethod
    def _abstain(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) > MAX_WARNING_LEN:
            raise ValueError("abstain_reason is too long")
        return v


class ValidationIssue(StrictModel):
    code: str
    severity: IssueSeverity
    message: str
    proposal_id: str = ""


class ValidationBlock(StrictModel):
    schema_valid: bool
    grounding_verified: bool
    source_coverage_complete: bool = False
    source_quotes_verified: bool = False
    semantic_approved: bool = False
    identity_mismatch: bool = False
    issues: list[ValidationIssue] = Field(default_factory=list)
    readiness: Readiness = "needs_review"
    material_proposal_count: int = 0
    covered_proposal_count: int = 0


class UsageMeta(StrictModel):
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None
    total_token_count: int | None = None


class DraftResponse(StrictModel):
    schema_version: Literal["draft_definition.v1"] = SCHEMA_VERSION
    request_id: str
    procedure_code: str
    definition: dict[str, Any]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    validation: ValidationBlock
    provider: str
    model: str
    prompt_version: str
    prompt_hash: str
    usage: UsageMeta = Field(default_factory=UsageMeta)
    latency_ms: int | None = None
    http_status: int | None = None
    abstain_reason: str | None = None
    original_procedure_code: str = ""
    identity_mismatch: bool = False
