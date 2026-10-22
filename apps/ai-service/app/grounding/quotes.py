"""Quotation membership and evidence classification for drafting."""

from __future__ import annotations

import unicodedata

from app.grounding.proposals import enumerate_from_model, known_ids, normalize_proposal_id
from app.models.draft import (
    DraftRequest,
    EvidenceItem,
    ModelDraft,
    ValidationBlock,
    ValidationIssue,
)


def normalize_quote(text: str) -> str:
    folded = unicodedata.normalize("NFC", text or "").casefold()
    return " ".join(folded.split())


def quotation_in_text(quotation: str, source: str) -> bool:
    needle = normalize_quote(quotation)
    haystack = normalize_quote(source)
    return bool(needle) and needle in haystack


def classify_evidence(request: DraftRequest, draft: ModelDraft) -> ValidationBlock:
    """Classify the ORIGINAL model output. Do not lock identity first."""
    issues: list[ValidationIssue] = []
    by_id = {item.excerpt_id: item for item in request.excerpts}
    matched = 0
    invalid = 0
    identity_mismatch = False

    if draft.procedure_code != request.procedure_code:
        identity_mismatch = True
        issues.append(
            ValidationIssue(
                code="PROCEDURE_CODE_LOCKED",
                severity="error",
                message="model attempted to change the locked procedure_code",
            )
        )
    if draft.definition.procedure_code != request.procedure_code:
        identity_mismatch = True
        issues.append(
            ValidationIssue(
                code="DEFINITION_PROCEDURE_MISMATCH",
                severity="error",
                message="definition.procedure_code does not match the locked target",
            )
        )

    proposals = enumerate_from_model(draft)
    known = known_ids(proposals)
    seen: dict[str, int] = {}
    covered: set[str] = set()
    unknown = 0
    duplicates = 0

    normalized_items: list[EvidenceItem] = []
    for item in draft.evidence:
        raw_id = item.proposal_id
        norm = normalize_proposal_id(raw_id, known)
        mapped = item.model_copy(update={"proposal_id": norm["canonical"]})
        if norm["normalized"]:
            issues.append(
                ValidationIssue(
                    code="PROPOSAL_ID_NORMALIZED",
                    severity="warning",
                    message=f"normalized {raw_id!r} to {norm['canonical']!r}",
                    proposal_id=norm["canonical"],
                )
            )
        if norm["ambiguous"] or norm["unknown"]:
            unknown += 1
            issues.append(
                ValidationIssue(
                    code="AMBIGUOUS_PROPOSAL_ID" if norm["ambiguous"] else "UNKNOWN_PROPOSAL_ID",
                    severity="error",
                    message="evidence refers to a proposal that is not in the current Definition",
                    proposal_id=raw_id,
                )
            )
            normalized_items.append(mapped)
            continue
        seen[mapped.proposal_id] = seen.get(mapped.proposal_id, 0) + 1
        if seen[mapped.proposal_id] > 1:
            duplicates += 1
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_PROPOSAL_MAPPING",
                    severity="error",
                    message="duplicate or conflicting evidence mapping for the same proposal",
                    proposal_id=mapped.proposal_id,
                )
            )
            normalized_items.append(mapped)
            continue
        excerpt = by_id.get(item.excerpt_id)
        if excerpt is None:
            invalid += 1
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_EXCERPT_ID",
                    severity="error",
                    message="evidence cites an excerpt_id that was not in the request",
                    proposal_id=mapped.proposal_id,
                )
            )
            normalized_items.append(mapped)
            continue
        if item.page < excerpt.page_start or item.page > excerpt.page_end:
            invalid += 1
            issues.append(
                ValidationIssue(
                    code="INVALID_PAGE",
                    severity="error",
                    message="evidence page is outside the cited excerpt page range",
                    proposal_id=mapped.proposal_id,
                )
            )
            normalized_items.append(mapped)
            continue
        if not quotation_in_text(item.quotation, excerpt.content):
            invalid += 1
            issues.append(
                ValidationIssue(
                    code="QUOTATION_NOT_FOUND",
                    severity="error",
                    message="quotation is not present in the cited excerpt",
                    proposal_id=mapped.proposal_id,
                )
            )
            normalized_items.append(mapped)
            continue
        matched += 1
        covered.add(mapped.proposal_id)
        issues.append(
            ValidationIssue(
                code="SOURCE_MATCHED_SEMANTIC_UNVERIFIED",
                severity="warning",
                message="quotation exists in the source; administrative interpretation is not approved",
                proposal_id=mapped.proposal_id,
            )
        )
        normalized_items.append(mapped)

    missing = 0
    for prop in proposals:
        if prop.id not in covered:
            missing += 1
            issues.append(
                ValidationIssue(
                    code="MISSING_EVIDENCE",
                    severity="error",
                    message="material proposal has no source evidence",
                    proposal_id=prop.id,
                )
            )

    schema_valid = not identity_mismatch
    quotes_verified = invalid == 0 and unknown == 0 and duplicates == 0
    coverage_complete = missing == 0 and unknown == 0 and duplicates == 0
    grounding_verified = schema_valid and quotes_verified and coverage_complete
    if not schema_valid:
        readiness = "invalid"
    else:
        readiness = "schema_valid_unapproved" if grounding_verified else "needs_review"
    return ValidationBlock(
        schema_valid=schema_valid,
        grounding_verified=grounding_verified,
        source_coverage_complete=coverage_complete,
        source_quotes_verified=quotes_verified,
        semantic_approved=False,
        identity_mismatch=identity_mismatch,
        issues=issues,
        readiness=readiness,
        material_proposal_count=len(proposals),
        covered_proposal_count=len(covered),
    )


def rewrite_canonical_ids(draft: ModelDraft, items: list[EvidenceItem]) -> list[EvidenceItem]:
    known = known_ids(enumerate_from_model(draft))
    out: list[EvidenceItem] = []
    for item in items:
        norm = normalize_proposal_id(item.proposal_id, known)
        if norm["normalized"] and not norm["unknown"] and not norm["ambiguous"]:
            out.append(item.model_copy(update={"proposal_id": norm["canonical"]}))
        else:
            out.append(item)
    return out


def annotate_evidence(request: DraftRequest, items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Stamp a review warning on matched quotes without changing identity fields."""
    by_id = {item.excerpt_id: item for item in request.excerpts}
    out: list[EvidenceItem] = []
    for item in items:
        excerpt = by_id.get(item.excerpt_id)
        warning = item.review_warning
        if excerpt is None:
            warning = warning or "source_reference_invalid"
        elif not quotation_in_text(item.quotation, excerpt.content):
            warning = warning or "quotation_not_found"
        else:
            warning = warning or "source_matched_semantic_unverified"
        out.append(item.model_copy(update={"review_warning": warning}))
    return out
