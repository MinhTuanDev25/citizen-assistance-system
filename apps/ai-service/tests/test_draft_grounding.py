from app.grounding.quotes import classify_evidence, quotation_in_text
from app.models.draft import (
    DefinitionProposal,
    DraftRequest,
    EvidenceItem,
    GuidanceProposal,
    ModelDraft,
    SlotProposal,
    SourceExcerpt,
    content_sha256,
)

CONTENT = "Đăng ký khai sinh tại UBND cấp xã. Nộp giấy chứng sinh."


def _request() -> DraftRequest:
    excerpt = SourceExcerpt(
        excerpt_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        generation_id="33333333-3333-3333-3333-333333333333",
        page_start=10,
        page_end=10,
        text_sha256=content_sha256(CONTENT),
        content=CONTENT,
    )
    return DraftRequest(
        request_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        procedure_code="dk_khai_sinh",
        procedure_name="Đăng ký khai sinh",
        xa_id="xa_chu_se",
        document_id=excerpt.document_id,
        generation_id=excerpt.generation_id,
        source_page_start=10,
        source_page_end=21,
        excerpts=[excerpt],
    )


def _draft(evidence, procedure_code="dk_khai_sinh") -> ModelDraft:
    return ModelDraft(
        procedure_code=procedure_code,
        definition=DefinitionProposal(
            procedure_code=procedure_code,
            name="Đăng ký khai sinh",
            intent_examples=["Đăng ký khai sinh"],
            slots=[SlotProposal(key="noi_sinh", type="string", question="Nơi sinh?")],
            required_slots=["noi_sinh"],
            guidance=GuidanceProposal(summary="Nộp tại xã", checklist=["Giấy chứng sinh"]),
        ),
        evidence=evidence,
    )


def test_quotation_membership_normalized():
    assert quotation_in_text("giấy chứng sinh", CONTENT)
    assert not quotation_in_text("lệ phí một trăm triệu", CONTENT)


def test_unknown_excerpt_is_invalid():
    draft = _draft(
        [
            EvidenceItem(
                proposal_id="slot:noi_sinh",
                kind="slot",
                excerpt_id="99999999-9999-9999-9999-999999999999",
                page=10,
                quotation="giấy chứng sinh",
            )
        ]
    )
    result = classify_evidence(_request(), draft)
    assert any(item.code == "UNKNOWN_EXCERPT_ID" for item in result.issues)
    assert result.grounding_verified is False


def test_invalid_page_is_rejected():
    draft = _draft(
        [
            EvidenceItem(
                proposal_id="slot:noi_sinh",
                kind="slot",
                excerpt_id="11111111-1111-1111-1111-111111111111",
                page=21,
                quotation="giấy chứng sinh",
            )
        ]
    )
    result = classify_evidence(_request(), draft)
    assert any(item.code == "INVALID_PAGE" for item in result.issues)


def test_semantic_approval_never_set():
    quote = "giấy chứng sinh"
    draft = _draft(
        [
            EvidenceItem(
                proposal_id="slot:noi_sinh",
                kind="slot",
                excerpt_id="11111111-1111-1111-1111-111111111111",
                page=10,
                quotation=quote,
            ),
            EvidenceItem(
                proposal_id="required:noi_sinh",
                kind="required_slot",
                excerpt_id="11111111-1111-1111-1111-111111111111",
                page=10,
                quotation=quote,
            ),
            EvidenceItem(
                proposal_id="checklist:0",
                kind="checklist",
                excerpt_id="11111111-1111-1111-1111-111111111111",
                page=10,
                quotation=quote,
            ),
            EvidenceItem(
                proposal_id="guidance:summary",
                kind="guidance",
                excerpt_id="11111111-1111-1111-1111-111111111111",
                page=10,
                quotation="UBND cấp xã",
            ),
        ]
    )
    result = classify_evidence(_request(), draft)
    assert result.semantic_approved is False
    assert result.schema_valid is True


def test_partial_evidence_is_not_fully_grounded():
    result = classify_evidence(
        _request(),
        _draft(
            [
                EvidenceItem(
                    proposal_id="slot:noi_sinh",
                    kind="slot",
                    excerpt_id="11111111-1111-1111-1111-111111111111",
                    page=10,
                    quotation="giấy chứng sinh",
                )
            ]
        ),
    )
    assert result.source_coverage_complete is False
    assert result.grounding_verified is False
    assert result.material_proposal_count > 1


def test_wrong_procedure_identity_is_not_hidden():
    result = classify_evidence(_request(), _draft([], procedure_code="chung_thuc_ban_sao"))
    assert result.schema_valid is False
    assert result.identity_mismatch is True
    assert any(item.code == "PROCEDURE_CODE_LOCKED" for item in result.issues)


def test_legacy_prop_slot_is_normalized():
    from app.grounding.proposals import known_ids, enumerate_from_model, normalize_proposal_id

    draft = _draft([])
    known = known_ids(enumerate_from_model(draft))
    norm = normalize_proposal_id("prop_slot_noi_sinh", known)
    assert norm["canonical"] == "slot:noi_sinh"
    assert norm["normalized"] is True
    assert norm["unknown"] is False
