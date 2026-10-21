from app.models.draft import (
    DefinitionProposal,
    DraftRequest,
    EvidenceItem,
    GuidanceProposal,
    ModelDraft,
    SlotProposal,
    SourceExcerpt,
    UsageMeta,
    content_sha256,
)
from types import SimpleNamespace

from app.services.draft_service import DraftService

CONTENT = "Đăng ký khai sinh tại UBND cấp xã. Nộp giấy chứng sinh."


class _MismatchProvider:
    name = "mock"
    model = "mock"
    draft_model = "mock"

    def draft_definition(self, request: DraftRequest):
        draft = ModelDraft(
            procedure_code="chung_thuc_ban_sao",
            definition=DefinitionProposal(
                procedure_code="chung_thuc_ban_sao",
                name="Chứng thực",
                intent_examples=["x"],
                slots=[SlotProposal(key="noi_sinh", type="string", question="Nơi sinh?")],
                required_slots=["noi_sinh"],
                guidance=GuidanceProposal(summary="Nộp tại xã", checklist=["Giấy chứng sinh"]),
            ),
            evidence=[
                EvidenceItem(
                    proposal_id="slot:noi_sinh",
                    kind="slot",
                    excerpt_id=request.excerpts[0].excerpt_id,
                    page=10,
                    quotation="giấy chứng sinh",
                )
            ],
        )
        return draft, UsageMeta(), 200


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


def test_original_mismatch_is_preserved_after_server_lock():
    svc = DraftService(SimpleNamespace(provider="mock"), provider=_MismatchProvider())
    response = svc.draft(_request())
    assert response.procedure_code == "dk_khai_sinh"
    assert response.definition["procedure_code"] == "dk_khai_sinh"
    assert response.original_procedure_code == "chung_thuc_ban_sao"
    assert response.identity_mismatch is True
    assert response.validation.schema_valid is False
    assert response.validation.identity_mismatch is True
