"""Step 02: extract.v1 contract on the thesis synthetic Vietnamese utterance.

Uses the mock provider (no network). Live Gemini is recorded separately
when GEMINI_API_KEY is present.
"""

from app.models.extract import Candidate, ExtractRequest, ExtractResponse, SlotSpec
from app.providers.mock import MockProvider

SYNTHETIC = (
    "Tôi muốn đăng ký khai sinh cho con, đã có giấy chứng sinh và muốn nộp trực tiếp."
)

CANDIDATE = Candidate(
    procedure_code="dk_khai_sinh",
    name="Đăng ký khai sinh",
    intent_examples=["Tôi muốn đăng ký khai sinh cho con", "Đăng ký khai sinh"],
    slots={
        "noi_sinh": SlotSpec(
            type="string",
            question="Bé sinh ở đâu (bệnh viện/cơ sở y tế hay tại nhà, thuộc xã/phường nào)?",
        ),
        "co_giay_chung_sinh": SlotSpec(
            type="boolean",
            question="Anh/chị có giấy chứng sinh không?",
        ),
        "nguoi_di_dang_ky": SlotSpec(
            type="enum",
            question="Ai sẽ đi đăng ký khai sinh?",
            enum_values=["cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"],
        ),
    },
)


def _request() -> ExtractRequest:
    return ExtractRequest(
        request_id="22222222-2222-2222-2222-222222222222",
        message=SYNTHETIC,
        candidates=[CANDIDATE],
        pinned_context=None,
    )


def _assert_go_like_allowlist(resp: ExtractResponse, req: ExtractRequest) -> None:
    codes = {c.procedure_code for c in req.candidates}
    if resp.intent and resp.intent.procedure_code:
        assert resp.intent.procedure_code in codes
    if resp.slots_for_procedure_code:
        assert resp.slots_for_procedure_code in codes
    by_code = {c.procedure_code: c for c in req.candidates}
    target = resp.slots_for_procedure_code
    for slot in resp.slots:
        assert target is not None
        assert slot.key in by_code[target].slots
        spec = by_code[target].slots[slot.key]
        if spec.type == "enum":
            assert slot.value in spec.enum_values
        if spec.type == "boolean":
            assert isinstance(slot.value, bool)


def test_synthetic_message_is_valid_extract_request():
    req = _request()
    assert "giấy chứng sinh" in req.message
    dumped = req.model_dump(mode="json")
    roundtrip = ExtractRequest.model_validate(dumped)
    assert roundtrip.message == SYNTHETIC


def test_mock_provider_does_not_invent_slots_on_synthetic():
    req = _request()
    resp = MockProvider().extract(req)
    roundtrip = ExtractResponse.model_validate(resp.model_dump(mode="json"))
    _assert_go_like_allowlist(roundtrip, req)
    invented = {s.key for s in roundtrip.slots} - set(CANDIDATE.slots)
    assert not invented
    assert roundtrip.provider == "mock"
