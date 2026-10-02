"""Behavioural contract tests for the deterministic mock provider.

These are the exact scenarios called out in the P2 spec: "ba" -> cha,
"tôi" must never be guessed, "chưa đăng ký kết hôn" -> False (including a
typo'd variant), an out-of-scope message abstains, and a slot key outside
allowed_slot_keys is never returned.
"""

from app.models.extract import Candidate, ExtractRequest, PinnedContext, SlotSpec
from app.providers.mock import MockProvider

DK_KHAI_SINH = Candidate(
    procedure_code="dk_khai_sinh",
    name="Đăng ký khai sinh",
    intent_examples=["tôi muốn đăng ký khai sinh cho con", "làm giấy khai sinh"],
    slots={
        "nguoi_di_dang_ky": SlotSpec(
            type="enum",
            question="Ai là người đi đăng ký khai sinh?",
            enum_values=["cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"],
        ),
        "da_ket_hon": SlotSpec(
            type="boolean",
            question="Cha mẹ bé đã đăng ký kết hôn chưa?",
        ),
        "ho_ten_be": SlotSpec(type="string", question="Họ tên bé là gì?"),
    },
)

DK_BHYT = Candidate(
    procedure_code="dk_bhyt_ho_gia_dinh",
    name="Đăng ký bảo hiểm y tế hộ gia đình",
    intent_examples=["tôi muốn đăng ký bảo hiểm y tế cho cả nhà"],
    slots={
        "so_thanh_vien": SlotSpec(type="number", question="Hộ có bao nhiêu thành viên?"),
    },
)


def _pinned_request(message: str, allowed: list[str]) -> ExtractRequest:
    return ExtractRequest(
        request_id="11111111-1111-1111-1111-111111111111",
        message=message,
        candidates=[DK_KHAI_SINH, DK_BHYT],
        pinned_context=PinnedContext(
            procedure_code="dk_khai_sinh",
            allowed_slot_keys=allowed,
            slot_state={},
        ),
    )


def test_ba_resolves_to_cha_enum():
    req = _pinned_request("ba", ["nguoi_di_dang_ky"])
    resp = MockProvider().extract(req)
    assert resp.schema_version == "extract.v1"
    assert resp.slots_for_procedure_code == "dk_khai_sinh"
    keyed = {s.key: s.value for s in resp.slots}
    assert keyed.get("nguoi_di_dang_ky") == "cha"


def test_toi_alone_is_never_guessed():
    req = _pinned_request("tôi", ["ho_ten_be"])
    resp = MockProvider().extract(req)
    assert resp.slots == []
    assert resp.slots_for_procedure_code is None


def test_toi_alone_not_guessed_even_for_boolean_or_enum_slot():
    for allowed in (["nguoi_di_dang_ky"], ["da_ket_hon"]):
        req = _pinned_request("tôi", allowed)
        resp = MockProvider().extract(req)
        assert resp.slots == [], f"allowed={allowed} must not be filled from a bare pronoun"


def test_chua_dang_ky_ket_hon_is_false():
    req = _pinned_request("chưa đăng ký kết hôn", ["da_ket_hon"])
    resp = MockProvider().extract(req)
    keyed = {s.key: s.value for s in resp.slots}
    assert keyed.get("da_ket_hon") is False


def test_typo_chua_dang_ky_ket_hon_still_resolves_false():
    req = _pinned_request("à quên chưa đưng kí kéth ôn", ["da_ket_hon"])
    resp = MockProvider().extract(req)
    keyed = {s.key: s.value for s in resp.slots}
    assert keyed.get("da_ket_hon") is False


def test_out_of_scope_message_abstains():
    req = ExtractRequest(
        request_id="22222222-2222-2222-2222-222222222222",
        message="hôm nay thời tiết thế nào",
        candidates=[DK_KHAI_SINH, DK_BHYT],
        pinned_context=None,
    )
    resp = MockProvider().extract(req)
    assert resp.intent is not None
    assert resp.intent.procedure_code is None
    assert resp.slots == []
    assert resp.abstain_reason is not None


def test_never_returns_a_slot_outside_allowed_keys():
    # Only ho_ten_be is allowed even though the message also plausibly hints
    # at nguoi_di_dang_ky ("ba") — the disallowed key must never be returned.
    req = _pinned_request("ba, tên bé là Nguyễn Văn A", ["ho_ten_be"])
    resp = MockProvider().extract(req)
    keys = {s.key for s in resp.slots}
    assert "nguoi_di_dang_ky" not in keys


def test_fresh_selection_picks_intent_from_examples():
    req = ExtractRequest(
        request_id="33333333-3333-3333-3333-333333333333",
        message="tôi muốn đăng ký khai sinh cho con",
        candidates=[DK_KHAI_SINH, DK_BHYT],
        pinned_context=None,
    )
    resp = MockProvider().extract(req)
    assert resp.intent is not None
    assert resp.intent.procedure_code == "dk_khai_sinh"
    assert resp.intent.confidence >= 0.6


def test_pinned_switch_to_strongly_matched_other_procedure():
    req = _pinned_request("tôi muốn đăng ký bảo hiểm y tế cho cả nhà", [])
    resp = MockProvider().extract(req)
    assert resp.intent is not None
    assert resp.intent.procedure_code == "dk_bhyt_ho_gia_dinh"
    # Switching away must never carry slots for the old/new procedure in the
    # same response — the caller (Go) is responsible for routing this
    # through CONFIRM_INTENT rather than applying anything silently.
    assert resp.slots == []
