import pytest
from pydantic import ValidationError

from app.models.draft import SlotProposal
from app.providers.draft_schema import DRAFT_RESPONSE_JSON_SCHEMA
from app.providers.prompt_draft import DRAFT_BATCH_PROMPT_VERSION, DRAFT_PROMPT_VERSION


def test_enum_labels_follow_values():
    slot = SlotProposal(
        key="truong_hop_tre",
        type="enum",
        question="Trường hợp khai sinh của trẻ thuộc diện nào?",
        enum_values=["tre_sinh_thuong", "tre_bi_bo_roi"],
        enum_labels=["Trẻ sinh bình thường", "Trẻ bị bỏ rơi"],
    )
    assert slot.enum_labels == ["Trẻ sinh bình thường", "Trẻ bị bỏ rơi"]


@pytest.mark.parametrize(
    "labels",
    [["Chỉ một nhãn"], ["Một", "Hai", "Ba"], ["Một", "  "]],
)
def test_enum_labels_must_match_one_to_one(labels):
    with pytest.raises(ValidationError):
        SlotProposal(key="k", type="enum", question="?", enum_values=["a", "b"], enum_labels=labels)


def test_labels_only_on_enum_slots_and_optional():
    SlotProposal(key="k", type="enum", question="?", enum_values=["a"])  # labels optional
    with pytest.raises(ValidationError):
        SlotProposal(key="k", type="boolean", question="?", enum_labels=["Có"])


def test_schema_and_prompt_version_carry_labels():
    slot_schema = DRAFT_RESPONSE_JSON_SCHEMA["properties"]["definition"]["properties"]["slots"]["items"]
    assert "enum_labels" in slot_schema["properties"] and "enum_labels" in slot_schema["required"]
    assert DRAFT_PROMPT_VERSION == "draft.prompt.v5"
    assert DRAFT_BATCH_PROMPT_VERSION.startswith("draft.prompt.v5")


def test_conditional_guidance_reaches_go_definition():
    from app.models.draft import DefinitionProposal

    d = DefinitionProposal.model_validate(
        {
            "procedure_code": "khai_sinh_v1",
            "name": "Đăng ký khai sinh",
            "slots": [
                {
                    "key": "truong_hop_tre",
                    "type": "enum",
                    "question": "Trường hợp của trẻ?",
                    "enum_values": ["tre_sinh_thuong", "tre_bi_bo_roi"],
                    "enum_labels": ["Trẻ sinh bình thường", "Trẻ bị bỏ rơi"],
                }
            ],
            "guidance": {
                "summary": "s",
                "checklist": ["Tờ khai"],
                "conditional_checklist": [
                    {"when": {"slot": "truong_hop_tre", "equals": "tre_bi_bo_roi"}, "item": "Biên bản trẻ bị bỏ rơi"}
                ],
                "conditional_notes": [
                    {"when": {"slot": "truong_hop_tre", "equals": "tre_bi_bo_roi"}, "item": "Niêm yết 07 ngày"}
                ],
            },
        }
    )
    g = d.to_go_definition()["guidance"]
    assert g["checklist"] == ["Tờ khai"]
    assert g["conditional_checklist"] == [
        {"when": {"slot": "truong_hop_tre", "equals": "tre_bi_bo_roi"}, "item": "Biên bản trẻ bị bỏ rơi"}
    ]
    assert g["conditional_notes"][0]["item"] == "Niêm yết 07 ngày"


def test_conditional_guidance_is_optional_and_item_required():
    from app.models.draft import GuidanceProposal

    assert "conditional_checklist" not in GuidanceProposal(summary="", checklist=[]).model_dump(exclude_defaults=True)
    with pytest.raises(ValidationError):
        GuidanceProposal(summary="", checklist=[], conditional_checklist=[{"when": {"slot": "k", "equals": True}, "item": " "}])
