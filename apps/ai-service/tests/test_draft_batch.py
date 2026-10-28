"""Batch mode of POST /v1/draft_definition: one part of a long procedure section."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.models.draft import DraftRequest, ModelDraft
from app.providers.prompt_draft import (
    DRAFT_BATCH_PROMPT_VERSION,
    DRAFT_PROMPT_VERSION,
    build_draft_prompt,
    prompt_hash,
)
from tests.test_draft_endpoint import VALID_DRAFT_BODY, post

CONTEXT = {
    "partial": True,
    "section_title": "Thủ tục đăng ký khai sinh",
    "batch_index": 2,
    "batch_count": 3,
    "known_slots": [{"key": "hinh_thuc_nop", "type": "enum", "question": "Nộp trực tiếp hay trực tuyến?"}],
}


def batch_body(**context):
    body = deepcopy(VALID_DRAFT_BODY)
    body["context"] = {**CONTEXT, **context}
    return body


def model_draft(slots):
    return {
        "procedure_code": "dk_khai_sinh",
        "definition": {
            "procedure_code": "dk_khai_sinh",
            "name": "Đăng ký khai sinh",
            "slots": slots,
            "guidance": {"summary": "", "checklist": []},
        },
    }


def test_batch_request_is_accepted_and_versioned():
    r = post(batch_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt_version"] == DRAFT_BATCH_PROMPT_VERSION
    assert body["prompt_hash"] == prompt_hash(partial=True)
    assert body["prompt_hash"] != prompt_hash()


def test_single_call_request_keeps_legacy_prompt():
    r = post(VALID_DRAFT_BODY)
    assert r.status_code == 200, r.text
    assert r.json()["prompt_version"] == DRAFT_PROMPT_VERSION
    prompt = build_draft_prompt(DraftRequest.model_validate(VALID_DRAFT_BODY))
    assert "BATCH MODE" not in prompt
    assert '"batch"' not in prompt


def test_batch_prompt_carries_rules_and_known_slots():
    prompt = build_draft_prompt(DraftRequest.model_validate(batch_body()))
    assert "BATCH MODE" in prompt
    assert "Never reuse the slot's own quotation as evidence of being required" in prompt
    assert '"known_slots": [{"key": "hinh_thuc_nop"' in prompt
    assert '"batch_index": 2' in prompt


@pytest.mark.parametrize(
    "override",
    [
        {"batch_index": 4},
        {"batch_index": 0},
        {"known_slots": [{"key": "1bad", "type": "string", "question": "x"}]},
        {"known_slots": [{"key": f"k{i}", "type": "string", "question": "x"} for i in range(31)]},
        {"section_title": "x" * 301},
    ],
)
def test_batch_context_is_bounded(override):
    r = post(batch_body(**override))
    assert r.status_code == 422, r.text


def test_empty_slots_only_allowed_for_a_batch():
    with pytest.raises(ValidationError):
        ModelDraft.model_validate(model_draft([]))
    draft = ModelDraft.model_validate(model_draft([]), context={"partial": True})
    assert draft.definition.slots == []
    assert draft.definition.to_go_definition()["slots"] == {}
