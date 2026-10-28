"""Shared prompt/payload builders for the real (network) providers.

Kept separate from mock.py so the deterministic test path never imports
anything network-related, and so both real adapters build an identical,
minimal, schema-anchored payload — no chain-of-thought, no free text answer.
"""

from __future__ import annotations

from app.models.extract import ExtractRequest

EXTRACT_PROMPT_VERSION = "extract.prompt.v3"

SYSTEM_PROMPT = (
    "You extract structured intent and slot values for a Vietnamese citizen "
    "administrative-procedure assistant. You MUST only pick procedure_code "
    "from the given candidates and only fill slot keys from the given allowed "
    "keys. Extract ONLY information explicitly stated or clearly paraphrased "
    "in the citizen message. Each slot MUST include an evidence span copied "
    "verbatim from the message. Do not invent evidence. Do not infer civil "
    "status, document possession, or family relationship from unrelated "
    "clauses: 'mẹ đi đăng ký' does not mean the parents are married; naming "
    "a hospital does not establish other eligibility conditions. Boolean "
    "true/false requires an explicit positive or negative statement "
    "(đã/có vs chưa/không/không có). Negation applies to the nearest claim: "
    "'không có giấy chứng sinh' is false, not true. If a slot is not stated, "
    "omit it. An enum value MUST be one of enum_values; enum_labels (same "
    "order) say how a citizen names each value, so 'sinh bình thường' maps to "
    "the value whose label says so. If the request is too vague to pick a procedure, set "
    "procedure_code null, confidence 0, and abstain_reason. Never guess. "
    "Output must be a single JSON object matching this schema: "
    "{schema_version:'extract.v1', intent:{procedure_code, confidence 0..1, "
    "alternatives:[{procedure_code, confidence}]}, slots_for_procedure_code, "
    "slots:[{key, value, confidence, evidence, operation:'set'|'correct'}], "
    "abstain_reason}. Do not emit provider or model; the server sets those."
)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def prompt_version() -> str:
    return EXTRACT_PROMPT_VERSION


def build_user_payload(request: ExtractRequest) -> dict:
    """Non-PII-safe-ish payload: the citizen message is included (required
    for extraction) but nothing beyond what the model needs to answer.
    Caller (service layer) is responsible for not logging this payload.
    """
    return {
        "message": request.message,
        "candidates": [
            {
                "procedure_code": c.procedure_code,
                "name": c.name,
                "intent_examples": c.intent_examples,
                "slots": {
                    key: {
                        "type": spec.type,
                        "question": spec.question,
                        "enum_values": spec.enum_values,
                        **({"enum_labels": spec.enum_labels} if spec.enum_labels else {}),
                    }
                    for key, spec in c.slots.items()
                },
            }
            for c in request.candidates
        ],
        "pinned_context": (
            None
            if request.pinned_context is None
            else {
                "procedure_code": request.pinned_context.procedure_code,
                "allowed_slot_keys": request.pinned_context.allowed_slot_keys,
                "slot_state": {
                    key: {"value": entry.value, "status": entry.status}
                    for key, entry in request.pinned_context.slot_state.items()
                },
            }
        ),
    }
