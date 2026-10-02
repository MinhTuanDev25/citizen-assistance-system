"""Shared prompt/payload builders for the real (network) providers.

Kept separate from mock.py so the deterministic test path never imports
anything network-related, and so both real adapters build an identical,
minimal, schema-anchored payload — no chain-of-thought, no free text answer.
"""

from __future__ import annotations

from app.models.extract import ExtractRequest

SYSTEM_PROMPT = (
    "You extract structured intent and slot values for a Vietnamese citizen "
    "administrative-procedure assistant. You MUST only pick procedure_code "
    "from the given candidates and only fill slot keys from the given allowed "
    "keys. Never invent a procedure or a slot. If unsure, leave fields null "
    "and set abstain_reason. Output must be a single JSON object matching "
    "this schema: {schema_version:'extract.v1', intent:{procedure_code, "
    "confidence 0..1, alternatives:[{procedure_code, confidence}]}, "
    "slots_for_procedure_code, slots:[{key, value, confidence, evidence, "
    "operation:'set'|'correct'}], abstain_reason}. Do not emit provider or "
    "model; the server sets those."
)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


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
