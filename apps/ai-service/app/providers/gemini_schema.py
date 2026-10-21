"""Gemini-facing JSON Schema for extract.v1 structured output.

Gemini Structured Outputs accept a subset of JSON Schema. The Pydantic
ExtractResponse schema is not sent verbatim: `provider` and `model` are
server-owned (the model must not emit them), and nullable unions use
Gemini's `type: [T, "null"]` form instead of Pydantic `anyOf`.
"""

from __future__ import annotations

from typing import Any

# Provider/model are injected after parse. See GeminiProvider.extract.
EXTRACT_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "intent",
        "slots_for_procedure_code",
        "slots",
        "abstain_reason",
    ],
    "properties": {
        "schema_version": {"type": "string", "enum": ["extract.v1"]},
        "intent": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "procedure_code": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["procedure_code", "confidence"],
                        "properties": {
                            "procedure_code": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                    },
                },
            },
        },
        "slots_for_procedure_code": {"type": ["string", "null"]},
        "slots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "value", "confidence", "evidence", "operation"],
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": ["string", "number", "boolean"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                    "operation": {"type": "string", "enum": ["set", "correct"]},
                },
            },
        },
        "abstain_reason": {"type": ["string", "null"]},
    },
}


def parse_usage_metadata(payload: dict[str, Any]) -> dict[str, int | None]:
    """Return token counts from generateContent. Never includes request text."""
    usage = payload.get("usageMetadata") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {
            "prompt_token_count": None,
            "candidates_token_count": None,
            "total_token_count": None,
        }

    def _int(name: str) -> int | None:
        raw = usage.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        return int(raw)

    return {
        "prompt_token_count": _int("promptTokenCount"),
        "candidates_token_count": _int("candidatesTokenCount"),
        "total_token_count": _int("totalTokenCount"),
    }


def redact_gemini_error(text: str) -> str:
    """Strip API keys that httpx may embed in URLs or exception text."""
    import re

    cleaned = re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", text or "")
    cleaned = re.sub(r"(AIza[0-9A-Za-z_-]{10,})", "<redacted>", cleaned)
    cleaned = re.sub(r"(AQ\.[0-9A-Za-z_-]{10,})", "<redacted>", cleaned)
    return cleaned[:300]
