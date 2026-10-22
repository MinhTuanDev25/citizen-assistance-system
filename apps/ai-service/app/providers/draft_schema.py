"""Gemini-facing JSON Schema for draft_definition.v1.

Slots are an array (Gemini structured output does not reliably support
open-ended object maps). The server converts the array into the Go
Definition object map after validation.
"""

from __future__ import annotations

from typing import Any

DRAFT_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "procedure_code",
        "definition",
        "evidence",
        "warnings",
        "missing_requirements",
        "abstain_reason",
    ],
    "properties": {
        "schema_version": {"type": "string", "enum": ["draft_definition.v1"]},
        "procedure_code": {"type": "string"},
        "definition": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "procedure_code",
                "name",
                "intent_examples",
                "slots",
                "required_slots",
                "conditional_slots",
                "guidance",
                "citations",
            ],
            "properties": {
                "procedure_code": {"type": "string"},
                "name": {"type": "string"},
                "intent_examples": {"type": "array", "items": {"type": "string"}},
                "slots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["key", "type", "question", "enum_values"],
                        "properties": {
                            "key": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["string", "boolean", "number", "enum"],
                            },
                            "question": {"type": "string"},
                            "enum_values": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "required_slots": {"type": "array", "items": {"type": "string"}},
                "conditional_slots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["when", "require"],
                        "properties": {
                            "when": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["slot", "equals"],
                                "properties": {
                                    "slot": {"type": "string"},
                                    "equals": {"type": ["string", "number", "boolean"]},
                                },
                            },
                            "require": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "guidance": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "summary",
                        "checklist",
                        "where_to_submit",
                        "notes",
                        "fee_hint",
                        "processing_time_hint",
                    ],
                    "properties": {
                        "summary": {"type": "string"},
                        "checklist": {"type": "array", "items": {"type": "string"}},
                        "where_to_submit": {"type": "string"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                        "fee_hint": {"type": "string"},
                        "processing_time_hint": {"type": "string"},
                    },
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["doc_id", "title", "source_type"],
                        "properties": {
                            "doc_id": {"type": "string"},
                            "title": {"type": "string"},
                            "source_type": {
                                "type": "string",
                                "enum": ["ai_draft_unapproved"],
                            },
                        },
                    },
                },
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "proposal_id",
                    "kind",
                    "excerpt_id",
                    "page",
                    "quotation",
                    "review_warning",
                ],
                "properties": {
                    "proposal_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "slot",
                            "required_slot",
                            "conditional",
                            "checklist",
                            "guidance",
                            "citation",
                        ],
                    },
                    "excerpt_id": {"type": "string"},
                    "page": {"type": "integer"},
                    "quotation": {"type": "string"},
                    "review_warning": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "missing_requirements": {"type": "array", "items": {"type": "string"}},
        "abstain_reason": {"type": ["string", "null"]},
    },
}
