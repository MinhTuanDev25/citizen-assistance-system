"""Canonical proposal identity shared with Go drafting.proposals.

Material administrative claims are derived from the Definition, never from
model-invented IDs. Legacy Wave 1 prefixes are normalized only when unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.models.draft import DefinitionProposal, ModelDraft

SLOT_ID = "slot:{key}"
REQUIRED_ID = "required:{key}"
CHECKLIST_ID = "checklist:{i}"
CONDITIONAL_ID = "conditional:{i}"
GUIDANCE_ID = "guidance:{field}"

_LEGACY_SLOT = re.compile(r"^prop_slot_(.+)$")
_LEGACY_REQUIRED = re.compile(r"^prop_required_(.+)$")
_LEGACY_CHECKLIST = re.compile(r"^prop_checklist_(\d+)$")
_LEGACY_CONDITIONAL = re.compile(r"^prop_conditional_(\d+)$")
_AMBIGUOUS_PREFIX = re.compile(r"^prop_")


class MaterialProposal:
    def __init__(self, pid: str, ptype: str, location: str, content: str) -> None:
        self.id = pid
        self.type = ptype
        self.location = location
        self.content = content
        canonical = f"{ptype}\n{pid}\n{content}"
        self.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "type": self.type,
            "location": self.location,
            "content": self.content,
            "fingerprint": self.fingerprint,
        }


def enumerate_material_proposals(definition: DefinitionProposal) -> list[MaterialProposal]:
    out: list[MaterialProposal] = []
    for item in definition.slots:
        content = item.type + "\t" + item.question.strip()
        if item.enum_values:
            content += "\t" + ",".join(item.enum_values)
        out.append(MaterialProposal(f"slot:{item.key}", "slot", f"slots.{item.key}", content))
    for key in definition.required_slots:
        out.append(MaterialProposal(f"required:{key}", "required", f"required_slots.{key}", key))
    for idx, rule in enumerate(definition.conditional_slots):
        equals = json.dumps(rule.when.equals, ensure_ascii=False, separators=(",", ":"))
        content = f"{rule.when.slot}={equals} require={','.join(rule.require)}"
        out.append(
            MaterialProposal(
                f"conditional:{idx}",
                "conditional",
                f"conditional_slots[{idx}]",
                content,
            )
        )
    for idx, text in enumerate(definition.guidance.checklist):
        out.append(
            MaterialProposal(
                f"checklist:{idx}",
                "checklist",
                f"guidance.checklist[{idx}]",
                text.strip(),
            )
        )
    for field, value in (
        ("summary", definition.guidance.summary),
        ("where_to_submit", definition.guidance.where_to_submit),
        ("fee_hint", definition.guidance.fee_hint),
        ("processing_time_hint", definition.guidance.processing_time_hint),
    ):
        text = (value or "").strip()
        if not text:
            continue
        out.append(MaterialProposal(f"guidance:{field}", "guidance", f"guidance.{field}", text))
    return out


def enumerate_from_model(draft: ModelDraft) -> list[MaterialProposal]:
    return enumerate_material_proposals(draft.definition)


def known_ids(proposals: list[MaterialProposal]) -> dict[str, MaterialProposal]:
    return {item.id: item for item in proposals}


def normalize_proposal_id(raw: str, known: dict[str, MaterialProposal]) -> dict[str, Any]:
    original = (raw or "").strip()
    result = {
        "original": original,
        "canonical": original,
        "normalized": False,
        "ambiguous": False,
        "unknown": False,
    }
    if not original:
        result["unknown"] = True
        return result
    if original in known:
        return result
    mapped, ok, ambiguous = _legacy_proposal_id(original)
    if ambiguous:
        result["ambiguous"] = True
        result["unknown"] = True
        return result
    if not ok:
        result["unknown"] = True
        return result
    result["canonical"] = mapped
    result["normalized"] = True
    if mapped not in known:
        result["unknown"] = True
    return result


def _legacy_proposal_id(raw: str) -> tuple[str, bool, bool]:
    if m := _LEGACY_SLOT.fullmatch(raw):
        return f"slot:{m.group(1)}", True, False
    if m := _LEGACY_REQUIRED.fullmatch(raw):
        return f"required:{m.group(1)}", True, False
    if m := _LEGACY_CHECKLIST.fullmatch(raw):
        return f"checklist:{m.group(1)}", True, False
    if m := _LEGACY_CONDITIONAL.fullmatch(raw):
        return f"conditional:{m.group(1)}", True, False
    if raw in {"prop_guidance_summary", "prop_summary"}:
        return "guidance:summary", True, False
    if raw in {"prop_guidance_where_to_submit", "prop_where_to_submit"}:
        return "guidance:where_to_submit", True, False
    if raw in {"prop_guidance_fee_hint", "prop_fee_hint"}:
        return "guidance:fee_hint", True, False
    if raw in {"prop_guidance_processing_time_hint", "prop_processing_time_hint"}:
        return "guidance:processing_time_hint", True, False
    if _AMBIGUOUS_PREFIX.match(raw):
        return "", False, True
    return "", False, False
