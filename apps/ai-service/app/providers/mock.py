"""Deterministic, network-free mock provider.

Used as the default LLM_PROVIDER for local dev and for every automated test.
It intentionally mirrors (a simplified version of) the same folding / keyword
heuristics as the Go keyword engine (`internal/decision`) so behaviour is
predictable, but it is a *separate* implementation: the Go side always
re-validates everything independently (fail-closed) before trusting it.

Explicit behavioural contract (see apps/ai-service/tests/test_mock_provider.py):
  - "ba"                                        -> enum slot resolves to "cha"
  - "tôi" alone                                  -> never guessed into a slot
  - "chưa đăng ký kết hôn"                       -> boolean slot -> False
  - "à quên chưa đưng kí kéth ôn" (typo'd)        -> boolean slot -> False
  - message unrelated to any candidate            -> procedure_code=null, no slots
  - a slot key outside allowed_slot_keys/definition -> never returned
"""

from __future__ import annotations

import unicodedata

from app.models.extract import (
    Alternative,
    Candidate,
    ExtractRequest,
    ExtractResponse,
    IntentResult,
    SlotResult,
    SlotSpec,
)

MODEL_NAME = "mock-extract-v1"

# Colloquial / regional synonyms the mock resolves onto a canonical enum value.
_ENUM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cha": ("ba", "bo", "cha"),
    "me": ("ma", "me"),
}

_STOPWORDS = {
    "anh", "chi", "da", "chua", "khong", "co", "cua", "cho", "mot", "cac",
    "nao", "gi", "la", "duoc", "hay", "va", "hoac", "ban", "be", "toi",
    "minh", "voi", "trong", "khi", "neu", "theo", "tai", "den", "de", "thi",
    "nay", "kia", "do", "the", "roi", "dung", "yes", "ko", "sai", "phai",
    "a", "o",
}

# Pure pronoun/chatter messages must never be guessed into a string slot.
_WEAK_ONLY_TOKENS = {
    "toi", "minh", "ban", "anh", "chi", "em", "a", "da", "vang", "roi", "ok",
    "hello", "hi", "chao",
}


def fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    out: list[str] = []
    prev_space = True
    for ch in stripped:
        if ch.isdigit() or ("a" <= ch <= "z"):
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out).strip()


def _tokens(folded: str) -> list[str]:
    return folded.split() if folded else []


def _contains_phrase(folded: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {folded} "


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    d = _levenshtein(a, b)
    if d > 2 or d > max_len // 2:
        return 0.0
    return 1 - d / max_len


def _polarity(folded: str) -> tuple[bool, bool]:
    """Mirrors Go's polarity(): returns (value, matched)."""
    if (
        _contains_phrase(folded, "khong co")
        or _contains_phrase(folded, "chua co")
        or _contains_phrase(folded, "chua")
    ):
        return False, True
    if _contains_phrase(folded, "khong") or _contains_phrase(folded, "ko") or _contains_phrase(folded, "sai"):
        return False, True
    if (
        _contains_phrase(folded, "co")
        or _contains_phrase(folded, "roi")
        or _contains_phrase(folded, "dung")
        or _contains_phrase(folded, "da")
        or _contains_phrase(folded, "phai")
    ):
        return True, True
    return False, False


def _keyword_score(question: str, folded_message: str) -> int:
    score = 0
    for token in _tokens(fold(question)):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if _contains_phrase(folded_message, token):
            score += 1
            continue
        for mt in _tokens(folded_message):
            if _token_similarity(token, mt) >= 0.75:
                score += 1
                break
    return score


def _fuzzy_keyword_score(question: str, folded_message: str) -> int:
    """Typo-tolerant variant used only when the exact scorer finds nothing,
    so garbled input like "đưng kí kéth ôn" (đăng ký kết hôn) still resolves
    to the correct boolean slot.
    """
    q_tokens = [t for t in _tokens(fold(question)) if len(t) >= 3 and t not in _STOPWORDS]
    m_tokens = _tokens(folded_message)
    if not q_tokens or not m_tokens:
        return 0
    score = 0
    for qt in q_tokens:
        best = 0.0
        for mt in m_tokens:
            best = max(best, _token_similarity(qt, mt))
        if best >= 0.6:
            score += 1
    return score


def _score_text(folded_message: str, raw: str) -> float:
    folded = fold(raw)
    if not folded or not folded_message:
        return 0.0
    if folded_message == folded:
        return 1.0
    if folded in folded_message and len(folded) >= 6:
        return 0.95
    if folded_message in folded and len(folded_message) >= 6:
        return 0.9
    et = _tokens(folded)
    mt = set(_tokens(folded_message))
    if not et:
        return 0.0
    shared = 0
    for t in et:
        if t in mt:
            shared += 1
            continue
        for m in mt:
            if _token_similarity(t, m) >= 0.75:
                shared += 1
                break
    return shared / len(et) if et else 0.0


def _score_candidate(folded_message: str, candidate: Candidate) -> float:
    best = _score_text(folded_message, candidate.name)
    for example in candidate.intent_examples:
        best = max(best, _score_text(folded_message, example))
    return best


def _extract_enum(spec: SlotSpec, folded_message: str) -> str | None:
    for value in spec.enum_values:
        phrase = fold(value).replace("_", " ")
        if phrase and (folded_message == phrase or _contains_phrase(folded_message, phrase)):
            return value
    for canonical, synonyms in _ENUM_SYNONYMS.items():
        if canonical not in spec.enum_values:
            continue
        for syn in synonyms:
            if _contains_phrase(folded_message, syn) or folded_message == syn:
                return canonical
    return None


def _extract_boolean(spec: SlotSpec, folded_message: str) -> bool | None:
    value, matched = _polarity(folded_message)
    if not matched:
        return None
    score = _keyword_score(spec.question, folded_message)
    if score < 2:
        score = _fuzzy_keyword_score(spec.question, folded_message)
    if score < 1:
        return None
    return value


def _is_weak_only(folded_message: str) -> bool:
    toks = _tokens(folded_message)
    if not toks:
        return True
    return all(t in _WEAK_ONLY_TOKENS for t in toks)


def _extract_slots(
    candidate: Candidate, allowed_keys: list[str], message: str, folded_message: str
) -> list[SlotResult]:
    out: list[SlotResult] = []
    if _is_weak_only(folded_message):
        # "tôi" and similar bare pronouns/chatter must never be guessed into
        # any slot, regardless of type.
        return out

    boolean_hits: list[tuple[str, bool, int]] = []
    for key in allowed_keys:
        spec = candidate.slots.get(key)
        if spec is None:
            continue
        if spec.type == "enum":
            value = _extract_enum(spec, folded_message)
            if value is not None:
                out.append(
                    SlotResult(key=key, value=value, confidence=0.9, evidence=message[:200], operation="set")
                )
        elif spec.type == "boolean":
            score = _keyword_score(spec.question, folded_message)
            fuzzy = score < 2
            if fuzzy:
                score = _fuzzy_keyword_score(spec.question, folded_message)
            value, matched = _polarity(folded_message)
            if matched and score >= 1:
                boolean_hits.append((key, value, score))
        elif spec.type == "number":
            digits = "".join(ch for ch in folded_message if ch.isdigit())
            if digits and len(allowed_keys) == 1:
                try:
                    out.append(
                        SlotResult(
                            key=key,
                            value=float(digits),
                            confidence=0.7,
                            evidence=message[:200],
                            operation="set",
                        )
                    )
                except ValueError:
                    pass
        elif spec.type == "string":
            if len(allowed_keys) == 1 and len(folded_message) >= 2:
                trimmed = message.strip()
                if trimmed and not _is_weak_only(folded_message):
                    out.append(
                        SlotResult(
                            key=key,
                            value=trimmed[:500],
                            confidence=0.6,
                            evidence=message[:200],
                            operation="set",
                        )
                    )

    if boolean_hits:
        best_score = max(h[2] for h in boolean_hits)
        best = [h for h in boolean_hits if h[2] == best_score]
        if len(best) == 1:
            key, value, _ = best[0]
            out.append(
                SlotResult(key=key, value=value, confidence=0.85, evidence=message[:200], operation="set")
            )
    return out


class MockProvider:
    name = "mock"
    model = MODEL_NAME

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        folded_message = fold(request.message)
        by_code = {c.procedure_code: c for c in request.candidates}

        if request.pinned_context is not None and request.pinned_context.procedure_code in by_code:
            return self._extract_pinned(request, folded_message, by_code)
        return self._extract_fresh(request, folded_message, by_code)

    def _extract_fresh(
        self, request: ExtractRequest, folded_message: str, by_code: dict[str, Candidate]
    ) -> ExtractResponse:
        scored = sorted(
            (
                (candidate, _score_candidate(folded_message, candidate))
                for candidate in request.candidates
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if not scored or scored[0][1] < 0.6:
            return ExtractResponse(
                intent=IntentResult(procedure_code=None, confidence=0.0, alternatives=[]),
                slots_for_procedure_code=None,
                slots=[],
                abstain_reason="out_of_scope",
                provider=self.name,
                model=self.model,
            )

        best_candidate, best_score = scored[0]
        alternatives = [
            Alternative(procedure_code=c.procedure_code, confidence=s)
            for c, s in scored[1:4]
            if s >= 0.3
        ]
        allowed_keys = list(best_candidate.slots.keys())
        slots = _extract_slots(best_candidate, allowed_keys, request.message, folded_message)
        return ExtractResponse(
            intent=IntentResult(
                procedure_code=best_candidate.procedure_code,
                confidence=best_score,
                alternatives=alternatives,
            ),
            slots_for_procedure_code=best_candidate.procedure_code if slots else None,
            slots=slots,
            abstain_reason=None,
            provider=self.name,
            model=self.model,
        )

    def _extract_pinned(
        self, request: ExtractRequest, folded_message: str, by_code: dict[str, Candidate]
    ) -> ExtractResponse:
        pinned_code = request.pinned_context.procedure_code  # type: ignore[union-attr]
        pinned_candidate = by_code[pinned_code]

        # Detect a strong switch away from the pinned procedure.
        best_other_code: str | None = None
        best_other_score = 0.0
        for code, candidate in by_code.items():
            if code == pinned_code:
                continue
            score = _score_candidate(folded_message, candidate)
            if score > best_other_score:
                best_other_score = score
                best_other_code = code

        if best_other_code is not None and best_other_score >= 0.85:
            return ExtractResponse(
                intent=IntentResult(
                    procedure_code=best_other_code,
                    confidence=best_other_score,
                    alternatives=[Alternative(procedure_code=pinned_code, confidence=0.5)],
                ),
                slots_for_procedure_code=None,
                slots=[],
                abstain_reason=None,
                provider=self.name,
                model=self.model,
            )

        allowed_keys = list(request.pinned_context.allowed_slot_keys)  # type: ignore[union-attr]
        slots = _extract_slots(pinned_candidate, allowed_keys, request.message, folded_message)
        return ExtractResponse(
            intent=IntentResult(procedure_code=pinned_code, confidence=0.95, alternatives=[]),
            slots_for_procedure_code=pinned_code if slots else None,
            slots=slots,
            abstain_reason=None,
            provider=self.name,
            model=self.model,
        )
