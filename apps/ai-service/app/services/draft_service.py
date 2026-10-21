"""Definition-drafting wrapper around the shared Gemini/mock provider.

Does not write to PostgreSQL. Does not claim semantic/legal approval.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from app.config import Settings
from app.grounding.quotes import annotate_evidence, classify_evidence, rewrite_canonical_ids
from app.models.draft import DraftRequest, DraftResponse, ModelDraft, UsageMeta, ValidationBlock
from app.providers.base import ProviderError
from app.providers.prompt_draft import DRAFT_PROMPT_VERSION, prompt_hash

logger = logging.getLogger("ai_service.draft")


class DraftProvider(Protocol):
    name: str
    model: str

    def draft_definition(self, request: DraftRequest) -> tuple[ModelDraft, UsageMeta, int]:
        """Return (proposal, usage, http_status) or raise ProviderError."""
        ...


class DraftServiceError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def build_draft_provider(settings: Settings) -> DraftProvider:
    if settings.provider == "openai":
        raise DraftServiceError(
            "config_error",
            "definition drafting is implemented for mock and gemini only",
        )
    if settings.provider == "gemini":
        from app.providers.gemini import GeminiProvider

        return GeminiProvider(settings)
    from app.providers.mock import MockProvider

    return MockProvider()


class DraftService:
    def __init__(self, settings: Settings, provider: DraftProvider | None = None) -> None:
        self._settings = settings
        self._provider = provider

    def _provider_or_build(self) -> DraftProvider:
        if self._provider is not None:
            return self._provider
        return build_draft_provider(self._settings)

    def draft(self, request: DraftRequest) -> DraftResponse:
        started = time.perf_counter()
        logger.info(
            "draft_request request_id=%s procedure=%s excerpts=%d provider=%s",
            request.request_id,
            request.procedure_code,
            len(request.excerpts),
            self._settings.provider,
        )
        try:
            provider = self._provider_or_build()
            proposal, usage, http_status = provider.draft_definition(request)
        except ProviderError as exc:
            logger.warning(
                "draft_provider_error request_id=%s reason=%s provider=%s",
                request.request_id,
                exc.reason,
                self._settings.provider,
            )
            raise DraftServiceError(exc.reason, str(exc)) from exc

        original_code = proposal.procedure_code
        validation = classify_evidence(request, proposal)
        locked = proposal.model_copy(
            update={
                "procedure_code": request.procedure_code,
                "definition": proposal.definition.model_copy(
                    update={"procedure_code": request.procedure_code}
                ),
            }
        )
        evidence = rewrite_canonical_ids(proposal, annotate_evidence(request, locked.evidence))
        latency_ms = int((time.perf_counter() - started) * 1000)
        response = DraftResponse(
            request_id=str(request.request_id),
            procedure_code=request.procedure_code,
            original_procedure_code=original_code,
            identity_mismatch=validation.identity_mismatch,
            definition=locked.definition.to_go_definition(),
            evidence=evidence,
            warnings=list(locked.warnings),
            missing_requirements=list(locked.missing_requirements),
            validation=validation,
            provider=provider.name,
            model=getattr(provider, "draft_model", None) or provider.model,
            prompt_version=DRAFT_PROMPT_VERSION,
            prompt_hash=prompt_hash(),
            usage=usage,
            latency_ms=latency_ms,
            http_status=http_status,
            abstain_reason=locked.abstain_reason,
        )
        logger.info(
            "draft_response request_id=%s procedure=%s schema_valid=%s grounding=%s "
            "provider=%s model=%s prompt_version=%s prompt_tokens=%s output_tokens=%s latency_ms=%s",
            request.request_id,
            response.procedure_code,
            validation.schema_valid,
            validation.grounding_verified,
            response.provider,
            response.model,
            response.prompt_version,
            usage.prompt_token_count,
            usage.candidates_token_count,
            latency_ms,
        )
        return response


def empty_validation(schema_valid: bool = False) -> ValidationBlock:
    return ValidationBlock(
        schema_valid=schema_valid,
        grounding_verified=False,
        semantic_approved=False,
        issues=[],
        readiness="invalid" if not schema_valid else "needs_review",
    )
