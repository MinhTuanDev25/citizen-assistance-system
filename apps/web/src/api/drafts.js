import { apiRequest } from './client.js'

export const DRAFT_TIMEOUT_MS = 330000

export function listProcedureDrafts(procedureId) {
  return apiRequest(`/api/v1/admin/procedures/${encodeURIComponent(procedureId)}/drafts`, { auth: true })
}

export function listAllDrafts(procedureId) {
  const q = procedureId ? `?procedure_id=${encodeURIComponent(procedureId)}` : ''
  return apiRequest(`/api/v1/admin/procedure-drafts${q}`, { auth: true })
}

export function listDraftSources(procedureId) {
  return apiRequest(`/api/v1/admin/procedures/${encodeURIComponent(procedureId)}/draft-sources`, { auth: true })
}

export function getDraft(draftId) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}`, { auth: true })
}

/** Preview how a long document is drafted: detected sections, chosen scope, batches. */
export function planDraft(procedureId, body) {
  return apiRequest(`/api/v1/admin/procedures/${encodeURIComponent(procedureId)}/draft-plan`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** Queue a multi-batch draft job (202). The same requestId replays the same job. */
export function createDraftJob(procedureId, body, requestId) {
  return apiRequest(`/api/v1/admin/procedures/${encodeURIComponent(procedureId)}/draft-jobs`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'X-Request-ID': requestId },
  })
}

export function listDraftJobs(procedureId) {
  return apiRequest(`/api/v1/admin/procedures/${encodeURIComponent(procedureId)}/draft-jobs`, { auth: true })
}

export function getDraftJob(jobId) {
  return apiRequest(`/api/v1/admin/procedure-draft-jobs/${encodeURIComponent(jobId)}`, { auth: true })
}

/** Re-run only the failed batches (or the merge) of a FAILED job. */
export function retryDraftJob(jobId) {
  return apiRequest(`/api/v1/admin/procedure-draft-jobs/${encodeURIComponent(jobId)}/retry`, {
    auth: true,
    method: 'POST',
    body: '{}',
  })
}

/** Record why a coverage-gap field stays empty (the source does not state it). */
export function acknowledgeDraftGap(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/acknowledge-gap`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** Put a citation dropped by the merge limit back (replace: index of a current one, or null to append). */
export function keepDraftCitation(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/keep-citation`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** choice: candidate index, or -1 to keep the field as currently edited. */
export function resolveDraftConflict(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/resolve-conflict`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** Keep (or with remove: true withdraw) a proposal without a source, on the Admin's reason. */
export function decideWithoutSource(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/admin-decision`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** Copy an approved/published draft into a new editable draft (no AI call). */
export function cloneDraft(draftId) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/clone`, {
    auth: true,
    method: 'POST',
  })
}

export function patchDraft(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}`, {
    auth: true,
    method: 'PATCH',
    body: JSON.stringify(body),
  })
}

export function validateDraft(draftId) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/validate`, {
    auth: true,
    method: 'POST',
    body: '{}',
  })
}

export function reviewDraft(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/review`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function approveDraft(draftId, body) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/approve`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function activateDraft(draftId, body, requestId) {
  return apiRequest(`/api/v1/admin/procedure-drafts/${encodeURIComponent(draftId)}/activate`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
    timeoutMs: DRAFT_TIMEOUT_MS,
    headers: { 'X-Request-ID': requestId },
  })
}

export function canApproveDraft(draft) {
  const v = draft?.validation || {}
  return Boolean(
    v.schema_valid &&
      v.procedural_ready !== false &&
      v.source_coverage_complete &&
      v.source_quotes_verified &&
      v.grounding_verified &&
      !v.identity_mismatch &&
      !draft?.identity_mismatch &&
      !(draft?.conflicts || []).some((c) => c.status !== 'RESOLVED' && !c.inactive) &&
      !(draft?.coverage?.gaps || []).some((g) => !g.acknowledged) &&
      !(draft?.coverage?.truncated || []).some((t) => t.blocking && !(draft.coverage.acknowledged || []).some((a) => a.field === `truncated:${t.field}`)) &&
      (draft?.coverage?.outside_mentions_reviewed ?? draft?.coverage?.legal_sources_verified) !== false &&
      !(draft?.coverage?.boundary_uncertain && !draft?.coverage?.boundary_reviewed) &&
      draft?.status === 'DRAFT',
  )
}
