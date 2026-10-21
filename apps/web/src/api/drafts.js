import { apiRequest } from './client.js'

export const DRAFT_TIMEOUT_MS = 120000

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

export function generateAIDraft(procedureId, body, requestId) {
  return apiRequest(`/api/v1/admin/procedures/${encodeURIComponent(procedureId)}/ai-drafts`, {
    auth: true,
    method: 'POST',
    body: JSON.stringify(body),
    timeoutMs: DRAFT_TIMEOUT_MS,
    headers: { 'X-Request-ID': requestId },
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
      draft?.status === 'DRAFT',
  )
}
