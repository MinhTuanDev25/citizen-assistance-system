import { apiRequest } from './client.js'

export async function listCommunes({ active = true } = {}) {
  const q = active ? '?active=true' : ''
  return apiRequest(`/api/v1/communes${q}`, { auth: false })
}

export async function listDomains({ active = true } = {}) {
  const q = active ? '?active=true' : ''
  return apiRequest(`/api/v1/domains${q}`, { auth: false })
}

export async function listProcedures({ xaId, domainId } = {}) {
  if (!xaId) throw new Error('xa_id is required')
  const params = new URLSearchParams()
  params.set('xa_id', xaId)
  if (domainId) params.set('domain_id', domainId)
  return apiRequest(`/api/v1/procedures?${params}`, { auth: false })
}

export async function getProcedureByCode(code, { xaId } = {}) {
  if (!xaId) throw new Error('xa_id is required')
  const params = new URLSearchParams()
  params.set('xa_id', xaId)
  return apiRequest(
    `/api/v1/procedures/by-code/${encodeURIComponent(code)}?${params}`,
    { auth: false },
  )
}

export async function getActiveVersion(procedureId) {
  return apiRequest(
    `/api/v1/procedures/${encodeURIComponent(procedureId)}/active-version`,
    { auth: false },
  )
}
