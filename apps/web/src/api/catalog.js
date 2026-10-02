import { apiRequest } from './client.js'

/** Dedupe concurrent catalog reads (React StrictMode double-mount). */
const procedureListInflight = new Map()

export async function listCommunes({ active = true } = {}) {
  const q = active ? '?active=true' : ''
  return apiRequest(`/api/v1/communes${q}`, { auth: false })
}

export async function listDomains({ active = true } = {}) {
  const q = active ? '?active=true' : ''
  return apiRequest(`/api/v1/domains${q}`, { auth: false })
}

/**
 * List procedures.
 * citizen=true → public CitizenDomainIDs (no auth required).
 * citizen=false → admin full catalog; Authorization Bearer ADMIN required.
 */
export async function listProcedures({ xaId, domainId, citizen }) {
  if (!xaId) throw new Error('xa_id is required')
  if (typeof citizen !== 'boolean') {
    throw new Error('citizen must be an explicit boolean (true|false)')
  }
  const params = new URLSearchParams()
  params.set('xa_id', xaId)
  params.set('citizen', citizen ? 'true' : 'false')
  if (domainId) params.set('domain_id', domainId)
  const path = `/api/v1/procedures?${params}`
  const auth = !citizen
  const key = `${path}|auth:${auth}`
  const pending = procedureListInflight.get(key)
  if (pending) return pending
  const request = apiRequest(path, {
    // Full catalog is an admin privilege — always attach JWT when citizen=false.
    auth,
  }).finally(() => {
    procedureListInflight.delete(key)
  })
  procedureListInflight.set(key, request)
  return request
}

export async function getProcedureByCode(code, { xaId, auth = false } = {}) {
  if (!xaId) throw new Error('xa_id is required')
  const params = new URLSearchParams()
  params.set('xa_id', xaId)
  return apiRequest(
    `/api/v1/procedures/by-code/${encodeURIComponent(code)}?${params}`,
    { auth },
  )
}

/** Admin active definition — always send Authorization Bearer. */
export async function getActiveVersion(procedureId) {
  return apiRequest(
    `/api/v1/procedures/${encodeURIComponent(procedureId)}/active-version`,
    { auth: true },
  )
}
