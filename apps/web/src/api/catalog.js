async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
    ...options,
  })

  const payload = await res.json().catch(() => ({}))
  if (!res.ok) {
    const msg = payload?.error?.message || `HTTP ${res.status}`
    const err = new Error(msg)
    err.code = payload?.error?.code
    err.status = res.status
    err.payload = payload
    throw err
  }
  return payload?.data
}

export async function listCommunes({ active = true } = {}) {
  const q = active ? '?active=true' : ''
  return request(`/api/v1/communes${q}`)
}

export async function listDomains({ active = true } = {}) {
  const q = active ? '?active=true' : ''
  return request(`/api/v1/domains${q}`)
}

export async function listProcedures({ xaId, domainId } = {}) {
  if (!xaId) throw new Error('xa_id is required')
  const params = new URLSearchParams()
  params.set('xa_id', xaId)
  if (domainId) params.set('domain_id', domainId)
  return request(`/api/v1/procedures?${params}`)
}

export async function getProcedureByCode(code, { xaId } = {}) {
  if (!xaId) throw new Error('xa_id is required')
  const params = new URLSearchParams()
  params.set('xa_id', xaId)
  return request(
    `/api/v1/procedures/by-code/${encodeURIComponent(code)}?${params}`,
  )
}

export async function getActiveVersion(procedureId) {
  return request(`/api/v1/procedures/${encodeURIComponent(procedureId)}/active-version`)
}
