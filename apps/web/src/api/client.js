import { loadAuth } from '../auth/auth.js'

/**
 * Shared JSON API client (Vite proxy /api → :8080).
 * options.guestToken → X-Guest-Token
 * options.auth === false → skip Bearer
 */
export async function apiRequest(path, options = {}) {
  const { guestToken, auth = true, headers: extraHeaders, ...rest } = options
  const headers = {
    Accept: 'application/json',
    ...(rest.body ? { 'Content-Type': 'application/json' } : {}),
    ...extraHeaders,
  }

  if (auth) {
    const session = loadAuth()
    if (session?.accessToken) {
      headers.Authorization = `Bearer ${session.accessToken}`
    }
  }
  if (guestToken) {
    headers['X-Guest-Token'] = guestToken
  }

  const res = await fetch(path, { ...rest, headers })
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
