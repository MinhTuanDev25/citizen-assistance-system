import { loadAuth } from '../auth/auth.js'
import { apiErrorMessage } from '../lib/apiErrors.js'

/**
 * Shared JSON API client (Vite proxy /api → :8080).
 * options.guestToken → X-Guest-Token
 * options.auth === false → skip Bearer
 */
export async function apiRequest(path, options = {}) {
  const { guestToken, auth = true, headers: extraHeaders, timeoutMs, ...rest } = options
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

  const controller = timeoutMs && !rest.signal ? new AbortController() : null
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null
  let res
  try {
    res = await fetch(path, { ...rest, headers, signal: rest.signal || controller?.signal })
  } catch (cause) {
    // Keep the name (AbortError, TypeError): callers use it to tell "no answer" from a refusal.
    const err = new Error('Không kết nối được máy chủ. Kiểm tra mạng rồi thử lại.')
    err.name = cause?.name || 'Error'
    err.serverMessage = cause?.message || ''
    err.cause = cause
    throw err
  } finally {
    if (timer) clearTimeout(timer)
  }
  const payload = await res.json().catch(() => ({}))
  if (!res.ok) {
    const serverMessage = payload?.error?.message || ''
    const code = payload?.error?.code
    const err = new Error(apiErrorMessage({ status: res.status, code, message: serverMessage }))
    err.code = code
    err.status = res.status
    err.serverMessage = serverMessage
    err.payload = payload
    throw err
  }
  return payload?.data
}
