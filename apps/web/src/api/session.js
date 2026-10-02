import { loadAuth } from '../auth/auth.js'
import { apiRequest } from './client.js'

const SESSION_KEY = 'cas.chatSession'
const PENDING_TURN_KEY = 'cas.pendingTurn'

/** Dedupe concurrent ensureSession (React StrictMode double-mount). */
const inflight = new Map()

export function loadChatSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed?.sessionId || !parsed?.xaId) return null
    return {
      sessionId: parsed.sessionId,
      xaId: parsed.xaId,
      guestToken: parsed.guestToken || null,
      userId: parsed.userId || null,
    }
  } catch {
    return null
  }
}

export function saveChatSession(session) {
  if (!session) {
    localStorage.removeItem(SESSION_KEY)
    return
  }
  localStorage.setItem(
    SESSION_KEY,
    JSON.stringify({
      sessionId: session.sessionId,
      xaId: session.xaId,
      guestToken: session.guestToken || null,
      userId: session.userId || null,
    }),
  )
}

export function clearChatSession() {
  localStorage.removeItem(SESSION_KEY)
  clearPendingTurn()
}

function sessionCacheKey(xaId) {
  const auth = loadAuth()
  return `${xaId}:${auth?.id || 'guest'}`
}

function cacheMatchesAuth(cached) {
  const auth = loadAuth()
  if (auth?.accessToken) {
    return !cached.guestToken && cached.userId === auth.id
  }
  return Boolean(cached.guestToken) && !cached.userId
}

async function tryReuseCached(xaId) {
  const cached = loadChatSession()
  if (!cached || cached.xaId !== xaId || !cacheMatchesAuth(cached)) {
    return null
  }
  try {
    await listSessionMessages(cached.sessionId, cached.guestToken)
    return cached
  } catch {
    clearChatSession()
    return null
  }
}

async function createOrResumeSession(xaId) {
  const auth = loadAuth()
  const cached = loadChatSession()
  const body = { xa_id: xaId }

  if (!auth?.accessToken && cached?.xaId === xaId && cached.guestToken) {
    body.guest_token = cached.guestToken
  }

  const data = await apiRequest('/api/v1/sessions', {
    method: 'POST',
    body: JSON.stringify(body),
  })

  const next = {
    sessionId: data.id,
    xaId: data.xa_id,
    guestToken: data.guest_token || null,
    userId: data.user_id || auth?.id || null,
  }
  saveChatSession(next)
  return next
}

export async function startFreshSession(xaId) {
  if (!xaId) throw new Error('xa_id is required')
  clearChatSession()
  const auth = loadAuth()
  const data = await apiRequest('/api/v1/sessions', {
    method: 'POST',
    body: JSON.stringify({ xa_id: xaId }),
  })
  const next = {
    sessionId: data.id,
    xaId: data.xa_id,
    guestToken: data.guest_token || null,
    userId: data.user_id || auth?.id || null,
  }
  saveChatSession(next)
  return next
}

export async function ensureSession(xaId) {
  if (!xaId) throw new Error('xa_id is required')

  const key = sessionCacheKey(xaId)
  if (inflight.has(key)) {
    return inflight.get(key)
  }

  const promise = (async () => {
    const reused = await tryReuseCached(xaId)
    if (reused) return reused
    return createOrResumeSession(xaId)
  })().finally(() => {
    inflight.delete(key)
  })

  inflight.set(key, promise)
  return promise
}

export async function listSessionMessages(sessionId, guestToken, limit) {
  const q =
    typeof limit === 'number' && limit > 0
      ? `?limit=${encodeURIComponent(String(limit))}`
      : ''
  return apiRequest(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages${q}`, {
    guestToken: guestToken || undefined,
  })
}

/** @deprecated Write path removed — use postTurn. */
export async function postUserMessage() {
  throw new Error('POST /sessions/:id/messages is deprecated; use postTurn')
}

export function loadPendingTurn() {
  try {
    const raw = sessionStorage.getItem(PENDING_TURN_KEY)
    if (!raw) return null
    const p = JSON.parse(raw)
    if (!p?.requestId || !p?.message || !p?.sessionId) return null
    return p
  } catch {
    return null
  }
}

export function savePendingTurn(pending) {
  if (!pending) {
    clearPendingTurn()
    return
  }
  sessionStorage.setItem(PENDING_TURN_KEY, JSON.stringify(pending))
}

export function clearPendingTurn() {
  sessionStorage.removeItem(PENDING_TURN_KEY)
}

/**
 * One decision turn. Reuses requestId for an in-flight / failed send of the same message.
 * Does not clear pending — callers use executeChatTurn for UI semantics.
 */
export async function postTurn(sessionId, message, guestToken, requestId) {
  const headers = {}
  if (requestId) {
    headers['X-Request-ID'] = requestId
  }
  return apiRequest(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/turns`,
    {
      method: 'POST',
      guestToken: guestToken || undefined,
      headers,
      body: JSON.stringify({ message }),
    },
  )
}

/**
 * Chat send flow used by the UI:
 * 1) resolve/reuse pending X-Request-ID for session+message
 * 2) postTurn — on failure pending is kept for retry
 * 3) on success clear pending, then reload history (history failure must not re-post)
 */
export async function executeChatTurn({
  sessionId,
  message,
  guestToken,
  post = postTurn,
  loadHistory = listSessionMessages,
}) {
  const requestId = resolveTurnRequestId(sessionId, message)
  let turnResult
  try {
    turnResult = await post(sessionId, message, guestToken, requestId)
  } catch (err) {
    return {
      ok: false,
      stage: 'turn',
      requestId,
      error: err,
      pending: loadPendingTurn(),
      turnResult: null,
      history: null,
    }
  }
  clearPendingTurn()
  try {
    const history = await loadHistory(sessionId, guestToken)
    return {
      ok: true,
      stage: 'done',
      requestId,
      error: null,
      pending: loadPendingTurn(),
      turnResult,
      history,
    }
  } catch (histErr) {
    return {
      ok: true,
      stage: 'history',
      requestId,
      error: histErr,
      pending: loadPendingTurn(),
      turnResult,
      history: null,
      historyFailed: true,
    }
  }
}

/** Resolve request id: reuse pending for same session+message, else mint new. */
export function resolveTurnRequestId(sessionId, message) {
  const pending = loadPendingTurn()
  if (
    pending &&
    pending.sessionId === sessionId &&
    pending.message === message &&
    pending.requestId
  ) {
    return pending.requestId
  }
  const id = newTurnRequestId()
  savePendingTurn({ sessionId, message, requestId: id })
  return id
}

export function newTurnRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function mapApiMessage(m) {
  const meta = m.message_metadata || m.metadata || {}
  return {
    id: m.id,
    role: m.role === 'USER' ? 'user' : 'assistant',
    text: m.content,
    action: m.action || null,
    candidates: meta.candidates || null,
    createdAt: m.created_at,
  }
}

const ROLE_ORDER = { USER: 0, SYSTEM: 1, ASSISTANT: 2 }

export function orderedMessages(items) {
  return [...(items || [])]
    .sort((a, b) => {
      const ta = a.created_at || ''
      const tb = b.created_at || ''
      if (ta < tb) return -1
      if (ta > tb) return 1
      return (ROLE_ORDER[a.role] ?? 9) - (ROLE_ORDER[b.role] ?? 9)
    })
    .map(mapApiMessage)
}
