import { loadAuth } from '../auth/auth.js'
import { apiRequest } from './client.js'

const SESSION_KEY = 'cas.chatSession'

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
}

function sessionCacheKey(xaId) {
  const auth = loadAuth()
  return `${xaId}:${auth?.id || 'guest'}`
}

function cacheMatchesAuth(cached) {
  const auth = loadAuth()
  if (auth?.accessToken) {
    // Logged-in cache: no guest token, same user id
    return !cached.guestToken && cached.userId === auth.id
  }
  // Guest cache
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

  // Guest resume via guest_token (server returns same OPEN session)
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

/** Create or resume session for current xã (guest_token or Bearer). */
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

export async function listSessionMessages(sessionId, guestToken) {
  return apiRequest(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
    guestToken: guestToken || undefined,
  })
}

export async function postUserMessage(sessionId, message, guestToken) {
  return apiRequest(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      method: 'POST',
      guestToken: guestToken || undefined,
      body: JSON.stringify({ message }),
    },
  )
}

export function mapApiMessage(m) {
  return {
    id: m.id,
    role: m.role === 'USER' ? 'user' : 'assistant',
    text: m.content,
    action: m.action || null,
    createdAt: m.created_at,
  }
}
