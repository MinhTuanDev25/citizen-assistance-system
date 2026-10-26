import { describe, expect, it, beforeEach, vi } from 'vitest'
import { activateProcedureVersion, getActiveVersion, listAdminProcedures, listProcedures } from './catalog.js'
import {
  executeChatTurn,
  loadPendingTurn,
  resolveTurnRequestId,
} from './session.js'

vi.mock('./client.js', () => ({
  apiRequest: vi.fn(),
}))

vi.mock('../auth/auth.js', () => ({
  loadAuth: vi.fn(() => null),
}))

import { apiRequest } from './client.js'

describe('listProcedures catalog auth flags', () => {
  beforeEach(() => {
    apiRequest.mockReset()
    apiRequest.mockResolvedValue({ domains: [], count: 0 })
  })

  it('citizen=true sends flag and skips Authorization', async () => {
    await listProcedures({ xaId: 'xa_chu_se', citizen: true })
    expect(apiRequest).toHaveBeenCalledTimes(1)
    const [path, opts] = apiRequest.mock.calls[0]
    expect(path).toContain('citizen=true')
    expect(opts.auth).toBe(false)
  })

  it('admin citizen=false requires auth option true', async () => {
    await listProcedures({ xaId: 'xa_chu_se', citizen: false })
    const [path, opts] = apiRequest.mock.calls[0]
    expect(path).toContain('citizen=false')
    expect(opts.auth).toBe(true)
  })

  it('rejects omitted citizen flag', async () => {
    await expect(listProcedures({ xaId: 'xa_chu_se' })).rejects.toThrow(/citizen/)
  })

  it('shares one request when two identical calls overlap', async () => {
    let resolveRequest
    apiRequest.mockImplementation(
      () => new Promise((resolve) => {
        resolveRequest = resolve
      }),
    )
    const first = listProcedures({ xaId: 'xa_chu_se', citizen: true })
    const second = listProcedures({ xaId: 'xa_chu_se', citizen: true })
    expect(apiRequest).toHaveBeenCalledTimes(1)
    resolveRequest({ domains: [{ id: 'ho_tich_chung_thuc' }] })
    await expect(first).resolves.toEqual({ domains: [{ id: 'ho_tich_chung_thuc' }] })
    await expect(second).resolves.toEqual({ domains: [{ id: 'ho_tich_chung_thuc' }] })
    apiRequest.mockResolvedValue({ domains: [] })
    await listProcedures({ xaId: 'xa_chu_se', citizen: true })
    expect(apiRequest).toHaveBeenCalledTimes(2)
  })
})

describe('getActiveVersion admin auth', () => {
  beforeEach(() => {
    apiRequest.mockReset()
    apiRequest.mockResolvedValue({ version: '1' })
  })

  it('always requests with Authorization', async () => {
    await getActiveVersion('proc-uuid-1')
    expect(apiRequest).toHaveBeenCalledWith(
      '/api/v1/procedures/proc-uuid-1/active-version',
      { auth: true },
    )
  })
})

describe('admin activation client', () => {
  beforeEach(() => {
    apiRequest.mockReset()
    apiRequest.mockResolvedValue({ items: [] })
  })

  it('lists admin procedures with auth', async () => {
    await listAdminProcedures()
    expect(apiRequest).toHaveBeenCalledWith('/api/v1/admin/procedures', { auth: true })
  })

  it('sends expected active version and request id', async () => {
    await activateProcedureVersion('p1', 'v2', 'v1', '11111111-1111-1111-1111-111111111111')
    const [path, opts] = apiRequest.mock.calls[0]
    expect(path).toContain('/admin/procedures/p1/versions/v2/activate')
    expect(opts.method).toBe('POST')
    expect(opts.headers['X-Request-ID']).toBe('11111111-1111-1111-1111-111111111111')
    expect(JSON.parse(opts.body)).toEqual({ expected_active_version_id: 'v1' })
  })
})

describe('executeChatTurn pending request-id flow', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('retries the same message with the same X-Request-ID after turn failure', async () => {
    const posts = []
    const failingPost = async (sessionId, message, guestToken, requestId) => {
      posts.push({ sessionId, message, guestToken, requestId })
      throw new Error('network down')
    }
    const first = await executeChatTurn({
      sessionId: 'sess-1',
      message: 'Xin chào',
      guestToken: 'g',
      post: failingPost,
      loadHistory: async () => ({ items: [] }),
    })
    expect(first.ok).toBe(false)
    expect(first.stage).toBe('turn')
    expect(first.pending?.requestId).toBe(posts[0].requestId)

    const succeedingPost = async (sessionId, message, guestToken, requestId) => {
      posts.push({ sessionId, message, guestToken, requestId })
      return { action: 'ASK_MISSING_SLOTS' }
    }
    const second = await executeChatTurn({
      sessionId: 'sess-1',
      message: 'Xin chào',
      guestToken: 'g',
      post: succeedingPost,
      loadHistory: async () => ({ items: [{ id: '1', role: 'ASSISTANT', content: 'ok' }] }),
    })
    expect(second.ok).toBe(true)
    expect(posts).toHaveLength(2)
    expect(posts[1].requestId).toBe(posts[0].requestId)
    expect(resolveTurnRequestId('sess-1', 'Xin chào')).not.toBe(posts[0].requestId)
  })

  it('keeps pending when postTurn fails', async () => {
    const ridHolder = { id: null }
    const result = await executeChatTurn({
      sessionId: 's',
      message: 'm',
      guestToken: null,
      post: async (_s, _m, _g, requestId) => {
        ridHolder.id = requestId
        throw new Error('503')
      },
      loadHistory: async () => {
        throw new Error('should not load history')
      },
    })
    expect(result.ok).toBe(false)
    expect(loadPendingTurn()?.requestId).toBe(ridHolder.id)
  })

  it('forwards an explicit speech_request_id without inventing one for typed chat', async () => {
    const seen = []
    await executeChatTurn({
      sessionId: 's-speech',
      message: 'tôi muốn khai sinh',
      guestToken: 'g',
      speechRequestId: '11111111-1111-1111-1111-111111111111',
      post: async (_s, message, _g, requestId, extras) => {
        seen.push({ message, requestId, extras })
        return { action: 'OUT_OF_SCOPE' }
      },
      loadHistory: async () => ({ items: [] }),
    })
    expect(seen).toHaveLength(1)
    expect(seen[0].extras.speechRequestId).toBe('11111111-1111-1111-1111-111111111111')

    seen.length = 0
    sessionStorage.clear()
    await executeChatTurn({
      sessionId: 's-typed',
      message: 'xin chào',
      guestToken: 'g',
      post: async (_s, message, _g, requestId, extras) => {
        seen.push({ message, extras })
        return { action: 'OUT_OF_SCOPE' }
      },
      loadHistory: async () => ({ items: [] }),
    })
    expect(seen[0].extras.speechRequestId).toBeUndefined()
  })

  it('retries the same pending turn with the first speech_request_id', async () => {
    const seen = []
    const first = '11111111-1111-1111-1111-111111111111'
    const swapped = '22222222-2222-2222-2222-222222222222'
    await executeChatTurn({
      sessionId: 's-pin',
      message: 'tôi muốn khai sinh',
      guestToken: 'g',
      speechRequestId: first,
      post: async (_s, _m, _g, requestId, extras) => {
        seen.push({ requestId, extras })
        throw new Error('network down')
      },
      loadHistory: async () => ({ items: [] }),
    })
    expect(loadPendingTurn()?.speechRequestId).toBe(first)

    await executeChatTurn({
      sessionId: 's-pin',
      message: 'tôi muốn khai sinh',
      guestToken: 'g',
      speechRequestId: swapped,
      post: async (_s, _m, _g, requestId, extras) => {
        seen.push({ requestId, extras })
        return { action: 'OUT_OF_SCOPE' }
      },
      loadHistory: async () => ({ items: [] }),
    })
    expect(seen).toHaveLength(2)
    expect(seen[1].requestId).toBe(seen[0].requestId)
    expect(seen[1].extras.speechRequestId).toBe(first)
    expect(seen[1].extras.speechRequestId).not.toBe(swapped)
  })

  it('does not re-post when history reload fails after successful turn', async () => {
    let postCount = 0
    const result = await executeChatTurn({
      sessionId: 's2',
      message: 'hello',
      guestToken: 'gt',
      post: async () => {
        postCount += 1
        return { action: 'OUT_OF_SCOPE' }
      },
      loadHistory: async () => {
        throw new Error('history unavailable')
      },
    })
    expect(result.ok).toBe(true)
    expect(result.historyFailed).toBe(true)
    expect(postCount).toBe(1)
    expect(loadPendingTurn()).toBeNull()
    const again = resolveTurnRequestId('s2', 'hello')
    expect(again).toBeTruthy()
  })
})
