import { describe, expect, it, beforeEach, vi } from 'vitest'
import { getActiveVersion, listProcedures } from './catalog.js'
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
