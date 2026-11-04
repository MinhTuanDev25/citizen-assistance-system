import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiErrorMessage, reasonsText } from './apiErrors.js'
import { apiRequest } from '../api/client.js'

describe('apiErrorMessage', () => {
  it('translates known server messages', () => {
    expect(apiErrorMessage({ status: 401, code: 'UNAUTHORIZED', message: 'email or password is incorrect' })).toBe('Email hoặc mật khẩu không đúng.')
    expect(apiErrorMessage({ status: 409, code: 'CONFLICT', message: 'email already registered' })).toBe('Email này đã được đăng ký. Hãy đăng nhập.')
  })

  it('translates wrapped server details', () => {
    expect(apiErrorMessage({ status: 422, code: 'REVIEW_INVALID', message: 'drafting: review decision rejected: reason must be 10-500 characters' }))
      .toBe('Lý do phải dài từ 10 đến 500 ký tự.')
    expect(apiErrorMessage({ status: 422, code: 'REVIEW_INVALID', message: 'drafting: review decision rejected: only DRAFT rows can be changed' }))
      .toContain('không sửa được')
  })

  it('falls back to the code, then the HTTP status, never to English', () => {
    expect(apiErrorMessage({ status: 422, code: 'APPROVAL_BLOCKED', message: 'drafting: approval blocked: something new' })).toBe('Chưa đủ điều kiện phê duyệt.')
    expect(apiErrorMessage({ status: 500, code: 'WHATEVER', message: 'boom' })).toBe('Hệ thống gặp lỗi. Thử lại sau.')
    expect(apiErrorMessage({ status: 404 })).toBe('Không tìm thấy dữ liệu.')
  })

  it('joins reason codes as Vietnamese and drops unknown codes', () => {
    expect(reasonsText(['INDEXING_PENDING', 'DOCUMENT_INDEX_PROCESSING', 'X_UNKNOWN'])).toBe('Tài liệu đang được lập chỉ mục.')
    expect(reasonsText(['X_UNKNOWN'])).toBe('')
  })
})

describe('apiRequest errors', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('shows Vietnamese and keeps the server text and code', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { code: 'UNAUTHORIZED', message: 'email or password is incorrect' } }), { status: 401 })))
    const err = await apiRequest('/api/v1/auth/login', { method: 'POST', auth: false, body: '{}' }).catch((e) => e)
    expect(err.message).toBe('Email hoặc mật khẩu không đúng.')
    expect(err.serverMessage).toBe('email or password is incorrect')
    expect(err.code).toBe('UNAUTHORIZED')
    expect(err.status).toBe(401)
  })

  it('reports a network failure in Vietnamese and keeps its name', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch') }))
    const err = await apiRequest('/api/v1/auth/me').catch((e) => e)
    expect(err.message).toBe('Không kết nối được máy chủ. Kiểm tra mạng rồi thử lại.')
    expect(err.name).toBe('TypeError')
    expect(err.status).toBeUndefined()
  })
})
