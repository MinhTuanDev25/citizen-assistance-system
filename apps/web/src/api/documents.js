import { loadAuth } from '../auth/auth.js'
import { apiRequest } from './client.js'

export function validateUpload({ file, title, domainId, maxBytes }) {
  if (!file) return 'Chọn file PDF.'
  const name = file.name || ''
  if (!name.toLowerCase().endsWith('.pdf') || name.includes('/') || name.includes('\\') || name.includes('..')) {
    return 'Chỉ nhận một file .pdf.'
  }
  if (!file.size) return 'File rỗng.'
  if (typeof maxBytes !== 'number' || maxBytes <= 0) return 'Chưa biết giới hạn dung lượng.'
  if (file.size > maxBytes) return 'File vượt quá dung lượng cho phép.'
  if (!String(title || '').trim()) return 'Nhập tiêu đề.'
  if (!domainId) return 'Chọn lĩnh vực.'
  return ''
}

export function buildUploadForm({ file, title, domainId, documentNumber, issuer }) {
  const body = new FormData()
  body.append('file', file, file.name)
  body.append('title', title.trim())
  body.append('domain_id', domainId)
  if (documentNumber) body.append('document_number', documentNumber.trim())
  if (issuer) body.append('issuer', issuer.trim())
  return body
}

export function uploadErrorMessage(status) {
  switch (status) {
    case 401:
      return 'Phiên đăng nhập hết hạn.'
    case 403:
      return 'Bạn không có quyền admin.'
    case 409:
      return 'Tài liệu đã tồn tại (trùng nội dung).'
    case 413:
      return 'File vượt quá dung lượng cho phép.'
    case 422:
      return 'Dữ liệu không hợp lệ.'
    default:
      return 'Không tải lên được. Thử lại sau.'
  }
}

export async function uploadDocument(fields) {
  const headers = { Accept: 'application/json' }
  const session = loadAuth()
  if (session?.accessToken) headers.Authorization = `Bearer ${session.accessToken}`
  const res = await fetch('/api/v1/admin/documents', {
    method: 'POST',
    headers,
    body: buildUploadForm(fields),
  })
  const payload = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = new Error(uploadErrorMessage(res.status))
    err.status = res.status
    err.code = payload?.error?.code
    throw err
  }
  return payload?.data
}

export function listDocuments(params = {}) {
  const q = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value) q.set(key, value)
  }
  const suffix = q.toString() ? `?${q}` : ''
  return apiRequest(`/api/v1/admin/documents${suffix}`)
}

export function listLinkTargets() {
  return apiRequest('/api/v1/admin/documents/link-targets')
}

export function listDocumentLinks(id) {
  return apiRequest(`/api/v1/admin/documents/${id}/links`)
}

export function linkDocument(id, body) {
  return apiRequest(`/api/v1/admin/documents/${id}/links`, {
    method: 'POST',
    headers: { 'X-Request-ID': crypto.randomUUID() },
    body: JSON.stringify(body),
  })
}

export function unlinkDocument(id, versionId) {
  return apiRequest(`/api/v1/admin/documents/${id}/links/${versionId}`, {
    method: 'DELETE',
    headers: { 'X-Request-ID': crypto.randomUUID() },
  })
}

export function requestIndexing(id, versionId, retry = false) {
  const path = retry ? 'index/retry' : 'index'
  return apiRequest(`/api/v1/admin/documents/${id}/links/${versionId}/${path}`, {
    method: 'POST',
    headers: { 'X-Request-ID': crypto.randomUUID() },
  })
}

export function requestReindex(id, versionId) {
  return apiRequest(`/api/v1/admin/documents/${id}/links/${versionId}/index/reindex`, {
    method: 'POST',
    headers: { 'X-Request-ID': crypto.randomUUID() },
  })
}

export async function downloadDocument(id, filename) {
  const headers = {}
  const session = loadAuth()
  if (session?.accessToken) headers.Authorization = `Bearer ${session.accessToken}`
  const res = await fetch(`/api/v1/admin/documents/${id}/content`, { headers })
  if (!res.ok) {
    const err = new Error(uploadErrorMessage(res.status))
    err.status = res.status
    throw err
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename || 'document.pdf'
  a.click()
  URL.revokeObjectURL(url)
}
