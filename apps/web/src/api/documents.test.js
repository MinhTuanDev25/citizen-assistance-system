import { createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  buildUploadForm,
  uploadErrorMessage,
  validateUpload,
} from './documents.js'
import DocumentsPage from '../pages/admin/DocumentsPage.jsx'

vi.mock('./catalog.js', () => ({
  listDomains: vi.fn(async () => ({ items: [{ id: 'ho_tich_chung_thuc', name: 'Hộ tịch' }] })),
}))

vi.mock('./client.js', () => ({
  apiRequest: vi.fn(async () => ({
    max_bytes: 1048576,
    items: [
      {
        id: 'doc-1',
        title: 'Giấy khai sinh',
        filename: 'a.pdf',
        processing_status: 'UPLOADED',
        validity_status: 'PENDING',
        checksum: 'abc',
        file_size_bytes: 12,
      },
    ],
  })),
}))

const uploadDocument = vi.fn()
const downloadDocument = vi.fn()
vi.mock('./documents.js', async () => {
  const actual = await vi.importActual('./documents.js')
  return {
    ...actual,
    uploadDocument: (...args) => uploadDocument(...args),
    downloadDocument: (...args) => downloadDocument(...args),
  }
})

function file(name, size = 4) {
  return { name, size }
}

describe('document upload validation', () => {
  it('rejects a missing pdf, a bad name, and an empty title', () => {
    expect(validateUpload({ file: null, title: 'a', domainId: 'd' })).toMatch(/PDF/)
    expect(validateUpload({ file: file('a.txt'), title: 'a', domainId: 'd' })).toMatch(/pdf/)
    expect(validateUpload({ file: file('../a.pdf'), title: 'a', domainId: 'd' })).toMatch(/pdf/)
    expect(validateUpload({ file: file('a.pdf', 0), title: 'a', domainId: 'd' })).toMatch(/rỗng/)
    expect(validateUpload({ file: file('a.pdf', 10), title: '  ', domainId: 'd', maxBytes: 100 })).toMatch(/tiêu đề/)
    expect(validateUpload({ file: file('a.pdf', 10), title: 'a', domainId: 'd' })).toMatch(/giới hạn/)
    expect(validateUpload({ file: file('a.pdf', 30), title: 'a', domainId: 'd', maxBytes: 20 })).toMatch(/dung lượng/)
  })

  it('builds one file field and does not send xa_id', () => {
    const pdf = new File(['%PDF'], 'a.pdf', { type: 'application/pdf' })
    const body = buildUploadForm({ file: pdf, title: ' Giấy ', domainId: 'ho_tich_chung_thuc' })
    expect(body.get('file').name).toBe('a.pdf')
    expect(body.get('title')).toBe('Giấy')
    expect(body.get('domain_id')).toBe('ho_tich_chung_thuc')
    expect(body.get('xa_id')).toBeNull()
  })

  it('maps 409, 413 and 422', () => {
    expect(uploadErrorMessage(409)).toMatch(/trùng/)
    expect(uploadErrorMessage(413)).toMatch(/dung lượng/)
    expect(uploadErrorMessage(422)).toMatch(/không hợp lệ/)
    expect(uploadErrorMessage(401)).toMatch(/hết hạn/)
    expect(uploadErrorMessage(403)).toMatch(/quyền/)
    expect(uploadErrorMessage(500)).toMatch(/Thử lại/)
  })
})

describe('DocumentsPage', () => {
  beforeEach(() => {
    uploadDocument.mockReset()
  })

  it('shows the list and blocks a second submit while the first is in flight', async () => {
    let release
    uploadDocument.mockImplementation(() => new Promise((resolve) => {
      release = resolve
    }))
    const host = document.createElement('div')
    document.body.appendChild(host)
    const root = createRoot(host)
    await act(async () => {
      root.render(createElement(DocumentsPage))
    })
    await act(async () => {})
    expect(host.textContent).toContain('Giấy khai sinh')
    expect(host.textContent).toContain('UPLOADED')
    expect(host.textContent).toContain('PENDING')
    expect(host.textContent).not.toContain('Extract')
    const detail = [...host.querySelectorAll('button')].find((el) => el.textContent === 'Chi tiết')
    await act(async () => {
      detail.click()
    })
    expect(host.textContent).not.toContain('Activate')
    expect(host.textContent).not.toContain('Lập chỉ mục')
    expect(host.textContent).not.toContain('Thủ tục / phiên bản')

    const fileInput = host.querySelector('input[name=file]')
    const pdf = new File(['%PDF'], 'a.pdf', { type: 'application/pdf' })
    await act(async () => {
      Object.defineProperty(fileInput, 'files', { value: [pdf] })
      fileInput.dispatchEvent(new Event('change', { bubbles: true }))
    })
    const title = host.querySelector('input[name=title]')
    const setValue = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
    await act(async () => {
      setValue.call(title, 'Giấy')
      title.dispatchEvent(new Event('input', { bubbles: true }))
    })
    const form = host.querySelector('form')
    await act(async () => {
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    })
    expect(uploadDocument).toHaveBeenCalledTimes(1)
    expect(host.querySelector('button[type=submit]').disabled).toBe(true)
    await act(async () => {
      release({ id: 'new' })
    })
    root.unmount()
    host.remove()
  })

  it('shows a PDF download error on the page', async () => {
    downloadDocument.mockRejectedValue(new Error('Không tải được PDF.'))
    const host = document.createElement('div')
    document.body.appendChild(host)
    const root = createRoot(host)
    await act(async () => {
      root.render(createElement(DocumentsPage))
    })
    await act(async () => {})
    const button = [...host.querySelectorAll('button')].find((el) => el.textContent === 'Tải PDF')
    await act(async () => {
      button.click()
    })
    expect(host.querySelector('.form-error').textContent).toContain('Không tải được PDF.')
    root.unmount()
    host.remove()
  })
})
