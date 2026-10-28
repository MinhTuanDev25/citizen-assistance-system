import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.stubEnv('VITE_ADMIN_INDEXING', 'true')

const apiRequest = vi.fn()

vi.mock('../../api/catalog.js', () => ({
  listDomains: vi.fn(async () => ({ items: [{ id: 'ho_tich_chung_thuc', name: 'Hộ tịch' }] })),
}))

vi.mock('../../api/client.js', () => ({
  apiRequest: (...args) => apiRequest(...args),
}))

const base = {
  procedure_version_id: 'src-1',
  procedure_code: 'khai_sinh_v1',
  version: '0.0.0',
  recoverable: true,
}

let links = []

function install(initial, onRetry) {
  links = initial
  apiRequest.mockImplementation(async (path) => {
    const url = String(path)
    if (url.includes('link-targets')) {
      return { items: [{ procedure_version_id: 'src-1', procedure_name: 'Đăng ký khai sinh', version: '0.0.0', status: 'SOURCE' }] }
    }
    if (url.endsWith('/links/src-1/index/retry')) return onRetry()
    if (url.includes('/links')) return { items: links }
    return {
      max_bytes: 1048576,
      items: [{ id: 'doc-1', title: 'QĐ 163', filename: 'PL_163.pdf', processing_status: 'FAILED', validity_status: 'PENDING', checksum: 'abc', file_size_bytes: 6265417 }],
    }
  })
}

function buttonNamed(host, label) {
  return [...host.querySelectorAll('button')].find((node) => node.textContent.trim() === label)
}

async function flush() {
  for (let i = 0; i < 5; i += 1) {
    await act(async () => {
      await Promise.resolve()
    })
  }
}

async function openDetail() {
  const { default: DocumentsPage } = await import('./DocumentsPage.jsx')
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(MemoryRouter, null, createElement(DocumentsPage)))
  })
  await flush()
  await act(async () => {
    buttonNamed(host, 'Chi tiết').click()
  })
  await flush()
  return host
}

function linkCalls() {
  return apiRequest.mock.calls.filter((call) => String(call[0]).endsWith('/documents/doc-1/links')).length
}

describe('progressNote', () => {
  it('describes queued, page and stage progress', async () => {
    const { progressNote, indexJobActive } = await import('./DocumentsPage.jsx')
    expect(progressNote({})).toBe('')
    expect(progressNote({ progress: {} })).toBe('Đang chờ xử lý…')
    expect(progressNote({ progress: { stage: 'extract', pages_total: 451, pages_done: 120, pages_ocr: 98 } }))
      .toBe('Nhận dạng chữ · trang 120/451 · nhận dạng 98 trang')
    expect(progressNote({ progress: { stage: 'embed', pages_total: 451, pages_done: 451, pages_rejected: 2 } }))
      .toBe('Tạo dữ liệu tìm kiếm · trang 451/451 · 2 trang không đạt')
    expect(indexJobActive({ index_status: 'READY', progress: { stage: 'extract' } })).toBe(true)
    expect(indexJobActive({ index_status: 'READY' })).toBe(false)
  })
})

describe('DocumentsPage background indexing', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    apiRequest.mockReset()
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('retries without re-linking, shows page progress, and stops polling when READY', async () => {
    install(
      [{ ...base, index_status: 'FAILED', last_error_code: 'extract_unusable' }],
      () => {
        links = [{ ...base, index_status: 'PROCESSING', recoverable: false, progress: { stage: 'extract', pages_total: 451, pages_done: 120, pages_ocr: 98 } }]
        return { link_status: 'PROCESSING', processing_status: 'PROCESSING', job_status: 'CLAIMED', accepted: true }
      },
    )
    const host = await openDetail()
    expect(host.textContent).toContain('Đăng ký khai sinh · phiên nguồn')
    expect(host.textContent).toContain('chưa đọc được tiếng Việt có dấu')

    await act(async () => {
      buttonNamed(host, 'Thử lại lập chỉ mục').click()
    })
    await flush()
    expect(apiRequest.mock.calls.some((call) => String(call[0]).endsWith('/links/src-1/index/retry'))).toBe(true)
    expect(apiRequest.mock.calls.some((call) => call[1]?.method === 'POST' && String(call[0]).endsWith('/links'))).toBe(false)
    expect(host.textContent).toContain('Đang lập chỉ mục trong nền')
    expect(host.textContent).toContain('Nhận dạng chữ · trang 120/451 · nhận dạng 98 trang')
    expect(host.querySelector('progress')?.getAttribute('max')).toBe('451')

    links = [{ ...base, index_status: 'PROCESSING', recoverable: false, progress: { stage: 'embed', pages_total: 451, pages_done: 451, pages_ocr: 363 } }]
    await act(async () => {
      vi.advanceTimersByTime(3000)
    })
    await flush()
    expect(host.textContent).toContain('Tạo dữ liệu tìm kiếm · trang 451/451 · nhận dạng 363 trang')

    links = [{ ...base, index_status: 'READY', recoverable: false, page_count: 451, chunk_count: 1200 }]
    await act(async () => {
      vi.advanceTimersByTime(3000)
    })
    await flush()
    expect(host.textContent).toContain('Sẵn sàng')
    expect(host.querySelector('progress')).toBeNull()
    const afterReady = linkCalls()
    await act(async () => {
      vi.advanceTimersByTime(9000)
    })
    await flush()
    expect(linkCalls()).toBe(afterReady)
  })

  it('names the pages rejected by the OCR quality gate', async () => {
    install(
      [{ ...base, index_status: 'FAILED', last_error_code: 'ocr_quality_low', last_report: { rejected_pages: [7, 9], max_rejected_pages: 0 } }],
      () => ({}),
    )
    const host = await openDetail()
    expect(host.textContent).toContain('Kết quả nhận dạng chữ không đạt chất lượng ở trang 7, 9')
    expect(host.textContent).not.toContain('ocr_quality_low')
  })
})
