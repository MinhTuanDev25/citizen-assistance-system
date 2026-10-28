import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  listAdminProcedures: vi.fn(),
}))
const drafts = vi.hoisted(() => ({
  listDraftSources: vi.fn(),
  listProcedureDrafts: vi.fn(),
  listDraftJobs: vi.fn(),
  getDraftJob: vi.fn(),
  planDraft: vi.fn(),
  createDraftJob: vi.fn(),
  retryDraftJob: vi.fn(),
}))

vi.mock('../../api/catalog.js', () => api)
vi.mock('../../api/drafts.js', () => drafts)
vi.mock('../../commune/CommuneContext.jsx', () => ({
  useCommune: () => ({ xaId: 'xa_chu_se', commune: { name: 'Chư Sê' } }),
}))

const SOURCE = {
  document_id: 'doc-1', generation_id: 'gen-all', title: 'QĐ 163', version: '0.0.0', version_status: 'SOURCE',
  suggested_page_start: 1, suggested_page_end: 451,
}

const PLAN = {
  ready: true,
  document_pages: 451,
  document_chunks: 3215,
  section_title: 'Thủ tục đăng ký khai sinh',
  selected_section: 0,
  sections: [
    { index: 0, title: 'Thủ tục đăng ký khai sinh', page_start: 10, page_end: 21 },
    { index: 1, title: 'Thủ tục đăng ký khai sinh có yếu tố nước ngoài', page_start: 21, page_end: 33 },
  ],
  page_start: 10,
  page_end: 21,
  selected_chunks: 83,
  new_chunks: 83,
  mode: 'FULL',
  warnings: [],
  batches: [
    { index: 0, page_start: 10, page_end: 15, excerpts: 40 },
    { index: 1, page_start: 15, page_end: 20, excerpts: 40 },
    { index: 2, page_start: 20, page_end: 21, excerpts: 3 },
  ],
}

function job(status, extra = {}) {
  return {
    id: 'job-1', status, mode: 'FULL', batches_total: 3, batches_done: 0, batches_failed: 0,
    created_at: '2026-10-27T08:00:00Z', plan: { section_title: PLAN.section_title, page_start: 10, page_end: 21, new_chunks: 83 },
    batches: [], ...extra,
  }
}

async function flush() {
  for (let i = 0; i < 6; i += 1) {
    await act(async () => {
      await Promise.resolve()
    })
  }
}

async function renderPage() {
  const { default: Page } = await import('./DraftsPage.jsx')
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(MemoryRouter, null, createElement(Page)))
  })
  await flush()
  return host
}

function button(host, text) {
  return [...host.querySelectorAll('button')].find((node) => node.textContent.includes(text))
}

async function click(node) {
  await act(async () => {
    node.click()
  })
  await flush()
}

describe('DraftsPage long-document drafting', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    for (const fn of [...Object.values(api), ...Object.values(drafts)]) fn.mockReset()
    api.listAdminProcedures.mockResolvedValue({ items: [{ id: 'proc-1', procedure_code: 'khai_sinh_v1', name: 'Đăng ký khai sinh' }] })
    drafts.listDraftSources.mockResolvedValue({ items: [SOURCE] })
    drafts.listProcedureDrafts.mockResolvedValue({ items: [] })
    drafts.listDraftJobs.mockResolvedValue({ items: [] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('analyses the whole document and shows the procedure section and batches', async () => {
    drafts.planDraft.mockResolvedValue(PLAN)
    const host = await renderPage()
    expect(host.textContent).toContain('QĐ 163')
    await click(button(host, 'Phân tích phạm vi'))
    expect(drafts.planDraft).toHaveBeenCalledWith('proc-1', { document_id: 'doc-1', generation_id: 'gen-all', mode: 'FULL' })
    const summary = host.querySelector('.dj-plan-summary').textContent
    expect(summary).toContain('Tài liệu 451 trang · 3215 đoạn')
    expect(summary).toContain('Thủ tục đăng ký khai sinh')
    expect(summary).toContain('trang 10–21, 83 đoạn')
    expect(host.querySelectorAll('.dj-plan .dj-batch')).toHaveLength(3)
    expect(button(host, 'Bắt đầu soạn (3 đợt)')).toBeTruthy()
  })

  it('queues one job with the analysed body and a stable request id', async () => {
    drafts.planDraft.mockResolvedValue(PLAN)
    drafts.createDraftJob.mockResolvedValue(job('QUEUED'))
    drafts.getDraftJob.mockResolvedValue(job('RUNNING', { batches_done: 1 }))
    const host = await renderPage()
    await click(button(host, 'Phân tích phạm vi'))
    await click(button(host, 'Bắt đầu soạn'))
    expect(drafts.createDraftJob).toHaveBeenCalledTimes(1)
    const [proc, body, requestId] = drafts.createDraftJob.mock.calls[0]
    expect(proc).toBe('proc-1')
    expect(body).toEqual({ document_id: 'doc-1', generation_id: 'gen-all', mode: 'FULL' })
    expect(requestId).toMatch(/^[0-9a-f-]{36}$/)
    expect(host.textContent).toContain('Đã xếp hàng 3 đợt')
    expect(host.querySelector('.dj-job').textContent).toContain('0/3 đợt xong')
  })

  it('explains a blocked plan in Vietnamese and offers no start button', async () => {
    drafts.planDraft.mockResolvedValue({
      ...PLAN, ready: false, blocked_reason: 'MIXED_PROCEDURES', batches: [],
      warnings: ['Phạm vi trang chứa nội dung của thủ tục khác: Thủ tục đăng ký lại khai sinh.'],
    })
    const host = await renderPage()
    await click(button(host, 'Phân tích phạm vi'))
    expect(host.textContent).toContain('lẫn nội dung của thủ tục khác')
    expect(host.textContent).toContain('Thủ tục đăng ký lại khai sinh')
    expect(button(host, 'Bắt đầu soạn')).toBeUndefined()
  })

  it('sends a manual page range only when chosen and validates it first', async () => {
    drafts.planDraft.mockResolvedValue(PLAN)
    const host = await renderPage()
    await click(host.querySelector('input[name="scope"][value="manual"]'))
    const [start, end] = host.querySelectorAll('.draft-page-range input')
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
    await act(async () => {
      setter.call(start, '10')
      start.dispatchEvent(new Event('input', { bubbles: true }))
      setter.call(end, '500')
      end.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await click(button(host, 'Phân tích phạm vi'))
    expect(drafts.planDraft).not.toHaveBeenCalled()
    expect(host.textContent).toContain('Tài liệu chỉ có 451 trang.')
    await act(async () => {
      setter.call(end, '21')
      end.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await click(button(host, 'Phân tích phạm vi'))
    expect(drafts.planDraft.mock.calls[0][1]).toEqual({ document_id: 'doc-1', generation_id: 'gen-all', mode: 'FULL', page_start: 10, page_end: 21 })
  })

  it('shows a failed job with kept batches and retries only through the retry action', async () => {
    drafts.listDraftJobs.mockResolvedValue({
      items: [job('FAILED', {
        batches_done: 2, batches_failed: 1, error_message: '1/3 đợt chưa xong: đợt 3 (trang 20–21)',
        batches: [
          { batch_index: 0, page_start: 10, page_end: 15, status: 'SUCCEEDED', attempts: 1 },
          { batch_index: 1, page_start: 15, page_end: 20, status: 'SUCCEEDED', attempts: 1 },
          { batch_index: 2, page_start: 20, page_end: 21, status: 'FAILED', attempts: 3, error_code: 'DRAFT_TIMEOUT' },
        ],
      })],
    })
    drafts.retryDraftJob.mockResolvedValue(job('QUEUED', { batches_done: 2 }))
    drafts.getDraftJob.mockResolvedValue(job('QUEUED', { batches_done: 2 }))
    const host = await renderPage()
    const card = host.querySelector('.dj-job')
    expect(card.textContent).toContain('2/3 đợt xong · 1 đợt lỗi')
    expect(card.textContent).toContain('Đợt 3 · tr 20–21 · lỗi · lần 3 · Dịch vụ tự động xử lý quá thời gian')
    expect(card.textContent).toContain('chưa tạo bản nháp từ một phần nguồn')
    await click(button(host, 'Thử lại các đợt lỗi'))
    expect(drafts.retryDraftJob).toHaveBeenCalledWith('job-1')
  })

  it('polls a running job and links the merged draft when it succeeds', async () => {
    vi.useFakeTimers()
    drafts.listDraftJobs.mockResolvedValue({ items: [job('RUNNING', { batches_done: 1 })] })
    drafts.getDraftJob
      .mockResolvedValueOnce(job('RUNNING', { batches_done: 1 }))
      .mockResolvedValue(job('SUCCEEDED', { batches_done: 3, draft_id: 'draft-9' }))
    const host = await renderPage()
    expect(host.querySelector('.dj-job').textContent).toContain('1/3 đợt xong')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3100)
    })
    await flush()
    expect(host.querySelector('.dj-job').textContent).toContain('3/3 đợt xong')
    expect(host.querySelector('.dj-job a').getAttribute('href')).toBe('/admin/drafts/draft-9')
    expect(drafts.listProcedureDrafts).toHaveBeenCalledTimes(2)
  })

  it('blocks a range outside the procedure section and lists outside mentions', async () => {
    drafts.planDraft.mockResolvedValue({
      ...PLAN, ready: false, blocked_reason: 'OUTSIDE_SECTION', batches: [], warnings: [],
      outside_mentions: [
        { kind: 'name', section: '', page_start: 1, page_end: 9, chunks: 3 },
        { kind: 'name', section: 'Thủ tục đăng ký lại khai sinh', page_start: 33, page_end: 45, chunks: 20 },
        { kind: 'general', section: '', page_start: 2, page_end: 450, chunks: 3, pages: [{ start: 2, end: 2 }, { start: 100, end: 100 }, { start: 450, end: 450 }] },
      ],
    })
    const host = await renderPage()
    await click(button(host, 'Phân tích phạm vi'))
    expect(host.textContent).toContain('không chứa nội dung của mục thủ tục')
    const outside = host.querySelector('.dj-outside').textContent
    expect(outside).toContain('Các nguồn có thể liên quan ngoài phạm vi đã soạn (3 nhóm')
    expect(outside).toContain('Quy định chung (từ khóa lệ phí/thời hạn/biểu mẫu) · trang 2, 100, 450')
    expect(outside).toContain('Ngoài các mục thủ tục — nhắc tên thủ tục · trang 1–9 · 3 đoạn')
    expect(outside).toContain('Thủ tục đăng ký lại khai sinh · trang 33–45')
    expect(button(host, 'Bắt đầu soạn')).toBeUndefined()
  })
})
