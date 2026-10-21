import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  listAdminProcedures: vi.fn(),
}))
const drafts = vi.hoisted(() => ({
  listDraftSources: vi.fn(),
  listProcedureDrafts: vi.fn(),
  generateAIDraft: vi.fn(),
}))

vi.mock('../../api/catalog.js', () => api)
vi.mock('../../api/drafts.js', () => drafts)
vi.mock('../../commune/CommuneContext.jsx', () => ({
  useCommune: () => ({ xaId: 'xa_chu_se', commune: { name: 'Chư Sê' } }),
}))

async function renderPage() {
  const { default: Page } = await import('./DraftsPage.jsx')
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(MemoryRouter, null, createElement(Page)))
    await Promise.resolve()
  })
  return host
}

describe('DraftsPage', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    api.listAdminProcedures.mockReset()
    drafts.listDraftSources.mockReset()
    drafts.listProcedureDrafts.mockReset()
    drafts.generateAIDraft.mockReset()
    api.listAdminProcedures.mockResolvedValue({
      items: [{ id: 'proc-1', procedure_code: 'dk_khai_sinh', name: 'Đăng ký khai sinh' }],
    })
    drafts.listDraftSources.mockResolvedValue({
      items: [{
        document_id: 'doc-1',
        generation_id: 'gen-1',
        title: 'QĐ 163',
        version: '1.0.2',
        version_status: 'ARCHIVED',
        suggested_page_start: 10,
        suggested_page_end: 21,
      }],
    })
    drafts.listProcedureDrafts.mockResolvedValue({ items: [] })
  })

  it('shows generate action and empty drafts', async () => {
    const host = await renderPage()
    expect(host.textContent).toContain('Generate AI Draft')
    expect(host.textContent).toContain('Chưa có bản nháp')
    expect(host.textContent).toContain('QĐ 163')
  })
})
