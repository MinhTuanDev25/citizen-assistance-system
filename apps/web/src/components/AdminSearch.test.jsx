import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const listDocuments = vi.fn()
const listAdminProcedures = vi.fn()

vi.mock('../api/documents.js', () => ({
  listDocuments: (...args) => listDocuments(...args),
}))

vi.mock('../api/catalog.js', () => ({
  listAdminProcedures: (...args) => listAdminProcedures(...args),
}))

describe('AdminSearch', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    listDocuments.mockReset()
    listAdminProcedures.mockReset()
    listDocuments.mockResolvedValue({
      items: [{ id: 'd1', title: 'hi', filename: 'architecture.pdf' }],
    })
    listAdminProcedures.mockResolvedValue({
      items: [{
        id: 'p1',
        name: 'Chứng thực bản sao',
        procedure_code: 'chung_thuc_ban_sao',
        active_version: '2.2.3',
      }],
    })
  })

  it('lists matching documents and procedures', async () => {
    const { default: AdminSearch } = await import('./AdminSearch.jsx')
    const host = document.createElement('div')
    document.body.appendChild(host)
    await act(async () => {
      createRoot(host).render(createElement(MemoryRouter, null, createElement(AdminSearch)))
      await Promise.resolve()
    })
    await act(async () => {
      await Promise.resolve()
    })
    const input = host.querySelector('input[type=search]')
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
      setter.call(input, 'chung')
      input.dispatchEvent(new Event('input', { bubbles: true }))
      await Promise.resolve()
    })
    expect(host.textContent).toContain('Chứng thực bản sao')
    expect(host.textContent).toContain('Thủ tục')
    expect(host.textContent).not.toContain('architecture.pdf')
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
      setter.call(input, 'hi')
      input.dispatchEvent(new Event('input', { bubbles: true }))
      await Promise.resolve()
    })
    expect(host.textContent).toContain('Tài liệu')
    expect(host.textContent).toContain('hi')
  })
})
