import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.stubEnv('VITE_ADMIN_INDEXING', 'true')

const apiRequest = vi.fn()

vi.mock('../../api/catalog.js', () => ({
  listDomains: vi.fn(async () => ({ items: [{ id: 'ho_tich_chung_thuc', name: 'Hộ tịch' }] })),
}))

vi.mock('../../api/client.js', () => ({
  apiRequest: (...args) => apiRequest(...args),
}))

const uploadedLink = {
  procedure_version_id: 'ver-1',
  procedure_code: 'chung_thuc',
  version: '1.0.0',
  index_status: 'UPLOADED',
  recoverable: false,
}

let links = [uploadedLink]

function installApi() {
  links = [{ ...uploadedLink }]
  apiRequest.mockImplementation(async (path, options = {}) => {
    const url = String(path)
    if (url.includes('link-targets')) {
      return {
        items: [{
          procedure_version_id: 'ver-1',
          procedure_name: 'Chứng thực',
          version: '1.0.0',
        }],
      }
    }
    if (url.endsWith('/links/ver-1/index/reindex')) {
      links = [{ ...uploadedLink, index_status: 'READY', recoverable: false }]
      return { link_status: 'READY', processing_status: 'READY' }
    }
    if (url.endsWith('/links/ver-1/index/retry')) {
      links = [{ ...uploadedLink, index_status: 'READY', recoverable: false, last_error_code: undefined }]
      return { link_status: 'READY', processing_status: 'READY' }
    }
    if (url.endsWith('/links/ver-1/index')) {
      links = [{ ...uploadedLink, index_status: 'FAILED', recoverable: true, last_error_code: 'mock_failed' }]
      return { link_status: 'FAILED', processing_status: 'UPLOADED', error_code: 'mock_failed' }
    }
    if (options.method === 'DELETE' && url.includes('/links/ver-1')) {
      links = []
      return {}
    }
    if (url.includes('/links')) return { items: links }
    return {
      max_bytes: 1048576,
      items: [{
        id: 'doc-1',
        title: 'Giấy',
        filename: 'a.pdf',
        processing_status: 'UPLOADED',
        validity_status: 'PENDING',
        checksum: 'abc',
        file_size_bytes: 8,
      }],
    }
  })
}

function buttonNamed(host, label) {
  return [...host.querySelectorAll('button')].find((node) => node.textContent.trim() === label)
}

describe('DocumentsPage per-link indexing', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    installApi()
  })

  it('indexes, retries, and unlinks the selected link without Activate', async () => {
    const { default: DocumentsPage } = await import('./DocumentsPage.jsx')
    const host = document.createElement('div')
    document.body.appendChild(host)
    const root = createRoot(host)
    await act(async () => {
      root.render(createElement(DocumentsPage))
    })
    await act(async () => {
      await Promise.resolve()
    })
    await act(async () => {
      buttonNamed(host, 'Chi tiết').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(host.textContent).toContain('chung_thuc · 1.0.0 · UPLOADED')
    expect(host.textContent).not.toContain('Activate')
    const indexButton = buttonNamed(host, 'Lập chỉ mục')
    await act(async () => {
      indexButton.click()
      indexButton.click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    const indexCalls = apiRequest.mock.calls.filter((call) => String(call[0]).endsWith('/links/ver-1/index'))
    expect(indexCalls).toHaveLength(1)
    expect(indexCalls[0][1].method).toBe('POST')
    expect(host.textContent).toContain('FAILED')
    expect(host.textContent).toContain('mock_failed')
    await act(async () => {
      buttonNamed(host, 'Thử lại lập chỉ mục').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(apiRequest.mock.calls.some((call) => String(call[0]).endsWith('/links/ver-1/index/retry'))).toBe(true)
    expect(host.textContent).toContain('READY')
    expect(host.textContent).toContain('Lập lại chỉ mục')
    await act(async () => {
      buttonNamed(host, 'Lập lại chỉ mục').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(apiRequest.mock.calls.some((call) => String(call[0]).endsWith('/links/ver-1/index/reindex'))).toBe(true)
    await act(async () => {
      buttonNamed(host, 'Bỏ liên kết').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    const unlinked = apiRequest.mock.calls.find((call) => call[1]?.method === 'DELETE')
    expect(String(unlinked[0])).toContain('/documents/doc-1/links/ver-1')
    expect(host.textContent).not.toContain('chung_thuc')
    expect(host.textContent).not.toContain('Activate')
    act(() => root.unmount())
  })

  it('hides unlink while the link is PROCESSING and shows retry when the lease is recoverable', async () => {
    links = [{
      ...uploadedLink,
      index_status: 'PROCESSING',
      recoverable: true,
      last_error_code: 'lease_expired',
    }]
    const { default: DocumentsPage } = await import('./DocumentsPage.jsx')
    const host = document.createElement('div')
    document.body.appendChild(host)
    const root = createRoot(host)
    await act(async () => {
      root.render(createElement(DocumentsPage))
    })
    await act(async () => {
      buttonNamed(host, 'Chi tiết').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(host.textContent).toContain('PROCESSING')
    expect(buttonNamed(host, 'Bỏ liên kết')).toBeUndefined()
    expect(buttonNamed(host, 'Thử lại lập chỉ mục')).toBeTruthy()
    expect(host.textContent).not.toContain('Activate')
    act(() => root.unmount())
  })

  it('disables only the link whose index request is in flight', async () => {
    let releaseFirst
    links = [
      { ...uploadedLink, procedure_version_id: 'ver-1', procedure_code: 'mot' },
      { ...uploadedLink, procedure_version_id: 'ver-2', procedure_code: 'hai' },
    ]
    apiRequest.mockImplementation(async (path) => {
      const url = String(path)
      if (url.includes('link-targets')) return { items: [] }
      if (url.endsWith('/links/ver-1/index')) {
        await new Promise((resolve) => {
          releaseFirst = resolve
        })
        return { link_status: 'READY', processing_status: 'UPLOADED' }
      }
      if (url.endsWith('/links/ver-2/index')) {
        links = links.map((link) => (
          link.procedure_version_id === 'ver-2' ? { ...link, index_status: 'READY' } : link
        ))
        return { link_status: 'READY', processing_status: 'UPLOADED' }
      }
      if (url.includes('/links')) return { items: links }
      return {
        max_bytes: 1048576,
        items: [{
          id: 'doc-1',
          title: 'Giấy',
          filename: 'a.pdf',
          processing_status: 'UPLOADED',
          validity_status: 'PENDING',
          checksum: 'abc',
          file_size_bytes: 8,
        }],
      }
    })
    const { default: DocumentsPage } = await import('./DocumentsPage.jsx')
    const host = document.createElement('div')
    document.body.appendChild(host)
    const root = createRoot(host)
    await act(async () => {
      root.render(createElement(DocumentsPage))
    })
    await act(async () => {
      buttonNamed(host, 'Chi tiết').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    const indexButtons = () => [...host.querySelectorAll('button')].filter((node) => node.textContent.trim() === 'Lập chỉ mục')
    expect(indexButtons()).toHaveLength(2)
    await act(async () => {
      indexButtons()[0].click()
    })
    expect(indexButtons()[0].disabled).toBe(true)
    expect(indexButtons()[1].disabled).toBe(false)
    await act(async () => {
      indexButtons()[1].click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(apiRequest.mock.calls.filter((call) => String(call[0]).endsWith('/links/ver-2/index'))).toHaveLength(1)
    await act(async () => {
      releaseFirst()
    })
    act(() => root.unmount())
  })

  it('shows pipeline stats for one link and no search or Activate', async () => {
    links = [{
      ...uploadedLink,
      index_status: 'READY',
      pipeline_version: 'p4b.1',
      page_count: 2,
      chunk_count: 3,
      ocr_page_count: 1,
      embedding_model_id: 'intfloat/multilingual-e5-small',
    }]
    const { default: DocumentsPage } = await import('./DocumentsPage.jsx')
    const host = document.createElement('div')
    document.body.appendChild(host)
    const root = createRoot(host)
    await act(async () => {
      root.render(createElement(DocumentsPage))
    })
    await act(async () => {
      await Promise.resolve()
    })
    await act(async () => {
      buttonNamed(host, 'Chi tiết').click()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(host.textContent).toContain('2 trang')
    expect(host.textContent).toContain('3 đoạn')
    expect(host.textContent).toContain('OCR 1')
    expect(host.textContent).toContain('p4b.1')
    expect(host.textContent).toContain('intfloat/multilingual-e5-small')
    expect(host.textContent).not.toContain('Activate')
    expect(buttonNamed(host, 'Tìm kiếm')).toBeUndefined()
    act(() => root.unmount())
  })
})
