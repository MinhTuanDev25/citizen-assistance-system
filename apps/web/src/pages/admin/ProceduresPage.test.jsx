import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { isUnknownActivationOutcome, validateNewProcedure } from './ProceduresPage.jsx'

const api = vi.hoisted(() => ({
  listAdminProcedures: vi.fn(),
  listProcedureVersions: vi.fn(),
  listActivationHistory: vi.fn(),
  activateProcedureVersion: vi.fn(),
  createProcedureVersion: vi.fn(),
  createProcedure: vi.fn(),
  listDomains: vi.fn(),
  getActiveVersion: vi.fn(),
}))

vi.mock('../../api/catalog.js', () => api)
vi.mock('../../commune/CommuneContext.jsx', () => ({
  useCommune: () => ({ xaId: 'xa_chu_se', commune: { name: 'Chư Sê' } }),
}))

function buttonNamed(host, label) {
  return [...host.querySelectorAll('button')].find((node) => node.textContent.trim() === label)
}

const v1Proc = {
  id: 'proc-1',
  procedure_code: 'chung_thuc_ban_sao',
  name: 'Chứng thực bản sao',
  active_version: '1.0.0',
  active_version_id: 'v1',
  version_count: 2,
  legacy_active: true,
}

const v2Proc = {
  ...v1Proc,
  active_version: '2.0.0',
  active_version_id: 'v2',
  legacy_active: false,
}

const versionsBefore = [
  { id: 'v1', version: '1.0.0', status: 'ACTIVE', legacy_active: true, documents: [], eligibility: { can_activate: false, reasons: ['VERSION_ALREADY_ACTIVE'] } },
  { id: 'v2', version: '2.0.0', status: 'APPROVED', documents: [{ document_id: 'd1', title: 'PDF', index_status: 'READY' }], eligibility: { can_activate: true, reasons: [] } },
]

const versionsAfter = [
  { id: 'v1', version: '1.0.0', status: 'ARCHIVED', legacy_active: false, documents: [], eligibility: { can_activate: false, reasons: ['VERSION_ARCHIVED'] } },
  { id: 'v2', version: '2.0.0', status: 'ACTIVE', legacy_active: false, documents: [{ document_id: 'd1', title: 'PDF', index_status: 'READY' }], eligibility: { can_activate: false, reasons: ['VERSION_ALREADY_ACTIVE'] } },
]

async function renderPage() {
  const { default: Page } = await import('./ProceduresPage.jsx')
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(MemoryRouter, null, createElement(Page)))
    await Promise.resolve()
  })
  return host
}

async function openVersions(host) {
  await act(async () => {
    buttonNamed(host, 'Xem phiên bản').click()
    await Promise.resolve()
  })
}

async function openConfirm(host) {
  await openVersions(host)
  await act(async () => {
    buttonNamed(host, 'Kích hoạt').click()
    await Promise.resolve()
  })
}

async function confirm(host) {
  await act(async () => {
    buttonNamed(host, 'Xác nhận').click()
    await Promise.resolve()
  })
}

describe('isUnknownActivationOutcome', () => {
  it('treats timeout and network errors as unknown', () => {
    expect(isUnknownActivationOutcome(Object.assign(new Error('timeout'), { name: 'TimeoutError' }))).toBe(true)
    expect(isUnknownActivationOutcome(Object.assign(new Error('Failed to fetch'), { name: 'TypeError' }))).toBe(true)
    expect(isUnknownActivationOutcome(Object.assign(new Error('active version changed'), { status: 409, code: 'STALE_ACTIVE_VERSION' }))).toBe(false)
    expect(isUnknownActivationOutcome(Object.assign(new Error('ineligible'), { status: 422, code: 'DEFINITION_IDENTITY_MISMATCH' }))).toBe(false)
  })
})

describe('ProceduresPage activation', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    api.listAdminProcedures.mockReset()
    api.listProcedureVersions.mockReset()
    api.listActivationHistory.mockReset()
    api.activateProcedureVersion.mockReset()
    api.createProcedureVersion.mockReset()
    api.listAdminProcedures.mockResolvedValue({ items: [v1Proc] })
    api.listActivationHistory.mockResolvedValue({ items: [] })
    api.listProcedureVersions.mockResolvedValue({ items: versionsBefore })
    api.activateProcedureVersion.mockResolvedValue({ active_version_id: 'v2', replayed: false })
  })

  it('shows candidate and disables activate when ineligible', async () => {
    const host = await renderPage()
    expect(host.textContent).toContain('chung_thuc_ban_sao')
    await openVersions(host)
    expect(host.textContent).toContain('2.0.0')
    expect(buttonNamed(host, 'Kích hoạt')).toBeTruthy()
    expect(host.textContent).toContain('dữ liệu mẫu')
    expect(host.textContent).toContain('Phiên bản này đang được dùng.')
  })

  it('names the failed document instead of SOURCE jargon', async () => {
    api.listProcedureVersions.mockResolvedValue({
      items: [{
        id: 'v2',
        version: '2.2.2',
        status: 'APPROVED',
        documents: [
          { document_id: 'd1', title: 'hi', index_status: 'READY' },
          { document_id: 'd2', title: 'Test nè', index_status: 'FAILED' },
        ],
        eligibility: { can_activate: false, reasons: ['DOCUMENT_INDEX_FAILED', 'PARTIAL_SOURCE_INDEX'] },
      }],
    })
    const host = await renderPage()
    await openVersions(host)
    expect(host.textContent).toContain('«Test nè» lỗi')
    expect(host.textContent).toContain('hi · Sẵn sàng')
    expect(host.textContent).not.toContain('SOURCE')
    expect(host.textContent).not.toContain('DOCUMENT_INDEX_FAILED')
  })

  it('confirms activate once and blocks double click', async () => {
    let resolveActivate
    api.activateProcedureVersion.mockImplementation(
      () => new Promise((resolve) => {
        resolveActivate = resolve
      }),
    )
    const host = await renderPage()
    await openConfirm(host)
    expect(host.textContent).toContain('Chuyển phiên bản đang dùng?')
    expect(host.textContent).not.toContain('X-Request-ID')
    await act(async () => {
      buttonNamed(host, 'Xác nhận').click()
      buttonNamed(host, 'Xác nhận')?.click()
      await Promise.resolve()
    })
    expect(api.activateProcedureVersion).toHaveBeenCalledTimes(1)
    const args = api.activateProcedureVersion.mock.calls[0]
    expect(args[0]).toBe('proc-1')
    expect(args[1]).toBe('v2')
    expect(args[2]).toBe('v1')
    expect(args[3]).toBeTruthy()
    await act(async () => {
      resolveActivate({ active_version_id: 'v2', replayed: false })
      await Promise.resolve()
    })
  })

  it('activate success with refresh success updates ACTIVE display', async () => {
    api.listAdminProcedures
      .mockResolvedValueOnce({ items: [v1Proc] })
      .mockResolvedValueOnce({ items: [v2Proc] })
    api.listProcedureVersions
      .mockResolvedValueOnce({ items: versionsBefore })
      .mockResolvedValueOnce({ items: versionsAfter })
    api.listActivationHistory
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ items: [{ request_id: 'rid', created_at: 'now', new_active_version_id: 'v2' }] })
    const host = await renderPage()
    await openConfirm(host)
    await confirm(host)
    expect(host.textContent).toContain('Đã chuyển sang phiên bản mới')
    expect(host.textContent).not.toContain('Kích hoạt thất bại')
    expect(host.textContent).toContain('2.0.0')
    expect(host.textContent).toContain('Đang dùng')
    expect(host.querySelector('[role="dialog"]')).toBeNull()
  })

  it('keeps activate success when the procedure list refresh fails', async () => {
    let listCalls = 0
    api.listAdminProcedures.mockImplementation(async () => {
      listCalls += 1
      if (listCalls === 1) return { items: [v1Proc] }
      throw new Error('refresh failed')
    })
    const host = await renderPage()
    await openConfirm(host)
    await confirm(host)
    expect(host.textContent).toContain('Đã chuyển sang phiên bản mới')
    expect(host.textContent).toContain('chưa tải lại được dữ liệu mới')
    expect(host.textContent).not.toContain('Kích hoạt thất bại')
  })

  it('keeps activate success when history API fails after commit', async () => {
    api.listAdminProcedures
      .mockResolvedValueOnce({ items: [v1Proc] })
      .mockResolvedValueOnce({ items: [v2Proc] })
    api.listProcedureVersions
      .mockResolvedValueOnce({ items: versionsBefore })
      .mockResolvedValueOnce({ items: versionsAfter })
    let histCalls = 0
    api.listActivationHistory.mockImplementation(async () => {
      histCalls += 1
      if (histCalls === 1) return { items: [] }
      throw new Error('history failed')
    })
    const host = await renderPage()
    await openConfirm(host)
    await confirm(host)
    expect(host.textContent).toContain('Đã chuyển sang phiên bản mới')
    expect(host.textContent).toContain('chưa tải lại được lịch sử')
    expect(host.textContent).not.toContain('Kích hoạt thất bại')
  })

  it('shows 409 conflict without claiming success', async () => {
    const err = new Error('active version changed')
    err.status = 409
    err.code = 'STALE_ACTIVE_VERSION'
    api.activateProcedureVersion.mockRejectedValue(err)
    const host = await renderPage()
    await openConfirm(host)
    await confirm(host)
    expect(host.textContent).toContain('Phiên bản đang dùng đã đổi')
    expect(host.textContent).not.toContain('Đã chuyển sang phiên bản mới')
  })

  it('shows 422 identity mismatch in Vietnamese', async () => {
    const err = new Error('definition mismatch')
    err.status = 422
    err.code = 'DEFINITION_IDENTITY_MISMATCH'
    api.activateProcedureVersion.mockRejectedValue(err)
    const host = await renderPage()
    await openConfirm(host)
    await confirm(host)
    expect(host.textContent).toContain('Nội dung hướng dẫn không khớp thủ tục')
    expect(host.textContent).not.toContain('Đã chuyển sang phiên bản mới')
  })

  it('treats activate timeout as unknown and retries the same request id', async () => {
    const timeout = Object.assign(new Error('timeout'), { name: 'TimeoutError' })
    api.activateProcedureVersion
      .mockRejectedValueOnce(timeout)
      .mockResolvedValueOnce({ active_version_id: 'v2', replayed: true })
    api.listAdminProcedures
      .mockResolvedValueOnce({ items: [v1Proc] })
      .mockResolvedValueOnce({ items: [v2Proc] })
    api.listProcedureVersions
      .mockResolvedValueOnce({ items: versionsBefore })
      .mockResolvedValueOnce({ items: versionsAfter })
    const host = await renderPage()
    await openConfirm(host)
    await confirm(host)
    expect(host.textContent).toContain('Chưa rõ đã kích hoạt thành công chưa')
    expect(host.textContent).not.toContain('X-Request-ID')
    expect(host.textContent).not.toContain('Kích hoạt thất bại')
    expect(host.querySelector('[role="dialog"]')).toBeTruthy()
    const firstKey = api.activateProcedureVersion.mock.calls[0][3]
    await confirm(host)
    expect(api.activateProcedureVersion).toHaveBeenCalledTimes(2)
    expect(api.activateProcedureVersion.mock.calls[1][3]).toBe(firstKey)
    expect(host.textContent).toContain('Đã chuyển sang phiên bản mới')
  })

  it('explains that creating a candidate requires an ACTIVE version', async () => {
    api.listAdminProcedures.mockResolvedValue({
      items: [{
        id: 'proc-1',
        procedure_code: 'chung_thuc_ban_sao',
        name: 'Chứng thực bản sao',
        active_version: null,
        active_version_id: null,
        version_count: 0,
      }],
    })
    api.listProcedureVersions.mockResolvedValue({ items: [] })
    const host = await renderPage()
    await openVersions(host)
    await act(async () => {
      buttonNamed(host, 'Tạo phiên bản mới').click()
      await Promise.resolve()
    })
    expect(api.createProcedureVersion).not.toHaveBeenCalled()
    expect(host.textContent).toContain('chưa có phiên bản đang dùng')
  })

  it('shows a user-facing version error inside the create dialog', async () => {
    const host = await renderPage()
    await openVersions(host)
    const createButton = buttonNamed(host, 'Tạo phiên bản mới')
    expect(createButton.className).toContain('btn-primary')
    expect(createButton.className).not.toContain('linkish')
    await act(async () => {
      createButton.click()
      await Promise.resolve()
    })
    const input = host.querySelector('input')
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
      setter.call(input, 'v2')
      input.dispatchEvent(new Event('input', { bubbles: true }))
      await Promise.resolve()
    })
    await act(async () => {
      buttonNamed(host, 'Tạo phiên bản').click()
      await Promise.resolve()
    })
    expect(api.createProcedureVersion).not.toHaveBeenCalled()
    const alert = host.querySelector('[role="alert"]')
    expect(alert?.textContent).toContain('Số phiên bản phải có dạng n.n.n, ví dụ 2.0.0.')
    expect(host.querySelector('[role="dialog"]')).toBeTruthy()
  })

  it('shows the server validation error inside the create dialog', async () => {
    const err = new Error('validation: version must be n.n.n')
    err.status = 400
    err.code = 'VALIDATION'
    api.createProcedureVersion.mockRejectedValue(err)
    const host = await renderPage()
    await openVersions(host)
    await act(async () => {
      buttonNamed(host, 'Tạo phiên bản mới').click()
      await Promise.resolve()
    })
    await act(async () => {
      buttonNamed(host, 'Tạo phiên bản').click()
      await Promise.resolve()
    })
    const alert = host.querySelector('[role="alert"]')
    expect(alert?.textContent).toContain('Số phiên bản phải có dạng n.n.n, ví dụ 2.0.0.')
    expect(alert?.textContent).toContain('Để trống nếu muốn hệ thống tự tăng.')
    expect(host.textContent).not.toContain('validation: version must be n.n.n')
    expect(host.querySelector('[role="dialog"]')).toBeTruthy()
  })
})

function setField(host, name, value) {
  const node = host.querySelector(`[name="${name}"]`)
  const proto = node.tagName === 'SELECT' ? window.HTMLSelectElement.prototype : window.HTMLInputElement.prototype
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(node, value)
  node.dispatchEvent(new Event(node.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }))
}

async function flush() {
  for (let i = 0; i < 4; i += 1) await Promise.resolve()
}

const domainItems = [
  { id: 'ho_tich_chung_thuc', name: 'Hộ tịch & Chứng thực' },
  { id: 'dat_dai_nha_o_quy_hoach', name: 'Đất đai, Nhà ở & Quy hoạch' },
]

const createdProc = {
  id: 'proc-new',
  procedure_code: 'dk_khai_sinh',
  name: 'Đăng ký khai sinh',
  active_version: '',
  active_version_id: null,
  version_count: 1,
  legacy_active: false,
}

const sourceVersion = {
  id: 'src-1', version: '0.0.0', status: 'SOURCE', documents: [],
  eligibility: { can_activate: false, reasons: ['SOURCE_VERSION'] },
}

async function openNewProcedureDialog(host) {
  await act(async () => {
    buttonNamed(host, '+ Tạo thủ tục').click()
    await flush()
  })
}

async function fillNewProcedure(host, { code, name, domainId }) {
  await act(async () => {
    setField(host, 'procedure_code', code)
    setField(host, 'name', name)
    setField(host, 'domain_id', domainId)
    await flush()
  })
}

async function submitNewProcedure(host) {
  await act(async () => {
    buttonNamed(host, 'Tạo thủ tục').click()
    await flush()
  })
}

describe('validateNewProcedure', () => {
  it('matches the API code rule and requires name and domain', () => {
    expect(validateNewProcedure({ code: 'dk_khai_sinh', name: 'Đăng ký', domainId: 'ho_tich_chung_thuc' })).toBe('')
    expect(validateNewProcedure({ code: 'DK_Khai', name: 'x', domainId: 'd' })).toContain('chữ thường')
    expect(validateNewProcedure({ code: 'dk-khai', name: 'x', domainId: 'd' })).toContain('chữ thường')
    expect(validateNewProcedure({ code: 'a', name: 'x', domainId: 'd' })).toContain('chữ thường')
    expect(validateNewProcedure({ code: 'dk_khai', name: '  ', domainId: 'd' })).toBe('Nhập tên thủ tục.')
    expect(validateNewProcedure({ code: 'dk_khai', name: 'x', domainId: '' })).toBe('Chọn lĩnh vực.')
  })
})

describe('ProceduresPage create procedure', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    for (const fn of Object.values(api)) fn.mockReset()
    api.listAdminProcedures.mockResolvedValue({ items: [] })
    api.listActivationHistory.mockResolvedValue({ items: [] })
    api.listProcedureVersions.mockResolvedValue({ items: [sourceVersion] })
    api.listDomains.mockResolvedValue({ items: domainItems, count: 2 })
  })

  it('offers + Tạo thủ tục on an empty commune and creates the first procedure', async () => {
    api.createProcedure.mockResolvedValue({ ...createdProc, source_version_id: 'src-1', source_version: '0.0.0', replayed: false })
    const host = await renderPage()
    expect(host.textContent).toContain('Chưa có thủ tục trong xã này.')
    await openNewProcedureDialog(host)
    expect(api.listDomains).toHaveBeenCalledTimes(1)
    expect(host.querySelectorAll('select[name="domain_id"] option').length).toBe(3)

    api.listAdminProcedures.mockResolvedValue({ items: [createdProc] })
    await fillNewProcedure(host, { code: 'dk_khai_sinh', name: 'Đăng ký khai sinh', domainId: 'ho_tich_chung_thuc' })
    await submitNewProcedure(host)

    expect(api.createProcedure).toHaveBeenCalledTimes(1)
    const [body, requestId] = api.createProcedure.mock.calls[0]
    expect(body).toEqual({ procedure_code: 'dk_khai_sinh', name: 'Đăng ký khai sinh', domain_id: 'ho_tich_chung_thuc' })
    expect(requestId).toMatch(/^[0-9a-f-]{36}$/)
    expect(host.querySelector('[aria-labelledby="new-proc-title"]')).toBeNull()
    expect(host.textContent).toContain('Đã tạo thủ tục «Đăng ký khai sinh»')
    expect(host.textContent).toContain('dk_khai_sinh')
    expect(api.listProcedureVersions).toHaveBeenCalledWith('proc-new')
    expect(host.textContent).toContain('Phiên nguồn')
    expect(host.textContent).toContain('Phiên nguồn chỉ để gắn tài liệu và tạo bản nháp tự động, không kích hoạt.')
    expect(buttonNamed(host, 'Kích hoạt')).toBeUndefined()
    expect([...host.querySelectorAll('a')].map((a) => a.getAttribute('href'))).toContain('/admin/documents')
  })

  it('blocks an invalid code before calling the API', async () => {
    const host = await renderPage()
    await openNewProcedureDialog(host)
    await fillNewProcedure(host, { code: 'DK Khai Sinh', name: 'Đăng ký', domainId: 'ho_tich_chung_thuc' })
    await submitNewProcedure(host)
    expect(api.createProcedure).not.toHaveBeenCalled()
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('chữ thường')
  })

  it('shows a duplicate code in Vietnamese and keeps the dialog open', async () => {
    api.createProcedure.mockRejectedValue(Object.assign(new Error('procedure_code already exists in this commune'), {
      status: 409, code: 'PROCEDURE_CODE_EXISTS',
    }))
    const host = await renderPage()
    await openNewProcedureDialog(host)
    await fillNewProcedure(host, { code: 'dk_khai_sinh', name: 'Đăng ký', domainId: 'ho_tich_chung_thuc' })
    await submitNewProcedure(host)
    expect(host.querySelector('[role="alert"]')?.textContent).toBe('Mã thủ tục này đã có trong xã. Hãy dùng mã khác.')
    expect(host.querySelector('[aria-labelledby="new-proc-title"]')).toBeTruthy()
  })

  it('reuses the request id when retrying the same body after a network error', async () => {
    api.createProcedure
      .mockRejectedValueOnce(Object.assign(new Error('Failed to fetch'), { name: 'TypeError' }))
      .mockResolvedValueOnce({ ...createdProc, source_version: '0.0.0', replayed: true })
    const host = await renderPage()
    await openNewProcedureDialog(host)
    await fillNewProcedure(host, { code: 'dk_khai_sinh', name: 'Đăng ký khai sinh', domainId: 'ho_tich_chung_thuc' })
    await submitNewProcedure(host)
    await submitNewProcedure(host)
    expect(api.createProcedure).toHaveBeenCalledTimes(2)
    expect(api.createProcedure.mock.calls[1][1]).toBe(api.createProcedure.mock.calls[0][1])
  })
})
