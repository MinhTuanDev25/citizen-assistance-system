import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const drafts = vi.hoisted(() => ({
  cloneDraft: vi.fn(),
  decideWithoutSource: vi.fn(),
  acknowledgeDraftGap: vi.fn(),
  keepDraftCitation: vi.fn(),
  activateDraft: vi.fn(),
  approveDraft: vi.fn(),
  canApproveDraft: vi.fn(() => false),
  getDraft: vi.fn(),
  patchDraft: vi.fn(),
  resolveDraftConflict: vi.fn(),
  reviewDraft: vi.fn(),
  validateDraft: vi.fn(),
}))

vi.mock('../../api/drafts.js', () => drafts)

const catalog = vi.hoisted(() => ({ listAdminProcedures: vi.fn() }))
vi.mock('../../api/catalog.js', () => catalog)

const baseDraft = {
  id: 'draft-1',
  procedure_id: 'proc-1',
  procedure_code: 'khai_sinh_v1',
  document_id: 'doc-1',
  status: 'DRAFT',
  model: 'gemini-3.8-flash',
  updated_at: '2026-10-26T15:00:00Z',
  source_page_start: 10,
  source_page_end: 17,
  definition: {
    name: 'Đăng ký khai sinh',
    slots: {
      hinh_thuc_nop: { type: 'enum', question: 'Bạn muốn nộp trực tiếp hay trực tuyến?', enum_values: ['truc_tiep', 'truc_tuyen'] },
      yeu_cau_ban_sao: { type: 'boolean', question: 'Bạn có yêu cầu cấp bản sao không?' },
    },
    required_slots: ['hinh_thuc_nop'],
    guidance: { summary: 'Nộp tại Trung tâm Phục vụ hành chính công.', checklist: ['Tờ khai đăng ký khai sinh'], fee_hint: 'Miễn lệ phí' },
  },
  proposals: [
    { id: 'slot:hinh_thuc_nop', type: 'slot', content: 'enum\tBạn muốn nộp trực tiếp hay trực tuyến?\ttruc_tiep,truc_tuyen' },
    { id: 'required:hinh_thuc_nop', type: 'required', content: 'hinh_thuc_nop' },
    { id: 'guidance:fee_hint', type: 'guidance', content: 'Miễn lệ phí' },
  ],
  evidence: [
    { proposal_id: 'slot:hinh_thuc_nop', quote_valid: true, page: 10, quotation: 'nộp hồ sơ trực tiếp hoặc trực tuyến', review_status: 'accepted' },
    { proposal_id: 'guidance:fee_hint', quote_valid: true, page: 15, quotation: 'Miễn lệ phí', review_status: 'pending' },
  ],
  missing_requirements: ['Thời hạn giải quyết'],
  warnings: ['Cần đối chiếu lệ phí với địa phương'],
  validation: {
    schema_valid: true,
    source_quotes_verified: true,
    covered_proposal_count: 2,
    material_proposal_count: 3,
    issues: [
      { code: 'MISSING_EVIDENCE', severity: 'error', proposal_id: 'required:hinh_thuc_nop', message: 'no evidence' },
      { code: 'SOURCE_MATCHED_SEMANTIC_UNVERIFIED', severity: 'warning', proposal_id: 'guidance:fee_hint', message: 'x' },
    ],
  },
}

async function flush() {
  for (let i = 0; i < 4; i += 1) {
    await act(async () => {
      await Promise.resolve()
    })
  }
}

async function renderPage() {
  const { default: Page } = await import('./DraftReviewPage.jsx')
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(MemoryRouter, { initialEntries: ['/admin/drafts/draft-1'] },
      createElement(Routes, null, createElement(Route, { path: '/admin/drafts/:draftId', element: createElement(Page) }))))
  })
  await flush()
  return host
}

function button(host, text) {
  return [...host.querySelectorAll('button')].find((node) => node.textContent.trim().startsWith(text))
}

describe('DraftReviewPage', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    for (const fn of Object.values(drafts)) fn.mockReset()
    catalog.listAdminProcedures.mockReset()
    drafts.canApproveDraft.mockReturnValue(false)
    drafts.getDraft.mockResolvedValue(baseDraft)
  })

  it('shows readable proposal labels, sources, preview and AI notes', async () => {
    const host = await renderPage()
    expect(host.querySelector('h1').textContent).toBe('Đăng ký khai sinh')
    expect(host.textContent).toContain('Câu hỏi thông tin')
    expect(host.textContent).toContain('Bắt buộc')
    expect(host.textContent).toContain('Có nguồn · trang 10')
    expect(host.textContent).toContain('Thiếu nguồn')
    expect(host.textContent).toContain('Hệ thống không tìm thấy câu trích')
    expect(host.textContent).toContain('Người dân sẽ thấy')
    expect(host.textContent).toContain('Chưa có — nguồn không nêu')
    expect(host.textContent).toContain('Thời hạn giải quyết')
    expect(host.textContent).toContain('Thiếu trích dẫn nguồn')
    expect(host.textContent).not.toContain('MISSING_EVIDENCE')
    expect(host.querySelector('.dr-steps li.current')?.textContent).toContain('Duyệt đề xuất')
  })

  it('shows answer formats in Vietnamese instead of raw slot types', async () => {
    const host = await renderPage()
    const cards = [...host.querySelectorAll('.dr-card')]
    const slotCard = cards.find((card) => card.textContent.includes('Câu hỏi thông tin'))
    expect(slotCard.querySelector('.dr-card-content').textContent).toBe('Bạn muốn nộp trực tiếp hay trực tuyến?')
    expect(slotCard.querySelector('.dr-answer').textContent).toBe('Chọn 1 lựa chọntruc_tieptruc_tuyen')
    const requiredCard = cards.find((card) => card.textContent.includes('Bắt buộc hỏi'))
    expect(requiredCard.querySelector('.dr-card-content').textContent).toBe('Bắt buộc hỏi: «Bạn muốn nộp trực tiếp hay trực tuyến?»')
    const preview = host.querySelector('.dr-slots').textContent
    expect(preview).toContain('Có / Không')
    expect(preview).toContain('Chọn 1 lựa chọn: truc_tiep · truc_tuyen · bắt buộc')
    // The raw JSON editor (Nâng cao) may show types; the review cards and preview must not.
    const visible = host.querySelector('.dr-cards').textContent + host.querySelector('.dr-side').textContent
    expect(visible).not.toMatch(/\b(enum|boolean|string)\b/)
  })

  it('removes a no-evidence slot with its required flag and conditional references', async () => {
    const { removeProposalFromDefinition } = await import('./DraftReviewPage.jsx')
    const def = {
      slots: { a: { type: 'boolean' }, b: { type: 'string' } },
      required_slots: ['a', 'b'],
      conditional_slots: [{ when: { slot: 'a', equals: true }, require: ['b'] }, { when: { slot: 'b', equals: 'x' }, require: ['a', 'b'] }],
      guidance: { checklist: ['x', 'y'], fee_hint: 'free' },
    }
    const out = removeProposalFromDefinition(def, 'slot:a')
    expect(Object.keys(out.slots)).toEqual(['b'])
    expect(out.required_slots).toEqual(['b'])
    expect(out.conditional_slots).toEqual([{ when: { slot: 'b', equals: 'x' }, require: ['b'] }])
    expect(def.slots.a).toBeDefined()
    expect(removeProposalFromDefinition(def, 'checklist:0').guidance.checklist).toEqual(['y'])
    expect(removeProposalFromDefinition(def, 'guidance:fee_hint').guidance.fee_hint).toBeUndefined()
  })

  it('removes a missing-source proposal through the existing save API', async () => {
    drafts.patchDraft.mockResolvedValue({ ...baseDraft })
    const host = await renderPage()
    const missingCard = [...host.querySelectorAll('.dr-card')].find((card) => card.classList.contains('is-missing'))
    await act(async () => {
      [...missingCard.querySelectorAll('button')].find((b) => b.textContent === 'Bỏ khỏi bản nháp').click()
    })
    await flush()
    const [id, body] = drafts.patchDraft.mock.calls[0]
    expect(id).toBe('draft-1')
    expect(body.expected_updated_at).toBe(baseDraft.updated_at)
    expect(body.definition.required_slots).toEqual([])
    expect(body.definition.slots.hinh_thuc_nop).toBeDefined()
  })

  it('rejecting a proposal removes it from the draft without JSON editing', async () => {
    const draft = {
      ...baseDraft,
      definition: { ...baseDraft.definition, guidance: { ...baseDraft.definition.guidance, checklist: ['Tờ khai đăng ký khai sinh', 'Xuất trình giấy tờ tùy thân'] } },
      proposals: [...baseDraft.proposals, { id: 'checklist:1', type: 'checklist', content: 'Xuất trình giấy tờ tùy thân' }],
      evidence: [...baseDraft.evidence, { proposal_id: 'checklist:1', quote_valid: false, page: 13, quotation: 'Hộ chiếu/Thẻ căn', review_status: 'rejected' }],
    }
    drafts.getDraft.mockResolvedValue(draft)
    drafts.patchDraft.mockResolvedValue({ ...draft })
    const host = await renderPage()
    const card = [...host.querySelectorAll('.dr-card')].find((c) => c.textContent.includes('Xuất trình giấy tờ tùy thân'))
    expect(card.textContent).toContain('Câu trích không khớp văn bản nguồn')
    await act(async () => {
      [...card.querySelectorAll('button')].find((b) => b.textContent === 'Từ chối').click()
    })
    await flush()
    expect(drafts.reviewDraft).not.toHaveBeenCalled()
    const [, body] = drafts.patchDraft.mock.calls.at(-1)
    expect(body.definition.guidance.checklist).toEqual(['Tờ khai đăng ký khai sinh'])
    expect(host.textContent).toContain('Đã bỏ mục khỏi bản nháp')
  })

  it('shows enum choices with their citizen-facing labels', async () => {
    const { describeProposal } = await import('./DraftReviewPage.jsx')
    const shown = describeProposal({ id: 'slot:truong_hop_tre', content: 'enum\tThuộc diện nào?\ttre_sinh_thuong,tre_bi_bo_roi\tTrẻ sinh bình thường|Trẻ bị bỏ rơi' })
    expect(shown.options).toEqual(['Trẻ sinh bình thường (tre_sinh_thuong)', 'Trẻ bị bỏ rơi (tre_bi_bo_roi)'])
    expect(describeProposal({ id: 'slot:k', content: 'enum\tQ?\ta,b' }).options).toEqual(['a', 'b'])
  })

  it('shows and removes checklist items that apply to one answer only', async () => {
    const { describeProposal, removeProposalFromDefinition, proposalLabel } = await import('./DraftReviewPage.jsx')
    const slots = { truong_hop_tre: { question: 'Trường hợp của trẻ?', enum_values: ['tre_sinh_thuong', 'tre_bi_bo_roi'], enum_labels: ['Trẻ sinh bình thường', 'Trẻ bị bỏ rơi'] } }
    const shown = describeProposal({ id: 'conditional_checklist:0', content: 'truong_hop_tre="tre_bi_bo_roi"\tBiên bản về việc trẻ bị bỏ rơi' }, slots)
    expect(shown.text).toBe('Chỉ khi «Trường hợp của trẻ?» = Trẻ bị bỏ rơi (tre_bi_bo_roi): Biên bản về việc trẻ bị bỏ rơi')
    expect(proposalLabel('conditional_checklist:0').group).toBe('Giấy tờ theo trường hợp')
    const def = { guidance: { checklist: ['A'], conditional_checklist: [{ when: { slot: 'truong_hop_tre', equals: 'tre_bi_bo_roi' }, item: 'B' }, { when: { slot: 'x', equals: true }, item: 'C' }] } }
    expect(removeProposalFromDefinition(def, 'conditional_checklist:0').guidance.conditional_checklist).toEqual([{ when: { slot: 'x', equals: true }, item: 'C' }])
  })

  it('removing a question also removes the items shown only for its answers', async () => {
    const { removeProposalFromDefinition } = await import('./DraftReviewPage.jsx')
    const def = {
      slots: { co_uy_quyen: { type: 'boolean' }, truong_hop_tre: { type: 'enum' } },
      required_slots: [],
      guidance: {
        conditional_checklist: [{ when: { slot: 'co_uy_quyen', equals: true }, item: 'Văn bản ủy quyền' }, { when: { slot: 'truong_hop_tre', equals: 'x' }, item: 'B' }],
        conditional_notes: [{ when: { slot: 'co_uy_quyen', equals: true }, item: 'Ủy quyền phải chứng thực' }],
      },
    }
    const out = removeProposalFromDefinition(def, 'slot:co_uy_quyen')
    expect(out.guidance.conditional_checklist).toEqual([{ when: { slot: 'truong_hop_tre', equals: 'x' }, item: 'B' }])
    expect(out.guidance.conditional_notes).toEqual([])
    expect(removeProposalFromDefinition(def, 'conditional_notes:0').guidance.conditional_notes).toEqual([])
  })

  it('offers to remove an item whose condition is broken', async () => {
    const draft = {
      ...baseDraft,
      definition: { ...baseDraft.definition, guidance: { ...baseDraft.definition.guidance, conditional_notes: [{ when: { slot: 'khong_con', equals: true }, item: 'X' }] } },
      validation: { ...baseDraft.validation, issues: [{ code: 'INVALID_GUIDANCE_CONDITION', severity: 'error', proposal_id: 'conditional_notes:0', message: 'm' }] },
    }
    drafts.getDraft.mockResolvedValue(draft)
    drafts.patchDraft.mockResolvedValue({ ...draft })
    const host = await renderPage()
    const issue = [...host.querySelectorAll('.dr-issues li')].find((li) => li.textContent.includes('không còn tồn tại'))
    expect(issue.textContent).toContain('Lưu ý theo trường hợp · #1')
    await act(async () => {
      [...issue.querySelectorAll('button')].find((b) => b.textContent === 'Bỏ mục này').click()
    })
    await flush()
    const [, body] = drafts.patchDraft.mock.calls.at(-1)
    expect(body.definition.guidance.conditional_notes).toEqual([])
  })

  it('makes a never-asked question asked after a chosen answer', async () => {
    const draft = {
      ...baseDraft,
      definition: {
        ...baseDraft.definition,
        slots: {
          truong_hop_tre: { type: 'enum', question: 'Trường hợp của trẻ?', enum_values: ['tre_sinh_thuong', 'tre_bi_bo_roi'], enum_labels: ['Trẻ sinh bình thường', 'Trẻ bị bỏ rơi'] },
          co_giay_chung_sinh: { type: 'boolean', question: 'Trẻ đã có Giấy chứng sinh chưa?' },
        },
        required_slots: ['truong_hop_tre'],
        conditional_slots: [],
      },
      validation: { ...baseDraft.validation, issues: [{ code: 'UNREACHABLE_SLOT', severity: 'error', proposal_id: 'slot:co_giay_chung_sinh', message: 'm' }] },
    }
    drafts.getDraft.mockResolvedValue(draft)
    drafts.patchDraft.mockResolvedValue({ ...draft })
    const host = await renderPage()
    const issue = [...host.querySelectorAll('.dr-issues li')].find((li) => li.textContent.includes('không bao giờ được hỏi'))
    const select = issue.querySelector('select')
    const option = [...select.options].find((o) => o.textContent.includes('Trẻ sinh bình thường'))
    expect(option.textContent).toBe('Chỉ hỏi khi «Trường hợp của trẻ?» = Trẻ sinh bình thường (tre_sinh_thuong)')
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value').set
      setter.call(select, option.value)
      select.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => {
      [...issue.querySelectorAll('button')].find((b) => b.textContent === 'Áp dụng').click()
    })
    await flush()
    const [, body] = drafts.patchDraft.mock.calls.at(-1)
    expect(body.definition.conditional_slots).toEqual([{ when: { slot: 'truong_hop_tre', equals: 'tre_sinh_thuong' }, require: ['co_giay_chung_sinh'] }])
    expect(body.definition.required_slots).toEqual(['truong_hop_tre'])
  })

  it('can make a never-asked question always asked', async () => {
    const { addAskRule } = await import('./DraftReviewPage.jsx')
    expect(addAskRule({ required_slots: ['a'] }, 'b', null).required_slots).toEqual(['a', 'b'])
  })

  it('copies a published draft into a new editable draft', async () => {
    drafts.getDraft.mockResolvedValue({ ...baseDraft, status: 'PUBLISHED' })
    drafts.cloneDraft.mockResolvedValue({ ...baseDraft, id: 'draft-2', status: 'DRAFT' })
    const host = await renderPage()
    const btn = [...host.querySelectorAll('button')].find((b) => b.textContent === 'Tạo bản nháp để sửa')
    expect(btn).toBeTruthy()
    await act(async () => {
      btn.click()
    })
    await flush()
    expect(drafts.cloneDraft).toHaveBeenCalledWith('draft-1')
  })

  it('removing a question also removes questions asked only after its answer', async () => {
    const { removeProposalFromDefinition } = await import('./DraftReviewPage.jsx')
    const def = {
      slots: { truong_hop_tre: {}, co_giay_chung_sinh: {}, co_nguoi_lam_chung: {} },
      required_slots: ['truong_hop_tre'],
      conditional_slots: [
        { when: { slot: 'truong_hop_tre', equals: 'tre_sinh_thuong' }, require: ['co_giay_chung_sinh'] },
        { when: { slot: 'co_giay_chung_sinh', equals: false }, require: ['co_nguoi_lam_chung'] },
      ],
      guidance: { conditional_checklist: [{ when: { slot: 'co_nguoi_lam_chung', equals: true }, item: 'Văn bản người làm chứng' }] },
    }
    const out = removeProposalFromDefinition(def, 'slot:co_giay_chung_sinh')
    expect(Object.keys(out.slots)).toEqual(['truong_hop_tre'])
    expect(out.conditional_slots).toEqual([])
    expect(out.guidance.conditional_checklist).toEqual([])
  })

  it('keeps a proposal without a source on the Admin reason', async () => {
    drafts.patchDraft.mockResolvedValue({ ...baseDraft })
    drafts.decideWithoutSource.mockResolvedValue({ ...baseDraft })
    const host = await renderPage()
    const card = [...host.querySelectorAll('.dr-card')].find((c) => c.classList.contains('is-missing'))
    const keep = () => [...card.querySelectorAll('button')].find((b) => b.textContent === 'Giữ theo quyết định của cán bộ')
    await act(async () => {
      keep().click()
    })
    await flush()
    expect(drafts.decideWithoutSource).not.toHaveBeenCalled()
    expect(host.textContent).toContain('Ghi lý do (ít nhất 10 ký tự)')
    const area = card.querySelector('textarea.dr-reason')
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
      setter.call(area, 'Quy định riêng của UBND xã')
      area.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => {
      keep().click()
    })
    await flush()
    expect(drafts.decideWithoutSource).toHaveBeenCalledWith('draft-1', {
      expected_updated_at: baseDraft.updated_at,
      proposal_id: 'required:hinh_thuc_nop',
      reason: 'Quy định riêng của UBND xã',
      remove: false,
    })
  })

  it('shows a decided proposal as an Admin decision that counts as accepted', async () => {
    drafts.getDraft.mockResolvedValue({
      ...baseDraft,
      admin_decisions: [{ proposal_id: 'required:hinh_thuc_nop', reason: 'Quy định riêng của UBND xã', by: 'u', at: 'x' }],
    })
    const host = await renderPage()
    const card = [...host.querySelectorAll('.dr-card')].find((c) => c.textContent.includes('Quyết định của cán bộ'))
    expect(card.textContent).toContain('Lý do: Quy định riêng của UBND xã')
    expect(card.classList.contains('is-missing')).toBe(false)
    const tab = [...host.querySelectorAll('.dr-tabs button')].find((b) => b.textContent.startsWith('Thiếu nguồn'))
    expect(tab.textContent).toBe('Thiếu nguồn0')
  })

  it('decodes conditional rules', async () => {
    const { describeProposal } = await import('./DraftReviewPage.jsx')
    const slots = { da_ket_hon: { question: 'Cha mẹ đã kết hôn chưa?' }, giay_ket_hon: { question: 'Có giấy kết hôn không?' } }
    expect(describeProposal({ id: 'conditional:0', content: 'da_ket_hon=true require=giay_ket_hon' }, slots).text)
      .toBe('Nếu «Cha mẹ đã kết hôn chưa?» = true thì hỏi thêm: Có giấy kết hôn không?')
  })

  it('filters proposals by tab with counts', async () => {
    const host = await renderPage()
    const tab = (label) => [...host.querySelectorAll('.dr-tabs button')].find((b) => b.textContent.startsWith(label))
    expect(tab('Tất cả').textContent).toBe('Tất cả3')
    expect(tab('Thiếu nguồn').textContent).toBe('Thiếu nguồn1')
    expect(tab('Đã chấp nhận').textContent).toBe('Đã chấp nhận1')
    await act(async () => {
      tab('Thiếu nguồn').click()
    })
    expect(host.querySelectorAll('.dr-card')).toHaveLength(1)
    expect(host.querySelector('.dr-card').textContent).toContain('Bắt buộc')
  })

  it('accepts one proposal and explains why approval is blocked', async () => {
    drafts.reviewDraft.mockResolvedValue({ ...baseDraft })
    const host = await renderPage()
    expect(host.querySelector('.dr-actionbar p').textContent).toContain('Còn 1 đề xuất thiếu nguồn')
    expect(button(host, 'Phê duyệt').disabled).toBe(true)
    const feeCard = [...host.querySelectorAll('.dr-card')].find((card) => card.textContent.includes('Lệ phí'))
    await act(async () => {
      [...feeCard.querySelectorAll('.dr-segment button')].find((b) => b.textContent === 'Chấp nhận').click()
    })
    await flush()
    expect(drafts.reviewDraft).toHaveBeenCalledWith('draft-1', {
      expected_updated_at: baseDraft.updated_at,
      decisions: [{ proposal_id: 'guidance:fee_hint', decision: 'accepted' }],
    })
    const missingCard = [...host.querySelectorAll('.dr-card')].find((card) => card.classList.contains('is-missing'))
    // No evidence row exists for it, so the server cannot record any decision: no buttons, only a fix hint.
    expect(missingCard.querySelector('.dr-segment')).toBeNull()
    expect(missingCard.querySelector('.dr-fix-hint').textContent).toContain('Chưa duyệt được vì không có nguồn')
  })

  it('locks review actions once the draft is approved', async () => {
    drafts.getDraft.mockResolvedValue({ ...baseDraft, status: 'APPROVED' })
    const host = await renderPage()
    expect(host.textContent).toContain('Đã phê duyệt')
    expect([...host.querySelectorAll('.dr-segment button')].every((b) => b.disabled)).toBe(true)
    expect(button(host, 'Kích hoạt').disabled).toBe(false)
  })

  it('sends the ACTIVE version it will replace when activating', async () => {
    drafts.getDraft.mockResolvedValue({ ...baseDraft, status: 'APPROVED' })
    catalog.listAdminProcedures.mockResolvedValue({ items: [{ id: 'proc-1', active_version_id: 'old-v', active_version: '8.1.1' }] })
    drafts.activateDraft.mockResolvedValue({ activated: true, version: { version: '8.2.2' }, draft: { ...baseDraft, status: 'PUBLISHED' } })
    const host = await renderPage()
    await act(async () => { button(host, 'Kích hoạt').click() })
    await flush()
    expect(host.querySelector('.dr-active-target').textContent).toContain('Sẽ thay thế phiên bản đang dùng: 8.1.1')
    await act(async () => { button(host, 'Xác nhận kích hoạt').click() })
    await flush()
    expect(drafts.activateDraft.mock.calls[0][1]).toEqual({
      confirm: true,
      expected_updated_at: baseDraft.updated_at,
      expected_active_version_id: 'old-v',
    })
    expect(host.textContent).toContain('Đã kích hoạt phiên bản 8.2.2')
  })

  it('shows coverage, source conflicts and provenance of a merged job draft', async () => {
    const merged = {
      ...baseDraft,
      coverage: {
        mode: 'FULL', batches: 3, complete: true, document_pages: 451, document_chunks: 3215,
        section_title: 'Thủ tục đăng ký khai sinh', page_start: 10, page_end: 21,
        selected_chunks: 83, processed_chunks: 83, duplicates_removed: 4,
        gaps: [{ topic: 'Cách thức / cơ quan thực hiện', field: 'guidance.where_to_submit', page: 12 }],
      },
      conflicts: [{
        id: 'guidance:processing_time_hint', field: 'guidance.processing_time_hint', status: 'OPEN',
        candidates: [
          { value: 'Trong ngày', contributor: 'batch:1', page_start: 10, page_end: 15, evidence: { page: 11, quotation: 'trả kết quả trong ngày' } },
          { value: 'Không quá 03 ngày làm việc', contributor: 'batch:3', page_start: 20, page_end: 21, evidence: { page: 20, quotation: 'không quá 03 ngày làm việc' } },
        ],
      }],
      provenance: [{
        proposal_id: 'slot:hinh_thuc_nop', content: baseDraft.proposals[0].content,
        sources: [
          { contributor: 'batch:1', excerpt_id: 'e1', page: 10, quotation: 'trực tiếp hoặc trực tuyến', quote_valid: true },
          { contributor: 'batch:2', excerpt_id: 'e2', page: 16, quotation: 'qua bưu chính', quote_valid: true },
        ],
      }],
    }
    drafts.getDraft.mockResolvedValue(merged)
    drafts.resolveDraftConflict.mockResolvedValue({ ...merged, conflicts: [{ ...merged.conflicts[0], status: 'RESOLVED', chosen: 1 }] })
    const host = await renderPage()
    const coverage = host.querySelector('.dr-coverage').textContent
    expect(coverage).toContain('Tài liệu 451 trang · 3215 đoạn')
    expect(coverage).toContain('Đã xử lý 83/83 đoạn trong 3 đợt')
    expect(coverage).toContain('Cách thức / cơ quan thực hiện')
    expect(host.querySelector('.dr-hero-meta').textContent).toContain('Gộp từ 3 đợt')
    expect(host.querySelector('.dr-actionbar p').textContent).toContain('Còn 1 mâu thuẫn nguồn')
    const conflict = host.querySelector('.dr-conflict')
    expect(conflict.textContent).toContain('Thời hạn giải quyết')
    expect(conflict.textContent).toContain('đợt 3 · trang 20–21')
    const prov = host.querySelector('.dr-prov summary').textContent
    expect(prov).toBe('2 nguồn ủng hộ · trang 10, 16')
    await act(async () => {
      button(host, 'Chọn phương án 2').click()
    })
    await flush()
    expect(drafts.resolveDraftConflict).toHaveBeenCalledWith('draft-1', {
      expected_updated_at: baseDraft.updated_at,
      conflict_id: 'guidance:processing_time_hint',
      choice: 1,
    })
    expect(host.querySelector('.dr-conflict').textContent).toContain('Đã chọn phương án 2')
  })

  it('labels a citation description conflict with its document', async () => {
    const cite = (title) => ({ doc_id: '8939d98f-ce2c-487f-9f28-a53510c29271', title, source_type: 'ai_draft_unapproved' })
    drafts.getDraft.mockResolvedValue({
      ...baseDraft,
      conflicts: [{
        id: 'citation:abc', field: 'citations', status: 'OPEN',
        candidates: [
          { value: 'Quyết định 163', contributor: 'batch:1', citation: cite('Quyết định 163') },
          { value: 'Quyết định số 163/QĐ-BTP · số 163/QĐ-BTP', contributor: 'batch:2', citation: cite('Quyết định số 163/QĐ-BTP') },
        ],
      }],
    })
    const host = await renderPage()
    expect(host.querySelector('.dr-conflict strong').textContent).toBe('Tên/số hiệu văn bản trích dẫn (mã 8939d98f) — các đợt ghi khác nhau')
    expect(host.querySelector('.dr-conflict').textContent).toContain('Quyết định số 163/QĐ-BTP · số 163/QĐ-BTP')
  })

  it('shows inactive, reopened and multi-batch citation conflicts', async () => {
    const cite = (title) => ({ doc_id: '8939d98f-ce2c-487f-9f28-a53510c29271', title, source_type: 'ai_draft_unapproved' })
    const candidates = [
      { value: 'Văn bản 10', contributor: 'batch:1', also_from: ['batch:3'], citation: cite('Văn bản 10') },
      { value: 'Quyết định số 163/QĐ-BTP', contributor: 'batch:2', citation: cite('Quyết định số 163/QĐ-BTP') },
    ]
    drafts.getDraft.mockResolvedValue({
      ...baseDraft,
      conflicts: [
        { id: 'citation:a', field: 'citations', status: 'OPEN', inactive: true, candidates },
        { id: 'citation:b', field: 'citations', status: 'OPEN', reopened: true, candidates },
      ],
    })
    const host = await renderPage()
    const [inactive, reopened] = host.querySelectorAll('.dr-conflict')
    expect(inactive.textContent).toContain('Chưa áp dụng')
    expect(inactive.textContent).toContain('hiện không có trong citations')
    expect(inactive.textContent).toContain('đợt 1 · cũng có ở đợt 3')
    expect(reopened.textContent).toContain('Chưa chọn')
    expect(reopened.textContent).toContain('Giá trị đã bị sửa sau lần chọn trước')
    expect(host.textContent).toContain('Còn 1 mâu thuẫn nguồn')
  })

  it('requires a reason to leave a coverage gap empty and labels slot-type conflicts', async () => {
    const merged = {
      ...baseDraft,
      coverage: {
        batches: 3, complete: true, document_pages: 451, document_chunks: 3215, page_start: 9, page_end: 21,
        selected_chunks: 84, processed_chunks: 84,
        gaps: [{ topic: 'Lệ phí', field: 'guidance.fee_hint', page: 15 }],
        outside_mentions: [{ section: '', page_start: 1, page_end: 9, chunks: 3 }],
      },
      conflicts: [{
        id: 'slot:hinh_thuc_nop', field: 'slots.hinh_thuc_nop', status: 'OPEN',
        candidates: [{ value: 'string: Hình thức nộp?', contributor: 'batch:1' }, { value: 'enum: Hình thức nộp? [truc_tiep, buu_chinh]', contributor: 'batch:2' }],
      }],
    }
    drafts.getDraft.mockResolvedValue(merged)
    drafts.acknowledgeDraftGap.mockResolvedValue({
      ...merged, conflicts: [], coverage: { ...merged.coverage, gaps: [{ ...merged.coverage.gaps[0], acknowledged: true, reason: 'Văn bản không quy định lệ phí' }] },
    })
    const host = await renderPage()
    expect(host.querySelector('.dr-conflict strong').textContent).toBe('Kiểu trả lời của câu hỏi hinh_thuc_nop')
    expect(host.querySelector('.dr-outside summary').textContent).toBe('Các nguồn có thể liên quan ngoài phạm vi đã soạn (1 nhóm)')
    await act(async () => {
      button(host, 'Xác nhận để trống').click()
    })
    await flush()
    expect(drafts.acknowledgeDraftGap).not.toHaveBeenCalled()
    expect(host.textContent).toContain('Ghi lý do (ít nhất 5 ký tự)')
    const input = host.querySelector('.dr-gap-form input')
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, 'Văn bản không quy định lệ phí')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => {
      button(host, 'Xác nhận để trống').click()
    })
    await flush()
    expect(drafts.acknowledgeDraftGap).toHaveBeenCalledWith('draft-1', {
      expected_updated_at: baseDraft.updated_at, field: 'guidance.fee_hint', reason: 'Văn bản không quy định lệ phí',
    })
    expect(host.querySelector('.dr-gap-ack').textContent).toContain('Văn bản không quy định lệ phí')
  })

  it('lists truncated items and separates scope completion from legal-source review', async () => {
    const merged = {
      ...baseDraft,
      coverage: {
        batches: 10, scope_complete: true, complete: true, outside_mentions_reviewed: false, boundary_uncertain: true,
        outside_criteria: 'Các đoạn ngoài phạm vi có tên thủ tục',
        document_pages: 451, document_chunks: 3215, page_start: 9, page_end: 21, selected_chunks: 84, processed_chunks: 84,
        gaps: [],
        truncated: [{ field: 'guidance.checklist', limit: 40, blocking: true, dropped: ['Giấy tờ số 41', 'Giấy tờ số 42'] }],
        outside_mentions: [{ section: '', page_start: 1, page_end: 8, chunks: 6 }],
      },
    }
    drafts.getDraft.mockResolvedValue(merged)
    drafts.acknowledgeDraftGap.mockResolvedValue({
      ...merged,
      coverage: { ...merged.coverage, acknowledged: [{ field: 'truncated:guidance.checklist', reason: 'Trùng nghĩa với mục đã có' }] },
    })
    const host = await renderPage()
    const scope = host.querySelector('.dr-scope').textContent
    expect(scope).toContain('Phạm vi đã chọn: đã xử lý hết')
    expect(scope).toContain('Kết quả tìm ngoài phạm vi: 1 nhóm chưa được cán bộ xem xét')
    expect(host.querySelector('.dr-criteria').textContent).toContain('Tiêu chí tìm: Các đoạn ngoài phạm vi có tên thủ tục')
    expect(button(host, 'Xác nhận ranh giới')).toBeTruthy()
    const trunc = host.querySelector('.dr-trunc')
    expect(trunc.textContent).toContain('Bị cắt 2 mục hồ sơ (giới hạn 40)')
    expect(trunc.textContent).toContain('Giấy tờ số 42')
    expect(host.querySelector('.dr-actionbar p').textContent).toContain('bị cắt do vượt giới hạn')
    const input = trunc.querySelector('input')
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, 'Trùng nghĩa với mục đã có')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => {
      trunc.querySelector('button').click()
    })
    await flush()
    expect(drafts.acknowledgeDraftGap).toHaveBeenCalledWith('draft-1', {
      expected_updated_at: baseDraft.updated_at, field: 'truncated:guidance.checklist', reason: 'Trùng nghĩa với mục đã có',
    })
    expect(host.querySelector('.dr-trunc').textContent).toContain('Cán bộ đã xem: “Trùng nghĩa với mục đã có”')
  })

  it('keeps a dropped citation in place of a current one', async () => {
    const withCites = {
      ...baseDraft,
      definition: { ...baseDraft.definition, citations: [{ doc_id: 'a1b2c3d4-0000', title: 'Quyết định 163' }, { doc_id: 'ffff0000-1111', title: 'Quyết định 163' }] },
      coverage: {
        batches: 2, scope_complete: true, complete: true, outside_mentions_reviewed: true, gaps: [],
        truncated: [{ field: 'citations', limit: 2, blocking: true, dropped: ['Quyết định 163'], dropped_citations: [{ doc_id: 'x', title: 'Quyết định 163', document_number: '163/QĐ-BTP' }] }],
      },
    }
    drafts.getDraft.mockResolvedValue(withCites)
    drafts.keepDraftCitation.mockResolvedValue({ ...withCites, coverage: { ...withCites.coverage, truncated: [] } })
    const host = await renderPage()
    const select = host.querySelector('.dr-keep select')
    // Same title, different sources: the Admin can tell them apart.
    expect([...select.options].map((o) => o.textContent)).toEqual([
      'Thêm vào (không thay)', 'Thay cho: Quyết định 163 · mã a1b2c3d4', 'Thay cho: Quyết định 163 · mã ffff0000',
    ])
    expect(host.querySelector('.dr-trunc li').textContent).toContain('Quyết định 163 (số 163/QĐ-BTP)')
    await act(async () => {
      select.value = '1'
      select.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => {
      host.querySelector('.dr-keep button').click()
    })
    await flush()
    expect(drafts.keepDraftCitation).toHaveBeenCalledWith('draft-1', { expected_updated_at: baseDraft.updated_at, dropped: 0, replace: 1 })
    expect(host.querySelector('.dr-trunc')).toBeNull()
  })
})
