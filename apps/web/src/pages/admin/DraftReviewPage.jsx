import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  cloneDraft,
  decideWithoutSource,
  activateDraft,
  approveDraft,
  canApproveDraft,
  getDraft,
  acknowledgeDraftGap,
  keepDraftCitation,
  patchDraft,
  resolveDraftConflict,
  reviewDraft,
  validateDraft,
} from '../../api/drafts.js'
import { listAdminProcedures } from '../../api/catalog.js'
import { citationLabel, mentionLabel, mentionPages, OUTSIDE_TITLE } from './draftSources.js'
import { adminDocumentPath, adminDraftPath, adminDraftsPath } from '../../routes.js'

function formatWhen(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso || ''
  return d.toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'medium' })
}

function asObject(raw) {
  if (!raw) return {}
  if (typeof raw === 'string') {
    try {
      return JSON.parse(raw)
    } catch {
      return {}
    }
  }
  return raw
}

const DRAFT_STATUS = {
  DRAFT: { label: 'Bản nháp', tone: 'mute' },
  REVIEWED: { label: 'Đang duyệt', tone: 'warn' },
  APPROVED: { label: 'Đã phê duyệt', tone: 'ok' },
  PUBLISHED: { label: 'Đã kích hoạt', tone: 'ok' },
  REJECTED: { label: 'Đã từ chối', tone: 'bad' },
}

const REVIEW_STATUS = {
  pending: { label: 'Chờ duyệt', tone: 'mute' },
  accepted: { label: 'Đã chấp nhận', tone: 'ok' },
  rejected: { label: 'Đã từ chối', tone: 'bad' },
  needs_correction: { label: 'Cần sửa', tone: 'warn' },
  stale: { label: 'Nguồn cũ', tone: 'warn' },
}

const GUIDANCE_LABEL = {
  summary: 'Tóm tắt thủ tục',
  where_to_submit: 'Nơi nộp hồ sơ',
  fee_hint: 'Lệ phí',
  processing_time_hint: 'Thời hạn giải quyết',
}

const ISSUE_LABEL = {
  UNREACHABLE_SLOT: 'Câu hỏi không bao giờ được hỏi — chọn khi nào hỏi',
  INVALID_GUIDANCE_CONDITION: 'Mục theo trường hợp trỏ tới câu hỏi hoặc đáp án không còn tồn tại',
  ENUM_LABELS_MISMATCH: 'Tên lựa chọn không khớp với các đáp án',
  MISSING_EVIDENCE: 'Thiếu trích dẫn nguồn',
  SOURCE_MATCHED_SEMANTIC_UNVERIFIED: 'Có nguồn, chờ cán bộ duyệt nội dung',
  QUOTE_NOT_FOUND: 'Câu trích không có trong tài liệu',
  STALE_EVIDENCE: 'Nguồn cũ sau khi sửa nội dung',
  IDENTITY_MISMATCH: 'Sai mã thủ tục',
  SCHEMA_INVALID: 'Cấu trúc định nghĩa không hợp lệ',
  SOURCE_CONFLICT: 'Các phần của nguồn nói khác nhau — cán bộ phải chọn',
  COVERAGE_GAP: 'Nguồn có mục này nhưng định nghĩa còn trống — bổ sung hoặc xác nhận lý do',
  COVERAGE_GAP_ACKNOWLEDGED: 'Cán bộ đã xác nhận để trống và nêu lý do',
  CONTENT_TRUNCATED: 'Một số đề xuất bị cắt do vượt giới hạn — cán bộ cần xem',
  CONTENT_TRUNCATED_ACKNOWLEDGED: 'Cán bộ đã xem nội dung bị cắt và nêu lý do',
  OUTSIDE_SOURCES_UNREVIEWED: 'Có trang ngoài các mục thủ tục cần cán bộ xem xét',
  SECTION_BOUNDARY_UNCERTAIN: 'Ranh giới cuối mục chưa chắc chắn — cán bộ cần xác nhận',
  SOURCE_PARTIALLY_PROCESSED: 'Chưa xử lý hết phạm vi nguồn',
}

const TRUNCATED_LABEL = {
  slots: 'câu hỏi',
  conditional_slots: 'quy tắc điều kiện',
  'guidance.checklist': 'mục hồ sơ',
  'guidance.notes': 'ghi chú',
  citations: 'trích dẫn văn bản',
  intent_examples: 'câu ví dụ',
}

/** The Admin decision currently recorded for a coverage finding, if any. */
export function findingAck(coverage, field) {
  return (coverage?.acknowledged || []).find((a) => a.field === field)
}

const CONTRIBUTOR_LABEL = (id) => (id === 'base' ? 'bản nền' : `đợt ${String(id || '').replace('batch:', '')}`)

/** Every quotation behind a merged proposal (job drafts), if its text is unchanged. */
export function provenanceFor(draft, prop) {
  const item = (draft?.provenance || []).find((p) => p.proposal_id === prop.id)
  if (!item || item.content !== prop.content) return []
  return item.sources || []
}

/** Human label for a canonical proposal id (slot:x, required:x, checklist:n, guidance:x, conditional:n). */
export function proposalLabel(id) {
  const [kind, rest = ''] = String(id || '').split(':')
  switch (kind) {
    case 'slot':
      return { group: 'Câu hỏi thông tin', name: rest, tone: 'info' }
    case 'required':
      return { group: 'Bắt buộc', name: rest, tone: 'violet' }
    case 'checklist':
      return { group: 'Giấy tờ / bước', name: `#${Number(rest) + 1}`, tone: 'teal' }
    case 'conditional_checklist':
      return { group: 'Giấy tờ theo trường hợp', name: `#${Number(rest) + 1}`, tone: 'teal' }
    case 'conditional_notes':
      return { group: 'Lưu ý theo trường hợp', name: `#${Number(rest) + 1}`, tone: 'teal' }
    case 'conditional':
      return { group: 'Điều kiện', name: `#${Number(rest) + 1}`, tone: 'amber' }
    case 'guidance':
      return { group: GUIDANCE_LABEL[rest] || 'Hướng dẫn', name: '', tone: 'indigo' }
    default:
      return { group: kind || 'Đề xuất', name: rest, tone: 'mute' }
  }
}

const ANSWER_LABEL = {
  boolean: 'Có / Không',
  enum: 'Chọn 1 lựa chọn',
  string: 'Trả lời tự do',
  number: 'Nhập số',
}

/** How the citizen answers a slot, in plain Vietnamese (never the raw type). */
export function answerLabel(type) {
  return ANSWER_LABEL[type] || 'Trả lời tự do'
}

/**
 * Readable view of a proposal. The server's proposal content is a fingerprint
 * input (slot: "type\tquestion\tv1,v2"; conditional: "slot=value require=a,b"),
 * so it is decoded here instead of being shown raw.
 */
/** Answers another question can be asked after: enum values and yes/no. */
export function askConditions(slotsByKey = {}, slotKey) {
  const out = []
  for (const [key, slot] of Object.entries(slotsByKey)) {
    if (key === slotKey) continue
    if (slot.type === 'enum') {
      ;(slot.enum_values || []).forEach((value, i) => {
        out.push({ slot: key, equals: value, label: `«${slot.question || key}» = ${optionText(value, (slot.enum_labels || [])[i])}` })
      })
    } else if (slot.type === 'boolean') {
      out.push({ slot: key, equals: true, label: `«${slot.question || key}» = Có` })
      out.push({ slot: key, equals: false, label: `«${slot.question || key}» = Không` })
    }
  }
  return out
}

/** Make a never-asked question reachable: always (when = null) or after one answer. */
export function addAskRule(definition, slotKey, when) {
  const def = JSON.parse(JSON.stringify(definition || {}))
  if (!when) {
    def.required_slots = [...(def.required_slots || []).filter((k) => k !== slotKey), slotKey]
    return def
  }
  def.conditional_slots = [...(def.conditional_slots || []), { when: { slot: when.slot, equals: when.equals }, require: [slotKey] }]
  return def
}

/** "Chỉ khi «Trường hợp của trẻ?» = Trẻ bị bỏ rơi" for a slot=value condition. */
export function whenText(cond, slotsByKey = {}) {
  const at = String(cond || '').indexOf('=')
  const slotKey = at < 0 ? String(cond || '') : cond.slice(0, at)
  let value = at < 0 ? '' : cond.slice(at + 1)
  try {
    value = JSON.parse(value)
  } catch {
    // keep the raw text
  }
  const slot = slotsByKey[slotKey] || {}
  let shown = value
  if (value === true) shown = 'Có'
  else if (value === false) shown = 'Không'
  else {
    const i = (slot.enum_values || []).indexOf(value)
    if (i >= 0) shown = optionText(value, (slot.enum_labels || [])[i])
  }
  return `Chỉ khi «${slot.question || slotKey}» = ${shown}`
}

/** "Trực tiếp (truc_tiep)": what the citizen reads, then the stored code. */
export function optionText(value, label) {
  return label && label.trim() ? `${label.trim()} (${value})` : value
}

export function describeProposal(prop, slotsByKey = {}) {
  const [kind, key = ''] = String(prop?.id || '').split(':')
  const content = String(prop?.content || '')
  if (kind === 'slot') {
    const [type, question, values, labels] = content.split('\t')
    const names = labels ? labels.split('|') : []
    return {
      text: question || slotsByKey[key]?.question || key,
      answer: answerLabel(type),
      options: values ? values.split(',').filter(Boolean).map((value, i) => optionText(value, names[i])) : [],
    }
  }
  if (kind === 'required') {
    const question = slotsByKey[content || key]?.question
    return { text: question ? `Bắt buộc hỏi: «${question}»` : `Bắt buộc hỏi thông tin «${content || key}»`, answer: '', options: [] }
  }
  if (kind === 'conditional_checklist') {
    const [cond, item = ''] = content.split('\t')
    return { text: `${whenText(cond, slotsByKey)}: ${item}`, answer: '', options: [] }
  }
  if (kind === 'conditional') {
    const match = content.match(/^([^=]+)=(\S*) require=(.*)$/)
    if (match) {
      const when = slotsByKey[match[1]]?.question || match[1]
      const needs = match[3].split(',').filter(Boolean).map((item) => slotsByKey[item]?.question || item)
      return { text: `Nếu «${when}» = ${match[2].replace(/^"|"$/g, '')} thì hỏi thêm: ${needs.join('; ')}`, answer: '', options: [] }
    }
  }
  return { text: content, answer: '', options: [] }
}

/** Slots the assistant can ask: required, or required by a conditional rule. */
export function reachableSlots(def) {
  const out = new Set(def?.required_slots || [])
  for (const rule of def?.conditional_slots || []) {
    for (const k of rule?.require || []) out.add(k)
  }
  return out
}

/**
 * Definition without one proposal. A removed slot also leaves required_slots and
 * any conditional rule that mentions it, so the definition stays consistent.
 */
export function removeProposalFromDefinition(definition, id) {
  const def = JSON.parse(JSON.stringify(definition || {}))
  const [kind, key = ''] = String(id || '').split(':')
  const index = Number(key)
  const dropSlotRefs = (slot) => {
    def.required_slots = (def.required_slots || []).filter((item) => item !== slot)
    def.conditional_slots = (def.conditional_slots || [])
      .filter((rule) => rule?.when?.slot !== slot)
      .map((rule) => ({ ...rule, require: (rule.require || []).filter((item) => item !== slot) }))
      .filter((rule) => rule.require.length > 0)
    // Items shown only for an answer to this question can no longer be shown.
    if (def.guidance) {
      for (const field of ['conditional_checklist', 'conditional_notes']) {
        if (Array.isArray(def.guidance[field])) {
          def.guidance[field] = def.guidance[field].filter((rule) => rule?.when?.slot !== slot)
        }
      }
    }
  }
  if (kind === 'slot') {
    const before = reachableSlots(def)
    if (def.slots && !Array.isArray(def.slots)) delete def.slots[key]
    dropSlotRefs(key)
    // A question asked only after an answer to the removed one is never asked
    // now: remove it too (and, in turn, what depended on it).
    for (;;) {
      const now = reachableSlots(def)
      const orphan = Object.keys(def.slots || {}).find((k) => before.has(k) && !now.has(k))
      if (!orphan) break
      delete def.slots[orphan]
      dropSlotRefs(orphan)
    }
  } else if (kind === 'required') {
    def.required_slots = (def.required_slots || []).filter((item) => item !== key)
  } else if (kind === 'checklist' && def.guidance?.checklist) {
    def.guidance.checklist = def.guidance.checklist.filter((_, i) => i !== index)
  } else if ((kind === 'conditional_checklist' || kind === 'conditional_notes') && def.guidance?.[kind]) {
    def.guidance[kind] = def.guidance[kind].filter((_, i) => i !== index)
  } else if (kind === 'conditional' && def.conditional_slots) {
    def.conditional_slots = def.conditional_slots.filter((_, i) => i !== index)
  } else if (kind === 'guidance' && def.guidance) {
    delete def.guidance[key]
  }
  return def
}

function Chip({ tone = 'mute', children }) {
  return <span className={`dr-chip dr-chip-${tone}`}>{children}</span>
}

function Meter({ value, total }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0
  return (
    <div className="dr-meter" aria-hidden="true">
      <span style={{ width: `${pct}%` }} />
    </div>
  )
}

const FILTERS = [
  { key: 'all', label: 'Tất cả' },
  { key: 'pending', label: 'Chờ duyệt' },
  { key: 'missing', label: 'Thiếu nguồn' },
  { key: 'accepted', label: 'Đã chấp nhận' },
]

export default function DraftReviewPage() {
  const { draftId } = useParams()
  const [draft, setDraft] = useState(null)
  const [definitionText, setDefinitionText] = useState('{}')
  const [parseError, setParseError] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [confirmApprove, setConfirmApprove] = useState(false)
  const [confirmActivate, setConfirmActivate] = useState(false)
  // The ACTIVE version this activation will replace; undefined while loading.
  const [activeTarget, setActiveTarget] = useState(undefined)
  const [filter, setFilter] = useState('all')
  const [gapReasons, setGapReasons] = useState({})
  const [keepReplace, setKeepReplace] = useState({})
  const [askWhen, setAskWhen] = useState({})
  const [decisionReason, setDecisionReason] = useState({})
  const navigate = useNavigate()

  async function load() {
    const view = await getDraft(draftId)
    setDraft(view)
    setDefinitionText(JSON.stringify(asObject(view.definition), null, 2))
    setParseError('')
    return view
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      setLoading(true)
      try {
        await load()
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được bản nháp')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [draftId])

  const definition = useMemo(() => asObject(draft?.definition), [draft])
  const slotsByKey = definition.slots || {}
  const slots = Object.entries(slotsByKey)
  const checklist = definition.guidance?.checklist || []
  const guidance = definition.guidance || {}

  async function withBusy(label, fn) {
    setBusy(label)
    setError('')
    setNotice('')
    try {
      const view = await fn()
      if (view?.draft) {
        setDraft(view.draft)
        setDefinitionText(JSON.stringify(asObject(view.draft.definition), null, 2))
        return view
      }
      if (view?.id) {
        setDraft(view)
        setDefinitionText(JSON.stringify(asObject(view.definition), null, 2))
      }
      return view
    } catch (err) {
      setError(err.message || 'Thao tác thất bại')
      throw err
    } finally {
      setBusy('')
    }
  }

  async function onSave() {
    let parsed
    try {
      parsed = JSON.parse(definitionText)
      setParseError('')
    } catch {
      setParseError('Dữ liệu cấu hình không hợp lệ nên chưa thể lưu.')
      return
    }
    await withBusy('save', () => patchDraft(draftId, {
      definition: parsed,
      expected_updated_at: draft.updated_at,
    }))
    setNotice('Đã lưu. Evidence cũ của đề xuất đã sửa bị đánh stale.')
  }

  async function onResolveConflict(conflictId, choice) {
    await withBusy('resolve', () => resolveDraftConflict(draftId, {
      expected_updated_at: draft.updated_at,
      conflict_id: conflictId,
      choice,
    }))
    setNotice(choice >= 0 ? 'Đã chọn giá trị theo nguồn. Duyệt lại trích dẫn của đề xuất này.' : 'Đã giữ giá trị hiện tại trong định nghĩa.')
  }

  async function onAcknowledgeGap(field) {
    const reason = String(gapReasons[field] || '').trim()
    if (reason.length < 5) {
      setError('Ghi lý do (ít nhất 5 ký tự), ví dụ: "Văn bản không quy định lệ phí cho thủ tục này".')
      return
    }
    await withBusy('gap', () => acknowledgeDraftGap(draftId, { expected_updated_at: draft.updated_at, field, reason }))
    setNotice('Đã ghi nhận lý do để trống.')
  }

  async function onKeepCitation(idx) {
    const replace = keepReplace[idx]
    await withBusy('citation', () => keepDraftCitation(draftId, {
      expected_updated_at: draft.updated_at,
      dropped: idx,
      replace: replace === undefined || replace === '' ? null : Number(replace),
    }))
    setKeepReplace({})
    setNotice('Đã giữ trích dẫn văn bản. Danh sách bị cắt đã đổi — xem lại và xác nhận.')
  }

  async function onDecide(proposalId, remove = false) {
    const reason = (decisionReason[proposalId] || '').trim()
    if (!remove && reason.length < 10) {
      setError('Ghi lý do (ít nhất 10 ký tự) — vì sao giữ mục này dù văn bản không nêu.')
      return
    }
    await withBusy('decide', () => decideWithoutSource(draftId, {
      expected_updated_at: draft.updated_at,
      proposal_id: proposalId,
      reason: remove ? '' : reason,
      remove,
    }))
    setDecisionReason({})
    setNotice(remove ? 'Đã huỷ quyết định — mục này lại cần nguồn.' : 'Đã giữ mục theo quyết định của cán bộ.')
  }

  async function onClone() {
    setBusy('clone')
    setError('')
    try {
      const copy = await cloneDraft(draftId)
      navigate(adminDraftPath(copy.id))
    } catch (err) {
      setError(err?.message || 'Không tạo được bản nháp mới.')
    } finally {
      setBusy('')
    }
  }

  async function onAskRule(slotKey) {
    const choice = askWhen[slotKey] ?? 'always'
    if (choice === 'remove') {
      await onRemoveProposal(`slot:${slotKey}`)
      return
    }
    const when = choice === 'always' ? null : askConditions(slotsByKey, slotKey)[Number(choice)]
    await withBusy('save', () => patchDraft(draftId, {
      definition: addAskRule(asObject(draft.definition), slotKey, when),
      expected_updated_at: draft.updated_at,
    }))
    setAskWhen({})
    setNotice(when ? `Đã đặt: chỉ hỏi khi ${when.label}.` : 'Đã đặt: luôn hỏi câu này.')
  }

  async function onRemoveProposal(proposalId) {
    const next = removeProposalFromDefinition(asObject(draft.definition), proposalId)
    await withBusy('save', () => patchDraft(draftId, {
      definition: next,
      expected_updated_at: draft.updated_at,
    }))
    setNotice('Đã bỏ mục khỏi bản nháp. Các mục khác giữ nguyên.')
  }

  async function onValidate() {
    await withBusy('validate', () => validateDraft(draftId))
    setNotice('Đã chạy lại validation.')
  }

  async function onReview(proposalId, decision) {
    await withBusy('review', () => reviewDraft(draftId, {
      expected_updated_at: draft.updated_at,
      decisions: [{ proposal_id: proposalId, decision }],
    }))
  }

  async function onAcceptAllValid() {
    const decisions = (draft.evidence || [])
      .filter((item) => item.quote_valid && item.review_status !== 'stale')
      .map((item) => ({ proposal_id: item.proposal_id, decision: 'accepted' }))
    if (!decisions.length) {
      setError('Không có evidence hợp lệ để chấp nhận.')
      return
    }
    await withBusy('review', () => reviewDraft(draftId, {
      expected_updated_at: draft.updated_at,
      decisions,
    }))
    setNotice('Đã ghi nhận review. Đây không phải duyệt pháp lý.')
  }

  async function onApprove() {
    await withBusy('approve', () => approveDraft(draftId, {
      confirm: true,
      expected_updated_at: draft.updated_at,
    }))
    setConfirmApprove(false)
    setNotice('Cán bộ đã duyệt bản nháp. Hệ thống tự động không thể tự phê duyệt.')
  }

  async function openActivate() {
    setConfirmActivate(true)
    setActiveTarget(undefined)
    try {
      const res = await listAdminProcedures()
      const proc = (res?.items || []).find((item) => item.id === draft.procedure_id)
      if (!proc) throw new Error('procedure not found')
      setActiveTarget({ id: proc.active_version_id || null, version: proc.active_version || '' })
    } catch {
      setActiveTarget({ error: true })
    }
  }

  async function onActivate() {
    let result
    try {
      result = await withBusy('activate', () => activateDraft(draftId, {
        confirm: true,
        expected_updated_at: draft.updated_at,
        expected_active_version_id: activeTarget?.id ?? null,
      }, crypto.randomUUID()))
    } catch (err) {
      setConfirmActivate(false)
      if (err?.code === 'STALE_ACTIVE_VERSION') {
        setError('Phiên bản đang dùng vừa thay đổi. Hãy bấm Kích hoạt lại để xem phiên bản sẽ bị thay thế.')
      }
      return
    }
    setConfirmActivate(false)
    if (result?.activated && result?.reconciliation_required) {
      setNotice('Phiên bản đã được kích hoạt nhưng trạng thái bản nháp chưa được đồng bộ. Hãy bấm Kích hoạt để hoàn tất; hệ thống sẽ không tạo thêm phiên bản mới.')
    } else if (result?.activated) {
      setNotice(`Đã kích hoạt phiên bản ${result.version?.version || ''}.`)
    } else if (result?.indexing_pending) {
      setNotice('Đang lập chỉ mục. Hãy bấm Kích hoạt lại sau khi xử lý xong; hệ thống sẽ không tạo lượt xử lý trùng.')
    } else {
      setNotice(`Chưa kích hoạt: ${result?.blocked_reason || (result?.reasons || []).join(', ') || 'chưa đủ điều kiện'}.`)
    }
  }

  if (loading) return <div className="admin-page draft-review"><p className="muted">Đang tải bản nháp…</p></div>
  if (!draft) return <div className="admin-page draft-review"><p className="form-error">{error || 'Không tìm thấy bản nháp.'}</p></div>

  const v = draft.validation || {}
  const evidence = draft.evidence || []
  const proposals = draft.proposals || []
  const evidenceFor = (id) => evidence.find((item) => item.proposal_id === id)
  const decisionFor = (id) => (draft.admin_decisions || []).find((item) => item.proposal_id === id)
  const approveReady = canApproveDraft(draft) && evidence.every((item) => {
    if (!item.quote_valid) return true
    return item.review_status === 'accepted'
  }) && proposals.every((p) => decisionFor(p.id) || evidence.some((e) => e.proposal_id === p.id && e.review_status === 'accepted'))

  const missingCount = proposals.filter((p) => !decisionFor(p.id) && !evidenceFor(p.id)?.quote_valid).length
  const acceptedCount = proposals.filter((p) => decisionFor(p.id) || evidenceFor(p.id)?.review_status === 'accepted').length
  const pendingCount = proposals.length - acceptedCount
  const errorIssues = (v.issues || []).filter((issue) => issue.severity === 'error')
  const warningIssues = (v.issues || []).filter((issue) => issue.severity !== 'error')
  const counts = { all: proposals.length, pending: pendingCount, missing: missingCount, accepted: acceptedCount }
  const visible = proposals.filter((p) => {
    const ev = evidenceFor(p.id)
    if (filter === 'pending') return ev?.review_status !== 'accepted'
    if (filter === 'missing') return !ev?.quote_valid
    if (filter === 'accepted') return ev?.review_status === 'accepted'
    return true
  })
  const status = DRAFT_STATUS[draft.status] || { label: draft.status, tone: 'mute' }
  const editable = draft.status === 'DRAFT'
  const stepIndex = draft.status === 'PUBLISHED' ? 4 : draft.status === 'APPROVED' ? 3 : acceptedCount === proposals.length && proposals.length ? 2 : 1
  const conflicts = draft.conflicts || []
  const openConflicts = conflicts.filter((c) => c.status !== 'RESOLVED' && !c.inactive)
  const coverage = draft.coverage
  const openGaps = (coverage?.gaps || []).filter((g) => !g.acknowledged)
  const openTruncations = (coverage?.truncated || []).filter((t) => t.blocking && !findingAck(coverage, `truncated:${t.field}`))
  // Older drafts stored legal_sources_verified; the meaning is the same narrow check.
  const outsideReviewed = coverage ? (coverage.outside_mentions_reviewed ?? coverage.legal_sources_verified) : true
  const outsideUnreviewed = outsideReviewed === false
  const boundaryPending = Boolean(coverage?.boundary_uncertain && !coverage?.boundary_reviewed)
  const scopeComplete = coverage ? (coverage.scope_complete ?? coverage.complete) : true
  const unsectioned = (coverage?.outside_mentions || []).filter((m) => !m.section)
  const approveHint = approveReady
    ? 'Đủ điều kiện phê duyệt.'
    : openConflicts.length
      ? `Còn ${openConflicts.length} mâu thuẫn nguồn — chọn giá trị đúng ở khung "Mâu thuẫn nguồn".`
      : openGaps.length
        ? `Còn ${openGaps.length} mục nguồn chưa có trong định nghĩa — bổ sung hoặc ghi lý do để trống.`
        : openTruncations.length
          ? 'Có đề xuất bị cắt do vượt giới hạn — xem danh sách và xác nhận.'
          : outsideUnreviewed
            ? 'Có trang ngoài các mục thủ tục cần xem (nhắc tên thủ tục hoặc quy định chung) — xem và ghi nhận trước khi duyệt.'
            : boundaryPending
              ? 'Ranh giới cuối mục chưa được xác nhận.'
      : missingCount
      ? `Còn ${missingCount} đề xuất thiếu nguồn — sửa dữ liệu hoặc soạn lại với phạm vi trang đúng.`
      : pendingCount
        ? `Còn ${pendingCount} đề xuất chưa chấp nhận.`
        : errorIssues.length
          ? `Còn ${errorIssues.length} lỗi ở khung "Kiểm tra tự động" bên phải — xem và sửa (có nút sửa nhanh nếu có).`
          : 'Chưa đủ điều kiện phê duyệt — bấm "Kiểm tra lại".'

  return (
    <div className="admin-page draft-review">
      <header className="dr-hero">
        <div className="dr-hero-main">
          <Link className="dr-back" to={adminDraftsPath(draft.procedure_id)}>← Bản nháp của thủ tục</Link>
          <h1>{definition.name || draft.procedure_code}</h1>
          <div className="dr-hero-meta">
            <Chip tone="code"><code>{draft.procedure_code}</code></Chip>
            <Chip tone={status.tone}>{status.label}</Chip>
            <span>Mô hình tự động: {draft.model || '—'}</span>
            {draft.document_id ? (
              <span>
                Nguồn: <Link to={adminDocumentPath(draft.document_id)}>tài liệu</Link> · trang {draft.source_page_start}–{draft.source_page_end}
              </span>
            ) : null}
            {coverage ? <span>Gộp từ {coverage.batches} đợt</span> : null}
            {draft.parent_draft_id ? (
              <span>Bổ sung từ <Link to={adminDraftPath(draft.parent_draft_id)}>bản nháp {String(draft.parent_draft_id).slice(0, 8)}</Link></span>
            ) : null}
            <span>Cập nhật {formatWhen(draft.updated_at)}</span>
          </div>
        </div>
        <ol className="dr-steps" aria-label="Quy trình">
          {['Tự động soạn', 'Duyệt đề xuất', 'Phê duyệt', 'Kích hoạt'].map((label, idx) => (
            <li key={label} className={idx < stepIndex ? 'done' : idx === stepIndex ? 'current' : ''}>
              <span className="dr-step-dot">{idx < stepIndex ? '✓' : idx + 1}</span>
              <span>{label}</span>
            </li>
          ))}
        </ol>
      </header>

      {error ? <p className="dr-alert dr-alert-bad" role="alert">{error}</p> : null}
      {notice ? <p className="dr-alert dr-alert-ok">{notice}</p> : null}
      {draft.identity_mismatch || v.identity_mismatch ? (
        <p className="dr-alert dr-alert-bad">Gemini/editor đề xuất sai procedure identity. Server đã khóa mã thủ tục; không duyệt được.</p>
      ) : null}

      <section className="dr-metrics">
        <div className="dr-metric">
          <span>Đề xuất có nguồn</span>
          <strong>{v.covered_proposal_count || 0}<small>/{v.material_proposal_count || 0}</small></strong>
          <Meter value={v.covered_proposal_count || 0} total={v.material_proposal_count || 0} />
        </div>
        <div className="dr-metric">
          <span>Đã chấp nhận</span>
          <strong>{acceptedCount}<small>/{proposals.length}</small></strong>
          <Meter value={acceptedCount} total={proposals.length} />
        </div>
        <div className={`dr-metric ${errorIssues.length ? 'is-bad' : 'is-ok'}`}>
          <span>Lỗi cần xử lý</span>
          <strong>{errorIssues.length}</strong>
          <small>{warningIssues.length} lưu ý</small>
        </div>
        <div className={`dr-metric ${v.schema_valid && v.source_quotes_verified ? 'is-ok' : 'is-bad'}`}>
          <span>Cấu trúc · trích dẫn</span>
          <strong className="dr-metric-text">{v.schema_valid ? 'Hợp lệ' : 'Lỗi'} · {v.source_quotes_verified ? 'Khớp tài liệu' : 'Chưa khớp'}</strong>
          <small>{v.semantic_approved ? 'Cán bộ đã duyệt nội dung' : 'Chưa duyệt nội dung'}</small>
        </div>
      </section>

      {coverage ? (
        <section className={`dr-panel dr-coverage ${coverage.complete ? '' : 'is-bad'}`}>
          <h3>Phạm vi nguồn đã xử lý</h3>
          <p>
            Tài liệu {coverage.document_pages} trang · {coverage.document_chunks} đoạn.
            {' '}{coverage.section_title ? <>Mục <strong>{coverage.section_title}</strong>, </> : 'Khoảng trang tự chọn, '}
            trang {coverage.page_start}–{coverage.page_end}.
            {' '}Đã xử lý {coverage.processed_chunks}/{coverage.selected_chunks} đoạn trong {coverage.batches} đợt
            {coverage.base_chunks ? ` (thêm ${coverage.base_chunks} đoạn từ bản nền)` : ''}
            {coverage.duplicates_removed ? ` · gộp ${coverage.duplicates_removed} đề xuất trùng` : ''}.
          </p>
          <ul className="dr-scope">
            <li className={scopeComplete ? 'ok' : 'bad'}>
              Phạm vi đã chọn: {scopeComplete ? 'đã xử lý hết' : 'chưa xử lý hết — không được duyệt'}
            </li>
            <li className={outsideUnreviewed ? 'bad' : 'ok'}>
              Kết quả tìm ngoài phạm vi: {unsectioned.length === 0
                ? 'không thấy chỗ nào theo tiêu chí tìm'
                : outsideUnreviewed ? `${unsectioned.length} nhóm chưa được cán bộ xem xét` : 'Cán bộ đã xem xét'}
            </li>
          </ul>
          {coverage.outside_criteria ? (
            <p className="muted dr-criteria">Tiêu chí tìm: {coverage.outside_criteria}</p>
          ) : null}
          {coverage.boundary_uncertain ? (
            <div className={`dr-boundary ${boundaryPending ? 'is-bad' : 'is-ack'}`}>
              <p>
                Ranh giới cuối mục không chắc chắn (số thứ tự mục kế tiếp bị nhảy). Kiểm tra trang {coverage.page_start}–{coverage.page_end}
                {' '}có phải đều thuộc thủ tục này không.
              </p>
              {boundaryPending ? (
                <span className="dr-gap-form">
                  <input
                    type="text"
                    aria-label="Xác nhận ranh giới mục"
                    placeholder="Đã kiểm tra: các trang cuối thuộc thủ tục này vì…"
                    value={gapReasons.section_boundary || ''}
                    disabled={!!busy || !editable}
                    onChange={(e) => setGapReasons((prev) => ({ ...prev, section_boundary: e.target.value }))}
                  />
                  <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onAcknowledgeGap('section_boundary')}>
                    Xác nhận ranh giới
                  </button>
                </span>
              ) : (
                <p className="dr-gap-ack">Cán bộ đã xác nhận: “{findingAck(coverage, 'section_boundary')?.reason}”</p>
              )}
            </div>
          ) : null}
          {coverage.gaps?.length ? (
            <ul className="dr-gaps">
              {coverage.gaps.map((gap) => (
                <li key={gap.field} className={gap.acknowledged ? 'is-ack' : ''}>
                  Nguồn có mục <strong>{gap.topic}</strong> (trang {gap.page}) nhưng định nghĩa chưa có nội dung tương ứng.
                  {gap.acknowledged ? (
                    <span className="dr-gap-ack"> Cán bộ xác nhận để trống: “{gap.reason}”</span>
                  ) : (
                    <span className="dr-gap-form">
                      <input
                        type="text"
                        aria-label={`Lý do để trống ${gap.topic}`}
                        placeholder="Lý do để trống (nếu văn bản không quy định)"
                        value={gapReasons[gap.field] || ''}
                        disabled={!!busy || !editable}
                        onChange={(e) => setGapReasons((prev) => ({ ...prev, [gap.field]: e.target.value }))}
                      />
                      <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onAcknowledgeGap(gap.field)}>
                        Xác nhận để trống
                      </button>
                    </span>
                  )}
                </li>
              ))}
            </ul>
          ) : null}
          {coverage.truncated?.length ? (
            <div className="dr-truncated">
              {coverage.truncated.map((t) => {
                const field = `truncated:${t.field}`
                const ack = findingAck(coverage, field)
                return (
                  <div key={t.field} className={`dr-trunc ${ack ? 'is-ack' : t.blocking ? 'is-bad' : ''}`}>
                    <p>
                      Bị cắt {t.dropped.length} {TRUNCATED_LABEL[t.field] || t.field} (giới hạn {t.limit}):
                    </p>
                    <ul>
                      {t.dropped.map((item, idx) => (
                        <li key={`${item}-${idx}`}>
                          {t.field === 'citations' && t.dropped_citations?.[idx] ? citationLabel(t.dropped_citations[idx]) : item}
                          {t.field === 'citations' && t.dropped_citations?.[idx] ? (
                            <span className="dr-keep">
                              <select
                                aria-label={`Giữ ${citationLabel(t.dropped_citations[idx])} thay cho`}
                                value={keepReplace[idx] ?? ''}
                                disabled={!!busy || !editable}
                                onChange={(e) => setKeepReplace((prev) => ({ ...prev, [idx]: e.target.value }))}
                              >
                                <option value="">Thêm vào (không thay)</option>
                                {(definition.citations || []).map((c, ci) => (
                                  <option key={`${c.title}-${ci}`} value={ci}>Thay cho: {citationLabel(c)}</option>
                                ))}
                              </select>
                              <button type="button" className="dr-btn dr-btn-soft" disabled={!!busy || !editable} onClick={() => onKeepCitation(idx)}>
                                Giữ
                              </button>
                            </span>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                    {ack ? (
                      <p className="dr-gap-ack">Cán bộ đã xem: “{ack.reason}”</p>
                    ) : (
                      <span className="dr-gap-form">
                        <input
                          type="text"
                          aria-label={`Lý do cho nội dung bị cắt ${t.field}`}
                          placeholder="Đã xem các mục bị cắt: lý do (ví dụ trùng nghĩa, đã thêm thủ công)"
                          value={gapReasons[field] || ''}
                          disabled={!!busy || !editable}
                          onChange={(e) => setGapReasons((prev) => ({ ...prev, [field]: e.target.value }))}
                        />
                        <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onAcknowledgeGap(field)}>
                          Xác nhận đã xem
                        </button>
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
          ) : null}
          {coverage.outside_mentions?.length ? (
            <details className="dr-outside" open={outsideUnreviewed}>
              <summary>{OUTSIDE_TITLE} ({coverage.outside_mentions.length} nhóm)</summary>
              <ul>
                {coverage.outside_mentions.map((m) => (
                  <li key={`${m.kind}-${m.section}-${m.page_start}`}>{mentionLabel(m)} · trang {mentionPages(m)} · {m.chunks} đoạn</li>
                ))}
              </ul>
              {unsectioned.length ? (
                findingAck(coverage, 'outside_mentions') && !outsideUnreviewed ? (
                  <p className="dr-gap-ack">Cán bộ đã xem: “{findingAck(coverage, 'outside_mentions').reason}”</p>
                ) : (
                  <span className="dr-gap-form">
                    <input
                      type="text"
                      aria-label="Kết quả xem các trang ngoài mục thủ tục"
                      placeholder="Đã xem các trang ngoài mục: lý do không cần bổ sung (hoặc dùng chế độ Bổ sung)"
                      value={gapReasons.outside_mentions || ''}
                      disabled={!!busy || !editable}
                      onChange={(e) => setGapReasons((prev) => ({ ...prev, outside_mentions: e.target.value }))}
                    />
                    <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onAcknowledgeGap('outside_mentions')}>
                      Xác nhận đã xem
                    </button>
                  </span>
                )
              ) : null}
            </details>
          ) : null}
        </section>
      ) : null}

      {conflicts.length ? (
        <section className="dr-panel dr-conflicts">
          <h3>Mâu thuẫn nguồn</h3>
          <p className="muted">Các phần khác nhau của tài liệu nêu giá trị khác nhau. Hệ thống không tự chọn — cán bộ cần đối chiếu trích dẫn và quyết định.</p>
          {conflicts.map((c) => {
            const field = String(c.field || '').replace('guidance.', '')
            const slotKey = String(c.field || '').startsWith('slots.') ? c.field.slice(6) : ''
            const citeDoc = c.field === 'citations'
              ? String(c.candidates?.[0]?.citation?.doc_id || c.candidates?.[0]?.citation?.document_id || '').slice(0, 8)
              : ''
            const title = slotKey
              ? `Kiểu trả lời của câu hỏi ${slotKey}`
              : c.field === 'citations'
                ? `Tên/số hiệu văn bản trích dẫn${citeDoc ? ` (mã ${citeDoc})` : ''} — các đợt ghi khác nhau`
                : GUIDANCE_LABEL[field] || c.field
            return (
              <div key={c.id} className={`dr-conflict ${c.status === 'RESOLVED' ? 'is-resolved' : ''}`}>
                <header>
                  <strong>{title}</strong>
                  <Chip tone={c.status === 'RESOLVED' || c.inactive ? 'ok' : 'bad'}>
                    {c.status === 'RESOLVED'
                      ? (c.chosen >= 0 ? `Đã chọn phương án ${c.chosen + 1}` : 'Giữ giá trị hiện tại')
                      : c.inactive ? 'Chưa áp dụng' : 'Chưa chọn'}
                  </Chip>
                </header>
                {c.inactive ? (
                  <p className="muted">Văn bản này hiện không có trong citations (bị cắt khi gộp hoặc đã gỡ). Nếu giữ lại văn bản, mâu thuẫn này phải được chọn trước khi duyệt.</p>
                ) : null}
                {c.reopened && c.status !== 'RESOLVED' ? (
                  <p className="muted">Giá trị đã bị sửa sau lần chọn trước — hãy chọn lại hoặc giữ giá trị đang có trong dữ liệu cấu hình.</p>
                ) : null}
                <ol>
                  {c.candidates.map((cand, idx) => (
                    <li key={`${c.id}-${idx}`} className={c.chosen === idx ? 'is-chosen' : ''}>
                      <p>{cand.value}</p>
                      <small>
                        {CONTRIBUTOR_LABEL(cand.contributor)}{cand.page_start ? ` · trang ${cand.page_start}–${cand.page_end}` : ''}
                        {cand.also_from?.length ? ` · cũng có ở ${cand.also_from.map(CONTRIBUTOR_LABEL).join(', ')}` : ''}
                      </small>
                      {cand.evidence?.quotation ? (
                        <blockquote className="dr-quote"><span className="dr-quote-page">Tr. {cand.evidence.page}</span>“{cand.evidence.quotation}”</blockquote>
                      ) : null}
                      <button type="button" className="dr-btn dr-btn-soft" disabled={!!busy || !editable} onClick={() => onResolveConflict(c.id, idx)}>
                        Chọn phương án {idx + 1}
                      </button>
                    </li>
                  ))}
                </ol>
                <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onResolveConflict(c.id, -1)}>
                  Giữ giá trị đang có trong dữ liệu
                </button>
              </div>
            )
          })}
        </section>
      ) : null}

      <div className="dr-layout">
        <section className="dr-main">
          <div className="dr-section-head">
            <div>
              <h2>Đề xuất và nguồn</h2>
              <p className="muted">Khớp trích dẫn không phải xác nhận pháp lý — đối chiếu nghĩa trước khi chấp nhận.</p>
            </div>
            <button type="button" className="dr-btn dr-btn-soft" disabled={!!busy || !editable} onClick={onAcceptAllValid}>
              Chấp nhận mọi trích dẫn hợp lệ
            </button>
          </div>
          <div className="dr-tabs" role="tablist">
            {FILTERS.map((item) => (
              <button
                key={item.key}
                type="button"
                role="tab"
                aria-selected={filter === item.key}
                className={filter === item.key ? 'active' : ''}
                onClick={() => setFilter(item.key)}
              >
                {item.label}<span>{counts[item.key]}</span>
              </button>
            ))}
          </div>

          {visible.length === 0 ? <p className="dr-empty">Không có đề xuất trong mục này.</p> : null}
          <div className="dr-cards">
            {visible.map((prop) => {
              const ev = evidenceFor(prop.id)
              const label = proposalLabel(prop.id)
              const review = REVIEW_STATUS[ev?.review_status || 'pending'] || REVIEW_STATUS.pending
              const decided = ev?.review_status
              const shown = describeProposal(prop, slotsByKey)
              const adminDecision = decisionFor(prop.id)
              return (
                <article key={prop.id} className={`dr-card ${ev?.quote_valid || adminDecision ? '' : 'is-missing'} ${decided === 'accepted' || adminDecision ? 'is-accepted' : ''}`}>
                  <header>
                    <Chip tone={label.tone}>{label.group}</Chip>
                    {label.name ? <code className="dr-key">{label.name}</code> : null}
                    <span className="dr-card-status">
                      {adminDecision
                        ? <Chip tone="violet">Quyết định của cán bộ</Chip>
                        : ev?.quote_valid ? <Chip tone="ok">Có nguồn · trang {ev.page}</Chip> : <Chip tone="bad">Thiếu nguồn</Chip>}
                      {adminDecision ? null : <Chip tone={review.tone}>{review.label}</Chip>}
                    </span>
                  </header>
                  <p className="dr-card-content">{shown.text}</p>
                  {(() => {
                    const sources = provenanceFor(draft, prop)
                    if (sources.length < 2) return null
                    return (
                      <details className="dr-prov">
                        <summary>{sources.length} nguồn ủng hộ · trang {[...new Set(sources.map((src) => src.page))].join(', ')}</summary>
                        <ul>
                          {sources.map((src, idx) => (
                            <li key={`${src.excerpt_id}-${idx}`} className={src.quote_valid ? '' : 'is-bad'}>
                              {CONTRIBUTOR_LABEL(src.contributor)} · tr. {src.page}: “{src.quotation}”
                            </li>
                          ))}
                        </ul>
                      </details>
                    )
                  })()}
                  {shown.answer ? (
                    <div className="dr-answer">
                      <span className="dr-answer-kind">{shown.answer}</span>
                      {shown.options.map((opt) => <code key={opt}>{opt}</code>)}
                    </div>
                  ) : null}
                  {ev?.quotation ? (
                    <blockquote className="dr-quote">
                      <span className="dr-quote-page">Tr. {ev.page}</span>
                      “{ev.quotation}”
                    </blockquote>
                  ) : (
                    <p className="dr-quote dr-quote-missing">Hệ thống không tìm thấy câu trích cho đề xuất này. Nguồn đã chọn có thể không chứa thông tin đó.</p>
                  )}
                  {ev ? (
                    <div className="dr-segment" role="group" aria-label={`Quyết định cho ${prop.id}`}>
                      <button type="button" className={decided === 'accepted' ? 'on ok' : ''} disabled={!!busy || !ev.quote_valid || !editable} onClick={() => onReview(prop.id, 'accepted')}>Chấp nhận</button>
                      <button type="button" className={decided === 'needs_correction' ? 'on warn' : ''} disabled={!!busy || !editable} onClick={() => onReview(prop.id, 'needs_correction')}>Cần sửa</button>
                      <button type="button" className={decided === 'rejected' ? 'on bad' : ''} disabled={!!busy || !editable} title="Bỏ mục này khỏi bản nháp" onClick={() => onRemoveProposal(prop.id)}>Từ chối</button>
                    </div>
                  ) : null}
                  {adminDecision ? (
                    <div className="dr-fix">
                      <p className="dr-fix-hint">Văn bản nguồn không nêu mục này; giữ theo quyết định của cán bộ. Lý do: {adminDecision.reason}</p>
                      <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onDecide(prop.id, true)}>
                        Huỷ quyết định
                      </button>
                    </div>
                  ) : !ev?.quote_valid ? (
                    <div className="dr-fix">
                      <p className="dr-fix-hint">
                        {ev ? 'Câu trích không khớp văn bản nguồn nên mục này chưa duyệt được.' : 'Chưa duyệt được vì không có nguồn.'}
                        {' '}Bỏ mục này, hoặc nếu văn bản thật sự không nêu mà vẫn cần giữ, ghi lý do để giữ theo quyết định của cán bộ.
                      </p>
                      {ev ? null : (
                        <button type="button" className="dr-btn dr-btn-danger" disabled={!!busy || !editable} onClick={() => onRemoveProposal(prop.id)}>
                          Bỏ khỏi bản nháp
                        </button>
                      )}
                      <textarea
                        className="dr-reason"
                        rows={2}
                        placeholder="Lý do giữ (vd: quy định riêng của UBND xã, văn bản không nêu)"
                        value={decisionReason[prop.id] || ''}
                        disabled={!!busy || !editable}
                        onChange={(e) => setDecisionReason({ ...decisionReason, [prop.id]: e.target.value })}
                      />
                      <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onDecide(prop.id)}>
                        Giữ theo quyết định của cán bộ
                      </button>
                    </div>
                  ) : null}
                </article>
              )
            })}
          </div>

          <details className="dr-advanced">
            <summary>Nâng cao · chỉnh sửa dữ liệu cấu hình</summary>
            {parseError ? <p className="dr-alert dr-alert-bad">{parseError}</p> : null}
            <textarea
              className="draft-json"
              rows={18}
              value={definitionText}
              onChange={(e) => setDefinitionText(e.target.value)}
              disabled={!!busy || !editable}
              spellCheck={false}
            />
            <p className="muted">Lưu sẽ đánh dấu nguồn của đề xuất đã sửa là cũ; cần chạy lại kiểm tra.</p>
          </details>
        </section>

        <aside className="dr-side">
          <div className="dr-panel dr-preview">
            <h3>Người dân sẽ thấy</h3>
            {guidance.summary ? <p className="dr-preview-summary">{guidance.summary}</p> : <p className="muted">Chưa có tóm tắt.</p>}
            <dl>
              {['where_to_submit', 'processing_time_hint', 'fee_hint'].map((key) => (
                <div key={key}>
                  <dt>{GUIDANCE_LABEL[key]}</dt>
                  <dd className={guidance[key] ? '' : 'dr-missing'}>{guidance[key] || 'Chưa có — nguồn không nêu'}</dd>
                </div>
              ))}
            </dl>
            {checklist.length ? (
              <>
                <h4>Giấy tờ / bước</h4>
                <ol>{checklist.map((item, idx) => <li key={idx}>{item}</li>)}</ol>
              </>
            ) : null}
            {(guidance.conditional_checklist || []).length || (guidance.conditional_notes || []).length ? (
              <>
                <h4>Chỉ hiện theo câu trả lời</h4>
                <ul className="dr-slots">
                  {[...(guidance.conditional_checklist || []).map((r) => ({ ...r, kind: 'Giấy tờ' })),
                    ...(guidance.conditional_notes || []).map((r) => ({ ...r, kind: 'Lưu ý' }))].map((rule, idx) => (
                    <li key={idx}>
                      <span>{rule.kind}: {rule.item}</span>
                      <small>{whenText(`${rule.when?.slot}=${JSON.stringify(rule.when?.equals)}`, slotsByKey)}</small>
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            {slots.length ? (
              <>
                <h4>Trợ lý sẽ hỏi</h4>
                <ul className="dr-slots">
                  {slots.map(([key, slot]) => (
                    <li key={key}>
                      <span>{slot.question || key}</span>
                      <small>
                        {answerLabel(slot.type)}
                        {slot.type === 'enum' && (slot.enum_values || []).length
                          ? `: ${slot.enum_values.map((value, i) => optionText(value, (slot.enum_labels || [])[i])).join(' · ')}`
                          : ''}
                        {(definition.required_slots || []).includes(key) ? ' · bắt buộc' : ''}
                      </small>
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
          </div>

          {(draft.missing_requirements || []).length || (draft.warnings || []).length ? (
            <div className="dr-panel dr-notes">
              <h3>Ghi chú của hệ thống tự động</h3>
              {(draft.missing_requirements || []).length ? (
                <>
                  <h4>Không tìm thấy trong nguồn</h4>
                  <ul>{draft.missing_requirements.map((item, idx) => <li key={idx}>{item}</li>)}</ul>
                </>
              ) : null}
              {(draft.warnings || []).length ? (
                <>
                  <h4>Lưu ý</h4>
                  <ul>{draft.warnings.map((item, idx) => <li key={idx}>{item}</li>)}</ul>
                </>
              ) : null}
            </div>
          ) : null}

          {(v.issues || []).length ? (
            <div className="dr-panel dr-issues">
              <h3>Kiểm tra tự động</h3>
              <ul>
                {[...errorIssues, ...warningIssues].map((issue, idx) => {
                  const label = proposalLabel(issue.proposal_id)
                  return (
                    <li key={`${issue.code}-${idx}`} className={issue.severity === 'error' ? 'is-error' : ''} title={`${issue.code}: ${issue.message}`}>
                      <strong>{ISSUE_LABEL[issue.code] || issue.code}</strong>
                      {issue.proposal_id ? <span>{label.group}{label.name ? ` · ${label.name}` : ''}</span> : null}
                      {issue.code === 'UNREACHABLE_SLOT' && issue.proposal_id ? (() => {
                        const slotKey = issue.proposal_id.slice(5)
                        const conds = askConditions(slotsByKey, slotKey)
                        return (
                          <div className="dr-ask-rule">
                            <small>«{slotsByKey[slotKey]?.question || slotKey}»</small>
                            <select
                              value={askWhen[slotKey] ?? 'always'}
                              disabled={!!busy || !editable}
                              onChange={(e) => setAskWhen({ ...askWhen, [slotKey]: e.target.value })}
                              aria-label={`Khi nào hỏi ${slotKey}`}
                            >
                              <option value="always">Luôn hỏi</option>
                              {conds.map((c, i) => <option key={i} value={String(i)}>Chỉ hỏi khi {c.label}</option>)}
                              <option value="remove">Bỏ câu hỏi này</option>
                            </select>
                            <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={() => onAskRule(slotKey)}>Áp dụng</button>
                          </div>
                        )
                      })() : null}
                      {issue.code === 'INVALID_GUIDANCE_CONDITION' && issue.proposal_id ? (
                        <button type="button" className="dr-btn dr-btn-danger" disabled={!!busy || !editable} onClick={() => onRemoveProposal(issue.proposal_id)}>
                          Bỏ mục này
                        </button>
                      ) : null}
                    </li>
                  )
                })}
              </ul>
            </div>
          ) : null}
        </aside>
      </div>

      <footer className="dr-actionbar">
        <p className={approveReady ? 'ok' : ''}>{approveHint}</p>
        <div className="dr-actionbar-buttons">
          <button type="button" className="dr-btn" disabled={!!busy || !editable} onClick={onSave}>
            {busy === 'save' ? 'Đang lưu…' : 'Lưu dữ liệu'}
          </button>
          <button type="button" className="dr-btn" disabled={!!busy} onClick={onValidate}>
            {busy === 'validate' ? 'Đang kiểm…' : 'Kiểm tra lại'}
          </button>
          <button type="button" className="dr-btn dr-btn-primary" disabled={!!busy || !approveReady} onClick={() => setConfirmApprove(true)}>
            Phê duyệt
          </button>
          {draft.status === 'APPROVED' || draft.status === 'PUBLISHED' ? (
            <button type="button" className="dr-btn" disabled={!!busy} onClick={onClone} title="Chép bản này thành bản nháp mới để sửa rồi duyệt lại — không gọi AI">
              {busy === 'clone' ? 'Đang tạo…' : 'Tạo bản nháp để sửa'}
            </button>
          ) : null}
          <button type="button" className="dr-btn dr-btn-accent" disabled={!!busy || (draft.status !== 'APPROVED' && draft.status !== 'PUBLISHED')} onClick={openActivate}>
            {busy === 'activate' ? 'Đang kích hoạt…' : 'Kích hoạt / Tiếp tục'}
          </button>
        </div>
      </footer>

      {confirmApprove ? (
        <div className="confirm-overlay">
          <div className="panel dr-modal" role="dialog" aria-modal="true">
            <h2>Duyệt bản nháp?</h2>
            <p>Đây là bước duyệt nội dung trong ứng dụng nghiên cứu, không phải chứng nhận pháp lý của cơ quan nhà nước.</p>
            <div className="draft-actions">
              <button type="button" className="dr-btn" onClick={() => setConfirmApprove(false)}>Hủy</button>
              <button type="button" className="dr-btn dr-btn-primary" onClick={onApprove}>Xác nhận duyệt</button>
            </div>
          </div>
        </div>
      ) : null}

      {confirmActivate ? (
        <div className="confirm-overlay">
          <div className="panel dr-modal" role="dialog" aria-modal="true">
            <h2>Kích hoạt phiên bản mới?</h2>
            <p>
              Hệ thống sẽ tạo phiên bản thủ tục, gắn tài liệu rồi lập chỉ mục mới.
              Lịch sử vẫn được giữ nguyên và phiên bản đang dùng chỉ thay đổi khi đủ điều kiện.
            </p>
            <p className="dr-active-target">
              {activeTarget === undefined ? 'Đang kiểm tra phiên bản đang dùng…'
                : activeTarget.error ? 'Không đọc được phiên bản đang dùng. Hãy đóng và thử lại.'
                  : activeTarget.id ? `Sẽ thay thế phiên bản đang dùng: ${activeTarget.version || activeTarget.id} (bản cũ được lưu trữ, không bị xóa).`
                    : 'Thủ tục chưa có phiên bản đang dùng — đây sẽ là phiên bản đầu tiên.'}
            </p>
            <div className="draft-actions">
              <button type="button" className="dr-btn" onClick={() => setConfirmActivate(false)}>Hủy</button>
              <button type="button" className="dr-btn dr-btn-accent" disabled={activeTarget === undefined || activeTarget.error} onClick={onActivate}>Xác nhận kích hoạt</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
