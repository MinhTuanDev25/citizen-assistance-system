import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  activateDraft,
  approveDraft,
  canApproveDraft,
  getDraft,
  patchDraft,
  reviewDraft,
  validateDraft,
} from '../../api/drafts.js'
import { adminDocumentPath, adminDraftsPath } from '../../routes.js'

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

function issueClass(severity) {
  return severity === 'error' ? 'form-error' : 'muted'
}

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
  const slots = Object.entries(definition.slots || {})
  const checklist = definition.guidance?.checklist || []

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
      setParseError('JSON không hợp lệ. Không lưu.')
      return
    }
    await withBusy('save', () => patchDraft(draftId, {
      definition: parsed,
      expected_updated_at: draft.updated_at,
    }))
    setNotice('Đã lưu. Evidence cũ của đề xuất đã sửa bị đánh stale.')
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
    setNotice('Admin đã duyệt bản nháp. Gemini không thể tự duyệt.')
  }

  async function onActivate() {
    const result = await withBusy('activate', () => activateDraft(draftId, {
      confirm: true,
      expected_updated_at: draft.updated_at,
    }, crypto.randomUUID()))
    setConfirmActivate(false)
    if (result?.activated && result?.reconciliation_required) {
      setNotice('Phiên bản đã ACTIVE. Trạng thái draft PUBLISHED chưa ghi được — bấm Activate để reconcile. Không tạo version mới.')
    } else if (result?.activated) {
      setNotice(`Đã kích hoạt phiên bản ${result.version?.version || ''}.`)
    } else if (result?.indexing_pending) {
      setNotice(`Đang lập chỉ mục (${result.phase || 'INDEXING_PENDING'}). Bấm Activate lại sau khi generation READY — không tạo generation trùng.`)
    } else {
      setNotice(`Chưa kích hoạt: ${result?.blocked_reason || (result?.reasons || []).join(', ') || 'chưa đủ điều kiện'}.`)
    }
  }

  if (loading) return <p className="muted">Đang tải bản nháp…</p>
  if (!draft) return <p className="form-error">{error || 'Không tìm thấy bản nháp.'}</p>

  const v = draft.validation || {}
  const approveReady = canApproveDraft(draft) && (draft.evidence || []).every((item) => {
    if (!item.quote_valid) return true
    return item.review_status === 'accepted'
  }) && (draft.proposals || []).every((p) => (draft.evidence || []).some((e) => e.proposal_id === p.id && e.review_status === 'accepted'))

  return (
    <div className="admin-page draft-review">
      <header className="admin-page-head">
        <div>
          <h1>Review bản nháp</h1>
          <p>
            <code>{draft.procedure_code}</code> · {draft.status} · {draft.model || '—'} · {formatWhen(draft.updated_at)}
          </p>
        </div>
        <Link to={adminDraftsPath(draft.procedure_id)}>← Danh sách</Link>
      </header>
      {error ? <p className="form-error">{error}</p> : null}
      {notice ? <p className="banner ok">{notice}</p> : null}
      {draft.identity_mismatch || v.identity_mismatch ? (
        <p className="form-error">Gemini/editor đề xuất sai procedure identity. Server đã khóa mã thủ tục; không duyệt được.</p>
      ) : null}

      <div className="stat-grid-2">
        <div className="stat-card"><span>Schema</span><strong>{v.schema_valid ? 'hợp lệ' : 'lỗi'}</strong></div>
        <div className="stat-card"><span>Coverage</span><strong>{v.covered_proposal_count || 0}/{v.material_proposal_count || 0}</strong></div>
        <div className="stat-card"><span>Quotes</span><strong>{v.source_quotes_verified ? 'đúng' : 'chưa'}</strong></div>
        <div className="stat-card"><span>Duyệt ngữ nghĩa</span><strong>{v.semantic_approved ? 'Admin đã duyệt' : 'chưa'}</strong></div>
      </div>

      <div className="panel">
        <h2>Đề xuất và nguồn</h2>
        <p className="muted">Khớp trích dẫn không phải xác nhận pháp lý. PDF renderer có thể không xem được trang; dùng đúng đoạn văn bản đã hash bên dưới.</p>
        {(draft.proposals || []).map((prop) => {
          const ev = (draft.evidence || []).find((item) => item.proposal_id === prop.id)
          return (
            <article key={prop.id} className="draft-proposal">
              <header>
                <code>{prop.id}</code>
                <span className="pill mute">{prop.type}</span>
                {ev?.quote_valid ? <span className="pill ok">quote ok</span> : <span className="pill bad">thiếu/sai nguồn</span>}
                <span className="pill">{ev?.review_status || 'pending'}</span>
              </header>
              <p>{prop.content}</p>
              {ev ? (
                <blockquote>
                  Trang {ev.page}: “{ev.quotation}”
                </blockquote>
              ) : (
                <p className="form-error">Không có evidence.</p>
              )}
              <div className="draft-actions">
                <button type="button" disabled={!!busy || !ev?.quote_valid} onClick={() => onReview(prop.id, 'accepted')}>Chấp nhận</button>
                <button type="button" disabled={!!busy} onClick={() => onReview(prop.id, 'rejected')}>Từ chối</button>
                <button type="button" disabled={!!busy} onClick={() => onReview(prop.id, 'needs_correction')}>Cần sửa</button>
              </div>
            </article>
          )
        })}
        <button type="button" disabled={!!busy} onClick={onAcceptAllValid}>Chấp nhận mọi quote hợp lệ</button>
      </div>

      <div className="panel">
        <h2>Slots / checklist</h2>
        <ul>
          {slots.map(([key, slot]) => (
            <li key={key}><code>{key}</code> · {slot.type} · {slot.question}</li>
          ))}
        </ul>
        <ol>
          {checklist.map((item, idx) => (
            <li key={idx}>{item}</li>
          ))}
        </ol>
        <p>{definition.guidance?.summary}</p>
        {draft.document_id ? (
          <p>
            Nguồn: <Link to={adminDocumentPath(draft.document_id)}>mở tài liệu</Link>
            {draft.generation_id ? <> · generation <code>{String(draft.generation_id).slice(0, 8)}</code></> : null}
            · trang {draft.source_page_start}–{draft.source_page_end}
          </p>
        ) : null}
      </div>

      <div className="panel">
        <h2>JSON Definition</h2>
        {parseError ? <p className="form-error">{parseError}</p> : null}
        <textarea
          className="draft-json"
          rows={18}
          value={definitionText}
          onChange={(e) => setDefinitionText(e.target.value)}
          disabled={!!busy || draft.status !== 'DRAFT'}
        />
        <div className="draft-actions">
          <button type="button" className="btn-primary" disabled={!!busy || draft.status !== 'DRAFT'} onClick={onSave}>
            {busy === 'save' ? 'Đang lưu…' : 'Save Draft'}
          </button>
          <button type="button" disabled={!!busy} onClick={onValidate}>
            {busy === 'validate' ? 'Đang kiểm…' : 'Validate'}
          </button>
          <button type="button" disabled={!!busy || !approveReady} onClick={() => setConfirmApprove(true)}>
            Approve Draft
          </button>
          <button type="button" disabled={!!busy || (draft.status !== 'APPROVED' && draft.status !== 'PUBLISHED')} onClick={() => setConfirmActivate(true)}>
            {busy === 'activate' ? 'Đang kích hoạt…' : 'Activate / Resume'}
          </button>
        </div>
        <p className="muted">Nút disable không phải ranh giới bảo mật. Backend từ chối duyệt/kích hoạt khi thiếu điều kiện.</p>
      </div>

      {(v.issues || []).length ? (
        <div className="panel">
          <h2>Validation</h2>
          <ul>
            {v.issues.map((issue, idx) => (
              <li key={`${issue.code}-${idx}`} className={issueClass(issue.severity)}>
                {issue.code}: {issue.message} {issue.proposal_id ? <code>{issue.proposal_id}</code> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {confirmApprove ? (
        <div className="confirm-overlay">
          <div className="panel">
            <h2>Duyệt bản nháp?</h2>
            <p>Đây là duyệt ứng dụng cho demo luận văn, không phải chứng nhận pháp lý nhà nước.</p>
            <div className="draft-actions">
              <button type="button" className="btn-primary" onClick={onApprove}>Xác nhận duyệt</button>
              <button type="button" onClick={() => setConfirmApprove(false)}>Hủy</button>
            </div>
          </div>
        </div>
      ) : null}

      {confirmActivate ? (
        <div className="confirm-overlay">
          <div className="panel">
            <h2>Kích hoạt phiên bản mới?</h2>
            <p>
              Hệ thống sẽ tạo Procedure Version qua API hiện có, gắn document, rồi lập chỉ mục generation mới.
              Không clone G3 lịch sử. Phiên ACTIVE cũ chỉ đổi khi đủ điều kiện.
            </p>
            <div className="draft-actions">
              <button type="button" className="btn-primary" onClick={onActivate}>Xác nhận kích hoạt</button>
              <button type="button" onClick={() => setConfirmActivate(false)}>Hủy</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
