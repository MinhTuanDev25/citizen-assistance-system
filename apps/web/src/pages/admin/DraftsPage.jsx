import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { listAdminProcedures } from '../../api/catalog.js'
import {
  createDraftJob,
  getDraftJob,
  listDraftJobs,
  listDraftSources,
  listProcedureDrafts,
  planDraft,
  retryDraftJob,
} from '../../api/drafts.js'
import { adminDraftPath, ROUTES } from '../../routes.js'
import { useCommune } from '../../commune/CommuneContext.jsx'
import { mentionLabel, mentionPages, OUTSIDE_TITLE } from './draftSources.js'

function formatWhen(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso || ''
  return d.toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'medium' })
}

function pill(status) {
  const u = String(status || '').toUpperCase()
  if (u === 'APPROVED' || u === 'PUBLISHED' || u === 'SUCCEEDED') return 'pill ok'
  if (u === 'REJECTED' || u === 'FAILED') return 'pill bad'
  if (u === 'REVIEWED' || u === 'RUNNING' || u === 'QUEUED') return 'pill warn'
  return 'pill mute'
}

const JOB_POLL_MS = 3000

const JOB_STATUS = { QUEUED: 'Đang chờ', RUNNING: 'Đang soạn', SUCCEEDED: 'Hoàn tất', FAILED: 'Lỗi' }
const BATCH_STATUS = { PENDING: 'chờ', RUNNING: 'đang chạy', SUCCEEDED: 'xong', FAILED: 'lỗi' }
const DRAFT_STATUS_LABEL = {
  DRAFT: 'Bản nháp',
  REVIEWED: 'Đang duyệt',
  APPROVED: 'Đã phê duyệt',
  PUBLISHED: 'Đã kích hoạt',
  REJECTED: 'Đã từ chối',
}

const PLAN_BLOCKED = {
  SECTION_NOT_FOUND: 'Không tìm thấy mục có đúng tên thủ tục trong tài liệu. Chọn mục thủ công hoặc nhập khoảng trang.',
  SECTION_AMBIGUOUS: 'Tài liệu có nhiều mục cùng tên thủ tục. Chọn một mục cụ thể.',
  MIXED_PROCEDURES: 'Khoảng trang này lẫn nội dung của thủ tục khác — không soạn để tránh trộn thủ tục.',
  SCOPE_TOO_LARGE: 'Phạm vi quá lớn cho một lần soạn. Thu hẹp về đúng mục của thủ tục.',
  NOTHING_NEW: 'Bản nháp nền đã bao phủ toàn bộ phạm vi này — không có trang mới để bổ sung.',
  BASE_SOURCE_MISMATCH: 'Bản nháp nền dùng tài liệu hoặc lượt xử lý khác — không thể gộp.',
  EMPTY_SCOPE: 'Phạm vi đã chọn không có đoạn văn bản nào.',
  OUTSIDE_SECTION: 'Khoảng trang không chứa nội dung của mục thủ tục (ví dụ chỉ là bảng danh mục) — không soạn từ đó.',
}

const BATCH_ERROR = {
  DRAFT_TIMEOUT: 'Dịch vụ tự động xử lý quá thời gian',
  RATE_LIMITED: 'Dịch vụ tự động đang giới hạn tần suất',
  AI_PROVIDER: 'Lỗi gọi dịch vụ trí tuệ nhân tạo',
  AI_INVALID_OUTPUT: 'Dịch vụ trí tuệ nhân tạo trả kết quả sai định dạng',
  SOURCE_CHANGED: 'Nguồn đã thay đổi',
  SOURCE_UNAVAILABLE: 'Không đọc được nguồn',
}

export function planBlockedMessage(reason) {
  return PLAN_BLOCKED[reason] || 'Chưa thể soạn với phạm vi này.'
}

export function batchErrorMessage(code) {
  return BATCH_ERROR[code] || code || ''
}

/** Request body shared by plan preview and job creation (they must match). */
export function buildPlanBody({ source, mode, baseDraftId, scope, sectionIndex, pageStart, pageEnd }) {
  const body = { document_id: source.document_id, generation_id: source.generation_id, mode }
  if (mode === 'SUPPLEMENT') body.base_draft_id = baseDraftId
  if (scope === 'section' && sectionIndex !== '') body.section_index = Number(sectionIndex)
  if (scope === 'manual') {
    body.page_start = Number(pageStart)
    body.page_end = Number(pageEnd)
  }
  return body
}

export function validateManualRange(start, end, source) {
  const hi = Number(source?.suggested_page_end) || 1
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) {
    return 'Nhập trang bắt đầu và kết thúc hợp lệ (bắt đầu ≤ kết thúc).'
  }
  if (end > hi) return `Tài liệu chỉ có ${hi} trang.`
  return ''
}

export function draftErrorMessage(err) {
  if (err?.code === 'PLAN_BLOCKED') return planBlockedMessage(err.message)
  if (err?.code === 'IDEMPOTENCY_CONFLICT') return 'Yêu cầu này đã được gửi với nội dung khác. Phân tích lại phạm vi rồi thử lại.'
  return err?.message || 'Không tạo được bản nháp tự động'
}

function isActive(job) {
  return job?.status === 'QUEUED' || job?.status === 'RUNNING'
}

function JobCard({ job, onRetry, busy }) {
  const plan = job.plan || {}
  const done = job.batches_done || 0
  const total = job.batches_total || 1
  const pct = Math.round((done / total) * 100)
  return (
    <div className={`dj-job dj-${String(job.status).toLowerCase()}`}>
      <div className="dj-job-head">
        <span className={pill(job.status)}>{JOB_STATUS[job.status] || job.status}</span>
        <strong>{job.mode === 'SUPPLEMENT' ? 'Bổ sung' : 'Soạn toàn bộ'}</strong>
        <span className="muted">
          {plan.section_title || 'Khoảng trang tự chọn'} · trang {plan.page_start}–{plan.page_end} · {plan.new_chunks} đoạn
        </span>
        <span className="muted dj-when">{formatWhen(job.created_at)}</span>
      </div>
      <div className="dj-progress" role="progressbar" aria-valuenow={pct} aria-valuemin="0" aria-valuemax="100">
        <span style={{ width: `${pct}%` }} />
      </div>
      <p className="dj-progress-text">
        {done}/{total} đợt xong{job.batches_failed ? ` · ${job.batches_failed} đợt lỗi` : ''}
        {isActive(job) ? ' · mỗi đợt khoảng 1–2 phút' : ''}
      </p>
      {job.batches?.length ? (
        <ul className="dj-batches">
          {job.batches.map((b) => (
            <li key={b.batch_index} className={`dj-batch dj-b-${String(b.status).toLowerCase()}`} title={b.error_message || ''}>
              Đợt {b.batch_index + 1} · tr {b.page_start}–{b.page_end} · {BATCH_STATUS[b.status] || b.status}
              {b.attempts > 1 ? ` · lần ${b.attempts}` : ''}
              {b.error_code ? ` · ${batchErrorMessage(b.error_code)}` : ''}
            </li>
          ))}
        </ul>
      ) : null}
      {job.status === 'FAILED' ? (
        <p className="form-error">
          {job.error_message || 'Tiến trình gặp lỗi.'} Kết quả các đợt đã xong được giữ lại; chưa tạo bản nháp từ một phần nguồn.
        </p>
      ) : null}
      <div className="dj-actions">
        {job.status === 'FAILED' ? (
          <button type="button" className="btn-primary" disabled={busy} onClick={() => onRetry(job.id)}>
            Thử lại các đợt lỗi
          </button>
        ) : null}
        {job.status === 'SUCCEEDED' && job.draft_id ? <Link to={adminDraftPath(job.draft_id)}>Mở bản nháp gộp →</Link> : null}
      </div>
    </div>
  )
}

export default function DraftsPage() {
  const [params] = useSearchParams()
  const { commune } = useCommune()
  const focusProc = params.get('proc')
  const [procedures, setProcedures] = useState([])
  const [procedureId, setProcedureId] = useState(focusProc || '')
  const [sources, setSources] = useState([])
  const [sourceKey, setSourceKey] = useState('')
  const [drafts, setDrafts] = useState([])
  const [jobs, setJobs] = useState([])
  const [mode, setMode] = useState('FULL')
  const [baseDraftId, setBaseDraftId] = useState('')
  const [scope, setScope] = useState('auto')
  const [sectionIndex, setSectionIndex] = useState('')
  const [pageStart, setPageStart] = useState('')
  const [pageEnd, setPageEnd] = useState('')
  const [plan, setPlan] = useState(null)
  // Sections detected by the last analysis; kept so "Chọn mục khác" survives re-planning.
  const [sections, setSections] = useState([])
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const requestIdRef = useRef('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await listAdminProcedures()
        if (cancelled) return
        const items = data?.items || []
        setProcedures(items)
        if (!procedureId && items[0]) setProcedureId(focusProc || items[0].id)
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được thủ tục')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  async function refreshLists(id = procedureId) {
    const [list, jobList] = await Promise.all([listProcedureDrafts(id), listDraftJobs(id).catch(() => ({ items: [] }))])
    setDrafts(list?.items || [])
    const items = jobList?.items || []
    const detailed = await Promise.all(items.map((job) => (isActive(job) ? getDraftJob(job.id).catch(() => job) : job)))
    setJobs(detailed)
  }

  useEffect(() => {
    if (!procedureId) return
    let cancelled = false
    setError('')
    setPlan(null)
    setSections([])
    ;(async () => {
      try {
        const src = await listDraftSources(procedureId)
        if (cancelled) return
        const sourceItems = src?.items || []
        setSources(sourceItems)
        setSourceKey(sourceItems[0] ? `${sourceItems[0].document_id}:${sourceItems[0].generation_id}` : '')
        await refreshLists(procedureId)
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được bản nháp')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [procedureId])

  // Poll running jobs; refresh the draft list when one finishes.
  const activeIds = jobs.filter(isActive).map((job) => job.id).join(',')
  useEffect(() => {
    if (!activeIds) return undefined
    const timer = setInterval(async () => {
      try {
        const updated = await Promise.all(activeIds.split(',').map((id) => getDraftJob(id)))
        let finished = false
        setJobs((prev) => prev.map((job) => {
          const next = updated.find((item) => item.id === job.id)
          if (next && !isActive(next)) finished = true
          return next || job
        }))
        if (finished) setDrafts((await listProcedureDrafts(procedureId))?.items || [])
      } catch {
        // Keep polling; a transient error must not stop progress updates.
      }
    }, JOB_POLL_MS)
    return () => clearInterval(timer)
  }, [activeIds, procedureId])

  const currentSource = sources.find((item) => `${item.document_id}:${item.generation_id}` === sourceKey)

  useEffect(() => {
    setPlan(null)
  }, [sourceKey, mode, baseDraftId, scope, sectionIndex, pageStart, pageEnd])

  function planBody() {
    return buildPlanBody({ source: currentSource, mode, baseDraftId, scope, sectionIndex, pageStart, pageEnd })
  }

  async function onPlan() {
    if (!currentSource) {
      setError('Chọn nguồn đã lập chỉ mục trước.')
      return
    }
    if (mode === 'SUPPLEMENT' && !baseDraftId) {
      setError('Chọn bản nháp nền để bổ sung.')
      return
    }
    if (scope === 'manual') {
      const invalid = validateManualRange(Number(pageStart), Number(pageEnd), currentSource)
      if (invalid) {
        setError(invalid)
        return
      }
    }
    setBusy('plan')
    setError('')
    setNotice('')
    try {
      const result = await planDraft(procedureId, planBody())
      setPlan(result)
      setSections(result?.sections || [])
      requestIdRef.current = crypto.randomUUID()
    } catch (err) {
      setError(draftErrorMessage(err))
    } finally {
      setBusy('')
    }
  }

  async function onStart() {
    if (!plan?.ready || busy) return
    setBusy('start')
    setError('')
    try {
      const job = await createDraftJob(procedureId, planBody(), requestIdRef.current)
      setJobs((prev) => [job, ...prev.filter((item) => item.id !== job.id)])
      setPlan(null)
      setNotice(`Đã xếp hàng ${job.batches_total} đợt. Bản nháp gộp chỉ được tạo khi mọi đợt xong.`)
    } catch (err) {
      setError(draftErrorMessage(err))
    } finally {
      setBusy('')
    }
  }

  async function onRetry(jobId) {
    setBusy('retry')
    setError('')
    try {
      const job = await retryDraftJob(jobId)
      setJobs((prev) => prev.map((item) => (item.id === job.id ? job : item)))
    } catch (err) {
      setError(draftErrorMessage(err))
    } finally {
      setBusy('')
    }
  }

  const selected = procedures.find((item) => item.id === procedureId)
  const baseCandidates = drafts.filter((item) => item.document_id === currentSource?.document_id && item.generation_id === currentSource?.generation_id)

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <div>
          <h1>Bản nháp hướng dẫn tự động</h1>
          <p>Xã {commune?.name || ''} · nội dung có nguồn và chỉ cán bộ mới được phê duyệt</p>
        </div>
      </header>
      {error ? <p className="form-error">{error}</p> : null}
      {notice ? <p className="banner ok">{notice}</p> : null}
      {loading ? <p className="muted">Đang tải…</p> : null}

      <div className="panel draft-generate">
        <h2>Soạn bản nháp từ tài liệu</h2>
        <p className="muted">
          Hệ thống tìm đúng mục của thủ tục trong toàn bộ tài liệu đã lập chỉ mục, chia thành các đợt xử lý phù hợp,
          rồi gộp thành một định nghĩa có trích dẫn. Không đưa cả tài liệu vào một lần gọi, không lấy nội dung của thủ tục khác.
        </p>
        <label>
          Thủ tục
          <select value={procedureId} onChange={(e) => setProcedureId(e.target.value)} disabled={!!busy}>
            {procedures.map((item) => (
              <option key={item.id} value={item.id}>
                {item.procedure_code} — {item.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Nguồn đã lập chỉ mục
          <select value={sourceKey} onChange={(e) => setSourceKey(e.target.value)} disabled={!!busy}>
            {sources.length === 0 ? <option value="">Chưa có nguồn đã xử lý sẵn sàng</option> : null}
            {sources.map((item) => (
              <option key={`${item.document_id}:${item.generation_id}`} value={`${item.document_id}:${item.generation_id}`}>
                {item.title || item.document_id} · {item.version_status === 'SOURCE' ? 'phiên nguồn' : `v${item.version} (${item.version_status})`} · trang {item.suggested_page_start}–{item.suggested_page_end}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="dj-choice">
          <legend>Cách soạn</legend>
          <label>
            <input type="radio" name="mode" value="FULL" checked={mode === 'FULL'} onChange={() => setMode('FULL')} disabled={!!busy} />
            Soạn mới toàn bộ thủ tục
          </label>
          <label>
            <input type="radio" name="mode" value="SUPPLEMENT" checked={mode === 'SUPPLEMENT'} onChange={() => setMode('SUPPLEMENT')} disabled={!!busy || baseCandidates.length === 0} />
            Bổ sung vào bản nháp đã có (tạo bản nháp mới, bản cũ giữ nguyên)
          </label>
          {mode === 'SUPPLEMENT' ? (
            <select value={baseDraftId} onChange={(e) => setBaseDraftId(e.target.value)} disabled={!!busy} aria-label="Bản nháp nền">
              <option value="">— Chọn bản nháp nền —</option>
              {baseCandidates.map((item) => (
                <option key={item.id} value={item.id}>
                  {String(item.id).slice(0, 8)} · {DRAFT_STATUS_LABEL[item.status] || item.status} · trang {item.source_page_start}–{item.source_page_end}
                </option>
              ))}
            </select>
          ) : null}
        </fieldset>

        <fieldset className="dj-choice">
          <legend>Phạm vi nguồn</legend>
          <label>
            <input type="radio" name="scope" value="auto" checked={scope === 'auto'} onChange={() => setScope('auto')} disabled={!!busy} />
            Tự tìm mục có đúng tên thủ tục
          </label>
          <label>
            <input
              type="radio" name="scope" value="section" checked={scope === 'section'} disabled={!!busy || !sections.length}
              onChange={() => {
                setScope('section')
                if (sectionIndex === '') setSectionIndex(String(plan?.selected_section ?? 0))
              }}
            />
            Chọn mục khác
          </label>
          {scope === 'section' && sections.length ? (
            <select value={sectionIndex} onChange={(e) => setSectionIndex(e.target.value)} aria-label="Mục thủ tục">
              {sections.map((sec) => (
                <option key={sec.index} value={sec.index}>
                  {sec.title} · trang {sec.page_start}–{sec.page_end}
                </option>
              ))}
            </select>
          ) : null}
          <label>
            <input type="radio" name="scope" value="manual" checked={scope === 'manual'} onChange={() => setScope('manual')} disabled={!!busy} />
            Tự nhập khoảng trang
          </label>
          {scope === 'manual' ? (
            <div className="draft-page-range">
              <label>
                Từ trang
                <input type="number" min="1" value={pageStart} onChange={(e) => setPageStart(e.target.value)} />
              </label>
              <label>
                Đến trang
                <input type="number" min="1" value={pageEnd} onChange={(e) => setPageEnd(e.target.value)} />
              </label>
            </div>
          ) : null}
        </fieldset>

        <button type="button" className="btn-secondary" onClick={onPlan} disabled={!!busy || !sourceKey}>
          {busy === 'plan' ? 'Đang phân tích…' : 'Phân tích phạm vi'}
        </button>

        {plan ? (
          <div className={`dj-plan ${plan.ready ? 'ok' : 'blocked'}`}>
            <p className="dj-plan-summary">
              Tài liệu {plan.document_pages} trang · {plan.document_chunks} đoạn → {plan.section_title ? <strong>{plan.section_title}</strong> : 'khoảng trang đã nhập'}:
              {' '}trang {plan.page_start}–{plan.page_end}, {plan.selected_chunks} đoạn
              {plan.mode === 'SUPPLEMENT' ? `, bản nền đã có ${plan.base_chunks} đoạn, cần soạn thêm ${plan.new_chunks}` : ''}
              {plan.ready ? ` · ${plan.batches.length} đợt` : ''}
            </p>
            {!plan.ready ? <p className="form-error">{planBlockedMessage(plan.blocked_reason)}</p> : null}
            {plan.warnings?.length ? (
              <ul className="dj-warnings">
                {plan.warnings.map((w) => <li key={w}>{w}</li>)}
              </ul>
            ) : null}
            {plan.outside_mentions?.length ? (
              <details className="dj-outside">
                <summary>{OUTSIDE_TITLE} ({plan.outside_mentions.length} nhóm, không tự đưa vào)</summary>
                <ul>
                  {plan.outside_mentions.map((m) => (
                    <li key={`${m.section}-${m.page_start}`}>
                      {mentionLabel(m)} · trang {mentionPages(m)} · {m.chunks} đoạn
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
            {plan.ready ? (
              <>
                <ul className="dj-batches">
                  {plan.batches.map((b) => (
                    <li key={b.index} className="dj-batch">Đợt {b.index + 1} · tr {b.page_start}–{b.page_end} · {b.excerpts} đoạn</li>
                  ))}
                </ul>
                <button type="button" className="btn-primary" onClick={onStart} disabled={!!busy}>
                  {busy === 'start' ? 'Đang xếp hàng…' : `Bắt đầu soạn (${plan.batches.length} đợt)`}
                </button>
              </>
            ) : null}
          </div>
        ) : null}
      </div>

      {jobs.length ? (
        <div className="panel">
          <div className="panel-head">
            <h2>Tiến độ soạn</h2>
          </div>
          {jobs.map((job) => <JobCard key={job.id} job={job} onRetry={onRetry} busy={!!busy} />)}
        </div>
      ) : null}

      <div className="panel table-wrap">
        <div className="panel-head">
          <h2>Bản nháp {selected ? <code>{selected.procedure_code}</code> : null}</h2>
        </div>
        {drafts.length === 0 ? <p className="muted">Chưa có bản nháp cho thủ tục này.</p> : null}
        {drafts.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Mã</th>
                <th>Trạng thái</th>
                <th>Nguồn</th>
                <th>Hợp lệ</th>
                <th>Mô hình</th>
                <th>Cập nhật</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {drafts.map((item) => (
                <tr key={item.id}>
                  <td><code>{String(item.id).slice(0, 8)}</code></td>
                  <td><span className={pill(item.status)}>{DRAFT_STATUS_LABEL[item.status] || item.status}</span></td>
                  <td>
                    trang {item.source_page_start}–{item.source_page_end}
                    {item.coverage ? ` · ${item.coverage.batches} đợt` : ''}
                    {item.parent_draft_id ? ' · bổ sung' : ''}
                  </td>
                  <td>
                    {item.validation?.schema_valid ? 'đúng cấu trúc' : 'sai cấu trúc'}
                    {item.validation?.source_coverage_complete ? ' · đủ nguồn' : ' · thiếu nguồn'}
                    {(item.conflicts || []).some((c) => c.status !== 'RESOLVED') ? ' · có mâu thuẫn' : ''}
                  </td>
                  <td>{item.model || '—'}</td>
                  <td>{formatWhen(item.updated_at)}</td>
                  <td>
                    <Link to={adminDraftPath(item.id)}>Duyệt</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
      <p>
        <Link to={ROUTES.adminProcedures}>← Thủ tục</Link>
      </p>
    </div>
  )
}
