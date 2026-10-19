import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  activateProcedureVersion,
  createProcedureVersion,
  listActivationHistory,
  listAdminProcedures,
  listProcedureVersions,
} from '../../api/catalog.js'
import { useCommune } from '../../commune/CommuneContext.jsx'

function newRequestId() {
  return crypto.randomUUID()
}

const versionPattern = /^[0-9]+\.[0-9]+\.[0-9]+$/

export function createVersionErrorMessage(err) {
  const msg = String(err?.message || '')
  if (msg.includes('version must be n.n.n')) {
    return 'Số phiên bản phải có dạng n.n.n, ví dụ 2.0.0. Để trống nếu muốn hệ thống tự tăng.'
  }
  if (msg.includes('confirm is required')) {
    return 'Thiếu xác nhận tạo phiên bản.'
  }
  if (msg.includes('definition or copy_from_version_id is required')) {
    return 'Cần bản định nghĩa hoặc phiên bản nguồn để sao chép.'
  }
  if (msg.includes('version already exists')) {
    return 'Số phiên bản này đã tồn tại. Hãy nhập số khác.'
  }
  return msg || 'Không tạo được phiên bản.'
}

export function isUnknownActivationOutcome(err) {
  if (!err) return false
  if (err.status || err.code) return false
  const name = String(err.name || '')
  if (name === 'AbortError' || name === 'TimeoutError' || name === 'TypeError') return true
  const msg = String(err.message || '').toLowerCase()
  return msg.includes('failed to fetch') || msg.includes('network') || msg.includes('timeout') || msg.includes('abort')
}

function pillClass(status) {
  const u = String(status || '').toUpperCase()
  if (['READY', 'ACTIVE', 'PROCESSED', 'VALID'].includes(u)) return 'pill ok'
  if (['FAILED', 'EXPIRED', 'SUPERSEDED'].includes(u)) return 'pill bad'
  if (['PROCESSING', 'UPLOADED', 'APPROVED', 'INDEXING'].includes(u)) return 'pill warn'
  if (['PENDING', 'ARCHIVED'].includes(u)) return 'pill mute'
  return 'pill'
}

function asDefinition(raw) {
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

function formatWhen(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso || ''
  return d.toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'medium' })
}

function shortId(id) {
  const s = String(id || '')
  return s.length > 8 ? s.slice(0, 8) : s
}

function DefinitionPanel({ detail, onClose }) {
  const panelRef = useRef(null)
  useLayoutEffect(() => {
    const node = panelRef.current
    if (!node || typeof node.scrollIntoView !== 'function') return
    node.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [detail?.id, detail?.version])
  const def = asDefinition(detail.definition)
  const guidance = def.guidance || {}
  const slots = def.slots && typeof def.slots === 'object' && !Array.isArray(def.slots)
    ? Object.entries(def.slots)
    : []
  const checklist = Array.isArray(guidance.checklist) ? guidance.checklist : []
  const examples = Array.isArray(def.intent_examples) ? def.intent_examples : []
  const hasBody = def.name || def.description || guidance.summary || guidance.where_to_submit
    || checklist.length || slots.length || examples.length

  return (
    <div className="panel anchor-target" ref={panelRef}>
      <div className="panel-head">
        <h2>Hướng dẫn phiên {detail.version}</h2>
        <button type="button" className="linkish dark" onClick={onClose}>Đóng</button>
      </div>
      {hasBody ? (
        <div className="def-view">
          {def.name ? <h3>{def.name}</h3> : null}
          {def.description || guidance.summary ? (
            <p>{def.description || guidance.summary}</p>
          ) : null}
          {(guidance.where_to_submit || guidance.processing_time_hint) ? (
            <dl className="def-meta">
              {guidance.where_to_submit ? (
                <>
                  <dt>Nơi nộp</dt>
                  <dd>{guidance.where_to_submit}</dd>
                </>
              ) : null}
              {guidance.processing_time_hint ? (
                <>
                  <dt>Thời gian</dt>
                  <dd>{guidance.processing_time_hint}</dd>
                </>
              ) : null}
            </dl>
          ) : null}
          {checklist.length ? (
            <>
              <h4>Hồ sơ cần mang</h4>
              <ul className="def-list">
                {checklist.map((item, i) => <li key={i}>{item}</li>)}
              </ul>
            </>
          ) : null}
          {slots.length ? (
            <>
              <h4>Câu hỏi hệ thống sẽ hỏi</h4>
              <ul className="def-list">
                {slots.map(([key, slot]) => (
                  <li key={key}>
                    <strong>{slot?.question || key}</strong>
                    {slot?.description ? <div className="cell-muted">{slot.description}</div> : null}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
          {examples.length ? (
            <>
              <h4>Câu hỏi mẫu</h4>
              <ul className="def-list">
                {examples.map((ex, i) => <li key={i}>{ex}</li>)}
              </ul>
            </>
          ) : null}
        </div>
      ) : (
        <p className="muted">Phiên bản này chưa có nội dung hướng dẫn để hiển thị.</p>
      )}
    </div>
  )
}

const VERSION_STATUS_LABEL = {
  ACTIVE: 'Đang dùng',
  APPROVED: 'Đã duyệt',
  INDEXING: 'Đang lập chỉ mục',
  ARCHIVED: 'Lưu trữ',
}

const INDEX_STATUS_LABEL = {
  UPLOADED: 'Đã tải',
  PROCESSING: 'Đang xử lý',
  READY: 'Sẵn sàng',
  FAILED: 'Lỗi',
}

function statusLabel(map, status) {
  const code = String(status || '').toUpperCase()
  return map[code] || status || '—'
}

function reasonLabel(code) {
  const map = {
    VERSION_NOT_APPROVED: 'Phiên bản này chưa được duyệt.',
    VERSION_ARCHIVED: 'Phiên bản đã lưu trữ.',
    VERSION_ALREADY_ACTIVE: 'Phiên bản này đang được dùng.',
    INVALID_DEFINITION: 'Nội dung hướng dẫn của phiên bản không hợp lệ.',
    NO_SOURCE_DOCUMENT: 'Chưa gắn tài liệu nguồn.',
    PARTIAL_SOURCE_INDEX: 'Một số tài liệu gắn vào phiên bản chưa sẵn sàng.',
    DOCUMENT_INDEX_UPLOADED: 'Tài liệu mới gắn, chưa lập chỉ mục.',
    DOCUMENT_INDEX_PROCESSING: 'Tài liệu đang được lập chỉ mục.',
    DOCUMENT_INDEX_FAILED: 'Lập chỉ mục tài liệu thất bại.',
    DOCUMENT_INDEX_NOT_READY: 'Tài liệu chưa sẵn sàng.',
    ACTIVE_GENERATION_MISSING: 'Tài liệu chưa có bản chỉ mục hoàn chỉnh.',
    MOCK_INDEX_NOT_PUBLISHED: 'Chưa có nội dung chỉ mục để trả lời. Không kích hoạt được.',
    DOCUMENT_NOT_CURRENT: 'Tài liệu hết hiệu lực hoặc đã bị thay thế.',
    SOURCE_SCOPE_MISMATCH: 'Tài liệu không thuộc xã hoặc thủ tục này.',
    DEFINITION_IDENTITY_MISMATCH: 'Nội dung hướng dẫn không khớp thủ tục hoặc xã đang chọn. Tải lại trang rồi thử lại.',
    STALE_ACTIVE_VERSION: 'Phiên bản đang dùng đã đổi. Tải lại trang rồi thử lại.',
    IDEMPOTENCY_CONFLICT: 'Yêu cầu này đã được gửi với nội dung khác. Tải lại trang rồi thử lại.',
    ACTIVATION_CONFLICT: 'Không kích hoạt được vì còn phiên bản đang bật khác. Tải lại trang rồi thử lại.',
  }
  return map[code] || code
}

function activateHint(version) {
  const reasons = version?.eligibility?.reasons || []
  if (!reasons.length) return ''
  const docs = version.documents || []
  const failed = docs.filter((d) => String(d.index_status || '').toUpperCase() === 'FAILED')
  if (failed.length && (reasons.includes('DOCUMENT_INDEX_FAILED') || reasons.includes('PARTIAL_SOURCE_INDEX'))) {
    const names = failed.map((d) => `«${d.title}»`).join(', ')
    return `${names} lỗi. Bỏ liên kết hoặc thử lại lập chỉ mục rồi mới kích hoạt được.`
  }
  const waiting = docs.filter((d) => {
    const status = String(d.index_status || '').toUpperCase()
    return status && status !== 'READY'
  })
  if (waiting.length && (reasons.includes('PARTIAL_SOURCE_INDEX') || reasons.includes('DOCUMENT_INDEX_UPLOADED') || reasons.includes('DOCUMENT_INDEX_PROCESSING'))) {
    const names = waiting.map((d) => `«${d.title}»`).join(', ')
    return `${names} chưa sẵn sàng.`
  }
  return reasons.map(reasonLabel).join(' ')
}

export default function ProceduresPage() {
  const [params] = useSearchParams()
  const focusProc = params.get('proc')
  const focusApplied = useRef('')
  const { xaId, commune } = useCommune()
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [warning, setWarning] = useState('')
  const [selected, setSelected] = useState(null)
  const [versions, setVersions] = useState([])
  const [history, setHistory] = useState([])
  const [detail, setDetail] = useState(null)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const activateKeyRef = useRef('')
  const [confirmTarget, setConfirmTarget] = useState(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [createVersion, setCreateVersion] = useState('')
  const [createError, setCreateError] = useState('')
  const versionsRef = useRef(null)

  async function refreshList() {
    const data = await listAdminProcedures()
    setItems(data?.items || [])
  }

  useLayoutEffect(() => {
    if (!selected) return
    const node = versionsRef.current
    if (!node || typeof node.scrollIntoView !== 'function') return
    node.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [selected])

  async function openProcedure(proc) {
    if (selected?.id === proc.id) {
      const node = versionsRef.current
      if (node && typeof node.scrollIntoView === 'function') {
        node.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }
    setSelected(proc)
    setDetail(null)
    setError('')
    const [verData, hist] = await Promise.all([
      listProcedureVersions(proc.id),
      listActivationHistory(proc.id).catch(() => ({ items: [] })),
    ])
    setVersions(verData?.items || [])
    setHistory(hist?.items || [])
  }

  useEffect(() => {
    if (!focusProc || !items.length) return
    if (focusApplied.current === focusProc) return
    const found = items.find((p) => p.id === focusProc)
    if (!found) return
    focusApplied.current = focusProc
    openProcedure(found).catch((err) => setError(err.message))
  }, [focusProc, items])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      setError('')
      try {
        await refreshList()
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được thủ tục')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  async function confirmPublish() {
    if (!selected || !confirmTarget || busyRef.current) return
    busyRef.current = true
    setBusy(true)
    setError('')
    setNotice('')
    setWarning('')
    const key = activateKeyRef.current || newRequestId()
    activateKeyRef.current = key
    const expected = selected.active_version_id || null
    const procedureId = selected.id
    try {
      await activateProcedureVersion(procedureId, confirmTarget.id, expected, key)
    } catch (err) {
      if (isUnknownActivationOutcome(err)) {
        setError('Chưa rõ đã kích hoạt thành công chưa. Bấm Xác nhận lần nữa, hoặc tải lại trang để kiểm tra.')
      } else {
        activateKeyRef.current = ''
        setError(reasonLabel(err.code) || err.message || 'Kích hoạt thất bại')
      }
      busyRef.current = false
      setBusy(false)
      return
    }
    activateKeyRef.current = ''
    setConfirmTarget(null)
    setNotice('Đã chuyển sang phiên bản mới. Hội thoại mới sẽ dùng phiên này.')
    try {
      const data = await listAdminProcedures()
      setItems(data?.items || [])
      const latest = data?.items?.find((p) => p.id === procedureId)
      if (!latest) {
        setWarning('Phiên bản đã được kích hoạt thành công, nhưng chưa tải lại được dữ liệu mới. Vui lòng làm mới trang.')
        busyRef.current = false
        setBusy(false)
        return
      }
      setSelected(latest)
      const verData = await listProcedureVersions(latest.id)
      setVersions(verData?.items || [])
      try {
        const hist = await listActivationHistory(latest.id)
        setHistory(hist?.items || [])
      } catch {
        setWarning('Phiên bản đã được kích hoạt thành công, nhưng chưa tải lại được lịch sử. Vui lòng làm mới trang.')
      }
    } catch {
      setWarning('Phiên bản đã được kích hoạt thành công, nhưng chưa tải lại được dữ liệu mới. Vui lòng làm mới trang.')
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  async function submitCreate() {
    if (!selected || busyRef.current) return
    if (!selected.active_version_id) {
      setError('Thủ tục này chưa có phiên bản đang dùng nên chưa tạo được bản mới từ giao diện.')
      setCreateOpen(false)
      return
    }
    const label = createVersion.trim()
    if (label && !versionPattern.test(label)) {
      setCreateError('Số phiên bản phải có dạng n.n.n, ví dụ 2.0.0. Để trống nếu muốn hệ thống tự tăng.')
      return
    }
    busyRef.current = true
    setBusy(true)
    setError('')
    setCreateError('')
    try {
      await createProcedureVersion(
        selected.id,
        {
          version: createVersion.trim() || undefined,
          copy_from_version_id: selected.active_version_id,
          confirm: true,
        },
        newRequestId(),
      )
      setCreateOpen(false)
      setCreateVersion('')
      setCreateError('')
      setNotice('Đã tạo phiên bản mới. Gắn PDF và lập chỉ mục trước khi kích hoạt.')
      await openProcedure(selected)
      await refreshList()
    } catch (err) {
      setCreateError(createVersionErrorMessage(err))
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <div>
          <h1>Thủ tục</h1>
          <p>Xã {commune?.name || xaId} · chọn phiên bản công dân sẽ được hướng dẫn</p>
        </div>
      </header>

      {error ? <p className="form-error">{error}</p> : null}
      {warning ? <p className="banner warn">{warning}</p> : null}
      {notice ? <p className="banner ok">{notice}</p> : null}
      {loading ? <p className="muted">Đang tải…</p> : null}

      {!loading && items.length === 0 ? (
        <p className="muted">Chưa có thủ tục trong xã này.</p>
      ) : null}

      <div className="panel table-wrap">
        <table>
          <thead>
            <tr>
              <th>Mã</th>
              <th>Tên</th>
              <th>Đang dùng</th>
              <th>Số phiên bản</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((p) => (
              <tr key={p.id}>
                <td><code>{p.procedure_code}</code></td>
                <td>{p.name}</td>
                <td>
                  {p.active_version ? (
                    <span className="pill ok">{p.active_version}{p.legacy_active ? ' · seed' : ''}</span>
                  ) : (
                    <span className="pill warn">Chưa có</span>
                  )}
                </td>
                <td>{p.version_count}</td>
                <td>
                  <button type="button" onClick={() => openProcedure(p).catch((err) => setError(err.message))}>
                    Xem phiên bản
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected ? (
        <div className="panel anchor-target" ref={versionsRef}>
          <div className="panel-head procedure-detail-head">
            <h2>
              {selected.name} <code>{selected.procedure_code}</code>
            </h2>
            <button
              type="button"
              className="btn-primary"
              onClick={() => {
                setCreateError('')
                if (!selected.active_version_id) {
                  setError('Thủ tục này chưa có phiên bản đang dùng nên chưa tạo được bản mới từ giao diện.')
                  return
                }
                setCreateOpen(true)
              }}
              disabled={busy}
            >
              Tạo phiên bản mới
            </button>
          </div>
          <div className="table-wrap">
            <table className="version-table">
              <thead>
                <tr>
                  <th>Phiên bản</th>
                  <th>Trạng thái</th>
                  <th>Tài liệu</th>
                  <th>Kích hoạt</th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr key={v.id}>
                    <td>
                      {v.version}
                      {v.legacy_active ? <div className="cell-muted">Phiên đang dùng từ dữ liệu mẫu</div> : null}
                    </td>
                    <td><span className={pillClass(v.status)}>{statusLabel(VERSION_STATUS_LABEL, v.status)}</span></td>
                    <td>
                      {(v.documents || []).length === 0 ? '—' : v.documents.map((d) => (
                        <div key={d.document_id} className="cell-muted">
                          {d.title} · {statusLabel(INDEX_STATUS_LABEL, d.index_status)}
                        </div>
                      ))}
                    </td>
                    <td>
                      {v.eligibility?.can_activate ? (
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() => {
                            if (!activateKeyRef.current) activateKeyRef.current = newRequestId()
                            setConfirmTarget(v)
                          }}
                        >
                          Kích hoạt
                        </button>
                      ) : (
                        <span className="cell-muted">{activateHint(v)}</span>
                      )}
                      <div>
                        <button
                          type="button"
                          className="linkish dark"
                          onClick={() => setDetail(v)}
                        >
                          Xem hướng dẫn
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>Lịch sử kích hoạt</h3>
          {history.length === 0 ? <p className="muted">Chưa có lần kích hoạt ghi audit.</p> : (
            <ul className="history-list">
              {history.map((h) => {
                const next = versions.find((v) => v.id === h.new_active_version_id)
                const prev = versions.find((v) => v.id === h.previous_active_version_id)
                return (
                  <li key={h.request_id}>
                    <time dateTime={h.created_at}>{formatWhen(h.created_at)}</time>
                    <span>
                      Kích hoạt {next?.version || shortId(h.new_active_version_id)}
                      {prev || h.previous_active_version_id
                        ? ` · thay ${prev?.version || shortId(h.previous_active_version_id)}`
                        : ''}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      ) : null}

      {detail ? <DefinitionPanel detail={detail} onClose={() => setDetail(null)} /> : null}

      {confirmTarget ? (
        <div className="confirm-mask" role="dialog" aria-modal="true">
          <div className="panel modal-card">
            <h2>Chuyển phiên bản đang dùng?</h2>
            <p>
              Hiện đang dùng <strong>{selected?.active_version || 'chưa có'}</strong>.
              {' '}Sau khi xác nhận, hội thoại mới sẽ theo phiên <strong>{confirmTarget.version}</strong>.
              Các cuộc trò chuyện đang mở vẫn giữ phiên cũ.
            </p>
            <p className="muted">Nếu mạng bị gián đoạn, bấm Xác nhận lần nữa — hệ thống không kích hoạt trùng.</p>
            <div className="action-bar">
              <button type="button" disabled={busy} onClick={() => setConfirmTarget(null)}>Hủy</button>
              <button type="button" className="btn-primary" disabled={busy} onClick={confirmPublish}>
                {busy ? 'Đang kích hoạt…' : 'Xác nhận'}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {createOpen ? (
        <div className="confirm-mask" role="dialog" aria-modal="true">
          <div className="panel modal-card">
            <h2>Tạo phiên bản mới</h2>
            <p>Sao chép hướng dẫn từ phiên đang dùng. Phiên mới chỉ thay thế khi bạn kích hoạt.</p>
            {createError ? <p className="form-error" role="alert">{createError}</p> : null}
            <label>
              Số phiên bản (để trống thì hệ thống tự tăng)
              <input
                value={createVersion}
                onChange={(e) => {
                  setCreateVersion(e.target.value)
                  setCreateError('')
                }}
                placeholder="2.0.0"
                aria-invalid={createError ? 'true' : 'false'}
              />
            </label>
            <div className="action-bar">
              <button type="button" disabled={busy} onClick={() => { setCreateOpen(false); setCreateError('') }}>Hủy</button>
              <button type="button" className="btn-primary" disabled={busy} onClick={submitCreate}>Tạo phiên bản</button>
            </div>
          </div>
        </div>
      ) : null}

    </div>
  )
}
