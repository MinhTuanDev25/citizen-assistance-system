import { useEffect, useRef, useState } from 'react'
import {
  activateProcedureVersion,
  createProcedureVersion,
  getActiveVersion,
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
  if (['READY', 'ACTIVE', 'PROCESSED'].includes(u)) return 'pill ok'
  if (['FAILED', 'EXPIRED', 'SUPERSEDED'].includes(u)) return 'pill bad'
  if (['PROCESSING', 'UPLOADED', 'PENDING', 'APPROVED', 'INDEXING'].includes(u)) return 'pill warn'
  if (u === 'ARCHIVED') return 'pill mute'
  return 'pill'
}

function reasonLabel(code) {
  const map = {
    VERSION_NOT_APPROVED: 'Phiên bản chưa được xác nhận APPROVED.',
    VERSION_ARCHIVED: 'Phiên bản đã lưu trữ.',
    VERSION_ALREADY_ACTIVE: 'Phiên bản đang ACTIVE.',
    INVALID_DEFINITION: 'Definition không hợp lệ.',
    NO_SOURCE_DOCUMENT: 'Chưa gắn tài liệu nguồn.',
    PARTIAL_SOURCE_INDEX: 'Một phần tài liệu nguồn chưa index xong. Tất cả SOURCE phải READY.',
    DOCUMENT_INDEX_UPLOADED: 'Tài liệu mới gắn, chưa index.',
    DOCUMENT_INDEX_PROCESSING: 'Tài liệu đang index.',
    DOCUMENT_INDEX_FAILED: 'Index tài liệu thất bại.',
    DOCUMENT_INDEX_NOT_READY: 'Index chưa READY.',
    ACTIVE_GENERATION_MISSING: 'Thiếu generation READY đã publish.',
    MOCK_INDEX_NOT_PUBLISHED: 'Kết quả mock/index chưa có chunks và vectors đã publish. Không kích hoạt được.',
    DOCUMENT_NOT_CURRENT: 'Tài liệu hết hiệu lực hoặc đã bị thay thế.',
    SOURCE_SCOPE_MISMATCH: 'Tài liệu không cùng xã/domain/thủ tục.',
    DEFINITION_IDENTITY_MISMATCH: 'Definition không khớp mã thủ tục, xã, domain hoặc số version trong cơ sở dữ liệu.',
    STALE_ACTIVE_VERSION: 'Phiên bản ACTIVE đã đổi. Tải lại rồi thử lại.',
    IDEMPOTENCY_CONFLICT: 'X-Request-ID đã dùng với nội dung khác.',
    ACTIVATION_CONFLICT: 'Không kích hoạt được vì còn phiên bản đang bật khác. Tải lại trang rồi thử lại.',
  }
  return map[code] || code
}

export default function ProceduresPage() {
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

  async function refreshList() {
    const data = await listAdminProcedures()
    setItems(data?.items || [])
  }

  async function openProcedure(proc) {
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
        setError('Chưa xác định được kết quả kích hoạt. Giữ nguyên mã yêu cầu; bấm lại Xác nhận để gửi cùng X-Request-ID, hoặc làm mới trang để kiểm tra phiên ACTIVE.')
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
    setNotice('Đã kích hoạt phiên bản mới. Catalog công dân dùng phiên bản này cho hội thoại mới.')
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
      setError('Thủ tục này chưa có phiên bản ACTIVE. Giao diện hiện chỉ sao chép definition từ phiên ACTIVE. Tạo phiên bản đầu tiên nằm ở backlog.')
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
      setNotice('Đã tạo phiên bản APPROVED. Gắn PDF và lập chỉ mục trước khi kích hoạt.')
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
        <h1>Thủ tục</h1>
        <p>
          Xã {commune?.name || xaId}. Kích hoạt khi mọi tài liệu SOURCE đã READY.
          Phiên ACTIVE từ seed có thể chưa có PDF — không tự archive.
        </p>
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
              <th>ACTIVE</th>
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
                    <span className="pill warn">chưa ACTIVE</span>
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
        <div className="panel">
          <div className="admin-page-head procedure-detail-head">
            <h2>
              {selected.name} <code>{selected.procedure_code}</code>
            </h2>
            <button
              type="button"
              className="btn-primary"
              onClick={() => {
                setCreateError('')
                if (!selected.active_version_id) {
                  setError('Thủ tục này chưa có phiên bản ACTIVE. Giao diện hiện chỉ sao chép definition từ phiên ACTIVE.')
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
            <table>
              <thead>
                <tr>
                  <th>Version</th>
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
                      {v.legacy_active ? <div className="cell-muted">ACTIVE từ seed, chưa có audit kích hoạt</div> : null}
                    </td>
                    <td><span className={pillClass(v.status)}>{v.status}</span></td>
                    <td>
                      {(v.documents || []).length === 0 ? '—' : v.documents.map((d) => (
                        <div key={d.document_id} className="cell-muted">
                          {d.title} · {d.index_status}
                          {d.reasons?.length ? ` · ${d.reasons.join(', ')}` : ''}
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
                        <span className="cell-muted">
                          {(v.eligibility?.reasons || []).map(reasonLabel).join(' ')}
                        </span>
                      )}
                      <div>
                        <button
                          type="button"
                          className="linkish dark"
                          onClick={() => setDetail(v)}
                        >
                          Definition
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
            <ul>
              {history.map((h) => (
                <li key={h.request_id}>
                  {h.created_at} · {h.new_active_version_id}
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : null}

      {detail ? (
        <div className="panel">
          <div className="admin-page-head">
            <h2>{detail.version}</h2>
            <button type="button" className="linkish dark" onClick={() => setDetail(null)}>Đóng</button>
          </div>
          <pre className="json-block">{JSON.stringify(detail.definition, null, 2)}</pre>
        </div>
      ) : null}

      {confirmTarget ? (
        <div className="confirm-mask" role="dialog" aria-modal="true">
          <div className="panel modal-card">
            <h2>Xác nhận kích hoạt</h2>
            <p>
              Phiên bản chính thức hiện tại: {selected?.active_version || 'chưa có'}.
              Sẽ chuyển sang {confirmTarget.version}. Hội thoại mới dùng phiên bản này.
              Hội thoại đã ghim phiên bản cũ giữ nguyên.
              Nếu mạng lỗi, bấm lại Xác nhận sẽ gửi cùng X-Request-ID.
            </p>
            <div className="action-bar">
              <button type="button" className="btn-primary" disabled={busy} onClick={confirmPublish}>
                {busy ? 'Đang kích hoạt…' : 'Xác nhận kích hoạt'}
              </button>
              <button type="button" disabled={busy} onClick={() => setConfirmTarget(null)}>Hủy</button>
            </div>
          </div>
        </div>
      ) : null}

      {createOpen ? (
        <div className="confirm-mask" role="dialog" aria-modal="true">
          <div className="panel modal-card">
            <h2>Tạo phiên bản APPROVED</h2>
            <p>Sao chép definition từ phiên ACTIVE hiện tại. Chưa ACTIVE cho đến khi bạn kích hoạt.</p>
            {createError ? <p className="form-error" role="alert">{createError}</p> : null}
            <label>
              Số version (trống = tự tăng)
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
              <button type="button" className="btn-primary" disabled={busy} onClick={submitCreate}>Tạo candidate</button>
              <button type="button" disabled={busy} onClick={() => { setCreateOpen(false); setCreateError('') }}>Hủy</button>
            </div>
          </div>
        </div>
      ) : null}

      <p className="muted">
        Public catalog vẫn là <button type="button" className="linkish dark" onClick={() => selected && getActiveVersion(selected.id).then(setDetail)}>GET active-version</button>.
      </p>
    </div>
  )
}
