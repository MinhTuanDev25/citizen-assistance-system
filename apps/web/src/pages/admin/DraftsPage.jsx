import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { listAdminProcedures } from '../../api/catalog.js'
import { generateAIDraft, listDraftSources, listProcedureDrafts } from '../../api/drafts.js'
import { adminDraftPath, ROUTES } from '../../routes.js'
import { useCommune } from '../../commune/CommuneContext.jsx'

function newRequestId() {
  return crypto.randomUUID()
}

function formatWhen(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso || ''
  return d.toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'medium' })
}

function pill(status) {
  const u = String(status || '').toUpperCase()
  if (u === 'APPROVED' || u === 'PUBLISHED') return 'pill ok'
  if (u === 'REJECTED') return 'pill bad'
  if (u === 'REVIEWED') return 'pill warn'
  return 'pill mute'
}

export default function DraftsPage() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { commune } = useCommune()
  const focusProc = params.get('proc')
  const [procedures, setProcedures] = useState([])
  const [procedureId, setProcedureId] = useState(focusProc || '')
  const [sources, setSources] = useState([])
  const [sourceKey, setSourceKey] = useState('')
  const [drafts, setDrafts] = useState([])
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await listAdminProcedures()
        if (cancelled) return
        const items = data?.items || []
        setProcedures(items)
        if (!procedureId && items[0]) {
          const birth = items.find((item) => item.procedure_code === 'dk_khai_sinh')
          setProcedureId(focusProc || birth?.id || items[0].id)
        }
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

  useEffect(() => {
    if (!procedureId) return
    let cancelled = false
    setError('')
    ;(async () => {
      try {
        const [src, list] = await Promise.all([
          listDraftSources(procedureId),
          listProcedureDrafts(procedureId),
        ])
        if (cancelled) return
        const sourceItems = src?.items || []
        setSources(sourceItems)
        setDrafts(list?.items || [])
        if (sourceItems[0]) {
          setSourceKey(`${sourceItems[0].document_id}:${sourceItems[0].generation_id}`)
        } else {
          setSourceKey('')
        }
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được bản nháp')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [procedureId])

  async function onGenerate() {
    if (!procedureId || generating) return
    const source = sources.find((item) => `${item.document_id}:${item.generation_id}` === sourceKey)
    if (!source) {
      setError('Chọn nguồn đã xác minh trước khi tạo bản nháp.')
      return
    }
    setGenerating(true)
    setError('')
    setNotice('')
    try {
      const view = await generateAIDraft(
        procedureId,
        {
          document_id: source.document_id,
          generation_id: source.generation_id,
          source_page_start: source.suggested_page_start || 10,
          source_page_end: source.suggested_page_end || 21,
        },
        newRequestId(),
      )
      setNotice('Đã lưu bản nháp. Gemini không duyệt và không kích hoạt.')
      navigate(adminDraftPath(view.id))
    } catch (err) {
      setError(err.message || 'Không tạo được bản nháp AI')
    } finally {
      setGenerating(false)
    }
  }

  const selected = procedures.find((item) => item.id === procedureId)

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <div>
          <h1>Bản nháp định nghĩa AI</h1>
          <p>Xã {commune?.name || ''} · soạn thảo có nguồn, Admin mới được duyệt</p>
        </div>
      </header>
      {error ? <p className="form-error">{error}</p> : null}
      {notice ? <p className="banner ok">{notice}</p> : null}
      {loading ? <p className="muted">Đang tải…</p> : null}

      <div className="panel draft-generate">
        <h2>Tạo bản nháp</h2>
        <label>
          Thủ tục
          <select value={procedureId} onChange={(e) => setProcedureId(e.target.value)} disabled={generating}>
            {procedures.map((item) => (
              <option key={item.id} value={item.id}>
                {item.procedure_code} — {item.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Nguồn đã xác minh
          <select value={sourceKey} onChange={(e) => setSourceKey(e.target.value)} disabled={generating}>
            {sources.length === 0 ? <option value="">Không có generation READY</option> : null}
            {sources.map((item) => (
              <option key={`${item.document_id}:${item.generation_id}`} value={`${item.document_id}:${item.generation_id}`}>
                {item.title || item.document_id} · v{item.version} ({item.version_status}) · trang {item.suggested_page_start}–{item.suggested_page_end}
              </option>
            ))}
          </select>
        </label>
        <p className="muted">
          Nguồn lịch sử (ví dụ G3 trên phiên bản ARCHIVED) chỉ dùng làm INPUT. Hệ thống không sao chép generation sang phiên bản mới.
        </p>
        <button type="button" className="btn-primary" onClick={onGenerate} disabled={generating || !sourceKey}>
          {generating ? 'Đang gọi Gemini… có thể 30–90 giây' : 'Generate AI Draft'}
        </button>
      </div>

      <div className="panel table-wrap">
        <div className="panel-head">
          <h2>Bản nháp {selected ? <code>{selected.procedure_code}</code> : null}</h2>
        </div>
        {drafts.length === 0 ? <p className="muted">Chưa có bản nháp cho thủ tục này.</p> : null}
        {drafts.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Trạng thái</th>
                <th>Mô hình</th>
                <th>Hợp lệ</th>
                <th>Nguồn</th>
                <th>Cập nhật</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {drafts.map((item) => (
                <tr key={item.id}>
                  <td><code>{String(item.id).slice(0, 8)}</code></td>
                  <td><span className={pill(item.status)}>{item.status}</span></td>
                  <td>{item.model || '—'}</td>
                  <td>
                    {item.validation?.schema_valid ? 'schema' : 'invalid'}
                    {item.validation?.source_coverage_complete ? ' · coverage' : ' · thiếu nguồn'}
                  </td>
                  <td><code>{String(item.document_id || '').slice(0, 8)}</code></td>
                  <td>{formatWhen(item.updated_at)}</td>
                  <td>
                    <Link to={adminDraftPath(item.id)}>Review</Link>
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
