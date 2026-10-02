import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listProcedures } from '../../api/catalog.js'
import { listDocuments } from '../../api/documents.js'
import { useCommune } from '../../commune/CommuneContext.jsx'

export default function AdminDashboardPage() {
  const { xaId } = useCommune()
  const [docs, setDocs] = useState(null)
  const [active, setActive] = useState(null)
  const [apiError, setApiError] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await listDocuments({})
        if (!cancelled) setDocs(data?.count ?? 0)
      } catch (err) {
        if (!cancelled) setApiError(err.message || 'API tài liệu lỗi')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listProcedures({ xaId, citizen: false })
        if (!cancelled) setActive(data?.count ?? 0)
      } catch (err) {
        if (!cancelled) setApiError(err.message || 'API procedures lỗi')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Tổng quan</h1>
        <p>Upload PDF đã có. OCR, bản nháp và RAG chưa triển khai.</p>
      </header>

      {apiError ? <p className="form-error">{apiError}</p> : null}

      <div className="stat-grid">
        <div className="stat-card">
          <span>Tài liệu</span>
          <strong>{docs == null ? '…' : docs}</strong>
        </div>
        <div className="stat-card">
          <span>Bản nháp</span>
          <strong>chưa triển khai</strong>
        </div>
        <div className="stat-card">
          <span>Thủ tục ACTIVE</span>
          <strong>{active == null ? '…' : active}</strong>
        </div>
      </div>

      <section className="panel">
        <h2>Luồng làm việc</h2>
        <ol className="flow-list">
          <li>
            <Link to="/admin/documents">Upload PDF</Link> → lưu documents
          </li>
          <li>OCR và extract draft chưa triển khai</li>
          <li>
            <Link to="/admin/procedures">Thủ tục ACTIVE</Link>
          </li>
        </ol>
      </section>
    </div>
  )
}
